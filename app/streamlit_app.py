"""Streamlit app: build a paper, add a topic, and look inside the model."""
import itertools
import json
import os
import subprocess
import tempfile
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd
import streamlit as st

from qpgen.config import DEFAULT_CONFIG, ROOT, load_config
from qpgen.data.labels import bloom_level
from qpgen.export import WRITERS, section_title
from qpgen.filters import filter_candidates, generate_questions, load_context, score
from qpgen.paper import (build_paper, check_paper, co_marks, fill_section, preset_rows,
                         replace_question, spec_from_pattern)
from qpgen.topics import check_new_topic, question_rows, retrain_commands, save_new_topic
from qpgen.upload import analyze_upload, read_upload

CONFIG = os.environ.get("QPGEN_CONFIG", str(DEFAULT_CONFIG))
BADGES = {"generated": "generated", "from_bank": "from bank", "from_notes": "from notes"}
PRESETS = {"Balanced": "balanced", "Easy-heavy": "easy_heavy", "Hard-heavy": "hard_heavy"}
CREATIVITY = {"Safe": 0.6, "Balanced": 0.8, "Creative": 1.0}   # sampling temperature
LEVELS = ["auto", "easy", "medium", "hard"]

st.set_page_config(page_title="Question Paper Generator", layout="wide")
# Wide enough for all four columns of the pattern table; set_page_config has no sidebar width option.
st.markdown('<style>[data-testid="stSidebar"] {min-width: 440px;}</style>', unsafe_allow_html=True)
cfg = load_config(CONFIG)


@st.cache_resource
def get_context():
    """Load the model, scorer and question bank once per server."""
    return load_context(CONFIG)


def uid() -> int:
    """A fresh id for each question shown, so its widgets reset when it is replaced."""
    st.session_state.setdefault("ids", itertools.count())
    return next(st.session_state["ids"])


def topic_inputs(ctx) -> dict | None:
    """Subject and topics, picked by hand or detected from uploaded notes."""
    names = [info["name"] for info in cfg.subjects.values()]
    if st.radio("Build from", ["Subject and topics", "Uploaded notes"], horizontal=True) == "Subject and topics":
        subject = st.selectbox("Subject", names)
        all_topics = trained_topics(ctx, subject)
        return {"subject": subject, "topics": st.multiselect("Topics", all_topics, default=all_topics),
                "weights": {}, "upload": None}
    file = st.file_uploader("Notes (PDF, TXT or MD)", type=["pdf", "txt", "md"])
    typed = st.text_area("Or type a chapter name, or paste notes", height=100)
    text = read_upload(file.getvalue(), file.name) if file else typed.strip()
    if not text:
        return None
    found = analyze_upload(text, cfg, ctx.scorer)
    if not found["supported"]:
        st.warning(found["message"])
        return None
    st.success(f"Detected {found['subject']}")
    all_topics = trained_topics(ctx, found["subject"])
    topics = st.multiselect("Topics", all_topics, default=list(found["topics"]))
    return {"subject": found["subject"], "topics": topics,
            "weights": {t: w for t, w in found["topics"].items() if t in topics},
            "upload": {"text": text, "key_terms": found["key_terms"]}}


def trained_topics(ctx, subject: str) -> list[str]:
    """Topics of the subject the loaded model knows, with a note about any still waiting for a retrain."""
    topics = cfg.subjects[cfg.subject_key(subject)]["topics"]
    waiting = [t for t in topics if t not in ctx.scorer.topics]
    if waiting:
        st.caption(f"Needs a retrain before use: {', '.join(waiting)}")
    return [t for t in topics if t in ctx.scorer.topics]


def pattern_input() -> list[dict]:
    """Editable table of marks, questions printed, questions to attempt and difficulty."""
    st.subheader("Paper pattern")
    preset = PRESETS[st.selectbox("Start from", list(PRESETS), help="Fills the table below. Edit any row, "
                                  "or add rows for other mark values.")]
    rows = pd.DataFrame([{**r, "difficulty": "auto"} for r in preset_rows(cfg, preset)])
    edited = st.data_editor(
        rows, num_rows="dynamic", hide_index=True, key=f"pattern_{preset}", width="stretch",
        column_config={
            "marks": st.column_config.NumberColumn("Marks", min_value=1, max_value=50, step=1, required=True),
            "count": st.column_config.NumberColumn("Questions", min_value=0, max_value=50, step=1, required=True,
                                                   help="How many questions are printed"),
            "attempt": st.column_config.NumberColumn("Attempt", min_value=0, max_value=50, step=1,
                                                     help="How many must be answered. Leave empty for all."),
            "difficulty": st.column_config.SelectboxColumn("Difficulty", options=LEVELS, default="auto",
                                                           help="auto picks it from the marks"),
        })
    return [{"marks": int(r["marks"]), "count": int(r["count"]),
             "attempt": 0 if pd.isna(r["attempt"]) else int(r["attempt"]),
             "difficulty": None if r["difficulty"] in (None, "auto") or pd.isna(r["difficulty"]) else r["difficulty"]}
            for r in edited.to_dict("records") if not (pd.isna(r["marks"]) or pd.isna(r["count"]))]


def co_input(picked: dict) -> dict[str, str]:
    """Course outcome for each chosen topic, defaulting to the subject's map in config."""
    default = cfg.subjects[cfg.subject_key(picked["subject"])].get("co", {})
    rows = pd.DataFrame([{"topic": t, "co": default.get(t, "")} for t in picked["topics"]])
    with st.expander("Course outcomes (CO)"):
        edited = st.data_editor(rows, hide_index=True, disabled=["topic"], width="stretch",
                                key=f"co_{picked['subject']}_{len(picked['topics'])}",
                                column_config={"topic": "Topic", "co": "CO"})
    return dict(zip(edited["topic"], edited["co"].fillna("").astype(str)))


def sidebar(ctx) -> None:
    """All paper settings and the Generate button."""
    with st.sidebar:
        st.header("Paper settings")
        picked = topic_inputs(ctx)
        co_map = co_input(picked) if picked and picked["topics"] else {}
        rows = pattern_input()
        try:
            spec = spec_from_pattern(cfg, next(iter(cfg.subjects.values()))["name"], [], rows)   # topics not needed for totals
            levels = ", ".join(f"{s.marks}m {s.difficulty}" for s in spec.sections)
            st.markdown(f"**Total: {spec.total_marks} marks** ({sum(s.count for s in spec.sections)} questions printed)")
            st.caption(f"Difficulty used: {levels}")
            pattern_ok = True
        except ValueError as e:
            st.error(str(e))
            pattern_ok = False
        st.subheader("Details")
        duration = st.number_input("Duration (minutes)", 30, 300, cfg.paper["duration_minutes"], step=15)
        institute = st.text_input("Institute name", cfg.paper["institute"])
        title = st.text_input("Exam title", cfg.paper["exam_title"])
        seed = int(st.number_input("Paper variant #", 0, 10**6, cfg.seed,
                                   help="The same number with the same settings gives the same paper. "
                                        "Change it for a different set, for example Set A = 1, Set B = 2."))
        creativity = st.select_slider("Creativity", list(CREATIVITY), value="Balanced",
                                      help="Safe keeps wording close to real questions and on topic. "
                                           "Creative writes newer questions but makes more mistakes.")
        ready = bool(picked and picked["topics"]) and pattern_ok
        if st.button("Generate paper", type="primary", disabled=not ready, width="stretch"):
            make_paper(ctx, picked, rows, co_map, seed, CREATIVITY[creativity],
                       institute=institute, exam_title=title, duration_minutes=int(duration))


def make_paper(ctx, picked: dict, rows: list[dict], co_map: dict, seed: int, temperature: float,
               **overrides) -> None:
    """Build the paper and keep it, with its generator, in session state."""
    gen = partial(generate_questions, ctx=ctx, upload=picked["upload"], temperature=temperature)
    spec = spec_from_pattern(cfg, picked["subject"], picked["topics"], rows, picked["weights"], co_map, **overrides)
    with st.spinner("Generating questions..."):
        paper = build_paper(spec, seed, cfg, gen)
    for s in paper["sections"]:
        for q in s["questions"]:
            q["uid"] = uid()
    st.session_state.update(paper=paper, gen=gen, seed=seed)


def clean(paper: dict) -> dict:
    """The paper without the app's widget ids, ready for export."""
    return {**paper, "sections": [{**s, "questions": [{k: v for k, v in q.items() if k != "uid"}
                                                        for q in s["questions"]]} for s in paper["sections"]]}


def downloads(paper: dict) -> None:
    """DOCX, PDF and JSON download buttons, built from the current (possibly edited) paper."""
    stem = f"{datetime.now():%Y%m%d_%H%M}_{cfg.subject_key(paper['spec'].subject)}"
    cols = st.columns(3)
    with tempfile.TemporaryDirectory() as tmp:
        for col, (fmt, write) in zip(cols, WRITERS.items()):
            data = write(clean(paper), Path(tmp) / f"{stem}.{fmt}").read_bytes()
            col.download_button(f"Download {fmt.upper()}", data, f"{stem}.{fmt}", width="stretch")


def question_row(paper: dict, s_idx: int, q_idx: int, number: int) -> None:
    """One question with its badge, scores, edit box and buttons."""
    q = paper["sections"][s_idx]["questions"][q_idx]
    text_col, act_col = st.columns([5, 1])
    new = text_col.text_input(f"Q{number}  [{q['marks']} marks]", q["text"], key=f"text_{q['uid']}")
    if new.strip() and new != q["text"]:
        q["text"], q["bloom"] = new.strip(), bloom_level(new)
    rel = "" if q.get("relevance") is None else f"relevance {q['relevance']:.2f}"
    prob = "" if q.get("difficulty_prob") is None else f"difficulty {q['difficulty_prob']:.2f}"
    text_col.caption(f"`{BADGES.get(q.get('origin'), q.get('origin'))}`  {q['topic']}  "
                     f"**{q.get('co', '')}**  **{q.get('bloom', '')}**  {rel}  {prob}")
    if act_col.button("Regenerate", key=f"regen_{q['uid']}", width="stretch"):
        st.session_state["seed"] += 7919
        if replace_question(paper, cfg, st.session_state["gen"], s_idx, q_idx, st.session_state["seed"]):
            paper["sections"][s_idx]["questions"][q_idx]["uid"] = uid()
        st.rerun()
    if act_col.button("Remove", key=f"remove_{q['uid']}", width="stretch"):
        paper["sections"][s_idx]["questions"].pop(q_idx)
        st.rerun()


def paper_tab() -> None:
    """Preview grouped by section, with per-question actions and downloads."""
    paper = st.session_state.get("paper")
    if not paper:
        st.info("Pick a subject and topics, or upload notes, then press Generate paper.")
        return
    spec = paper["spec"]
    st.subheader(f"{spec.institute}: {spec.exam_title}, {spec.subject}")
    number = 0
    for s_idx, (sec_spec, sec) in enumerate(zip(spec.sections, paper["sections"])):
        st.markdown(f"#### {section_title(sec)}")
        for q_idx in range(len(sec["questions"])):
            number += 1
            question_row(paper, s_idx, q_idx, number)
        if len(sec["questions"]) < sec_spec.count and st.button(f"Fill section {sec['name']}", key=f"fill_{s_idx}"):
            st.session_state["seed"] += 104729
            fill_section(paper, cfg, st.session_state["gen"], s_idx, st.session_state["seed"])
            for q in sec["questions"]:
                q.setdefault("uid", uid())
            st.rerun()
    st.caption("Marks set per CO: " + ",  ".join(f"{co or 'none'} {m}" for co, m in co_marks(paper).items())
               + ".  BL1 Remember, BL2 Understand, BL3 Apply, BL4 Analyze, BL5 Evaluate, BL6 Create.")
    try:
        check_paper(paper)
    except ValueError as e:
        st.warning(f"{e}. Fill the short sections before downloading.")
        return
    downloads(paper)


def try_tab(ctx) -> None:
    """Raw samples next to the ones that pass, with the reason for each rejection."""
    st.caption("What the LSTM writes before filtering, and why each rejected question was dropped.")
    names = [info["name"] for info in cfg.subjects.values()]
    c1, c2, c3, c4 = st.columns(4)
    subject = c1.selectbox("Subject", names, key="try_subject")
    topic = c2.selectbox("Topic", [t for t in cfg.subjects[cfg.subject_key(subject)]["topics"]
                                   if t in ctx.scorer.topics], key="try_topic")
    level = c3.selectbox("Difficulty", list(cfg.marks), key="try_level")
    n = c4.number_input("Samples", 5, 100, 20, key="try_n")
    if st.button("Sample"):
        raw = score(ctx.sampler(subject, topic, level, int(n), seed=uid()), ctx, topic, level)
        st.session_state["trial"] = filter_candidates(raw, ctx, topic)
    if "trial" in st.session_state:      # kept in state so the result survives later reruns
        passed, rejected = st.session_state["trial"]
        left, right = st.columns(2)
        left.markdown(f"**Passed ({len(passed)})**")
        left.dataframe(pd.DataFrame([{"question": c["text"], "relevance": round(c["relevance"], 2),
                                      "difficulty": round(c["difficulty_prob"], 2)} for c in passed]))
        right.markdown(f"**Rejected ({len(rejected)})**")
        right.dataframe(pd.DataFrame([{"question": c["text"], "reason": c["reason"]} for c in rejected]))


def report_tab() -> None:
    """Loss curve, metrics and baseline comparison from reports/."""
    reports = cfg.path("reports_dir")
    if (reports / "loss_curve.png").exists():
        st.image(str(reports / "loss_curve.png"), caption="Training and validation loss", width=600)
    if (reports / "metrics.md").exists():
        st.markdown((reports / "metrics.md").read_text(encoding="utf-8"))
    else:
        st.info("No metrics yet. Run `make eval` to create reports/metrics.md.")
    for name in ("baseline.json", "classifier_accuracy.json"):
        if (reports / name).exists():
            st.markdown(f"**{name}**")
            st.json(json.loads((reports / name).read_text(encoding="utf-8")), expanded=False)


def add_topic_page() -> None:
    """Paste questions for a new topic, save them, then retrain so the model learns it."""
    st.header("Add a topic")
    st.write("The model only writes questions for topics it was trained on. To add one, paste at least "
             f"{cfg.data['new_topic_min']} questions on it (one per line, with at least "
             f"{cfg.data['new_topic_min_per_level']} each of easy, medium and hard), save, then retrain.")
    names = [info["name"] for info in cfg.subjects.values()]
    c1, c2, c3 = st.columns([2, 3, 1])
    subject = c1.selectbox("Subject", names, key="new_subject")
    topic = c2.text_input("New topic name", key="new_topic")
    cos = sorted(set(cfg.subjects[cfg.subject_key(subject)].get("co", {}).values())) or ["CO1"]
    co = c3.text_input("CO", cos[-1], key=f"new_co_{subject}")
    rows = question_rows(st.text_area("Questions, one per line", height=250, key="new_questions"))
    if rows:
        st.caption("Difficulty is guessed from the opening verb. Fix any that are wrong.")
        edited = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch", key="new_rows",
                                column_config={"question": st.column_config.TextColumn("Question", disabled=True),
                                               "difficulty": st.column_config.SelectboxColumn(
                                                   "Difficulty", options=LEVELS[1:], required=True)})
        rows = edited.to_dict("records")
        st.caption("  ".join(f"{lv}: {sum(r['difficulty'] == lv for r in rows)}" for lv in LEVELS[1:]))
    errors = check_new_topic(cfg, subject, topic, rows) if topic.strip() and rows else []
    for e in errors:
        st.warning(e)
    if st.button("Save topic", type="primary", disabled=bool(errors) or not rows or not topic.strip()):
        save_new_topic(cfg, subject, topic, co.strip() or cos[-1], rows)
        st.success(f"Saved {len(rows)} questions for {topic.strip()}. Retrain below to start using it.")

    st.subheader("Retrain")
    st.write("Rebuilds the question bank, retrains the LSTM and refits the classifiers: about a minute on a GPU, "
             "4 to 5 on a CPU. This overwrites the model and the training reports in `reports/`.")
    if st.button("Retrain the model"):
        with st.status("Retraining...", expanded=True) as status:
            for cmd in retrain_commands(CONFIG):
                st.write(f"Running `{cmd[2]}`")
                done = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
                if done.returncode:
                    status.update(label=f"{cmd[2]} failed", state="error")
                    st.code(done.stderr[-3000:])
                    return
            status.update(label="Retrained. New topics are ready on Make a paper.", state="complete")
        get_context.clear()


def insights_page() -> None:
    """Raw model output and the evaluation report, for checking how the model behaves."""
    st.header("Model insights")
    trial, report = st.tabs(["Try the model", "Report"], key="insights_tab")
    with trial:
        try_tab(get_context())
    with report:
        report_tab()


def paper_page() -> None:
    """Settings in the sidebar, the paper in the main area."""
    sidebar(get_context())
    paper_tab()


def main() -> None:
    """Stop with a clear message when the model is missing, otherwise show the three pages."""
    st.title("Question Paper Generator")
    missing = [p for p in ("checkpoint", "classifiers") if not cfg.path(p).exists()]
    if missing:
        st.error("No trained model found. Run `make train` first, then reload this page.")
        st.stop()
    st.navigation([st.Page(paper_page, title="Make a paper", default=True),
                   st.Page(add_topic_page, title="Add a topic"),
                   st.Page(insights_page, title="Model insights")], position="top").run()


main()
