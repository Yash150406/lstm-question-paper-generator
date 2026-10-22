"""Streamlit app: build a paper, add a topic, and look inside the model."""
import html
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
from qpgen.export import WRITERS, header_lines, section_title
from qpgen.filters import filter_candidates, generate_questions, load_context, score
from qpgen.paper import (build_paper, check_paper, co_marks, fill_section, preset_rows,
                         replace_question, spec_from_pattern)
from qpgen.topics import check_new_topic, question_rows, retrain_commands, save_new_topic
from qpgen.upload import analyze_upload, read_upload

CONFIG = os.environ.get("QPGEN_CONFIG", str(DEFAULT_CONFIG))
ORIGINS = {"generated": "Written by the model", "from_bank": "Taken from the question bank",
           "from_notes": "Built from your notes"}
TAGS = {"from_bank": "bank", "from_notes": "notes"}      # shown on the sheet; model questions get no tag
STEPS = ["Subject", "Pattern", "Review", "Download"]
SERIF = "'Times New Roman', Georgia, serif"
STYLE = f"""<style>
@media (min-width: 900px) {{ [data-testid="stSidebar"] {{ min-width: 440px; }} }}
[data-testid="stMainBlockContainer"] {{ max-width: 960px; }}
h1 {{ color: #1f3a68; font-size: 1.15rem !important; font-weight: 600 !important; padding-bottom: .25rem !important; }}
.steps {{ display: flex; gap: .75rem; margin: 0 0 1.75rem; }}
.step {{ flex: 1; border-bottom: 3px solid #c9ced8; color: #6b7385; padding: .4rem 0; font-size: .92rem; }}
.step.done {{ border-color: #1f3a68; color: #1f3a68; }}
.step.now {{ border-color: #1f3a68; color: #1d2433; font-weight: 600; }}
.num {{ display: inline-block; width: 1.45rem; height: 1.45rem; line-height: 1.3rem; margin-right: .45rem;
        border: 1.5px solid currentColor; border-radius: 50%; text-align: center; font-size: .78rem; }}
.step.done .num {{ background: #1f3a68; color: #fff; border-color: #1f3a68; }}
.sidehead {{ font-weight: 600; color: #1f3a68; margin: .9rem 0 .1rem; }}
.st-key-sheet {{ background: #fff; padding: 3rem 3.5rem 3.25rem;
                 box-shadow: 0 1px 2px rgba(29,36,51,.08), 0 10px 30px rgba(29,36,51,.10); }}
.st-key-sheet [data-testid="stVerticalBlock"] {{ gap: .2rem; }}
.st-key-sheet p, .st-key-sheet input, .sheet, .q, .marks {{ font-family: {SERIF}; color: #1d2433; font-size: 1.03rem; }}
.sheet-head {{ text-align: center; border-bottom: 1px solid #1d2433; padding-bottom: .7rem; margin-bottom: .6rem; }}
.sheet-head .inst {{ font-size: 1.45rem; font-weight: 700; }}
.sheet-head .exam {{ font-weight: 700; font-size: 1.1rem; }}
.sheet-head .meta {{ font-size: .95rem; }}
.sec {{ font-weight: 700; font-style: italic; margin: 1.3rem 0 .15rem; }}
.sec-note {{ font-style: italic; font-size: .95rem; margin-bottom: .3rem; }}
.q {{ line-height: 1.45; }}
.marks {{ color: #4a5263; white-space: nowrap; text-align: right; font-size: .95rem; }}
.tag {{ font-family: sans-serif; font-size: .68rem; color: #6b7385; border: 1px solid #c9ced8;
        border-radius: 3px; padding: 0 .3rem; margin-left: .45rem; vertical-align: .15em; }}
.blank {{ color: #6b7385; text-align: center; padding: 4rem 0 3rem; font-style: italic; }}
.st-key-sheet button[kind="tertiary"] {{ color: #1f3a68; min-height: 1.7rem; padding: 0 .35rem; margin-top: .4rem; }}
.st-key-sheet [class*="st-key-del_"] button[kind="tertiary"] {{ color: #a3342b; }}
@media (max-width: 640px) {{ .st-key-sheet {{ padding: 1.25rem 1rem; }} .steps {{ font-size: .8rem; }} }}
</style>"""
PRESETS = {"Balanced": "balanced", "Easy-heavy": "easy_heavy", "Hard-heavy": "hard_heavy"}
CREATIVITY = {"Safe": 0.6, "Balanced": 0.8, "Creative": 1.0}   # sampling temperature
LEVELS = ["auto", "easy", "medium", "hard"]

st.set_page_config(page_title="Question Paper Generator", layout="wide")
# Theme colours are in .streamlit/config.toml. This adds what the theme cannot: the sidebar width
# (wide enough for all four pattern columns), the step bar and the paper sheet.
st.markdown(STYLE, unsafe_allow_html=True)
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


def side_heading(n: int, name: str) -> None:
    """A sidebar heading numbered like the step bar."""
    st.markdown(f'<div class="sidehead"><span class="num">{n}</span>{name}</div>', unsafe_allow_html=True)


def pattern_input() -> list[dict]:
    """Editable table of marks, questions printed, questions to attempt and difficulty."""
    side_heading(2, "Pattern")
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


def sidebar(ctx) -> tuple[bool, bool]:
    """All paper settings and the Generate button. Returns whether topics and the pattern are ready."""
    with st.sidebar:
        side_heading(1, "Subject")
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
        st.markdown('<div class="sidehead">Details</div>', unsafe_allow_html=True)
        duration = st.number_input("Duration (minutes)", 30, 300, cfg.paper["duration_minutes"], step=15)
        institute = st.text_input("Institute name", cfg.paper["institute"])
        title = st.text_input("Exam title", cfg.paper["exam_title"])
        seed = int(st.number_input("Paper variant #", 0, 10**6, cfg.seed,
                                   help="The same number with the same settings gives the same paper. "
                                        "Change it for a different set, for example Set A = 1, Set B = 2."))
        creativity = st.select_slider("Creativity", list(CREATIVITY), value="Balanced",
                                      help="Safe keeps wording close to real questions and on topic. "
                                           "Creative writes newer questions but makes more mistakes.")
        topics_ok = bool(picked and picked["topics"])
        if st.button("Generate paper", type="primary", disabled=not (topics_ok and pattern_ok), width="stretch"):
            make_paper(ctx, picked, rows, co_map, seed, CREATIVITY[creativity],
                       institute=institute, exam_title=title, duration_minutes=int(duration))
    return topics_ok, pattern_ok


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
    st.session_state.update(paper=paper, gen=gen, seed=seed, editing=set(), downloaded=False)


def changed() -> None:
    """The paper was edited, so an earlier download no longer matches it."""
    st.session_state["downloaded"] = False


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
            col.download_button(f"Download {fmt.upper()}", data, f"{stem}.{fmt}", width="stretch",
                                on_click=lambda: st.session_state.update(downloaded=True))


def question_row(paper: dict, s_idx: int, q_idx: int, number: int) -> None:
    """One line of the sheet: the question, its CO, BL and marks, and regenerate, edit and remove buttons."""
    q = paper["sections"][s_idx]["questions"][q_idx]
    editing = st.session_state.setdefault("editing", set())
    text, marks, regen, edit, drop = st.columns([16, 3.2, 1, 1, 1], gap="small", vertical_alignment="center")
    if q["uid"] in editing:
        new = text.text_input(f"Question {number}", q["text"], key=f"text_{q['uid']}", label_visibility="collapsed")
    else:
        tag = f'<span class="tag">{TAGS[q["origin"]]}</span>' if q.get("origin") in TAGS else ""
        text.markdown(f'<div class="q">{number}. {html.escape(q["text"])}{tag}</div>', unsafe_allow_html=True)
    marks.markdown(f'<div class="marks">{q.get("co", "")} {q.get("bloom", "")} [{q["marks"]}]</div>',
                   unsafe_allow_html=True)
    if regen.button("", icon=":material/refresh:", key=f"regen_{q['uid']}", type="tertiary",
                    help="Write a new question on this topic"):
        st.session_state["seed"] += 7919
        if replace_question(paper, cfg, st.session_state["gen"], s_idx, q_idx, st.session_state["seed"]):
            paper["sections"][s_idx]["questions"][q_idx]["uid"] = uid()
        changed()
        st.rerun()
    if q["uid"] in editing:
        if edit.button("", icon=":material/check:", key=f"save_{q['uid']}", type="tertiary",
                       help="Save the new wording"):
            if new.strip():
                q["text"], q["bloom"] = new.strip(), bloom_level(new)
            editing.discard(q["uid"])
            changed()
            st.rerun()
    elif edit.button("", icon=":material/edit:", key=f"edit_{q['uid']}", type="tertiary", help=details(q)):
        editing.add(q["uid"])
        st.rerun()
    if drop.button("", icon=":material/close:", key=f"del_{q['uid']}", type="tertiary", help="Remove this question"):
        paper["sections"][s_idx]["questions"].pop(q_idx)
        changed()
        st.rerun()


def details(q: dict) -> str:
    """Hover text for the edit button: where the question came from and how it scored."""
    scores = [f"{name} {q[key]:.2f}" for name, key in (("relevance", "relevance"), ("difficulty fit", "difficulty_prob"))
              if q.get(key) is not None]
    return f"Edit the wording. {ORIGINS.get(q.get('origin'), 'Edited by hand')}, topic {q['topic']}" + \
        (f", {', '.join(scores)}." if scores else ".")


def sheet_header(spec) -> None:
    """Institute, exam title and the duration and marks line, laid out like the printed paper."""
    inst, exam, info = (html.escape(x) for x in header_lines({"spec": spec}))
    st.markdown(f'<div class="sheet sheet-head"><div class="inst">{inst}</div><div class="exam">{exam}</div>'
                f'<div class="meta">{info}</div></div>', unsafe_allow_html=True)


def step_bar(done: list[bool]) -> None:
    """Subject, Pattern, Review, Download: ticked when done, the first open one marked as current."""
    now = done.index(False) if False in done else len(done)
    cells = "".join(f'<div class="step {"done" if d else "now" if i == now else ""}">'
                    f'<span class="num">{"✓" if d else i + 1}</span>{name}</div>'
                    for i, (name, d) in enumerate(zip(STEPS, done)))
    st.markdown(f'<div class="steps">{cells}</div>', unsafe_allow_html=True)


def paper_tab(topics_ok: bool, pattern_ok: bool) -> None:
    """The step bar, then the paper as a printed sheet with per-question controls, then downloads."""
    paper = st.session_state.get("paper")
    complete = False
    if paper:
        try:
            check_paper(paper)
            complete = True
        except ValueError as e:
            problem = str(e)
    step_bar([topics_ok, pattern_ok, complete, complete and st.session_state.get("downloaded", False)])
    with st.container(key="sheet"):
        if not paper:
            st.markdown('<div class="sheet blank">Your paper will appear here.<br>Pick a subject and topics, '
                        'set the pattern, then press Generate paper.</div>', unsafe_allow_html=True)
            return
        spec = paper["spec"]
        sheet_header(spec)
        number = 0
        for s_idx, (sec_spec, sec) in enumerate(zip(spec.sections, paper["sections"])):
            title = section_title(sec)
            st.markdown(f'<div class="sheet sec">{html.escape(title)}</div>', unsafe_allow_html=True)
            if sec["instruction"] not in title:
                st.markdown(f'<div class="sheet sec-note">{html.escape(sec["instruction"])}</div>',
                            unsafe_allow_html=True)
            for q_idx in range(len(sec["questions"])):
                number += 1
                question_row(paper, s_idx, q_idx, number)
            short = sec_spec.count - len(sec["questions"])
            if short > 0 and st.button(f"Fill section {sec['name']} ({short} missing)", key=f"fill_{s_idx}"):
                st.session_state["seed"] += 104729
                fill_section(paper, cfg, st.session_state["gen"], s_idx, st.session_state["seed"])
                for q in sec["questions"]:
                    q.setdefault("uid", uid())
                changed()
                st.rerun()
    st.write("")
    st.caption("Marks per course outcome: " + ", ".join(f"{co or 'none'} {m}" for co, m in co_marks(paper).items())
               + ". Bloom levels: BL1 Remember, BL2 Understand, BL3 Apply, BL4 Analyze, BL5 Evaluate, BL6 Create.")
    if not complete:
        st.warning(f"{problem}. Fill the short sections before downloading.")
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
    paper_tab(*sidebar(get_context()))


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
