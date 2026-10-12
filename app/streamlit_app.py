"""Streamlit app: build a paper, try the model, and read the report."""
import itertools
import json
import os
import tempfile
from datetime import datetime
from functools import partial
from pathlib import Path

import pandas as pd
import streamlit as st

from qpgen.config import DEFAULT_CONFIG, load_config
from qpgen.export import WRITERS, section_title
from qpgen.filters import filter_candidates, generate_questions, load_context, score
from qpgen.paper import build_paper, check_paper, fill_section, replace_question, spec_from_preset
from qpgen.upload import analyze_upload, read_upload

CONFIG = os.environ.get("QPGEN_CONFIG", str(DEFAULT_CONFIG))
BADGES = {"generated": "generated", "from_bank": "from bank", "from_notes": "from notes"}
PRESETS = {"Balanced": "balanced", "Easy-heavy": "easy_heavy", "Hard-heavy": "hard_heavy"}

st.set_page_config(page_title="Question Paper Generator", layout="wide")
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
        all_topics = cfg.subjects[cfg.subject_key(subject)]["topics"]
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
    all_topics = cfg.subjects[cfg.subject_key(found["subject"])]["topics"]
    topics = st.multiselect("Topics", all_topics, default=list(found["topics"]))
    return {"subject": found["subject"], "topics": topics,
            "weights": {t: w for t, w in found["topics"].items() if t in topics},
            "upload": {"text": text, "key_terms": found["key_terms"]}}


def sidebar(ctx) -> None:
    """All paper settings and the Generate button."""
    with st.sidebar:
        st.header("Paper settings")
        picked = topic_inputs(ctx)
        preset = PRESETS[st.selectbox("Preset", list(PRESETS))]
        duration = st.number_input("Duration (minutes)", 30, 300, cfg.paper["duration_minutes"], step=15)
        institute = st.text_input("Institute name", cfg.paper["institute"])
        title = st.text_input("Exam title", cfg.paper["exam_title"])
        seed = int(st.number_input("Seed", 0, 10**6, cfg.seed))
        temperature = st.slider("Temperature", 0.5, 1.2, float(cfg.generate["temperature"]), 0.05)
        ready = bool(picked and picked["topics"])
        if st.button("Generate paper", type="primary", disabled=not ready, use_container_width=True):
            make_paper(ctx, picked, preset, seed, temperature,
                       institute=institute, exam_title=title, duration_minutes=int(duration))


def make_paper(ctx, picked: dict, preset: str, seed: int, temperature: float, **overrides) -> None:
    """Build the paper and keep it, with its generator, in session state."""
    gen = partial(generate_questions, ctx=ctx, upload=picked["upload"], temperature=temperature)
    spec = spec_from_preset(cfg, picked["subject"], picked["topics"], preset, picked["weights"], **overrides)
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
            col.download_button(f"Download {fmt.upper()}", data, f"{stem}.{fmt}", use_container_width=True)


def question_row(paper: dict, s_idx: int, q_idx: int, number: int) -> None:
    """One question with its badge, scores, edit box and buttons."""
    q = paper["sections"][s_idx]["questions"][q_idx]
    text_col, act_col = st.columns([5, 1])
    new = text_col.text_input(f"Q{number}  [{q['marks']} marks]", q["text"], key=f"text_{q['uid']}")
    if new.strip() and new != q["text"]:
        q["text"] = new.strip()
    rel = "" if q.get("relevance") is None else f"relevance {q['relevance']:.2f}"
    prob = "" if q.get("difficulty_prob") is None else f"difficulty {q['difficulty_prob']:.2f}"
    text_col.caption(f"`{BADGES.get(q.get('origin'), q.get('origin'))}`  {q['topic']}  {rel}  {prob}")
    if act_col.button("Regenerate", key=f"regen_{q['uid']}", use_container_width=True):
        st.session_state["seed"] += 7919
        if replace_question(paper, cfg, st.session_state["gen"], s_idx, q_idx, st.session_state["seed"]):
            paper["sections"][s_idx]["questions"][q_idx]["uid"] = uid()
        st.rerun()
    if act_col.button("Remove", key=f"remove_{q['uid']}", use_container_width=True):
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
    try:
        check_paper(paper)
    except ValueError as e:
        st.warning(f"{e}. Fill the short sections before downloading.")
        return
    downloads(paper)


def try_tab(ctx) -> None:
    """Raw samples next to the ones that pass, with the reason for each rejection."""
    names = [info["name"] for info in cfg.subjects.values()]
    c1, c2, c3, c4 = st.columns(4)
    subject = c1.selectbox("Subject", names, key="try_subject")
    topic = c2.selectbox("Topic", cfg.subjects[cfg.subject_key(subject)]["topics"], key="try_topic")
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


def main() -> None:
    """Stop with a clear message when the model is missing, otherwise show the three tabs."""
    st.title("Question Paper Generator")
    missing = [p for p in ("checkpoint", "classifiers") if not cfg.path(p).exists()]
    if missing:
        st.error("No trained model found. Run `make train` first, then reload this page.")
        st.stop()
    ctx = get_context()
    sidebar(ctx)
    # A key makes Streamlit remember the open tab, so a button press does not jump back to Paper.
    paper, trial, report = st.tabs(["Paper", "Try the model", "Report"], key="tab")
    with paper:
        paper_tab()
    with trial:
        try_tab(ctx)
    with report:
        report_tab()


main()
