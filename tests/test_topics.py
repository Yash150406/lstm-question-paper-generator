import shutil
from dataclasses import replace

import pandas as pd
import pytest
import yaml

from qpgen.config import ROOT, load_config
from qpgen.topics import check_new_topic, question_rows, save_new_topic

CFG = load_config()
LINES = ([f"Define cache term number {i} clearly." for i in range(12)]
         + [f"Explain cache policy number {i} with an example." for i in range(12)]
         + [f"Design a cache for workload number {i} in detail." for i in range(6)])


@pytest.fixture
def tmp_cfg(tmp_path):
    """Config whose subject list and raw folder live in tmp_path."""
    shutil.copy(ROOT / CFG.paths["subjects_topics"], tmp_path / "subjects_topics.yaml")
    paths = {**CFG.paths, "subjects_topics": str(tmp_path / "subjects_topics.yaml"), "raw_dir": str(tmp_path)}
    return replace(CFG, paths=paths)


def test_question_rows_strip_numbers_and_label_difficulty():
    rows = question_rows("1. Define a cache line.\n\nQ2) Explain write-back caching with an example.\n")
    assert [r["question"] for r in rows] == ["Define a cache line.", "Explain write-back caching with an example."]
    assert [r["difficulty"] for r in rows] == ["easy", "medium"]


def test_good_topic_has_no_errors():
    assert check_new_topic(CFG, "Operating Systems", "Caching", question_rows("\n".join(LINES))) == []


@pytest.mark.parametrize("topic, lines, words", [
    ("Deadlocks", LINES, "already"),
    ("Bad: name", LINES, "letters"),
    ("Caching", LINES[:20], "at least 30"),
    ("Caching", LINES[:24] * 2, "hard"),
])
def test_bad_topic_is_refused(topic, lines, words):
    errors = check_new_topic(CFG, "Operating Systems", topic, question_rows("\n".join(lines)))
    assert any(words in e for e in errors), errors


def test_save_adds_topic_co_and_questions(tmp_cfg):
    save_new_topic(tmp_cfg, "Operating Systems", "Caching", "CO5", question_rows("\n".join(LINES)))
    data = yaml.safe_load(open(tmp_cfg.path("subjects_topics"), encoding="utf-8"))
    assert data["subjects"]["os"]["topics"][-1] == "Caching"
    assert data["subjects"]["os"]["co"]["Caching"] == "CO5"
    assert data["subjects"]["dbms"] == yaml.safe_load(open(ROOT / CFG.paths["subjects_topics"]))["subjects"]["dbms"]
    assert open(tmp_cfg.path("subjects_topics"), encoding="utf-8").read().startswith("# key is used")
    df = pd.read_csv(tmp_cfg.path("raw_dir") / "student_questions.csv")
    assert list(df.columns) == ["subject", "topic", "difficulty", "question"] and len(df) == 30
    save_new_topic(tmp_cfg, "DBMS", "Views", "CO2", question_rows("\n".join(LINES)))
    assert len(pd.read_csv(tmp_cfg.path("raw_dir") / "student_questions.csv")) == 60
