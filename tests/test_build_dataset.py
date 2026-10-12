import pandas as pd
import pytest

from qpgen.config import load_config
from qpgen.data.build_dataset import COLUMNS, build, clean, split


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def bank(cfg):
    return build(cfg)


def rows(*items):
    return pd.DataFrame(items, columns=["subject", "topic", "difficulty", "question", "source"])


def test_exact_duplicate_keeps_seed_over_template(cfg):
    df = clean(rows(
        ("DBMS", "Normalization", "easy", "What is a functional dependency in DBMS?", "template"),
        ("DBMS", "Normalization", "easy", "what is a functional dependency in DBMS? ", "seed"),
    ), cfg)
    assert len(df) == 1 and df.iloc[0]["source"] == "seed"


def test_near_duplicates_dropped_within_topic_only(cfg):
    a = "Explain the two phase locking protocol with an example."
    b = "Explain the two phase locking protocol with an example in detail."
    c = "Explain the two phase locking protocol with an example now."
    df = clean(rows(
        ("DBMS", "Concurrency control", "medium", a, "seed"),
        ("DBMS", "Concurrency control", "medium", b, "seed"),
        ("DBMS", "Recovery", "medium", c, "seed"),
    ), cfg)
    assert len(df) == 2
    assert set(df["topic"]) == {"Concurrency control", "Recovery"}


def test_length_filter_and_unknown_topic(cfg):
    df = clean(rows(
        ("DBMS", "Recovery", "easy", "What is a log?", "seed"),
        ("DBMS", "Recovery", "easy", "word " * 41 + "?", "seed"),
        ("DBMS", "Cooking", "easy", "What is the best way to boil rice?", "seed"),
        ("DBMS", "Recovery", "easy", "What is a checkpoint in DBMS?", "seed"),
    ), cfg)
    assert df["question"].tolist() == ["What is a checkpoint in DBMS?"]


def test_blank_difficulty_is_labelled_and_marks_filled(cfg):
    df = clean(rows(
        ("DBMS", "Recovery", None, "Discuss in detail the ARIES recovery algorithm steps.", "student"),
    ), cfg)
    assert df.iloc[0]["difficulty"] == "hard" and df.iloc[0]["marks"] == 10


def test_bank_matches_schema_and_size(bank, cfg):
    assert list(bank.columns) == COLUMNS
    assert len(bank) >= 3000
    assert bank["subject"].value_counts().min() >= cfg.data["min_per_subject"]
    assert not bank["question"].str.lower().duplicated().any()
    assert bank["id"].is_unique


def test_every_topic_and_level_present(bank, cfg):
    for info in cfg.subjects.values():
        for topic in info["topics"]:
            levels = set(bank.loc[bank["topic"] == topic, "difficulty"])
            assert levels == {"easy", "medium", "hard"}, topic


def test_split_sizes_and_no_overlap(bank, cfg):
    parts = split(bank, cfg)
    sizes = {k: len(v) / len(bank) for k, v in parts.items()}
    assert sizes["train"] == pytest.approx(0.8, abs=0.01)
    assert sizes["val"] == pytest.approx(0.1, abs=0.01)
    ids = [set(p["id"]) for p in parts.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
