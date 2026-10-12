import numpy as np
import pytest

from qpgen.classifiers import accuracy, fit_scorer, load_scorer, save_scorer
from qpgen.config import load_config
from qpgen.data.build_dataset import build, split


@pytest.fixture(scope="module")
def fitted():
    cfg = load_config()
    parts = split(build(cfg), cfg)
    return cfg, parts, fit_scorer(cfg, parts["train"], parts["val"])


def test_threshold_for_every_topic(fitted):
    cfg, _, scorer = fitted
    topics = [t for info in cfg.subjects.values() for t in info["topics"]]
    assert set(scorer.thresholds) == set(topics)
    assert all(0 < v < 1 for v in scorer.thresholds.values())


def test_relevance_prefers_matching_topic(fitted):
    _, _, scorer = fitted
    q = ["Explain the difference between 3NF and BCNF with an example."]
    assert scorer.relevance(q, "Normalization")[0] > scorer.relevance(q, "CPU scheduling")[0]
    assert scorer.predict_topic(q) == ["Normalization"]


def test_difficulty_probabilities_sum_to_one(fitted):
    cfg, _, scorer = fitted
    q = ["Discuss in detail the ARIES recovery algorithm."]
    probs = [scorer.difficulty_prob(q, lv)[0] for lv in cfg.marks]
    assert np.isclose(sum(probs), 1.0)
    assert scorer.difficulty_prob(q, "hard")[0] == max(probs)


def test_accuracy_on_real_test_questions(fitted):
    _, parts, scorer = fitted
    acc = accuracy(scorer, parts["test"])
    assert acc["topic_accuracy"] > 0.6 and acc["difficulty_accuracy"] > 0.6


def test_save_and_load(fitted, tmp_path):
    _, _, scorer = fitted
    save_scorer(scorer, tmp_path / "s.pkl")
    loaded = load_scorer(tmp_path / "s.pkl")
    q = ["What is a deadlock in operating systems?"]
    assert loaded.relevance(q, "Deadlocks")[0] == pytest.approx(scorer.relevance(q, "Deadlocks")[0])
