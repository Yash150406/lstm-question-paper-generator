from types import SimpleNamespace

import pytest

from qpgen.classifiers import fit_scorer
from qpgen.config import load_config
from qpgen.data.build_dataset import build, split
from qpgen.data.dataset import build_vocab
from qpgen.filters import (Context, check_clean, check_difficulty, check_duplicate, check_ending,
                           check_format, check_novel, check_relevance, filter_candidates,
                           generate_questions, rank)
from qpgen.model import QuestionLSTM
from qpgen.seed import set_seed

CFG = load_config()
F = CFG.filters


def q(text, **extra):
    return {"text": text, "tokens": text.lower().split(), **extra}


def test_format():
    assert check_format(q("What is a deadlock in operating systems?"), F)[0]
    assert not check_format(q("Define paging."), F)[0]
    assert not check_format(q("What is a deadlock in operating systems"), F)[0]


def test_clean():
    assert check_clean(q("Explain paging (with a diagram) in detail."), F)[0]
    assert not check_clean(q("What is is is a deadlock in detail?"), F)[0]
    assert not check_clean(q("Explain paging (with a diagram in detail."), F)[0]


def test_ending():
    assert check_ending(q("Explain the working of a semaphore."), F)[0]
    assert not check_ending(q("Explain the working of the."), F)[0]


def test_novel():
    train = {"what is a deadlock in operating systems?"}
    assert check_novel(q("What is a semaphore in operating systems?"), F, train)[0]
    c = q("What is a deadlock in operating systems?")
    assert not check_novel(c, F, train)[0] and c["novel"] is False
    assert check_novel(c, {**F, "allow_train_copies": True}, train)[0]


def test_duplicate():
    accepted = [q("Explain the two phase locking protocol with an example.")]
    assert check_duplicate(q("Explain shadow paging with a neat diagram."), F, accepted)[0]
    assert not check_duplicate(q("Explain the two phase locking protocol with one example."), F, accepted)[0]


def test_relevance():
    assert check_relevance({"relevance": 0.5}, 0.2)[0]
    assert not check_relevance({"relevance": 0.1}, 0.2)[0]


def test_difficulty():
    assert check_difficulty({"difficulty_prob": 0.9}, 0.4)[0]
    assert not check_difficulty({"difficulty_prob": 0.2}, 0.4)[0]


def test_filter_candidates_reports_first_failing_reason():
    ctx = SimpleNamespace(cfg=CFG, scorer=SimpleNamespace(thresholds={"Deadlocks": 0.2}), train_set=set())
    good = q("What is a resource allocation graph in deadlocks?", relevance=0.5, difficulty_prob=0.9)
    off_topic = q("What is a resource ordering scheme for printers?", relevance=0.05, difficulty_prob=0.9)
    short = q("Define it.", relevance=0.9, difficulty_prob=0.9)
    accepted, rejected = filter_candidates([good, off_topic, short], ctx, "Deadlocks")
    assert [a["text"] for a in accepted] == [good["text"]]
    assert rejected[0]["reason"].startswith("relevance") and rejected[1]["reason"].startswith("length")


def test_rank_by_relevance_then_logprob():
    items = [{"relevance": 0.3, "avg_logprob": -1.0}, {"relevance": 0.5, "avg_logprob": -2.0},
             {"relevance": 0.3, "avg_logprob": -0.5}]
    assert rank(items) == [items[1], items[2], items[0]]


@pytest.fixture(scope="module")
def weak_context():
    """An untrained model, so nearly every candidate fails and the fallback has to fill."""
    bank = build(CFG)
    parts = split(bank, CFG)
    vocab = build_vocab(CFG, parts["train"])
    set_seed(CFG.seed)
    model = QuestionLSTM.from_config({**CFG.model, "embed_dim": 16, "hidden_dim": 16}, len(vocab)).eval()
    scorer = fit_scorer(CFG, parts["train"], parts["val"])
    return Context(CFG, model, vocab, scorer, bank, set(parts["train"]["question"].str.lower()))


def test_generate_questions_falls_back_to_bank(weak_context):
    out = generate_questions("DBMS", "Recovery", "easy", 5, ctx=weak_context)
    assert len(out) == 5
    assert {"text", "avg_logprob", "relevance", "difficulty_prob", "novel", "origin"} == set(out[0])
    assert "from_bank" in {o["origin"] for o in out}
    assert len({o["text"] for o in out}) == 5
