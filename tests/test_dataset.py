import pandas as pd
import pytest

from qpgen.config import load_config
from qpgen.data.dataset import QuestionDataset, build_vocab, collate_fn, to_tokens
from qpgen.tokenizer import BOS, EOS, PAD_ID


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def frame():
    return pd.DataFrame({
        "subject": ["DBMS", "Operating Systems"],
        "topic": ["Normalization", "Deadlocks"],
        "difficulty": ["medium", "easy"],
        "question": ["Differentiate between 3NF and BCNF.", "What is a deadlock in operating systems?"],
    })


def test_sequence_starts_with_control_tokens(cfg):
    tokens = to_tokens(cfg, "DBMS", "Normalization", "medium", "Define 3NF.")
    assert tokens[:4] == [BOS, "<subj_dbms>", "<topic_normalization>", "<diff_medium>"]
    assert tokens[-1] == EOS


def test_target_is_input_shifted_with_controls_masked(cfg, frame):
    vocab = build_vocab(cfg, frame)
    ds = QuestionDataset(frame, vocab, cfg)
    x, y = ds[0]
    seq = ds.seqs[0]
    assert x.tolist() == seq[:-1]
    assert y[:3].tolist() == [PAD_ID] * 3          # subject, topic, difficulty targets
    assert y[3:].tolist() == seq[4:]


def test_collate_pads_to_longest(cfg, frame):
    vocab = build_vocab(cfg, frame)
    ds = QuestionDataset(frame, vocab, cfg)
    x, y = collate_fn([ds[0], ds[1]])
    assert x.shape == y.shape
    assert x.shape[1] == max(len(s) for s in ds.seqs) - 1
    shorter = min(range(2), key=lambda i: len(ds.seqs[i]))
    assert x[shorter, -1].item() == PAD_ID and y[shorter, -1].item() == PAD_ID


def test_long_sequences_are_dropped(cfg, frame):
    long = frame.iloc[:1].assign(question="word " * 60)
    ds = QuestionDataset(pd.concat([frame, long]), build_vocab(cfg, frame), cfg)
    assert len(ds) == 2
