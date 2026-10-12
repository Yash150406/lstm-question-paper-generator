import pandas as pd
import pytest
import torch

from qpgen.config import load_config
from qpgen.data.dataset import build_vocab, to_tokens
from qpgen.generate import mask_scores, sample
from qpgen.model import QuestionLSTM
from qpgen.seed import set_seed
from qpgen.tokenizer import BOS, PAD, UNK


@pytest.fixture(scope="module")
def setup():
    """An untrained small model: its output is random, which is what these tests need."""
    cfg = load_config()
    df = pd.read_csv(cfg.path("raw_dir") / "seed_dbms.csv").assign(subject="DBMS")
    vocab = build_vocab(cfg, df)
    set_seed(cfg.seed)
    model = QuestionLSTM.from_config({**cfg.model, "embed_dim": 32, "hidden_dim": 32}, len(vocab)).eval()
    prefix = to_tokens(cfg, "DBMS", "Normalization", "medium")
    return model, vocab, prefix


def run(setup, seed, n=8, top_k=40, max_new=20):
    model, vocab, prefix = setup
    return sample(model, vocab, prefix, n, temperature=1.0, top_k=top_k, max_new_tokens=max_new, seed=seed)


def test_same_seed_same_output(setup):
    assert run(setup, 42) == run(setup, 42)


def test_different_seed_changes_output(setup):
    assert [s["tokens"] for s in run(setup, 1)] != [s["tokens"] for s in run(setup, 2)]


def test_never_samples_special_or_control_tokens(setup):
    _, vocab, _ = setup
    banned = set(vocab.controls) | {PAD, BOS, UNK}
    for item in run(setup, 7, n=50):
        assert not banned & set(item["tokens"])


def test_stops_at_max_new_tokens(setup):
    for item in run(setup, 3, max_new=5):
        assert len(item["tokens"]) <= 5


def test_no_trigram_repeats_inside_a_question(setup):
    for item in run(setup, 11, n=50, max_new=40):
        t = item["tokens"]
        trigrams = list(zip(t, t[1:], t[2:]))
        assert len(trigrams) == len(set(trigrams))


def test_top_k_one_ignores_seed(setup):
    assert run(setup, 1, top_k=1) == run(setup, 99, top_k=1)


def test_mask_scores_blocks_seen_trigram_and_keeps_top_k():
    scores = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])
    blocked = torch.zeros(6, dtype=torch.bool)
    blocked[0] = True
    seen = [{(4, 5): {1}}]
    out = mask_scores(scores.clone(), blocked, [[4, 5]], seen, top_k=2)
    kept = (out[0] > float("-inf")).nonzero().flatten().tolist()
    assert kept == [2, 3]


def test_avg_logprob_is_negative(setup):
    for item in run(setup, 5):
        assert item["avg_logprob"] < 0
