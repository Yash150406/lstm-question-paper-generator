import math

import numpy as np
import pytest

from qpgen.baseline import TrigramModel, TrigramSampler
from qpgen.config import load_config
from qpgen.data.build_dataset import build, split
from qpgen.data.dataset import build_vocab
from qpgen.tokenizer import BOS, PAD, UNK


def test_probs_sum_to_one_and_prefer_seen_trigram():
    model = TrigramModel([[4, 5, 6], [4, 5, 7], [4, 5, 6]], vocab_size=10, alpha=0.4)
    p = model.probs(4, 5)
    assert np.isclose(p.sum(), 1.0)
    assert p[6] > p[7] > p[8]


def test_backs_off_to_bigram_for_unseen_context():
    model = TrigramModel([[4, 5, 6]], vocab_size=10, alpha=0.4)
    p = model.probs(9, 5)       # (9, 5) never seen, bigram 5 -> 6 has
    assert p.argmax() == 6
    assert np.all(model.probs(9, 9) > 0)


@pytest.fixture(scope="module")
def sampler():
    cfg = load_config()
    parts = split(build(cfg), cfg)
    train = parts["train"]
    return TrigramSampler(cfg, build_vocab(cfg, train), train), parts


def test_small_cells_use_topic_model(sampler):
    tri, parts = sampler
    train, cfg = parts["train"], tri.cfg
    counts = train.groupby(["topic", "difficulty"]).size()
    small = counts[counts < cfg.eval["baseline_min_cell"]].index[0]
    big = counts[counts >= cfg.eval["baseline_min_cell"]].index[0]
    tri.model_for(*small)
    tri.model_for(*big)
    assert (small[0], None) in tri.models and big in tri.models


def test_same_seed_same_samples_and_no_banned_tokens(sampler):
    tri, _ = sampler
    a = tri("DBMS", "Normalization", "medium", 10, seed=1)
    assert a == tri("DBMS", "Normalization", "medium", 10, seed=1)
    banned = set(tri.vocab.controls) | {PAD, BOS, UNK}
    assert all(not banned & set(item["tokens"]) for item in a)
    assert all(len(item["tokens"]) <= tri.cfg.generate["max_new_tokens"] for item in a)


def test_perplexity_is_finite_and_better_on_train(sampler):
    tri, parts = sampler
    val_ppl = tri.perplexity(parts["val"])
    train_ppl = tri.perplexity(parts["train"].head(300))
    assert math.isfinite(val_ppl) and 1 < train_ppl < val_ppl
