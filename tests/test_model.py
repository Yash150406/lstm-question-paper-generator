import math

import pandas as pd
import pytest
import torch
from torch import nn

from qpgen.config import load_config
from qpgen.data.dataset import QuestionDataset, build_vocab, collate_fn
from qpgen.model import QuestionLSTM
from qpgen.seed import set_seed
from qpgen.tokenizer import PAD_ID


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def make_model(cfg, vocab_size, **overrides):
    return QuestionLSTM.from_config({**cfg.model, **overrides}, vocab_size)


def test_output_shapes(cfg):
    model = make_model(cfg, 100)
    logits, (h, c) = model(torch.randint(0, 100, (3, 7)))
    assert logits.shape == (3, 7, 100)
    layers, hidden = cfg.model["num_layers"], cfg.model["hidden_dim"]
    assert h.shape == c.shape == (layers, 3, hidden)


def test_weights_are_tied(cfg):
    model = make_model(cfg, 100)
    assert model.decoder.weight is model.embedding.weight


def test_tying_needs_matching_sizes(cfg):
    with pytest.raises(ValueError):
        make_model(cfg, 100, hidden_dim=128)


def test_forget_gate_bias(cfg):
    model = make_model(cfg, 100)
    h = cfg.model["hidden_dim"]
    bias = model.lstms[0].bias_ih_l0 + model.lstms[0].bias_hh_l0
    assert torch.all(bias[h:2 * h] == cfg.model["forget_bias"])
    assert torch.all(bias[:h] == 0)


def test_parameter_count(cfg):
    model = make_model(cfg, 920)
    n = sum(p.numel() for p in model.parameters())
    print(f"\nParameters with a 920 token vocab: {n:,}")
    assert 1_000_000 < n < 3_000_000


def test_initial_loss_close_to_ln_vocab(cfg):
    set_seed(cfg.seed)
    vocab_size = 920
    model = make_model(cfg, vocab_size).eval()
    x = torch.randint(4, vocab_size, (16, 20))
    logits, _ = model(x)
    loss = nn.functional.cross_entropy(logits.reshape(-1, vocab_size), x.reshape(-1))
    assert abs(loss.item() - math.log(vocab_size)) < 0.1


def overfit_batch(cfg):
    """One seed question for each topic, so every control prefix is unique."""
    seeds = pd.concat(pd.read_csv(cfg.path("raw_dir") / f"seed_{k}.csv").assign(subject=v["name"])
                      for k, v in cfg.subjects.items())
    return seeds[seeds["difficulty"] == "medium"].groupby("topic").head(1)


def test_overfits_32_questions(cfg):
    set_seed(cfg.seed)
    df = overfit_batch(cfg)
    assert len(df) == 32
    vocab = build_vocab(cfg, df)
    x, y = collate_fn(list(QuestionDataset(df, vocab, cfg)))
    model = make_model(cfg, len(vocab), dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    for _ in range(300):
        opt.zero_grad()
        logits, _ = model(x)
        loss = loss_fn(logits.reshape(-1, len(vocab)), y.reshape(-1))
        loss.backward()
        opt.step()
    assert loss.item() < 0.05


def test_hidden_state_carries_over_steps(cfg):
    model = make_model(cfg, 50).eval()
    x = torch.randint(0, 50, (2, 6))
    full, _ = model(x)
    first, hidden = model(x[:, :4])
    rest, _ = model(x[:, 4:], hidden)
    assert torch.allclose(full, torch.cat([first, rest], dim=1), atol=1e-5)
