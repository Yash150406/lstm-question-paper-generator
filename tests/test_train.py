from pathlib import Path

import pandas as pd
import torch

from qpgen.config import load_config
from qpgen.data.dataset import QuestionDataset, build_vocab, collate_fn
from qpgen.model import QuestionLSTM
from qpgen.seed import set_seed
from qpgen.tokenizer import PAD_ID
from qpgen.train import load_model, masked_loss, pick_device, save_checkpoint, tagged


def test_padding_does_not_change_loss():
    set_seed(0)
    logits = torch.randn(1, 5, 10)
    target = torch.tensor([[PAD_ID, 4, 5, 6, 7]])
    padded_logits = torch.cat([logits, torch.randn(1, 3, 10)], dim=1)
    padded_target = torch.cat([target, torch.full((1, 3), PAD_ID)], dim=1)
    assert torch.allclose(masked_loss(logits, target), masked_loss(padded_logits, padded_target))


def test_control_target_positions_do_not_change_loss():
    cfg = load_config()
    df = pd.DataFrame({"subject": ["DBMS"], "topic": ["Recovery"], "difficulty": ["easy"],
                       "question": ["What is a checkpoint in DBMS?"]})
    vocab = build_vocab(cfg, pd.concat([df] * 2))
    _, y = collate_fn([QuestionDataset(df, vocab, cfg)[0]])
    logits = torch.randn(1, y.size(1), len(vocab))
    changed = logits.clone()
    changed[0, :3] = torch.randn(3, len(vocab)) * 10   # the three control-token target positions
    assert torch.allclose(masked_loss(logits, y), masked_loss(changed, y))


def test_tagged_path():
    assert tagged(Path("a/model_best.pt"), "cpu") == Path("a/model_best_cpu.pt")
    assert tagged(Path("a/model_best.pt"), "") == Path("a/model_best.pt")


def test_pick_device():
    assert pick_device("cpu").type == "cpu"
    assert pick_device("auto").type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_checkpoint_round_trip(tmp_path):
    cfg = load_config()
    df = pd.read_csv(cfg.path("raw_dir") / "seed_os.csv").assign(subject="Operating Systems").head(20)
    vocab = build_vocab(cfg, df)
    model_cfg = {**cfg.model, "embed_dim": 16, "hidden_dim": 16}
    model = QuestionLSTM.from_config(model_cfg, len(vocab)).eval()
    save_checkpoint(tmp_path / "m.pt", model, model_cfg, vocab, epoch=1, val_loss=1.0)

    loaded, loaded_vocab = load_model(tmp_path / "m.pt")
    x = torch.randint(0, len(vocab), (2, 6))
    assert loaded_vocab.itos == vocab.itos
    assert torch.equal(model(x)[0], loaded(x)[0])
