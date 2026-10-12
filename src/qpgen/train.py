"""Train the question LSTM and keep the best checkpoint."""
import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from qpgen.config import DEFAULT_CONFIG, Config, load_config
from qpgen.data.dataset import QuestionDataset, build_vocab, collate_fn
from qpgen.model import QuestionLSTM
from qpgen.seed import set_seed
from qpgen.tokenizer import PAD_ID, Vocab

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOG_FIELDS = ["epoch", "train_loss", "val_loss", "val_ppl", "lr", "seconds", "device"]


def pick_device(name: str) -> torch.device:
    """'auto' means CUDA when it is available, else CPU."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def tagged(path: Path, tag: str) -> Path:
    """model_best.pt -> model_best_<tag>.pt when a tag is given."""
    return path.with_name(f"{path.stem}_{tag}{path.suffix}") if tag else path


def masked_loss(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Cross entropy that skips pad positions, which also covers control-token targets."""
    return nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                                       ignore_index=PAD_ID, reduction=reduction)


@torch.no_grad()
def evaluate_loss(model: QuestionLSTM, loader: DataLoader, device: torch.device) -> float:
    """Mean cross entropy per predicted token over a whole split."""
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += masked_loss(model(x)[0], y, reduction="sum").item()
        count += (y != PAD_ID).sum().item()
    return total / count


def train_epoch(model: QuestionLSTM, loader: DataLoader, opt: torch.optim.Optimizer,
                clip: float, device: torch.device) -> float:
    """One pass over the training data with gradient clipping. Returns mean batch loss."""
    model.train()
    losses = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = masked_loss(model(x)[0], y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


def make_loaders(cfg: Config, vocab: Vocab, use_controls: bool = True) -> tuple[DataLoader, DataLoader]:
    """Shuffled train loader with a seeded generator and an ordered validation loader."""
    data = cfg.path("processed_dir")
    train_ds = QuestionDataset(pd.read_csv(data / "train.csv"), vocab, cfg, use_controls)
    val_ds = QuestionDataset(pd.read_csv(data / "val.csv"), vocab, cfg, use_controls)
    gen = torch.Generator().manual_seed(cfg.seed)
    bs = cfg.train["batch_size"]
    return (DataLoader(train_ds, bs, shuffle=True, collate_fn=collate_fn, generator=gen),
            DataLoader(val_ds, bs, collate_fn=collate_fn))


def save_checkpoint(path: Path, model: QuestionLSTM, model_cfg: dict, vocab: Vocab, epoch: int,
                    val_loss: float) -> None:
    """Weights, model config and vocabulary in one file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "model_cfg": model_cfg, "epoch": epoch,
                "val_loss": val_loss, "vocab": {"itos": vocab.itos, "controls": vocab.controls}}, path)


def load_model(path: Path, device: str = "cpu") -> tuple[QuestionLSTM, Vocab]:
    """Load a checkpoint written by save_checkpoint, ready for inference."""
    ckpt = torch.load(path, map_location=device)
    vocab = Vocab(ckpt["vocab"]["itos"], ckpt["vocab"]["controls"])
    model = QuestionLSTM.from_config(ckpt["model_cfg"], len(vocab))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), vocab


def plot_curve(rows: list[dict], path: Path) -> None:
    """Train and validation loss per epoch."""
    epochs = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [r["train_loss"] for r in rows], label="train")
    ax.plot(epochs, [r["val_loss"] for r in rows], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross entropy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_log(rows: list[dict], path: Path) -> None:
    """One CSV row per epoch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fit(cfg: Config, device: torch.device, tag: str = "", model_cfg: dict | None = None,
        use_controls: bool = True) -> list[dict]:
    """Full training run with LR schedule and early stopping. Returns the epoch log."""
    set_seed(cfg.seed)
    model_cfg = model_cfg or cfg.model
    vocab = build_vocab(cfg, pd.read_csv(cfg.path("processed_dir") / "train.csv"))
    vocab.save(tagged(cfg.path("vocab"), tag))
    train_dl, val_dl = make_loaders(cfg, vocab, use_controls)
    model = QuestionLSTM.from_config(model_cfg, len(vocab)).to(device)
    t = cfg.train
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=t["lr_factor"], patience=t["lr_patience"])

    rows, best, stale, start = [], math.inf, 0, time.perf_counter()
    for epoch in range(1, t["max_epochs"] + 1):
        train_loss = train_epoch(model, train_dl, opt, t["clip_norm"], device)
        val_loss = evaluate_loss(model, val_dl, device)
        sched.step(val_loss)
        rows.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4),
                     "val_ppl": round(math.exp(val_loss), 2), "lr": opt.param_groups[0]["lr"],
                     "seconds": round(time.perf_counter() - start, 1), "device": device.type})
        print(", ".join(f"{k} {v}" for k, v in rows[-1].items()))
        if val_loss < best:
            best, stale = val_loss, 0
            save_checkpoint(tagged(cfg.path("checkpoint"), tag), model, model_cfg, vocab, epoch, val_loss)
        else:
            stale += 1
            if stale >= t["patience"]:
                break
    write_log(rows, tagged(cfg.path("training_log"), tag))
    plot_curve(rows, tagged(cfg.path("loss_curve"), tag))
    return rows


def main(argv: list[str] | None = None) -> None:
    """Train from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", help="auto, cpu or cuda; overrides the config")
    parser.add_argument("--tag", default="", help="suffix for output files, used for extra runs")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    rows = fit(cfg, pick_device(args.device or cfg.train["device"]), args.tag)
    best = min(rows, key=lambda r: r["val_loss"])
    print(f"Best epoch {best['epoch']}: val loss {best['val_loss']}, val perplexity {best['val_ppl']}, "
          f"total time {rows[-1]['seconds']} s on {rows[-1]['device']}")


if __name__ == "__main__":
    main()
