"""PyTorch dataset of control-token sequences, plus vocab building."""
import argparse

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from qpgen.config import DEFAULT_CONFIG, Config, load_config
from qpgen.tokenizer import (BOS, EOS, PAD_ID, UNK_ID, Vocab, control_tokens,
                             difficulty_token, subject_token, tokenize, topic_token)


def to_tokens(cfg: Config, subject: str, topic: str, difficulty: str, question: str = "") -> list[str]:
    """<bos> <subj> <topic> <diff> followed by the question words and <eos> when given."""
    prefix = [BOS, subject_token(cfg.subject_key(subject)), topic_token(topic), difficulty_token(difficulty)]
    return prefix + tokenize(question) + [EOS] if question else prefix


def build_vocab(cfg: Config, train: pd.DataFrame) -> Vocab:
    """Build the vocabulary from training questions only."""
    controls = control_tokens(cfg.subjects, list(cfg.marks))
    sentences = [tokenize(q) for q in train["question"]]
    return Vocab.build(sentences, controls, cfg.vocab["min_freq"], cfg.vocab["max_words"])


class QuestionDataset(Dataset):
    """Input is the sequence without its last token, target is it shifted by one."""

    def __init__(self, df: pd.DataFrame, vocab: Vocab, cfg: Config) -> None:
        seqs = (vocab.encode(to_tokens(cfg, *row)) for row in
                df[["subject", "topic", "difficulty", "question"]].itertuples(index=False))
        self.seqs = [s for s in seqs if len(s) <= cfg.vocab["max_len"]]
        self.controls = set(vocab.control_ids)

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.seqs[i]
        # Control tokens are given at generation time, so the model is never asked to predict them.
        target = [PAD_ID if t in self.controls else t for t in seq[1:]]
        return torch.tensor(seq[:-1]), torch.tensor(target)


def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad inputs and targets in a batch to the same length with <pad>."""
    xs, ys = zip(*batch)
    return (pad_sequence(xs, batch_first=True, padding_value=PAD_ID),
            pad_sequence(ys, batch_first=True, padding_value=PAD_ID))


def unknown_rate(df: pd.DataFrame, vocab: Vocab) -> float:
    """Share of question words that map to <unk>."""
    ids = [i for q in df["question"] for i in vocab.encode(tokenize(q))]
    return sum(i == UNK_ID for i in ids) / len(ids)


def main(argv: list[str] | None = None) -> None:
    """Build and save the vocabulary, then print a few sequences and the unknown rate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    cfg = load_config(parser.parse_args(argv).config)
    data = cfg.path("processed_dir")
    train, val = pd.read_csv(data / "train.csv"), pd.read_csv(data / "val.csv")

    vocab = build_vocab(cfg, train)
    vocab.save(cfg.path("vocab"))
    ds = QuestionDataset(train, vocab, cfg)
    print(f"Vocab size: {len(vocab)} ({len(vocab.controls)} control tokens)")
    print(f"Train sequences: {len(ds)} of {len(train)}")
    for i in range(3):
        print(" ".join(vocab.decode(ds.seqs[i])))
    print(f"Unknown word rate on validation: {unknown_rate(val, vocab):.2%}")


if __name__ == "__main__":
    main()
