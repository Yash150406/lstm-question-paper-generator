"""Trigram baseline with stupid backoff, one model per topic and difficulty."""
import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from qpgen.config import DEFAULT_CONFIG, Config, load_config
from qpgen.data.dataset import build_vocab
from qpgen.generate import load_acronyms, mask_scores
from qpgen.tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID, Vocab, detokenize, tokenize


class TrigramModel:
    """Counts trigrams, bigrams and unigrams over word ids."""

    def __init__(self, seqs: list[list[int]], vocab_size: int, alpha: float) -> None:
        self.alpha, self.size = alpha, vocab_size
        self.tri: dict[tuple, Counter] = defaultdict(Counter)
        self.bi: dict[int, Counter] = defaultdict(Counter)
        uni = np.ones(vocab_size)   # add-one so no word gets zero probability
        for s in seqs:
            padded = [BOS_ID, BOS_ID] + s + [EOS_ID]
            for u, v, w in zip(padded, padded[1:], padded[2:]):
                self.tri[(u, v)][w] += 1
                self.bi[v][w] += 1
                uni[w] += 1
        self.uni = uni / uni.sum()

    def _vector(self, counts: Counter) -> np.ndarray:
        """Relative frequencies from a Counter as a dense vector."""
        vec = np.zeros(self.size)
        vec[list(counts)] = list(counts.values())
        return vec / vec.sum()

    def probs(self, u: int, v: int) -> np.ndarray:
        """Stupid backoff scores for the next word, normalised to sum to one."""
        scores = self.alpha ** 2 * self.uni
        if v in self.bi:
            b = self._vector(self.bi[v])
            scores = np.where(b > 0, self.alpha * b, scores)
        if (u, v) in self.tri:
            t = self._vector(self.tri[(u, v)])
            scores = np.where(t > 0, t, scores)
        return scores / scores.sum()


@dataclass
class TrigramSampler:
    """Same call signature as LSTMSampler, so the filters and evaluation treat both alike."""

    cfg: Config
    vocab: Vocab
    train: pd.DataFrame
    models: dict = field(default_factory=dict, init=False)

    def model_for(self, topic: str, difficulty: str) -> TrigramModel:
        """Cell model, or the whole-topic model when the cell is too small."""
        cell = self.train[(self.train["topic"] == topic) & (self.train["difficulty"] == difficulty)]
        key = (topic, difficulty)
        if len(cell) < self.cfg.eval["baseline_min_cell"]:
            cell, key = self.train[self.train["topic"] == topic], (topic, None)
        if key not in self.models:
            seqs = [self.vocab.encode(tokenize(q)) for q in cell["question"]]
            self.models[key] = TrigramModel(seqs, len(self.vocab), self.cfg.eval["backoff_alpha"])
        return self.models[key]

    def _one(self, model: TrigramModel, temperature: float, top_k: int, rng: np.random.Generator) -> dict:
        """Sample one question with the same masking rules as the LSTM."""
        blocked = torch.zeros(len(self.vocab), dtype=torch.bool)
        blocked[[PAD_ID, UNK_ID, BOS_ID] + self.vocab.control_ids] = True
        seq, logps, seen = [], [], [defaultdict(set)]
        for _ in range(self.cfg.generate["max_new_tokens"]):
            u, v = ([BOS_ID, BOS_ID] + seq)[-2:]
            p = model.probs(u, v)
            scores = mask_scores(torch.tensor(np.log(p) / temperature).unsqueeze(0), blocked, [seq], seen, top_k)
            q = torch.softmax(scores, dim=1)[0].numpy()
            tok = int(rng.choice(len(q), p=q / q.sum()))
            if len(seq) >= 2:
                seen[0][(seq[-2], seq[-1])].add(tok)
            seq.append(tok)
            logps.append(math.log(p[tok]))
            if tok == EOS_ID:
                break
        words = self.vocab.decode([t for t in seq if t != EOS_ID])
        return {"tokens": words, "finished": seq[-1] == EOS_ID, "avg_logprob": sum(logps) / len(logps)}

    def __call__(self, subject: str, topic: str, difficulty: str, n: int, temperature: float | None = None,
                 top_k: int | None = None, seed: int | None = None) -> list[dict]:
        g = self.cfg.generate
        rng = np.random.default_rng(self.cfg.seed if seed is None else seed)
        model = self.model_for(topic, difficulty)
        acronyms = load_acronyms(str(self.cfg.path("acronyms")))
        out = [self._one(model, temperature or g["temperature"], top_k or g["top_k"], rng) for _ in range(n)]
        for item in out:
            item["text"] = detokenize(item["tokens"], acronyms)
        return out

    def perplexity(self, df: pd.DataFrame) -> float:
        """exp of mean cross entropy over question words and <eos>, like the LSTM."""
        total, count = 0.0, 0
        for topic, level, q in df[["topic", "difficulty", "question"]].itertuples(index=False):
            model = self.model_for(topic, level)
            padded = [BOS_ID, BOS_ID] + self.vocab.encode(tokenize(q)) + [EOS_ID]
            for u, v, w in zip(padded, padded[1:], padded[2:]):
                total -= math.log(model.probs(u, v)[w])
                count += 1
        return math.exp(total / count)


def main(argv: list[str] | None = None) -> None:
    """Report validation perplexity and print a few baseline samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    cfg = load_config(parser.parse_args(argv).config)
    data = cfg.path("processed_dir")
    train, val = pd.read_csv(data / "train.csv"), pd.read_csv(data / "val.csv")
    sampler = TrigramSampler(cfg, build_vocab(cfg, train), train)
    ppl = sampler.perplexity(val)
    cfg.path("baseline_report").write_text(json.dumps({"val_perplexity": round(ppl, 2)}, indent=2), encoding="utf-8")
    print(f"Trigram validation perplexity: {ppl:.2f}")
    for info in cfg.subjects.values():
        topic = info["topics"][0]
        for item in sampler(info["name"], topic, "medium", 2):
            print(f"  [{info['name']} / {topic}] {item['text']}")


if __name__ == "__main__":
    main()
