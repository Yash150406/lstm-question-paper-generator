"""Sample questions from the trained LSTM."""
import argparse
import random
from collections import defaultdict
from functools import lru_cache

import torch
import yaml

from qpgen.config import DEFAULT_CONFIG, Config, load_config
from qpgen.data.dataset import to_tokens
from qpgen.model import QuestionLSTM
from qpgen.tokenizer import BOS_ID, EOS_ID, PAD_ID, UNK_ID, Vocab, detokenize
from qpgen.train import load_model, pick_device


@lru_cache
def load_acronyms(path: str) -> dict[str, str]:
    """Acronym map used when turning tokens back into text."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def mask_scores(scores: torch.Tensor, blocked: torch.Tensor, seqs: list[list[int]],
                seen: list[dict], top_k: int) -> torch.Tensor:
    """Block banned tokens and repeated trigrams, then keep only the top k per row."""
    scores[:, blocked] = float("-inf")
    for i, seq in enumerate(seqs):
        if len(seq) >= 2:
            for c in seen[i][(seq[-2], seq[-1])] - {EOS_ID}:
                scores[i, c] = float("-inf")
    kth = scores.topk(min(top_k, scores.size(1)), dim=1).values[:, -1:]
    return scores.masked_fill(scores < kth, float("-inf"))


@torch.no_grad()
def sample(model: QuestionLSTM, vocab: Vocab, prefix: list[str], n: int, temperature: float,
           top_k: int, max_new_tokens: int, seed: int) -> list[dict]:
    """Sample n continuations of the prefix in one batch."""
    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)
    blocked = torch.zeros(len(vocab), dtype=torch.bool, device=device)
    blocked[[PAD_ID, UNK_ID, BOS_ID] + vocab.control_ids] = True

    logits, hidden = model(torch.tensor([vocab.encode(prefix)] * n, device=device))
    seqs, logps = [[] for _ in range(n)], [[] for _ in range(n)]
    seen = [defaultdict(set) for _ in range(n)]
    done = [False] * n
    for _ in range(max_new_tokens):
        last = logits[:, -1]
        raw = torch.log_softmax(last, dim=1)
        scores = mask_scores(last / temperature, blocked, seqs, seen, top_k)
        nxt = torch.multinomial(torch.softmax(scores, dim=1), 1, generator=gen).squeeze(1)
        for i, tok in enumerate(nxt.tolist()):
            if done[i]:
                continue
            if len(seqs[i]) >= 2:
                seen[i][(seqs[i][-2], seqs[i][-1])].add(tok)
            seqs[i].append(tok)
            logps[i].append(raw[i, tok].item())
            done[i] = tok == EOS_ID
        if all(done):
            break
        logits, hidden = model(nxt.unsqueeze(1), hidden)
    return [{"tokens": vocab.decode([t for t in s if t != EOS_ID]), "finished": d,
             "avg_logprob": sum(lp) / len(lp)} for s, lp, d in zip(seqs, logps, done)]


def generate_raw(model: QuestionLSTM, vocab: Vocab, cfg: Config, subject: str, topic: str,
                 difficulty: str, n: int, temperature: float | None = None,
                 top_k: int | None = None, seed: int | None = None) -> list[dict]:
    """Unfiltered samples for one subject, topic and difficulty, with readable text."""
    g = cfg.generate
    out = sample(model, vocab, to_tokens(cfg, subject, topic, difficulty), n,
                 temperature or g["temperature"], top_k or g["top_k"], g["max_new_tokens"],
                 cfg.seed if seed is None else seed)
    acronyms = load_acronyms(str(cfg.path("acronyms")))
    for item in out:
        item["text"] = detokenize(item["tokens"], acronyms)
    return out


def main(argv: list[str] | None = None) -> None:
    """Print a few raw samples for every subject."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("-n", type=int, default=5)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    model, vocab = load_model(cfg.path("checkpoint"), pick_device(cfg.train["device"]))
    rng = random.Random(cfg.seed)
    for info in cfg.subjects.values():
        print(f"\n{info['name']}")
        for i in range(args.n):
            topic, level = rng.choice(info["topics"]), rng.choice(list(cfg.marks))
            item = generate_raw(model, vocab, cfg, info["name"], topic, level, 1, seed=cfg.seed + i)[0]
            print(f"  [{topic} / {level}] {item['text']}")


if __name__ == "__main__":
    main()
