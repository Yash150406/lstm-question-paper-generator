"""The seven quality filters, ranking and the fallback that fills short lists."""
import random
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
import yaml

from qpgen.classifiers import Scorer, load_scorer
from qpgen.config import DEFAULT_CONFIG, Config, load_config
from qpgen.data.build_dataset import overlap, words
from qpgen.generate import LSTMSampler
from qpgen.tokenizer import UNK
from qpgen.upload import upload_similarity
from qpgen.train import load_model, pick_device

Result = tuple[bool, str]
PAIRS = {")": "(", "]": "["}


@dataclass
class Context:
    """Everything generate_questions needs, loaded once."""

    cfg: Config
    sampler: Callable[..., list[dict]]   # LSTMSampler or TrigramSampler
    scorer: Scorer
    bank: pd.DataFrame
    train_set: set[str]


@lru_cache
def load_context(config_path: str = str(DEFAULT_CONFIG)) -> Context:
    """Load the model, scorer and question bank from the paths in config."""
    cfg = load_config(config_path)
    model, vocab = load_model(cfg.path("checkpoint"), pick_device(cfg.train["device"]))
    data = cfg.path("processed_dir")
    train = pd.read_csv(data / "train.csv")
    return Context(cfg, LSTMSampler(model, vocab, cfg), load_scorer(cfg.path("classifiers")),
                   pd.read_csv(data / "questions.csv"), set(train["question"].str.lower().str.strip()))


def check_format(c: dict, f: dict) -> Result:
    """5 to 30 words, starts with a letter, ends with ? or ."""
    n = len(words(c["text"]))
    if not f["min_words"] <= n <= f["max_words"]:
        return False, f"length {n} words"
    if not c["text"][:1].isalpha():
        return False, "does not start with a letter"
    if not c["text"].endswith(("?", ".")):
        return False, "does not end with ? or ."
    return True, ""


def check_clean(c: dict, f: dict) -> Result:
    """No <unk>, no word three times in a row, brackets and quotes balanced."""
    if UNK in c.get("tokens", []) or UNK in c["text"]:
        return False, "contains <unk>"
    ws = words(c["text"])
    run = f["max_repeat"]
    if any(len(set(ws[i:i + run])) == 1 for i in range(len(ws) - run + 1)):
        return False, f"word repeated {run} times in a row"
    stack = []
    for ch in c["text"]:
        if ch in "([":
            stack.append(ch)
        elif ch in PAIRS and (not stack or stack.pop() != PAIRS[ch]):
            return False, "unbalanced brackets"
    if stack or c["text"].count('"') % 2:
        return False, "unbalanced brackets or quotes"
    return True, ""


def check_ending(c: dict, f: dict) -> Result:
    """Last word is not a stop word."""
    ws = words(c["text"])
    if ws and ws[-1] in f["stop_words"]:
        return False, f"ends on '{ws[-1]}'"
    return True, ""


def check_novel(c: dict, f: dict, train_set: set[str]) -> Result:
    """Not an exact copy of a training question, unless copies are allowed."""
    c["novel"] = c["text"].lower().strip() not in train_set
    if not c["novel"] and not f["allow_train_copies"]:
        return False, "copy of a training question"
    return True, ""


def check_duplicate(c: dict, f: dict, accepted: list[dict]) -> Result:
    """Word overlap below the limit with every question already accepted."""
    ws = set(words(c["text"]))
    for a in accepted:
        if overlap(ws, set(words(a["text"]))) >= f["dup_overlap"]:
            return False, "too close to an accepted question"
    return True, ""


def check_relevance(c: dict, threshold: float) -> Result:
    """Topic relevance at least the topic threshold."""
    if c["relevance"] < threshold:
        return False, f"relevance {c['relevance']:.2f} below {threshold:.2f}"
    return True, ""


def check_difficulty(c: dict, min_prob: float) -> Result:
    """Difficulty classifier gives the requested level enough probability."""
    if c["difficulty_prob"] < min_prob:
        return False, f"difficulty probability {c['difficulty_prob']:.2f} below {min_prob:.2f}"
    return True, ""


def filter_candidates(cands: list[dict], ctx: Context, topic: str, relax: float = 1.0) -> tuple[list, list]:
    """Run the seven filters in order. Returns accepted and rejected (with reason)."""
    f = ctx.cfg.filters
    threshold, min_prob = ctx.scorer.thresholds[topic] * relax, f["min_difficulty_prob"] * relax
    accepted, rejected = [], []
    for c in cands:
        checks = (lambda: check_format(c, f), lambda: check_clean(c, f), lambda: check_ending(c, f),
                  lambda: check_novel(c, f, ctx.train_set), lambda: check_duplicate(c, f, accepted),
                  lambda: check_relevance(c, threshold), lambda: check_difficulty(c, min_prob))
        reason = next((r for ok, r in (check() for check in checks) if not ok), None)
        (rejected if reason else accepted).append({**c, "reason": reason} if reason else c)
    return accepted, rejected


def score(cands: list[dict], ctx: Context, topic: str, level: str) -> list[dict]:
    """Attach relevance and difficulty probability to each candidate."""
    texts = [c["text"] for c in cands]
    if not texts:
        return cands
    rel, prob = ctx.scorer.relevance(texts, topic), ctx.scorer.difficulty_prob(texts, level)
    return [{**c, "relevance": float(r), "difficulty_prob": float(p)} for c, r, p in zip(cands, rel, prob)]


def rank(accepted: list[dict]) -> list[dict]:
    """Most relevant first, ties broken by average log probability. With an upload, relevance is
    averaged with similarity to the notes."""
    def key(c: dict) -> tuple:
        rel = (c["relevance"] + c["upload_sim"]) / 2 if "upload_sim" in c else c["relevance"]
        return -rel, -c["avg_logprob"]
    return sorted(accepted, key=key)


def notes_fill(ctx: Context, topic: str, level: str, have: list[dict], need: int,
               upload: dict, seed: int) -> list[dict]:
    """Template questions built from key terms in the uploaded notes, marked from_notes."""
    templates = yaml.safe_load(ctx.cfg.path("concepts").read_text(encoding="utf-8"))["templates"][level]
    rng = random.Random(seed)
    texts = [rng.choice(templates).format(c=term) for term in upload["key_terms"].get(topic, [])]
    texts = [t[:1].upper() + t[1:] for t in texts]
    accepted, _ = filter_candidates(score([{"text": t, "avg_logprob": 0.0} for t in texts], ctx, topic, level),
                                    ctx, topic, relax=ctx.cfg.filters["relax_factor"])
    taken = {h["text"].lower() for h in have}
    picks = [c for c in accepted if c["text"].lower() not in taken][:need]
    return [{**c, "avg_logprob": None, "origin": "from_notes"} for c in picks]


def bank_fill(ctx: Context, subject: str, topic: str, level: str, have: list[dict],
              need: int, seed: int) -> list[dict]:
    """Real questions from the bank to fill a gap, marked from_bank."""
    bank = ctx.bank
    rows = bank[(bank["subject"] == subject) & (bank["topic"] == topic) & (bank["difficulty"] == level)]
    texts = rows["question"].tolist()
    random.Random(seed).shuffle(texts)
    taken = {h["text"].lower() for h in have}
    picks = [t for t in texts if t.lower() not in taken][:need]
    return [{**c, "avg_logprob": None, "novel": False, "origin": "from_bank"}
            for c in score([{"text": t} for t in picks], ctx, topic, level)]


def clean_output(c: dict) -> dict:
    """Keep only the fields in the generate_questions contract."""
    return {k: c.get(k) for k in ("text", "avg_logprob", "relevance", "difficulty_prob", "novel", "origin")}


def generate_questions(subject: str, topic: str, difficulty: str, n: int, temperature: float | None = None,
                       top_k: int | None = None, seed: int | None = None, ctx: Context | None = None,
                       upload: dict | None = None) -> list[dict]:
    """n filtered questions for one cell, with fallback rounds, relaxed thresholds and bank fill.
    upload is {"text": ..., "key_terms": {topic: [...]}} when the paper comes from notes."""
    ctx = ctx or load_context()
    cfg, f = ctx.cfg, ctx.cfg.filters
    seed = cfg.seed if seed is None else seed
    pool: list[dict] = []
    for r in range(f["max_rounds"]):
        raw = ctx.sampler(subject, topic, difficulty, cfg.generate["candidates_per_cell"],
                          temperature, top_k, seed + r)
        pool += score(raw, ctx, topic, difficulty)
        accepted, _ = filter_candidates(pool, ctx, topic)
        if len(accepted) >= n:
            break
    else:
        accepted, _ = filter_candidates(pool, ctx, topic, relax=f["relax_factor"])
    if upload and accepted:
        sims = upload_similarity([c["text"] for c in accepted], upload["text"], ctx.scorer)
        accepted = [{**c, "upload_sim": float(v)} for c, v in zip(accepted, sims)]
    chosen = [{**c, "origin": "generated"} for c in rank(accepted)[:n]]
    if upload and len(chosen) < n:
        chosen += notes_fill(ctx, topic, difficulty, chosen, n - len(chosen), upload, seed)
    if len(chosen) < n:
        chosen += bank_fill(ctx, subject, topic, difficulty, chosen, n - len(chosen), seed)
    return [clean_output(c) for c in chosen]
