"""Merge seed, template and student questions, clean them and split."""
import argparse
import random
import re
import sys

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from qpgen.config import DEFAULT_CONFIG, Config, load_config
from qpgen.data.labels import label_difficulty

COLUMNS = ["id", "subject", "topic", "difficulty", "marks", "question", "source"]
# When two rows are duplicates, the one from the earlier source is kept.
SOURCE_ORDER = ["student", "seed", "template", "mmlu"]


def words(text: str) -> list[str]:
    """Lowercase word tokens used for length and overlap checks."""
    return re.findall(r"\w+", text.lower())


def overlap(a: set[str], b: set[str]) -> float:
    """Jaccard overlap of two word sets."""
    return len(a & b) / len(a | b) if a | b else 0.0


def load_concepts(cfg: Config) -> dict:
    """Read templates and concept lists."""
    return yaml.safe_load(cfg.path("concepts").read_text(encoding="utf-8"))


def load_seeds(cfg: Config) -> pd.DataFrame:
    """Read data/raw/seed_<subject>.csv for every subject."""
    frames = []
    for key, info in cfg.subjects.items():
        df = pd.read_csv(cfg.path("raw_dir") / f"seed_{key}.csv")
        frames.append(df.assign(subject=info["name"], source="seed"))
    return pd.concat(frames, ignore_index=True)


def load_student(cfg: Config) -> pd.DataFrame:
    """Read the optional student_questions.csv."""
    path = cfg.path("raw_dir") / "student_questions.csv"
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path).assign(source="student")


def expand_templates(cfg: Config, concepts: dict, rng: random.Random) -> pd.DataFrame:
    """Fill a random subset of templates for every concept and pair."""
    templates, per = concepts["templates"], cfg.data["templates_per_concept"]
    rows = []
    for key, info in cfg.subjects.items():
        for topic, entry in concepts[key].items():
            for concept in entry["concepts"]:
                for level, n in per.items():
                    for t in rng.sample(templates[level], n):
                        rows.append((info["name"], topic, level, t.format(c=concept)))
            for a, b in entry["pairs"]:
                t = rng.choice(templates["pair_medium"])
                rows.append((info["name"], topic, "medium", t.format(a=a, b=b)))
    df = pd.DataFrame(rows, columns=["subject", "topic", "difficulty", "question"])
    return df.assign(source="template")


def drop_near_duplicates(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Within each topic, drop a row whose word overlap with a kept row is above threshold."""
    # ponytail: O(n^2) per topic, fine for ~150 rows a topic; use MinHash if the bank grows 100x
    keep = []
    for _, group in df.groupby("topic", sort=False):
        kept_sets = []
        for idx, q in group["question"].items():
            ws = set(words(q))
            if all(overlap(ws, k) <= threshold for k in kept_sets):
                kept_sets.append(ws)
                keep.append(idx)
    return df.loc[sorted(keep)]


def valid_topics(cfg: Config) -> set[tuple[str, str]]:
    """All allowed (subject name, topic) pairs."""
    return {(info["name"], t) for info in cfg.subjects.values() for t in info["topics"]}


def clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Normalise text, fill labels, filter length and remove duplicates."""
    df = df.copy()
    df["question"] = df["question"].astype(str).str.split().str.join(" ")
    blank = df["difficulty"].isna() | (df["difficulty"].astype(str).str.strip() == "")
    df.loc[blank, "difficulty"] = df.loc[blank, "question"].map(label_difficulty)
    df["marks"] = df["difficulty"].map(cfg.marks)

    allowed = valid_topics(cfg)
    ok = [(s, t) in allowed for s, t in zip(df["subject"], df["topic"])]
    if not all(ok):
        print(f"Dropping {ok.count(False)} rows with an unknown subject or topic")
    df = df[ok]

    n_words = df["question"].map(lambda q: len(words(q)))
    df = df[n_words.between(cfg.data["min_words"], cfg.data["max_words"])]

    df = df.assign(_rank=df["source"].map(SOURCE_ORDER.index))
    df = df.sort_values("_rank", kind="stable").drop(columns="_rank")
    df = df[~df["question"].str.lower().str.strip().duplicated()]
    df = drop_near_duplicates(df, cfg.data["near_dup_overlap"])
    df = df.reset_index(drop=True)
    df["id"] = df.index + 1
    return df[COLUMNS]


def split(df: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Stratified train, validation and test split by subject and difficulty."""
    _, val_frac, test_frac = cfg.data["split"]
    strata = df["subject"] + "|" + df["difficulty"]
    train, rest = train_test_split(df, test_size=val_frac + test_frac,
                                   stratify=strata, random_state=cfg.seed)
    val, test = train_test_split(rest, test_size=test_frac / (val_frac + test_frac),
                                 stratify=strata[rest.index], random_state=cfg.seed)
    return {"train": train, "val": val, "test": test}


def build(cfg: Config) -> pd.DataFrame:
    """Collect every source and return the cleaned question bank."""
    rng = random.Random(cfg.seed)
    concepts = load_concepts(cfg)
    parts = [load_student(cfg), load_seeds(cfg), expand_templates(cfg, concepts, rng)]
    parts = [p for p in parts if not p.empty]
    return clean(pd.concat(parts, ignore_index=True), cfg)


def report(df: pd.DataFrame) -> None:
    """Print counts by subject, difficulty and topic."""
    print(f"Total questions: {len(df)}\n")
    print(pd.crosstab(df["subject"], df["difficulty"], margins=True), "\n")
    print(pd.crosstab(df["subject"], df["source"], margins=True), "\n")
    print(df.groupby(["subject", "topic"]).size().to_string(), "\n")


def main(argv: list[str] | None = None) -> int:
    """Build questions.csv and the three splits. Returns an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    cfg = load_config(parser.parse_args(argv).config)

    df = build(cfg)
    report(df)
    small = df["subject"].value_counts().reindex(
        [s["name"] for s in cfg.subjects.values()], fill_value=0)
    small = small[small < cfg.data["min_per_subject"]]
    if not small.empty:
        print(f"Build failed, too few questions for: {dict(small)}")
        return 1

    out = cfg.path("processed_dir")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "questions.csv", index=False)
    for name, part in split(df, cfg).items():
        part.to_csv(out / f"{name}.csv", index=False)
        print(f"{name}: {len(part)} rows ({len(part) / len(df):.0%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
