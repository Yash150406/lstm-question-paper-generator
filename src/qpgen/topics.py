"""Add a new topic: check the pasted questions, save them, and retrain."""
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

from qpgen.config import Config
from qpgen.data.labels import label_difficulty
from qpgen.tokenizer import slug

NUMBERING = re.compile(r"^\s*(q\s*)?\d+\s*[.)\]:-]\s*", re.IGNORECASE)
TOPIC_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 +&()/-]{1,39}$")
COLUMNS = ["subject", "topic", "difficulty", "question"]
RETRAIN_MODULES = ["qpgen.data.build_dataset", "qpgen.train", "qpgen.classifiers"]


def question_rows(text: str) -> list[dict]:
    """One question per non-empty line, numbering stripped, difficulty guessed from the opening verb."""
    rows = []
    for line in text.splitlines():
        q = NUMBERING.sub("", line).strip()
        if q:
            rows.append({"question": q, "difficulty": label_difficulty(q)})
    return rows


def check_new_topic(cfg: Config, subject: str, topic: str, rows: list[dict]) -> list[str]:
    """Problems that stop the topic from being added; empty when it is fine."""
    errors = []
    if not TOPIC_NAME.match(topic.strip()):
        errors.append("The topic name should be 2 to 40 letters, digits, spaces or + & ( ) / -")
    taken = {slug(t) for info in cfg.subjects.values() for t in info["topics"]}
    if slug(topic) in taken:
        errors.append(f"A topic called '{topic.strip()}' already exists")
    lo, hi = cfg.data["min_words"], cfg.data["max_words"]
    usable = [r for r in rows if lo <= len(r["question"].split()) <= hi]
    if len(usable) < len(rows):
        errors.append(f"{len(rows) - len(usable)} questions are not between {lo} and {hi} words")
    need, per_level = cfg.data["new_topic_min"], cfg.data["new_topic_min_per_level"]
    if len({r["question"].lower() for r in usable}) < need:
        errors.append(f"Paste at least {need} different questions (found {len(usable)})")
    for level in cfg.marks:
        n = sum(r["difficulty"] == level for r in usable)
        if n < per_level:
            errors.append(f"Needs at least {per_level} {level} questions (found {n})")
    return errors


def save_new_topic(cfg: Config, subject: str, topic: str, co: str, rows: list[dict]) -> None:
    """Append the questions to student_questions.csv and the topic and its CO to subjects_topics.yaml.
    The YAML is edited as text so its comments survive."""
    key, topic = cfg.subject_key(subject), topic.strip()
    path = cfg.path("subjects_topics")
    lines = path.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"  {key}:")
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"^  \S", lines[i])), len(lines))
    block = range(start, end)
    last_topic = max(i for i in block if re.match(r"^      - ", lines[i]))
    last_co = max(i for i in block if re.match(r"^      \S.*: CO", lines[i]))
    lines.insert(last_co + 1, f"      {topic}: {co}")
    lines.insert(last_topic + 1, f"      - {topic}")
    text = "\n".join(lines) + "\n"
    info = yaml.safe_load(text)["subjects"][key]
    if info["topics"][-1] != topic or info["co"].get(topic) != co:
        raise ValueError("subjects_topics.yaml has an unexpected layout, add the topic by hand")
    path.write_text(text, encoding="utf-8")

    csv = cfg.path("raw_dir") / "student_questions.csv"
    df = pd.DataFrame([{"subject": cfg.subjects[key]["name"], "topic": topic, **r} for r in rows])[COLUMNS]
    df.to_csv(csv, mode="a", header=not csv.exists(), index=False)


def retrain_commands(config: str | Path) -> list[list[str]]:
    """Rebuild the dataset, train the LSTM and refit the classifiers, in that order."""
    return [[sys.executable, "-m", m, "--config", str(config)] for m in RETRAIN_MODULES]
