"""Paper spec, topic allocation and the paper builder."""
import math
from collections.abc import Callable
from dataclasses import dataclass, field

from qpgen.config import Config
from qpgen.data.build_dataset import overlap, words

SECTION_NAMES = {"easy": "A", "medium": "B", "hard": "C"}


@dataclass
class Section:
    """One section of a paper: all questions share a difficulty and marks."""

    name: str
    difficulty: str
    marks: int
    count: int
    instruction: str


@dataclass
class PaperSpec:
    """Everything that decides what a paper contains."""

    institute: str
    subject: str
    exam_title: str
    duration_minutes: int
    total_marks: int
    topics: list[str]
    sections: list[Section]
    weights: dict[str, float] = field(default_factory=dict)


def spec_from_preset(cfg: Config, subject: str, topics: list[str], preset: str,
                     weights: dict[str, float] | None = None, **overrides) -> PaperSpec:
    """Build a spec from a preset in config. overrides can set institute, exam_title or duration_minutes."""
    p = cfg.paper
    sections = [Section(SECTION_NAMES[lv], lv, cfg.marks[lv], n, p["instruction"])
                for lv, n in p["presets"][preset].items() if n > 0]
    fields = {"institute": p["institute"], "exam_title": p["exam_title"],
              "duration_minutes": p["duration_minutes"], **overrides}
    return PaperSpec(subject=subject, total_marks=p["total_marks"], topics=list(topics),
                     sections=sections, weights=weights or {}, **fields)


def allocate(count: int, topics: list[str], weights: dict[str, float]) -> dict[str, int]:
    """Spread count questions over topics by weight with the largest remainder method."""
    w = [weights.get(t, 1.0) for t in topics]
    share = [count * x / sum(w) for x in w]
    alloc = [math.floor(s) for s in share]
    by_remainder = sorted(range(len(topics)), key=lambda i: (-(share[i] - alloc[i]), i))
    for i in by_remainder[:count - sum(alloc)]:
        alloc[i] += 1
    if count >= len(topics):
        # Every topic gets at least one, taken from the topic furthest above its share.
        for i in [i for i, a in enumerate(alloc) if a == 0]:
            donor = max(range(len(topics)), key=lambda j: (alloc[j] - share[j], -j))
            alloc[donor] -= 1
            alloc[i] = 1
    return dict(zip(topics, alloc))


def pick_unique(candidates: list[dict], used: list[set[str]], need: int, limit: float) -> list[dict]:
    """Take candidates in rank order, skipping any too close to a question already on the paper."""
    picked = []
    for c in candidates:
        ws = set(words(c["text"]))
        if c["text"].strip() and all(overlap(ws, u) < limit for u in used):
            picked.append(c)
            used.append(ws)
            if len(picked) == need:
                break
    return picked


def build_paper(spec: PaperSpec, seed: int, cfg: Config, generate: Callable[..., list[dict]]) -> dict:
    """Ask generate (normally generate_questions) for each section and topic and assemble the paper."""
    used: list[set[str]] = []
    sections = []
    for s_idx, sec in enumerate(spec.sections):
        questions = []
        for t_idx, (topic, n) in enumerate(allocate(sec.count, spec.topics, spec.weights).items()):
            got: list[dict] = []
            for attempt in range(cfg.filters["max_rounds"]):
                if len(got) == n:
                    break
                cell_seed = seed + 1000 * s_idx + 100 * attempt + t_idx
                cands = generate(spec.subject, topic, sec.difficulty, n + cfg.paper["extra_candidates"], seed=cell_seed)
                got += pick_unique(cands, used, n - len(got), cfg.filters["dup_overlap"])
            questions += [{**q, "topic": topic, "marks": sec.marks} for q in got]
        sections.append({"name": sec.name, "difficulty": sec.difficulty, "marks": sec.marks,
                         "instruction": sec.instruction, "questions": questions})
    paper = {"spec": spec, "seed": seed, "sections": sections}
    check_paper(paper)
    return paper


def check_paper(paper: dict) -> None:
    """Raise ValueError if marks do not add up or a question is empty or repeated."""
    spec = paper["spec"]
    total = sum(q["marks"] for s in paper["sections"] for q in s["questions"])
    if total != spec.total_marks:
        raise ValueError(f"Paper totals {total} marks, expected {spec.total_marks}")
    for sec, s in zip(spec.sections, paper["sections"]):
        if len(s["questions"]) != sec.count:
            raise ValueError(f"Section {sec.name} has {len(s['questions'])} questions, expected {sec.count}")
    texts = [q["text"].strip().lower() for s in paper["sections"] for q in s["questions"]]
    if not all(texts):
        raise ValueError("Paper has an empty question")
    if len(set(texts)) != len(texts):
        raise ValueError("Paper repeats a question")
