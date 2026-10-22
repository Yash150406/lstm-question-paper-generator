"""Paper spec, topic allocation and the paper builder."""
import math
from collections.abc import Callable
from dataclasses import dataclass, field

from qpgen.config import Config
from qpgen.data.build_dataset import overlap, words
from qpgen.data.labels import bloom_level


@dataclass
class Section:
    """One section of a paper: all questions share a difficulty and marks."""

    name: str
    difficulty: str
    marks: int
    count: int          # questions printed
    instruction: str
    attempt: int = 0    # questions to answer, 0 means all

    def __post_init__(self):
        self.attempt = self.attempt or self.count


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
    co_map: dict[str, str] = field(default_factory=dict)   # topic -> course outcome


def difficulty_for_marks(cfg: Config, marks: int) -> str:
    """Default difficulty for a mark value, from paper.difficulty_by_marks."""
    for level, top in cfg.paper["difficulty_by_marks"].items():
        if marks <= top:
            return level
    return "hard"


def preset_rows(cfg: Config, preset: str) -> list[dict]:
    """A preset as pattern rows."""
    return [{"marks": cfg.marks[lv], "count": n, "attempt": n, "difficulty": lv}
            for lv, n in cfg.paper["presets"][preset].items() if n > 0]


def parse_pattern(text: str) -> list[dict]:
    """'2:10,5:6:4' -> rows of marks, questions printed and questions to attempt."""
    rows = []
    for part in text.split(","):
        nums = [int(x) for x in part.strip().split(":") if x.strip().isdigit()]
        if len(nums) not in (2, 3) or len(nums) != len(part.split(":")):
            raise ValueError(f"Pattern part '{part.strip()}' should be marks:questions or marks:questions:attempt")
        rows.append({"marks": nums[0], "count": nums[1], "attempt": nums[-1] if len(nums) == 3 else nums[1]})
    return rows


def spec_from_pattern(cfg: Config, subject: str, topics: list[str], rows: list[dict],
                      weights: dict[str, float] | None = None, co_map: dict[str, str] | None = None,
                      **overrides) -> PaperSpec:
    """Build a spec from rows of {marks, count, attempt, difficulty}; attempt and difficulty are optional.
    Rows with no questions are skipped. overrides can set institute, exam_title or duration_minutes."""
    p, sections = cfg.paper, []
    for r in (r for r in rows if int(r["count"]) > 0):
        marks, count = int(r["marks"]), int(r["count"])
        attempt = int(r.get("attempt") or count)
        if marks <= 0 or not 0 < attempt <= count:
            raise ValueError(f"Bad row: {marks} marks, attempt {attempt} of {count}")
        level = r.get("difficulty") or difficulty_for_marks(cfg, marks)
        note = p["instruction"] if attempt == count else f"Answer any {attempt} of {count}"
        sections.append(Section(chr(ord("A") + len(sections)), level, marks, count, note, attempt))
    if not sections:
        raise ValueError("The pattern has no questions")
    default_co = cfg.subjects[cfg.subject_key(subject)].get("co", {})
    fields = {"institute": p["institute"], "exam_title": p["exam_title"],
              "duration_minutes": p["duration_minutes"], **overrides}
    return PaperSpec(subject=subject, total_marks=sum(s.attempt * s.marks for s in sections),
                     topics=list(topics), sections=sections, weights=weights or {},
                     co_map={**default_co, **(co_map or {})}, **fields)


def spec_from_preset(cfg: Config, subject: str, topics: list[str], preset: str,
                     weights: dict[str, float] | None = None, **overrides) -> PaperSpec:
    """Build a spec from a preset in config."""
    return spec_from_pattern(cfg, subject, topics, preset_rows(cfg, preset), weights, **overrides)


def tag(q: dict, topic: str, marks: int, spec: PaperSpec) -> dict:
    """A picked question with its topic, marks, course outcome and Bloom level."""
    return {**q, "topic": topic, "marks": marks, "co": spec.co_map.get(topic, ""), "bloom": bloom_level(q["text"])}


def co_marks(paper: dict) -> dict[str, int]:
    """Marks of the printed questions for each course outcome."""
    out: dict[str, int] = {}
    for s in paper["sections"]:
        for q in s["questions"]:
            out[q.get("co", "")] = out.get(q.get("co", ""), 0) + q["marks"]
    return dict(sorted(out.items()))


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
            questions += [tag(q, topic, sec.marks, spec) for q in got]
        sections.append({"name": sec.name, "difficulty": sec.difficulty, "marks": sec.marks,
                         "attempt": sec.attempt, "instruction": sec.instruction, "questions": questions})
    paper = {"spec": spec, "seed": seed, "sections": sections}
    check_paper(paper)
    return paper


def check_paper(paper: dict) -> None:
    """Raise ValueError if a section is short, marks do not add up, or a question is empty or repeated."""
    spec = paper["spec"]
    for sec, s in zip(spec.sections, paper["sections"]):
        if len(s["questions"]) != sec.count:
            raise ValueError(f"Section {sec.name} has {len(s['questions'])} questions, expected {sec.count}")
        if any(q["marks"] != sec.marks for q in s["questions"]):
            raise ValueError(f"Section {sec.name} has a question that is not {sec.marks} marks")
    total = sum(sec.attempt * sec.marks for sec in spec.sections)
    if total != spec.total_marks:
        raise ValueError(f"Paper totals {total} marks, expected {spec.total_marks}")
    texts = [q["text"].strip().lower() for s in paper["sections"] for q in s["questions"]]
    if not all(texts):
        raise ValueError("Paper has an empty question")
    if len(set(texts)) != len(texts):
        raise ValueError("Paper repeats a question")


def paper_words(paper: dict, skip: dict | None = None) -> list[set[str]]:
    """Word sets of every question on the paper, leaving out one if given."""
    return [set(words(q["text"])) for s in paper["sections"] for q in s["questions"] if q is not skip]


def replace_question(paper: dict, cfg: Config, generate: Callable[..., list[dict]],
                     s_idx: int, q_idx: int, seed: int) -> bool:
    """Swap one question for a new one on the same topic. Returns False if none was found."""
    sec = paper["sections"][s_idx]
    old = sec["questions"][q_idx]
    used = paper_words(paper, skip=old)
    for attempt in range(cfg.filters["max_rounds"]):
        cands = generate(paper["spec"].subject, old["topic"], sec["difficulty"],
                         1 + cfg.paper["extra_candidates"], seed=seed + attempt)
        got = pick_unique([c for c in cands if c["text"] != old["text"]], used, 1, cfg.filters["dup_overlap"])
        if got:
            sec["questions"][q_idx] = tag(got[0], old["topic"], sec["marks"], paper["spec"])
            return True
    return False


def fill_section(paper: dict, cfg: Config, generate: Callable[..., list[dict]], s_idx: int, seed: int) -> None:
    """Add questions to a section until it is back to its planned count, topic by topic."""
    spec, sec = paper["spec"], paper["sections"][s_idx]
    target = allocate(spec.sections[s_idx].count, spec.topics, spec.weights)
    used = paper_words(paper)
    for t_idx, (topic, n) in enumerate(target.items()):
        missing = n - sum(q["topic"] == topic for q in sec["questions"])
        for attempt in range(cfg.filters["max_rounds"]):
            if missing <= 0:
                break
            cands = generate(spec.subject, topic, sec["difficulty"], missing + cfg.paper["extra_candidates"],
                             seed=seed + 100 * attempt + t_idx)
            got = pick_unique(cands, used, missing, cfg.filters["dup_overlap"])
            sec["questions"] += [tag(q, topic, sec["marks"], spec) for q in got]
            missing -= len(got)
    sec["questions"].sort(key=lambda q: spec.topics.index(q["topic"]))
