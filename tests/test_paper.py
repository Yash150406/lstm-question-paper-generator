import random
from collections import Counter

import pytest

from qpgen.config import load_config
from qpgen.paper import (allocate, build_paper, check_paper, co_marks, difficulty_for_marks, fill_section,
                         parse_pattern, replace_question, spec_from_pattern, spec_from_preset)

CFG = load_config()
TOPICS = CFG.subjects["dbms"]["topics"]


def stub_generate(subject, topic, difficulty, n, seed=0, **_):
    """Distinct fake questions that depend on the seed, like the real generator."""
    rng = random.Random(f"{topic}{difficulty}{seed}")
    return [{"text": f"Explain {fake_words(rng)} for {topic}.", "origin": "generated",
             "relevance": 0.5, "difficulty_prob": 0.9} for _ in range(n)]


def fake_words(rng, k=4):
    """A few made-up words, so two stub questions never look like near duplicates."""
    return " ".join(f"w{rng.randint(0, 10**9)}" for _ in range(k))


@pytest.mark.parametrize("preset", ["balanced", "easy_heavy", "hard_heavy"])
def test_every_preset_totals_70(preset):
    spec = spec_from_preset(CFG, "DBMS", TOPICS, preset)
    paper = build_paper(spec, 42, CFG, stub_generate)
    assert sum(q["marks"] for s in paper["sections"] for q in s["questions"]) == 70


def test_balanced_layout():
    spec = spec_from_preset(CFG, "DBMS", TOPICS, "balanced")
    assert [(s.name, s.difficulty, s.marks, s.count) for s in spec.sections] == [
        ("A", "easy", 2, 10), ("B", "medium", 5, 6), ("C", "hard", 10, 2)]
    assert spec.institute == "MMCOE"


def test_same_seed_same_paper_and_different_seed_differs():
    spec = spec_from_preset(CFG, "DBMS", TOPICS[:3], "balanced")
    a, b = build_paper(spec, 42, CFG, stub_generate), build_paper(spec, 42, CFG, stub_generate)
    assert a == b
    assert a != build_paper(spec, 43, CFG, stub_generate)


def test_allocation_rules():
    assert allocate(10, TOPICS, {}) == dict(zip(TOPICS, [2, 2, 1, 1, 1, 1, 1, 1]))
    alloc = allocate(6, TOPICS[:4], {TOPICS[0]: 3})
    assert sum(alloc.values()) == 6 and all(v >= 1 for v in alloc.values())
    assert allocate(2, TOPICS, {}) == dict(zip(TOPICS, [1, 1, 0, 0, 0, 0, 0, 0]))


@pytest.mark.parametrize("count, weights", [(10, {}), (7, {"ER model": 4, "Recovery": 0.5}), (30, {"SQL queries": 2})])
def test_no_topic_more_than_one_over_fair_share(count, weights):
    alloc = allocate(count, TOPICS, weights)
    w = {t: weights.get(t, 1.0) for t in TOPICS}
    for t, n in alloc.items():
        assert n <= count * w[t] / sum(w.values()) + 1


def test_questions_ordered_by_topic_and_unique():
    spec = spec_from_preset(CFG, "DBMS", TOPICS, "balanced")
    paper = build_paper(spec, 42, CFG, stub_generate)
    order = [TOPICS.index(q["topic"]) for q in paper["sections"][0]["questions"]]
    assert order == sorted(order)
    texts = [q["text"] for s in paper["sections"] for q in s["questions"]]
    assert len(set(texts)) == len(texts) and Counter(q["topic"] for q in paper["sections"][0]["questions"])


def test_repeats_from_generator_are_replaced():
    def repeating(subject, topic, difficulty, n, seed=0, **_):
        """Each call gives n copies of one question, so only one per call is usable."""
        return [{"text": f"Explain {fake_words(random.Random(seed))} for {topic}."}] * n
    spec = spec_from_preset(CFG, "DBMS", TOPICS[:4], "balanced")
    paper = build_paper(spec, 1, CFG, repeating)
    texts = [q["text"] for s in paper["sections"] for q in s["questions"]]
    assert len(set(texts)) == len(texts)


def test_check_paper_catches_bad_totals_and_repeats():
    spec = spec_from_preset(CFG, "DBMS", TOPICS, "balanced")
    paper = build_paper(spec, 42, CFG, stub_generate)
    paper["sections"][0]["questions"][1]["text"] = paper["sections"][0]["questions"][0]["text"]
    with pytest.raises(ValueError, match="repeats"):
        check_paper(paper)
    paper["sections"][0]["questions"].pop()
    with pytest.raises(ValueError):
        check_paper(paper)


def test_replace_question_changes_only_that_question():
    spec = spec_from_preset(CFG, "DBMS", TOPICS, "balanced")
    paper = build_paper(spec, 42, CFG, stub_generate)
    before = [q["text"] for s in paper["sections"] for q in s["questions"]]
    assert replace_question(paper, CFG, stub_generate, 1, 2, seed=999)
    after = [q["text"] for s in paper["sections"] for q in s["questions"]]
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert changed == [10 + 2]
    assert paper["sections"][1]["questions"][2]["topic"] == TOPICS[2]
    check_paper(paper)


def test_fill_section_restores_count_after_removal():
    spec = spec_from_preset(CFG, "DBMS", TOPICS, "balanced")
    paper = build_paper(spec, 42, CFG, stub_generate)
    removed = paper["sections"][0]["questions"].pop(3)
    fill_section(paper, CFG, stub_generate, 0, seed=500)
    check_paper(paper)
    assert sum(q["topic"] == removed["topic"] for q in paper["sections"][0]["questions"]) >= 1


@pytest.mark.parametrize("marks, level", [(1, "easy"), (3, "easy"), (4, "medium"), (7, "medium"), (8, "hard"), (15, "hard")])
def test_difficulty_follows_marks(marks, level):
    assert difficulty_for_marks(CFG, marks) == level


def test_custom_pattern_with_any_n_of_m():
    rows = [{"marks": 1, "count": 10}, {"marks": 5, "count": 6, "attempt": 4}, {"marks": 8, "count": 2, "difficulty": "medium"}]
    spec = spec_from_pattern(CFG, "DBMS", TOPICS, rows)
    assert [(s.name, s.difficulty, s.marks, s.count, s.attempt) for s in spec.sections] == [
        ("A", "easy", 1, 10, 10), ("B", "medium", 5, 6, 4), ("C", "medium", 8, 2, 2)]
    assert spec.total_marks == 10 + 20 + 16
    assert spec.sections[1].instruction == "Answer any 4 of 6"
    paper = build_paper(spec, 42, CFG, stub_generate)
    assert [len(s["questions"]) for s in paper["sections"]] == [10, 6, 2]


@pytest.mark.parametrize("rows", [[], [{"marks": 5, "count": 3, "attempt": 4}], [{"marks": 0, "count": 3}]])
def test_bad_pattern_is_refused(rows):
    with pytest.raises(ValueError):
        spec_from_pattern(CFG, "DBMS", TOPICS, rows)


def test_rows_with_no_questions_are_skipped():
    spec = spec_from_pattern(CFG, "DBMS", TOPICS, [{"marks": 2, "count": 0}, {"marks": 5, "count": 2}])
    assert [s.name for s in spec.sections] == ["A"] and spec.total_marks == 10


def test_parse_pattern():
    assert parse_pattern("2:10, 5:6:4") == [{"marks": 2, "count": 10, "attempt": 10}, {"marks": 5, "count": 6, "attempt": 4}]
    with pytest.raises(ValueError):
        parse_pattern("2x10")


def test_questions_carry_co_and_bloom():
    spec = spec_from_preset(CFG, "DBMS", TOPICS, "balanced", co_map={"Recovery": "CO9"})
    paper = build_paper(spec, 42, CFG, stub_generate)
    qs = [q for s in paper["sections"] for q in s["questions"]]
    assert all(q["bloom"] == "BL2" for q in qs)          # stub questions all open with "Explain"
    assert {q["co"] for q in qs if q["topic"] == "ER model"} == {"CO1"}
    assert {q["co"] for q in qs if q["topic"] == "Recovery"} == {"CO9"}
    assert sum(co_marks(paper).values()) == 70
    replace_question(paper, CFG, stub_generate, 0, 0, seed=7)
    assert paper["sections"][0]["questions"][0]["co"] == "CO1"
