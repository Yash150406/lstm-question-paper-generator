import pytest

from qpgen.data.labels import bloom_level, label_difficulty


@pytest.mark.parametrize("question, level", [
    ("What is a functional dependency?", "easy"),
    ("Define deadlock.", "easy"),
    ("Write the full form of ACID.", "easy"),
    ("Explain paging with an example.", "medium"),
    ("Differentiate between 3NF and BCNF.", "medium"),
    ("Write a query to find the second highest salary.", "medium"),
    ("Design a B+ tree of order 3 for the given keys.", "hard"),
    ("Discuss in detail the working of TCP congestion control.", "hard"),
    ("Prove that Dijkstra's algorithm is correct.", "hard"),
])
def test_opening_verb_sets_level(question, level):
    assert label_difficulty(question) == level


def test_unknown_opener_falls_back_to_medium():
    assert label_difficulty("Consider a relation R with five attributes.") == "medium"


@pytest.mark.parametrize("question, level", [
    ("Define deadlock.", "BL1"),
    ("What is a functional dependency?", "BL1"),
    ("Explain paging with an example.", "BL2"),
    ("Write a query to find the second highest salary.", "BL3"),
    ("Solve the given recurrence.", "BL3"),
    ("Differentiate between 3NF and BCNF.", "BL4"),
    ("Evaluate the cost of a nested loop join.", "BL5"),
    ("Design a B+ tree of order 3 for the given keys.", "BL6"),
    ("Consider a relation R with five attributes.", "BL2"),
])
def test_bloom_level_from_opening_verb(question, level):
    assert bloom_level(question) == level
