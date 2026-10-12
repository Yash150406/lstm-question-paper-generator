import pytest

from qpgen.data.labels import label_difficulty


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
