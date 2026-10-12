"""Guess difficulty from the verb a question opens with."""

# Checked hard first so "Discuss in detail" wins over shorter matches.
RULES = {
    "hard": ["design", "analyze", "analyse", "derive", "prove", "evaluate",
             "implement", "solve", "discuss in detail", "discuss"],
    "medium": ["explain", "differentiate", "compare", "describe", "illustrate",
               "write a query", "write an algorithm", "write a program",
               "write an sql", "write sql", "construct", "draw", "calculate", "find"],
    "easy": ["what is", "what are", "define", "list", "state", "name",
             "write the full form of", "expand", "give", "mention", "which"],
}
DEFAULT = "medium"


def label_difficulty(question: str) -> str:
    """Return easy, medium or hard based on the opening words."""
    text = question.strip().lower()
    for level, openers in RULES.items():
        if any(text.startswith(opener) for opener in openers):
            return level
    return DEFAULT
