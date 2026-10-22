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

# Bloom's taxonomy level from the opening verb. Checked top to bottom, first match wins.
BLOOM_RULES = {
    "BL6": ["design", "develop", "propose", "create"],
    "BL5": ["evaluate", "justify", "prove", "assess"],
    "BL4": ["analyze", "analyse", "differentiate", "compare", "distinguish", "contrast"],
    "BL3": ["solve", "calculate", "find", "compute", "implement", "construct", "draw", "derive",
            "apply", "illustrate", "write a", "write an", "write sql", "convert", "trace", "show"],
    "BL1": ["what is", "what are", "define", "list", "state", "name", "write the full form of",
            "expand", "mention", "which", "give", "recall"],
}
BLOOM_DEFAULT = "BL2"   # Understand: explain, describe, discuss and anything else


def bloom_level(question: str) -> str:
    """Return BL1 (Remember) to BL6 (Create) based on the opening words."""
    text = question.strip().lower()
    for level, openers in BLOOM_RULES.items():
        if any(text.startswith(opener) for opener in openers):
            return level
    return BLOOM_DEFAULT
