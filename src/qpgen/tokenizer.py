"""Word tokenizer, control tokens and the vocabulary."""
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

TOKEN_RE = re.compile(r"\w+|[^\w\s]")
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = [PAD, UNK, BOS, EOS]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = range(4)

# Spacing fixes applied after joining tokens with spaces, in order.
SPACING = [
    (re.compile(r" ([.,?!;:)\]'+])"), r"\1"),   # no space before closing punctuation
    (re.compile(r"([(\[]) "), r"\1"),          # no space after an opening bracket
    (re.compile(r" ?/ ?"), "/"),                # TCP/IP, 0/1
    (re.compile(r"' (s|t)\b"), r"'\1"),         # Banker's
    (re.compile(r"(\d)\. (\d)"), r"\1.\2"),     # 192.168.1.0, 802.11
]
SENTENCE_START = re.compile(r"([.?!] )([a-z])")    # "thread? give" -> "thread? Give"


def tokenize(text: str) -> list[str]:
    """Lowercase and split into words and single punctuation marks."""
    return TOKEN_RE.findall(text.lower())


def detokenize(tokens: list[str], acronyms: dict[str, str] | None = None) -> str:
    """Join tokens into a readable sentence with acronyms restored."""
    acronyms = acronyms or {}
    words = [acronyms.get(t, t) for t in tokens if not (t.startswith("<") and t.endswith(">"))]
    text = " ".join(words)
    for pattern, repl in SPACING:
        text = pattern.sub(repl, text)
    text = SENTENCE_START.sub(lambda m: m.group(1) + m.group(2).upper(), text)
    return text[:1].upper() + text[1:]


def slug(name: str) -> str:
    """Turn a topic name into a token-safe slug."""
    return re.sub(r"\W+", "_", name.lower()).strip("_")


def subject_token(key: str) -> str:
    """Control token for a subject key, such as <subj_dbms>."""
    return f"<subj_{key}>"


def topic_token(topic: str) -> str:
    """Control token for a topic name, such as <topic_normalization>."""
    return f"<topic_{slug(topic)}>"


def difficulty_token(level: str) -> str:
    """Control token for a difficulty level, such as <diff_medium>."""
    return f"<diff_{level}>"


def control_tokens(subjects: dict, levels: list[str]) -> list[str]:
    """All control tokens built from the subject config."""
    tokens = [subject_token(k) for k in subjects]
    tokens += [topic_token(t) for info in subjects.values() for t in info["topics"]]
    return tokens + [difficulty_token(lv) for lv in levels]


@dataclass
class Vocab:
    """Maps tokens to ids. Specials come first, then control tokens, then words."""

    itos: list[str]
    controls: list[str]
    stoi: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, sentences: list[list[str]], controls: list[str],
              min_freq: int, max_words: int) -> "Vocab":
        """Count words in tokenized sentences and keep the frequent ones."""
        counts = Counter(t for s in sentences for t in s)
        words = sorted((w for w, c in counts.items() if c >= min_freq), key=lambda w: (-counts[w], w))
        return cls(SPECIALS + controls + words[:max_words], controls)

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def control_ids(self) -> list[int]:
        """Ids of all control tokens."""
        return [self.stoi[t] for t in self.controls]

    def encode(self, tokens: list[str]) -> list[int]:
        """Token strings to ids, unknown words become <unk>."""
        return [self.stoi.get(t, UNK_ID) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        """Ids back to token strings."""
        return [self.itos[i] for i in ids]

    def save(self, path: Path) -> None:
        """Write the vocabulary to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"itos": self.itos, "controls": self.controls}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        """Read a vocabulary saved with save()."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["itos"], data["controls"])
