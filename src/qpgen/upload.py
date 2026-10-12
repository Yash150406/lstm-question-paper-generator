"""Read uploaded notes and map them onto the fixed subject and topic list."""
import io
import re
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.preprocessing import normalize

from qpgen.classifiers import Scorer
from qpgen.config import Config

SENTENCE_END = re.compile(r"[.\n?!]+")


def read_upload(source: str | Path | bytes, name: str = "") -> str:
    """Text from PDF bytes, text bytes, a file path, or a plain string."""
    if isinstance(source, bytes):
        if name.lower().endswith(".pdf"):
            return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(source)).pages)
        return source.decode("utf-8", errors="ignore")
    path = Path(source)
    if len(str(source)) < 260 and path.suffix.lower() in {".pdf", ".txt", ".md"} and path.is_file():
        return read_upload(path.read_bytes(), path.name)
    return str(source)


def content_words(text: str) -> str:
    """Drop English stop words, which otherwise make any English text look a bit like every topic."""
    return " ".join(w for w in re.findall(r"[\w+/']+", text.lower()) if w not in ENGLISH_STOP_WORDS)


def match_name(text: str, cfg: Config) -> dict | None:
    """A typed subject or topic name, or a comma separated list of topics, taken as is."""
    parts = [p.strip().lower() for p in text.split(",") if p.strip()]
    for key, info in cfg.subjects.items():
        if text.strip().lower() in (key, info["name"].lower()):
            return {"subject": info["name"], "topics": {t: 1.0 for t in info["topics"]}}
        topics = [t for t in info["topics"] if t.lower() in parts]
        if topics and len(topics) == len(parts):
            return {"subject": info["name"], "topics": {t: 1.0 for t in topics}}
    return None


def subject_scores(vec: np.ndarray, scorer: Scorer, cfg: Config) -> dict[str, float]:
    """Cosine of the text to each subject centroid (the mean of its topic centroids)."""
    out = {}
    for info in cfg.subjects.values():
        rows = [scorer.topics.index(t) for t in info["topics"]]
        centroid = normalize(scorer.centroids[rows].mean(axis=0, keepdims=True))[0]
        out[info["name"]] = float(vec @ centroid)
    return out


def key_terms(text: str, topics: list[str], scorer: Scorer, per_topic: int) -> dict[str, list[str]]:
    """Terms from the notes that sit closest to each chosen topic's centroid."""
    out: dict[str, list[str]] = {t: [] for t in topics}
    sentences = [s for s in (content_words(p) for p in SENTENCE_END.split(text)) if s]
    if not sentences:
        return out
    known = set(scorer.vectorizer.vocabulary_)   # only phrases that occur in real questions
    terms = [t for t in CountVectorizer(ngram_range=(1, 2)).fit(sentences).get_feature_names_out()
             if t in known and t.replace(" ", "").isalpha() and len(t) > 2]
    if not terms:
        return out
    rows = [scorer.topics.index(t) for t in topics]
    rel = scorer.vectors(terms) @ scorer.centroids[rows].T          # terms x chosen topics
    for i in np.argsort(-rel.max(axis=1), kind="stable"):
        topic = topics[int(rel[i].argmax())]
        covered = any(terms[i] in kept.split() for kept in out[topic])
        if rel[i].max() >= scorer.thresholds[topic] and len(out[topic]) < per_topic and not covered:
            out[topic].append(terms[i])
    return out


def analyze_upload(text: str, cfg: Config, scorer: Scorer) -> dict:
    """Detected subject, topic weights and key terms, or supported=False with a message."""
    u = cfg.upload
    names = [info["name"] for info in cfg.subjects.values()]
    if len(text.split()) <= u["max_name_words"] and (hit := match_name(text, cfg)):
        return {**hit, "supported": True, "key_terms": {t: [] for t in hit["topics"]}, "scores": {}}
    vec = scorer.vectors([content_words(text)])[0]
    subj = subject_scores(vec, scorer, cfg)
    best = max(subj, key=subj.get)
    if subj[best] < u["min_subject_score"]:
        return {"supported": False, "scores": subj,
                "message": f"This material does not match a supported subject. Supported: {', '.join(names)}."}
    topic_list = cfg.subjects[cfg.subject_key(best)]["topics"]
    weights = topic_shares(text, topic_list, scorer, u["min_topic_share"])
    if not weights:      # no single sentence is clearly on one topic, fall back to the whole text
        rel = {t: float(vec @ scorer.centroids[scorer.topics.index(t)]) for t in topic_list}
        weights = {max(rel, key=rel.get): 1.0}
    top = sorted(weights, key=lambda t: -weights[t])[:u["max_topics"]]
    chosen = {t: weights[t] for t in topic_list if t in top}      # keep syllabus order
    return {"supported": True, "subject": best, "topics": chosen,
            "key_terms": key_terms(text, list(chosen), scorer, u["key_terms_per_topic"]), "scores": subj}


def topic_shares(text: str, topics: list[str], scorer: Scorer, min_share: float) -> dict[str, float]:
    """Give each sentence to its closest topic when it clears that topic's threshold, then return
    each topic's share of those sentences. Long notes usually span several topics, and scoring
    the whole text at once hides the smaller ones."""
    sentences = [c for c in (content_words(p) for p in SENTENCE_END.split(text)) if len(c.split()) >= 3]
    if not sentences:
        return {}
    rows = [scorer.topics.index(t) for t in topics]
    rel = scorer.vectors(sentences) @ scorer.centroids[rows].T
    counts = {}
    for r in rel:
        t = topics[int(r.argmax())]
        if r.max() >= scorer.thresholds[t]:
            counts[t] = counts.get(t, 0) + 1
    total = sum(counts.values())
    return {t: round(c / total, 4) for t, c in counts.items() if c / total >= min_share}


def upload_similarity(texts: list[str], upload_text: str, scorer: Scorer) -> np.ndarray:
    """Cosine similarity of each question to the uploaded material."""
    return scorer.vectors(texts) @ scorer.vectors([content_words(upload_text)])[0]
