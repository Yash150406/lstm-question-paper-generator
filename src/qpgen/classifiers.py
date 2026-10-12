"""TF-IDF topic centroids and a difficulty classifier used by the filters and evaluation."""
import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

from qpgen.config import DEFAULT_CONFIG, Config, load_config


@dataclass
class Scorer:
    """Scores questions for topic relevance and difficulty."""

    vectorizer: TfidfVectorizer
    topics: list[str]
    centroids: np.ndarray
    thresholds: dict[str, float]
    difficulty: LogisticRegression

    def vectors(self, texts: list[str]) -> np.ndarray:
        """Row-normalised TF-IDF vectors."""
        return self.vectorizer.transform(texts).toarray()

    def relevance(self, texts: list[str], topic: str) -> np.ndarray:
        """Cosine similarity of each text to the topic centroid."""
        return self.vectors(texts) @ self.centroids[self.topics.index(topic)]

    def predict_topic(self, texts: list[str]) -> list[str]:
        """Nearest topic centroid for each text."""
        return [self.topics[i] for i in (self.vectors(texts) @ self.centroids.T).argmax(axis=1)]

    def difficulty_prob(self, texts: list[str], level: str) -> np.ndarray:
        """Probability the difficulty classifier gives to the level."""
        col = list(self.difficulty.classes_).index(level)
        return self.difficulty.predict_proba(self.vectorizer.transform(texts))[:, col]


def fit_scorer(cfg: Config, train: pd.DataFrame, val: pd.DataFrame) -> Scorer:
    """Fit on the training split. Thresholds come from validation questions."""
    c = cfg.classifiers
    vec = TfidfVectorizer(ngram_range=tuple(c["ngram_range"]), sublinear_tf=c["sublinear_tf"])
    x = vec.fit_transform(train["question"])
    topics = [t for info in cfg.subjects.values() for t in info["topics"]]
    centroids = normalize(np.vstack([np.asarray(x[(train["topic"] == t).to_numpy()].mean(axis=0))
                                     for t in topics]))
    clf = LogisticRegression(C=c["C"], max_iter=c["max_iter"], random_state=cfg.seed)
    clf.fit(x, train["difficulty"])
    scorer = Scorer(vec, topics, centroids, {}, clf)
    pct = cfg.filters["relevance_percentile"]
    for t in topics:
        scores = scorer.relevance(val.loc[val["topic"] == t, "question"].tolist(), t)
        scorer.thresholds[t] = float(np.percentile(scores, pct))
    return scorer


def accuracy(scorer: Scorer, df: pd.DataFrame) -> dict[str, float]:
    """Topic and difficulty accuracy on real questions."""
    texts = df["question"].tolist()
    topic_acc = np.mean(np.array(scorer.predict_topic(texts)) == df["topic"].to_numpy())
    diff_acc = np.mean(scorer.difficulty.predict(scorer.vectorizer.transform(texts)) == df["difficulty"].to_numpy())
    return {"topic_accuracy": round(float(topic_acc), 4), "difficulty_accuracy": round(float(diff_acc), 4)}


def save_scorer(scorer: Scorer, path: Path) -> None:
    """Pickle the scorer."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(pickle.dumps(scorer))


def load_scorer(path: Path) -> Scorer:
    """Load a scorer saved with save_scorer. Only load files this project wrote."""
    return pickle.loads(Path(path).read_bytes())


def main(argv: list[str] | None = None) -> None:
    """Fit, save and report accuracy on the real test questions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    cfg = load_config(parser.parse_args(argv).config)
    data = cfg.path("processed_dir")
    train, val, test = (pd.read_csv(data / f"{s}.csv") for s in ("train", "val", "test"))

    # Under `python -m` this file is __main__, so use the package copy of the class.
    # Otherwise pickle saves it as __main__.Scorer and nothing else can load it.
    from qpgen import classifiers
    scorer = classifiers.fit_scorer(cfg, train, val)
    classifiers.save_scorer(scorer, cfg.path("classifiers"))
    report = {"test": accuracy(scorer, test), "test_size": len(test),
              "relevance_thresholds": {t: round(v, 4) for t, v in scorer.thresholds.items()}}
    cfg.path("classifier_report").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
