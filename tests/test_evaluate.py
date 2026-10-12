from dataclasses import replace

import pandas as pd
import pytest

from qpgen.config import load_config
from qpgen.evaluate import average_ratings, cells, distinct, fmt, mean_std, self_bleu

CFG = load_config()


def test_distinct():
    assert distinct(["a b c", "a b c"], 1) == pytest.approx(3 / 6)
    assert distinct(["a b c", "d e f"], 2) == 1.0
    assert distinct([], 1) == 0.0


def test_self_bleu_is_high_for_copies_and_low_for_varied_text():
    copies = ["explain the working of two phase locking"] * 5
    varied = ["define paging in memory", "compare tcp and udp headers", "what is a heap sort",
              "draw an er diagram for a library", "explain bankers algorithm safe state"]
    assert self_bleu(copies, 5, 0) == pytest.approx(1.0)
    assert self_bleu(varied, 5, 0) < 0.2


def test_mean_std_skips_missing_values():
    out = mean_std([{"a": 1.0, "b": None}, {"a": 3.0, "b": None}])
    assert out == {"a": {"mean": 2.0, "std": 1.0}}
    assert fmt(out["a"]) == "2.000 ± 1.000" and fmt(None) == "-"


def test_cells_cover_every_combination():
    assert len(cells(CFG)) == 4 * 8 * 3


def test_average_ratings(tmp_path):
    cfg = replace(CFG, paths={**CFG.paths, "rating_key": str(tmp_path / "key.csv"),
                              "human_ratings_md": str(tmp_path / "ratings.md")})
    pd.DataFrame({"id": [1, 2], "source": ["LSTM", "Real"]}).to_csv(tmp_path / "key.csv", index=False)
    for name, scores in (("r1.csv", [2, 5]), ("r2.csv", [4, 5])):
        pd.DataFrame({"id": [1, 2], "question": ["q1", "q2"], "fluency": scores,
                      "relevance": scores, "difficulty_fit": scores}).to_csv(tmp_path / name, index=False)
    table = average_ratings(cfg, [str(tmp_path / "r1.csv"), str(tmp_path / "r2.csv")])
    assert table.loc["LSTM", "fluency"] == 3.0 and table.loc["Real", "relevance"] == 5.0
    assert "| LSTM | 3.0 |" in (tmp_path / "ratings.md").read_text(encoding="utf-8")
