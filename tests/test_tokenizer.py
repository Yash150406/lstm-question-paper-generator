import pandas as pd
import pytest
import yaml

from qpgen.config import load_config
from qpgen.tokenizer import (BOS_ID, EOS_ID, PAD_ID, UNK_ID, Vocab, control_tokens,
                             detokenize, tokenize, topic_token)


@pytest.fixture(scope="module")
def acronyms():
    return yaml.safe_load(load_config().path("acronyms").read_text(encoding="utf-8"))


def test_punctuation_is_its_own_token():
    assert tokenize("What is TCP/IP?") == ["what", "is", "tcp", "/", "ip", "?"]


@pytest.mark.parametrize("question", [
    "Differentiate between 3NF and BCNF with an example.",
    "What is TCP/IP?",
    "Explain the Banker's algorithm and check whether a given state is safe.",
    "Construct a B+ tree of order 4 for the given keys.",
    "Find the network address for 192.168.10.37/27.",
    "Explain the 0/1 knapsack problem using dynamic programming.",
    "Compare SCAN, C SCAN and LOOK.",
])
def test_round_trip_restores_text(question, acronyms):
    assert detokenize(tokenize(question), acronyms) == question


def test_round_trip_is_stable_on_whole_bank(acronyms):
    questions = pd.read_csv(load_config().path("raw_dir") / "seed_dbms.csv")["question"]
    for q in questions:
        assert tokenize(detokenize(tokenize(q), acronyms)) == tokenize(q), q


def test_detokenize_drops_special_tokens():
    assert detokenize(["<bos>", "<subj_os>", "define", "paging", ".", "<eos>"]) == "Define paging."


def test_vocab_special_ids_and_order():
    vocab = Vocab.build([["a", "b", "b"], ["b", "c"]], ["<diff_easy>"], min_freq=2, max_words=10)
    assert [vocab.stoi[t] for t in ["<pad>", "<unk>", "<bos>", "<eos>"]] == [PAD_ID, UNK_ID, BOS_ID, EOS_ID]
    assert vocab.itos[4:] == ["<diff_easy>", "b"]
    assert vocab.encode(["b", "zzz"]) == [5, UNK_ID]


def test_vocab_max_words():
    vocab = Vocab.build([["x"] * 3 + ["y"] * 2 + ["z"] * 2], [], min_freq=1, max_words=2)
    assert vocab.itos[4:] == ["x", "y"]


def test_vocab_save_load(tmp_path):
    vocab = Vocab.build([["a", "a"]], ["<diff_hard>"], min_freq=1, max_words=5)
    vocab.save(tmp_path / "vocab.json")
    loaded = Vocab.load(tmp_path / "vocab.json")
    assert loaded.itos == vocab.itos and loaded.control_ids == vocab.control_ids


def test_control_tokens_from_config():
    cfg = load_config()
    tokens = control_tokens(cfg.subjects, list(cfg.marks))
    assert len(tokens) == 4 + 32 + 3 and len(set(tokens)) == len(tokens)
    assert topic_token("Indexing and B+ trees") == "<topic_indexing_and_b_trees>"
