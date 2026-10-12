import json

import pytest

from qpgen.cli import main, parser
from qpgen.config import load_config

HAS_MODEL = load_config().path("checkpoint").exists()


def test_parser_defaults():
    args = parser().parse_args(["paper", "--subject", "DBMS"])
    assert args.preset == "balanced" and args.seed == 42 and args.format == "all"


def test_paper_needs_subject_or_file():
    with pytest.raises(SystemExit):
        main(["paper"])


@pytest.mark.skipif(not HAS_MODEL, reason="needs a trained model")
def test_generate_prints_clean_questions(capsys):
    main(["generate", "--subject", "DBMS", "--topic", "Normalization", "--difficulty", "medium",
          "-n", "5", "--seed", "42"])
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 5
    assert all(line.endswith(("?", ".")) and "<" not in line and ">" not in line for line in lines)


@pytest.mark.skipif(not HAS_MODEL, reason="needs a trained model")
def test_paper_from_typed_chapter_name(capsys, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("The Banker's algorithm keeps the system in a safe state and avoids deadlock "
                     "by checking every resource request against the maximum claim of each process.")
    main(["paper", "--from-file", str(notes), "--format", "json", "--seed", "3"])
    out = capsys.readouterr().out
    assert "Detected Operating Systems" in out
    path = out.strip().splitlines()[-1].removeprefix("Wrote ")
    data = json.loads(open(path, encoding="utf-8").read())
    assert sum(q["marks"] for s in data["sections"] for q in s["questions"]) == 70
