import pytest
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate

from qpgen.classifiers import fit_scorer
from qpgen.config import load_config
from qpgen.data.build_dataset import build, split
from qpgen.upload import analyze_upload, read_upload, upload_similarity

CFG = load_config()
OS_NOTES = ("A deadlock happens when processes wait for each other forever. The four conditions are mutual "
            "exclusion, hold and wait, no preemption and circular wait. The Banker's algorithm keeps the "
            "system in a safe state. Page replacement with LRU evicts the least recently used page.")


@pytest.fixture(scope="module")
def scorer():
    parts = split(build(CFG), CFG)
    return fit_scorer(CFG, parts["train"], parts["val"])


def test_read_pdf_text_and_plain_string(tmp_path):
    pdf = tmp_path / "notes.pdf"
    SimpleDocTemplate(str(pdf), pagesize=A4).build([Paragraph(OS_NOTES, getSampleStyleSheet()["Normal"])])
    assert "Banker" in read_upload(pdf)
    assert "Banker" in read_upload(pdf.read_bytes(), "notes.pdf")
    assert read_upload(b"paging and segmentation", "notes.txt") == "paging and segmentation"
    assert read_upload("Deadlocks") == "Deadlocks"


def test_typed_names(scorer):
    assert analyze_upload("Deadlocks", CFG, scorer)["topics"] == {"Deadlocks": 1.0}
    whole = analyze_upload("dbms", CFG, scorer)
    assert whole["subject"] == "DBMS" and len(whole["topics"]) == 8
    pair = analyze_upload("Normalization, SQL queries", CFG, scorer)
    assert set(pair["topics"]) == {"Normalization", "SQL queries"}


def test_notes_map_to_subject_topics_and_key_terms(scorer):
    found = analyze_upload(OS_NOTES, CFG, scorer)
    assert found["supported"] and found["subject"] == "Operating Systems"
    assert "Deadlocks" in found["topics"]
    assert max(found["topics"], key=found["topics"].get) == "Deadlocks"
    assert "circular wait" in found["key_terms"]["Deadlocks"]


def test_unrelated_text_is_not_supported(scorer):
    found = analyze_upload("The French revolution began in 1789 with the storming of the Bastille "
                           "and the fall of the monarchy in Paris.", CFG, scorer)
    assert not found["supported"] and "DBMS" in found["message"]


def test_upload_similarity_prefers_matching_question(scorer):
    sims = upload_similarity(["Explain the Banker's algorithm with an example.",
                              "Explain the HTTP request and response format."], OS_NOTES, scorer)
    assert sims[0] > sims[1]
