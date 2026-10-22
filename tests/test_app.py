from pathlib import Path

import pytest
import yaml
from streamlit.testing.v1 import AppTest

from qpgen.config import DEFAULT_CONFIG, ROOT, load_config

APP = str(ROOT / "app" / "streamlit_app.py")


def config_without_model(tmp_path: Path) -> Path:
    """A copy of the config whose checkpoint path points nowhere."""
    raw = yaml.safe_load(Path(DEFAULT_CONFIG).read_text(encoding="utf-8"))
    raw["paths"]["checkpoint"] = str(tmp_path / "missing.pt")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_missing_checkpoint_shows_message(tmp_path, monkeypatch):
    monkeypatch.setenv("QPGEN_CONFIG", str(config_without_model(tmp_path)))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "make train" in at.error[0].value


@pytest.mark.skipif(not load_config().path("checkpoint").exists(), reason="needs a trained model")
def test_generate_paper_end_to_end(monkeypatch):
    monkeypatch.delenv("QPGEN_CONFIG", raising=False)
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    lines = [m.value for m in at.markdown if m.value.startswith('<div class="q">')]
    assert len(lines) == 18 and lines[0].startswith('<div class="q">1. ')
    labels = [b.label for b in at.get("download_button")]
    assert labels == ["Download DOCX", "Download PDF", "Download JSON"]

    # Edit the first question in place: the pencil opens a text box, the tick saves it.
    [b for b in at.button if (b.key or "").startswith("edit_")][0].click().run()
    box = at.main.text_input[0]
    box.input("Define a weak entity set with an example.").run()
    [b for b in at.button if (b.key or "").startswith("save_")][0].click().run()
    assert not at.exception and not at.main.text_input
    assert any("1. Define a weak entity set with an example." in m.value for m in at.markdown)


def page_script(index: int) -> str:
    """The app with only one of its pages, since AppTest cannot switch to a function page."""
    app = str(APP)
    return (f"import streamlit as st\n"
            f"st.__dict__.setdefault('_orig_nav', st.navigation)\n"
            f"st.navigation = lambda pages, **kw: st._orig_nav([pages[{index}]], **kw)\n"
            f"exec(compile(open({app!r}, encoding='utf-8').read(), {app!r}, 'exec'))\n")


def test_add_topic_page_refuses_bad_input(monkeypatch):
    monkeypatch.delenv("QPGEN_CONFIG", raising=False)
    at = AppTest.from_string(page_script(1), default_timeout=120).run()
    at.text_input[0].input("Deadlocks").run()
    at.text_area[0].input("1. Define a deadlock in operating systems.").run()
    assert not at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "already exists" in warnings and "at least 30" in warnings
    assert [b.disabled for b in at.button if b.label == "Save topic"] == [True]
