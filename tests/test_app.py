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
    assert len(at.text_input) >= 18 + 2          # 18 questions plus institute and exam title
    labels = [b.label for b in at.get("download_button")]
    assert labels == ["Download DOCX", "Download PDF", "Download JSON"]
