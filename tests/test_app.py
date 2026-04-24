from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_without_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert any("Paste a GitHub URL" in info.value for info in at.info)
