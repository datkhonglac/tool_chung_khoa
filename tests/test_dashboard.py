from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def test_dashboard_renders_all_tabs_without_errors(warehouse):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6
    assert at.selectbox[0].options == ["DEMO_A", "DEMO_B", "DEMO_C"]
    labels = " | ".join(m.label for m in at.metric)
    for label in ("Điểm TA", "Điểm FA", "Giá trị DCF/cp", "Lợi nhuận/năm", "VaR 95% 1 ngày"):
        assert label in labels


def test_dashboard_symbol_without_statements(warehouse):
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox[0].select("DEMO_B").run()
    assert not at.exception, at.exception
    assert any("fetch-bctc DEMO_B" in i.value for i in at.info)


def test_dashboard_without_warehouse(tmp_path, monkeypatch):
    monkeypatch.setenv("VNSA_WAREHOUSE", str(tmp_path / "none.duckdb"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert at.error and "fetch" in at.error[0].value
