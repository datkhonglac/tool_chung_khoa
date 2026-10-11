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


def test_dashboard_fetches_any_symbol_from_sidebar(warehouse, monkeypatch):
    from vnsa import analysis
    from vnsa.data.autofetch import Step, UpdateReport

    calls = []
    monkeypatch.setattr(analysis, "update", lambda sym, *a, **k: calls.append(sym) or
                        UpdateReport(sym, [Step("giá", "ok", "thêm 1 phiên")], has_prices=True))
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.text_input[0].input("demo_c")               # ô trong form: gửi cùng lúc bấm nút
    at.button[0].click().run()
    assert not at.exception, at.exception
    assert calls == ["DEMO_C"]
    assert at.selectbox[0].value == "DEMO_C"
    assert any("DEMO_C: đã sẵn sàng" in s.value for s in at.success)


def test_dashboard_bank_shows_note_instead_of_dcf(warehouse):
    import pandas as pd
    from conftest import _fundamentals

    from vnsa.data.store import Store

    with Store(warehouse) as store:
        store.write_companies(pd.DataFrame({"symbol": ["DEMO_B"], "exchange": ["HNX"],
                                            "model": ["bank"]}), "test")
        store.write_fundamentals(_fundamentals("DEMO_B"), "test")
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.selectbox[0].select("DEMO_B").run()
    assert not at.exception, at.exception
    assert sum("ngân hàng" in i.value for i in at.info) == 2       # tab Tài chính và Định giá
    assert "P/B" in " | ".join(m.label for m in at.metric)
