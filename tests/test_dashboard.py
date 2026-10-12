from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


def test_dashboard_renders_all_tabs_without_errors(warehouse):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 9
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


def test_dashboard_bank_uses_residual_income(warehouse):
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
    assert any("ngân hàng" in i.value for i in at.info)            # tab Tài chính
    labels = " | ".join(m.label for m in at.metric)
    for label in ("P/B", "Thu nhập thặng dư", "P/B hợp lý theo ROE", "Vùng mua"):
        assert label in labels, label


def test_dashboard_macro_tab_with_data(warehouse):
    import numpy as np
    import pandas as pd

    from vnsa.data.store import Store

    today = pd.Timestamp.today().normalize()
    with Store(warehouse) as store:
        store.write_companies(pd.DataFrame({
            "symbol": ["DEMO_A"], "name": ["Thép Demo"], "exchange": ["HOSE"],
            "industry": ["Tài nguyên Cơ bản"], "model": ["general"], "industry3": ["Kim loại"],
            "industry4": ["Thép và sản phẩm thép"]}), "test")
        d = list(pd.date_range(end=today.replace(day=1), periods=72, freq="MS"))
        d[-1] = today
        for sid in ("iron_ore", "steel_hrc", "usd_vnd"):
            store.write_macro(pd.DataFrame({"series": sid, "date": d,
                                            "value": np.linspace(100, 130, 72), "source": "t"}))
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    text = " ".join(str(m.value) for m in at.markdown) + " ".join(str(s.value) for s in at.subheader)
    assert "Thép (ICB cấp 4" in text
    assert any("Thép" in str(x.label) or "Chuỗi" in str(x.label) for x in at.selectbox)


def test_dashboard_screen_tab(warehouse):
    import pandas as pd

    from vnsa.data.store import Store

    day = pd.Timestamp("2026-10-09")
    base = {"as_of": day, "exchange": "HOSE", "group_name": "Ngân hàng", "industry": "Ngân hàng",
            "price": 25_000.0, "value_20d": 9e9, "rs_6m": 0.05, "fa_score": 70.0,
            "ta_score": 55.0, "score": 64.0, "confidence": "Trung bình", "fair_value": 30_000.0,
            "expected_return": 0.22, "buy_low": 24_500.0, "buy_high": 25_000.0, "stop": 23_000.0,
            "eps_growth": 0.12, "group_rank": 1.0, "group_score": 80.0}
    table = pd.DataFrame([
        {**base, "symbol": "AAA", "stage": 4, "action": "MUA", "pick_rank": 1.0, "reason": ""},
        {**base, "symbol": "BBB", "stage": 3, "action": "NẮM GIỮ", "reason": "xếp hạng NẮM GIỮ"},
        {**base, "symbol": "CCC", "stage": 2, "action": "MUA", "group_name": "Thép",
         "reason": "đạt nhưng ngoài nhóm ngành dẫn đầu"},
        {**base, "symbol": "DDD", "stage": 0, "action": None, "reason": "thanh khoản thấp: …"},
    ])
    ind = pd.DataFrame([{"as_of": day, "group_name": "Ngân hàng", "rank": 1.0, "score": 80.0,
                         "members": 2, "rs": 0.05, "growth": 0.1, "expected": 0.2,
                         "pt_momentum": 100.0, "note": ""},
                        {"as_of": day, "group_name": "Thép", "rank": 2.0, "score": 20.0,
                         "members": 1, "rs": -0.02, "growth": -0.1, "expected": 0.3,
                         "pt_momentum": 0.0, "note": ""}])
    with Store(warehouse) as store:
        store.write_screen(table, ind)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    heads = [h.value for h in at.subheader]
    assert "Nhóm ngành tiềm năng (điểm 0–100)" in heads
    assert "Mã tiềm năng trong các nhóm dẫn đầu" in heads
    assert "Mã MUA ngoài các nhóm dẫn đầu" in heads
    labels = {m.label: m.value for m in at.metric}
    assert labels["Được chọn"] == "1" and labels["Vũ trụ"] == "4"


def test_dashboard_screen_tab_liquidity_only(warehouse):
    """Lượt chỉ chạy --stage liquidity: nói rõ chỉ có tầng 1, hiện danh sách mã qua lọc (trước đây
    báo "Không mã nào đạt đủ điều kiện" gây hiểu nhầm)."""
    import pandas as pd

    from vnsa.data.store import Store

    day = pd.Timestamp("2026-10-12")
    rows = [{"as_of": day, "symbol": s, "exchange": "HOSE", "industry": "Ngân hàng",
             "price": 25_000.0, "value_20d": v, "rs_3m": 0.02, "rs_6m": 0.05,
             "above_ma50": 1.0, "above_ma200": 1.0, "stage": st, "reason": r}
            for s, v, st, r in [("AAA", 9e9, 1, ""), ("BBB", 5e9, 1, ""),
                                ("CCC", 1e8, 0, "thanh khoản thấp: 0,1 tỷ")]]
    with Store(warehouse) as store:
        store.write_screen(pd.DataFrame(rows))
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert any("tầng 1 — lọc thanh khoản" in i.value for i in at.info)
    assert "2 mã qua lọc thanh khoản" in [h.value for h in at.subheader]
    assert "Mã tiềm năng trong các nhóm dẫn đầu" not in [h.value for h in at.subheader]
