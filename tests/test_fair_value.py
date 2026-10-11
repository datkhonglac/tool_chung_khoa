"""Giá trị hợp lý (fa/fair_value.py), vùng mua (ta/entry.py), lịch sử khuyến nghị (recommend/review.py)."""

import numpy as np
import pandas as pd
import pytest
from conftest import _fundamentals

from vnsa.fa import fair_value as fv
from vnsa.fa.scores import financial_score
from vnsa.recommend.review import evaluate, summarize
from vnsa.ta.entry import entry_zone, vwap
from vnsa.ta.levels import Level

B = 1_000_000_000
TICKS = [[0, 10], [10_000, 50], [50_000, 100]]
ENTRY = {"max_pullback_atr": 2.0, "zone_atr": 0.5, "chase_atr": 2.0, "vwap_sessions": 20,
         "margin_of_safety": 0.15, "weak_fa_extra": 0.10}


def _bank(roe=0.15, years=(2022, 2023, 2024, 2025)) -> pd.DataFrame:
    """Vốn chủ không đổi 10.000 tỷ, 1 tỷ cp (vốn góp 10.000 tỷ) -> B0 = 10.000 đ; ROE đúng `roe`."""
    eq = 10_000 * B
    return pd.DataFrame({"symbol": "BANK", "year": list(years), "equity": eq,
                         "minority_interest": 0.0, "net_income_parent": roe * eq,
                         "net_income": roe * eq, "total_assets": 10 * eq,
                         "paid_in_capital": 10_000 * B, "eps": roe * 10_000})


def _prices(years=(2022, 2023, 2024, 2025), price=15_000.0) -> pd.DataFrame:
    dates = pd.bdate_range(f"{years[0]}-01-01", f"{years[-1]}-12-31")
    return pd.DataFrame({"date": dates, "close": price, "open": price, "high": price,
                         "low": price, "volume": 1_000})


def test_residual_income_matches_closed_form():
    # ROE không đổi (persistence = 1) và vốn chủ tăng đúng g = ROE × tỷ lệ giữ lại:
    # V = B0 + (ROE − ke) B0 / (ke − g) = B0 × (ROE − g) / (ke − g) = P/B hợp lý × B0
    cfg = {"forecast_years": 5, "roe_years": 3, "roe_persistence": 1.0, "retention": 0.2}
    ri = fv.residual_income(_bank(0.15), ke=0.12, g=0.03, cfg=cfg)
    assert ri.book0 == pytest.approx(10_000)
    assert ri.roe0 == pytest.approx(0.15) and ri.roe_terminal == pytest.approx(0.15)
    assert ri.justified_pb == pytest.approx((0.15 - 0.03) / (0.12 - 0.03))
    assert ri.value_per_share == pytest.approx(ri.justified_pb * 10_000, rel=1e-9)


def test_residual_income_fades_roe_and_rejects_bad_inputs():
    cfg = {"forecast_years": 5, "roe_persistence": 0.5, "retention": 0.8}
    ri = fv.residual_income(_bank(0.20), ke=0.12, g=0.03, cfg=cfg)
    assert ri.roe_terminal == pytest.approx(0.16)                  # 12% + 50% × (20% − 12%)
    rising = _bank().assign(net_income_parent=[0.08e13, 0.10e13, 0.12e13, 0.14e13])
    r2 = fv.residual_income(rising, ke=0.12, g=0.03, cfg=cfg)
    assert r2.roe0 == pytest.approx((0.10 * 1 + 0.12 * 2 + 0.14 * 3) / 6)   # năm gần nặng hơn
    assert ri.roe_path[0] == pytest.approx(0.20) and ri.roe_path[-1] == pytest.approx(0.16)
    assert ri.value_per_share > ri.book0                            # ROE > ke -> P/B > 1
    low = fv.residual_income(_bank(0.08), ke=0.12, g=0.03, cfg=cfg)
    assert low.value_per_share < low.book0                          # ROE < ke -> P/B < 1
    with pytest.raises(ValueError):
        fv.residual_income(_bank(), ke=0.03, g=0.03, cfg=cfg)


def test_historical_multiples_use_current_shares_and_yearly_average():
    fin = _fundamentals("X")             # vốn góp 300 tỷ -> 30 triệu cp, LNST cty mẹ 75 tỷ × g
    hist = fv.historical_multiples(fin, _prices(price=40_000))
    shares = 300 * B / 10_000
    assert hist["eps"].iloc[0] == pytest.approx(75 * B / shares)
    assert hist["pe"].iloc[0] == pytest.approx(40_000 / (75 * B / shares))
    parent_eq = (900 - 30) * B * 1.35
    assert hist["bvps"].iloc[-1] == pytest.approx(parent_eq / shares)


def test_combine_takes_median_of_positive_methods():
    fin, px = _fundamentals("X"), _prices(price=40_000)
    fair = fv.combine(40_000, fin, px, dcf=10_000.0)
    used = sorted(fair.used.values())
    assert len(used) == 3 and fair.value == pytest.approx(used[1])
    no_dcf = fv.combine(40_000, fin, px, dcf=-5.0)                  # DCF âm bị bỏ
    assert set(no_dcf.used) == {"pe_hist", "pb_hist"}
    assert no_dcf.upside == pytest.approx(no_dcf.value / 40_000 - 1)


def test_financial_fair_value_balances_roe_models_and_history():
    cfg = {"forecast_years": 5, "roe_persistence": 0.5, "retention": 0.8}
    ri = fv.residual_income(_bank(0.15), ke=0.12, g=0.03, cfg=cfg)
    fair = fv.combine(15_000, _bank(0.15), _prices(price=15_000), ri=ri)
    m = fair.methods
    assert m["pb_hist"] == pytest.approx(1.5 * 10_000)              # P/B lịch sử 1,5 × B0
    assert fair.value == pytest.approx(((m["ri"] + m["pb_roe"]) / 2 + m["pb_hist"]) / 2)


def test_financial_score_uses_bank_thresholds():
    cfg = {"weights": {"sinh_loi": 35, "tang_truong": 20, "an_toan": 15, "dinh_gia": 30},
           "thresholds": {"roe": [0.08, 0.20], "ni_cagr": [0.0, 0.20], "pb": [3.0, 0.8],
                          "fair_upside": [-0.2, 0.3]},
           "equity_to_assets": {"bank": [0.05, 0.11], "securities": [0.25, 0.60]}}
    s = financial_score(_bank(0.20), cfg, "bank", pb=0.8, fair_upside=0.3)
    assert s.items["equity_to_assets"][1] == pytest.approx((0.10 - 0.05) / 0.06 * 100)
    assert s.groups["sinh_loi"] == 100 and s.groups["dinh_gia"] == 100
    sec = financial_score(_bank(0.20), cfg, "securities", pb=0.8, fair_upside=0.3)
    assert sec.items["equity_to_assets"][1] == 0                   # 10% quá mỏng với công ty CK


def _ind(close=30_000.0, sma20=29_500.0, atr=500.0, n=30) -> pd.DataFrame:
    df = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=n), "symbol": "X",
                       "open": close, "high": close + 200, "low": close - 200, "close": close,
                       "volume": 1_000, "atr": atr, "sma20": sma20})
    return df


def test_vwap_weights_typical_price_by_volume():
    df = pd.DataFrame({"high": [11.0, 22.0], "low": [9.0, 18.0], "close": [10.0, 20.0],
                       "volume": [1, 3]})
    assert vwap(df) == pytest.approx((10 * 1 + 20 * 3) / 4)


def test_entry_zone_anchors_below_price_and_rounds_to_tick():
    ind = _ind()                                       # VWAP20 = 30.000, SMA20 = 29.500
    z = entry_zone(ind, [Level(29_000, 3, pd.Timestamp("2026-01-05"))], ENTRY, TICKS)
    assert z.anchor_source == "VWAP20" and z.anchor == pytest.approx(30_000)
    assert z.status == "MUA TRONG VÙNG"
    assert z.high <= z.close and z.low <= z.high
    assert z.low % 50 == 0 and z.high % 50 == 0 and z.mid % 50 == 0
    assert "giá trị hợp lý" in " ".join(z.notes)       # chưa có giá trị hợp lý -> ghi chú


def test_entry_zone_value_cap_lowers_zone_or_waits():
    ind = _ind(close=30_000, sma20=29_000)             # VWAP20 = 30.000 → neo, vùng 30.000–30.000
    sup = [Level(28_000, 2, pd.Timestamp("2026-01-05"))]
    wait = entry_zone(ind, sup, ENTRY, TICKS, fair_value=30_000)   # trần 25.500 < neo
    assert wait.status == "CHỜ GIÁ" and wait.high == 25_500 and wait.entry == 25_500
    weak = entry_zone(ind, sup, ENTRY, TICKS, fair_value=30_000, fa_score=40)
    assert weak.margin_of_safety == pytest.approx(0.25) and weak.high == 22_500
    ok = entry_zone(ind, sup, ENTRY, TICKS, fair_value=60_000)
    assert ok.status == "MUA TRONG VÙNG" and ok.value_cap == pytest.approx(51_000)


def test_entry_zone_falls_back_when_price_below_all_anchors():
    ind = _ind(close=30_000, sma20=34_000).assign(high=36_000.0, low=35_000.0)   # VWAP > giá
    z = entry_zone(ind, [], ENTRY, TICKS)
    assert z.anchor_source == "giá − 1 ATR" and z.anchor == pytest.approx(29_500)
    assert any("xu hướng yếu" in n for n in z.notes)
    chase = entry_zone(_ind(close=32_000, sma20=30_000), [], ENTRY, TICKS)
    assert any("không đuổi giá" in n for n in chase.notes)


def test_review_scores_returns_touches_and_zone_fills():
    dates = pd.bdate_range("2026-01-01", periods=10)
    px = pd.DataFrame({"date": dates, "close": [100, 101, 99, 104, 111, 112, 113, 114, 115, 116.0],
                       "high": [100, 102, 100, 105, 112, 113, 114, 115, 116, 117.0],
                       "low": [99, 98, 97, 103, 110, 111, 112, 113, 114, 115.0]})
    index = pd.DataFrame({"date": dates, "close": np.linspace(1000, 1010, 10)})
    sig = pd.DataFrame([{"as_of": dates[0], "symbol": "X", "action": "MUA", "price": 100.0,
                         "buy_high": 98.0, "stop": 95.0, "target": 110.0},
                        {"as_of": dates[0], "symbol": "Y", "action": "TRÁNH", "price": 50.0,
                         "buy_high": np.nan, "stop": np.nan, "target": np.nan}])
    ev = evaluate(sig, {"X": px}, index, [3, 5, 20])
    assert set(ev["horizon"]) == {3, 5}                # 20 phiên chưa đủ dữ liệu; Y không có giá
    h5 = ev[ev["horizon"] == 5].iloc[0]
    assert h5["ret"] == pytest.approx(0.12)            # 112 / 100 − 1
    assert h5["index_ret"] == pytest.approx(np.linspace(1000, 1010, 10)[5] / 1000 - 1)
    assert h5["touch"] == "mục tiêu" and h5["zone_filled"]
    assert h5["ret_from_zone"] == pytest.approx(112 / 98 - 1)
    assert ev[ev["horizon"] == 3].iloc[0]["touch"] == "chưa chạm"
    summary = summarize(ev)
    row = summary[(summary["horizon"] == 5) & (summary["action"] == "MUA")].iloc[0]
    assert row["n"] == 1 and row["win_rate"] == 1 and row["target_first"] == 1


def test_recommend_records_signals_and_review_cli(warehouse, capsys, monkeypatch):
    from vnsa import analysis, cli
    from vnsa.config import path
    from vnsa.data.store import Store

    assert cli.main(["recommend", "DEMO_A", "DEMO_C"]) == 0
    out = capsys.readouterr().out
    assert "Vùng mua" in out and "Hợp lý" in out
    with Store(warehouse) as store:
        sig = store.signals()
        px = store.prices("DEMO_A")
    assert set(sig["symbol"]) == {"DEMO_A", "DEMO_C"} and sig["params"].str.len().eq(8).all()
    assert path("signals_csv").exists()

    # Khuyến nghị giả lập ở 70 phiên trước để có kết quả chấm 5/20/60 phiên
    past = px.iloc[-71]
    with Store(warehouse) as store:
        store.write_signals(pd.DataFrame([{"as_of": past["date"], "symbol": "DEMO_A",
                                           "action": "MUA", "price": past["close"],
                                           "buy_high": past["close"], "stop": past["close"] * 0.9,
                                           "target": past["close"] * 1.1}]))
    _, summary, n = analysis.review()
    assert n == 3 and set(summary["horizon"]) == {5, 20, 60}
    assert cli.main(["review-signals"]) == 0
    assert "Vượt VNI" in capsys.readouterr().out
    # Kho mới (đổi Codespace) nạp lại lịch sử từ CSV
    monkeypatch.setenv("VNSA_WAREHOUSE", str(warehouse.parent / "fresh.duckdb"))
    assert analysis.import_signals_csv() == 2
