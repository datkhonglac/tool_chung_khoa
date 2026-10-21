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


def test_combine_weights_methods_and_drops_outliers():
    fin, px = _fundamentals("X"), _prices(price=40_000)
    base = fv.combine(40_000, fin, px, dcf=float("nan"))
    pe, pb = base.methods["pe_hist"], base.methods["pb_hist"]
    assert base.weights == pytest.approx({"pe_hist": 0.5, "pb_hist": 0.5})   # chia lại 30/30
    assert base.value == pytest.approx((pe + pb) / 2)
    dcf = (pe + pb) / 2 * 1.2
    fair = fv.combine(40_000, fin, px, dcf=dcf)
    assert fair.value == pytest.approx(0.4 * dcf + 0.3 * pe + 0.3 * pb)
    wild = fv.combine(40_000, fin, px, dcf=pe * 10)                   # DCF lệch 10 lần -> bỏ
    assert "dcf" not in wild.weights and any("lệch quá xa" in n for n in wild.notes)
    no_dcf = fv.combine(40_000, fin, px, dcf=-5.0)                    # DCF âm bị bỏ
    assert set(no_dcf.used) == {"pe_hist", "pb_hist"}
    assert no_dcf.upside == pytest.approx(no_dcf.value / 40_000 - 1)


def test_scenario_range_brackets_center():
    assert fv.scenario_range([10, 20, 30, 40, 50], 30) == pytest.approx((14, 46))
    assert fv.scenario_range([10, 20, 30], 60)[1] == 60              # luôn bao giá trị trung tâm
    lo, hi = fv.scenario_range([10, -5], 20)
    assert np.isnan(lo) and np.isnan(hi)                               # quá ít kịch bản


def test_financial_fair_value_balances_roe_models_and_history():
    cfg = {"forecast_years": 5, "roe_persistence": 0.5, "retention": 0.8}
    ri = fv.residual_income(_bank(0.15), ke=0.12, g=0.03, cfg=cfg)
    fair = fv.combine(15_000, _bank(0.15), _prices(price=15_000), ri=ri)
    m = fair.methods
    assert m["pb_hist"] == pytest.approx(1.5 * 10_000)              # P/B lịch sử 1,5 × B0
    assert fair.value == pytest.approx(0.25 * m["ri"] + 0.25 * m["pb_roe"] + 0.5 * m["pb_hist"])


def test_financial_score_uses_bank_thresholds():
    cfg = {"weights": {"sinh_loi": 35, "tang_truong": 20, "an_toan": 15, "dinh_gia": 30},
           "thresholds": {"roe": [0.08, 0.20], "ni_cagr": [0.0, 0.20], "pb_rel": [1.3, 0.7],
                          "fair_upside": [-0.2, 0.3]},
           "equity_to_assets": {"bank": [0.05, 0.11], "securities": [0.25, 0.60]}}
    s = financial_score(_bank(0.20), cfg, "bank", pb=0.8, fair_upside=0.3, pb_hist=1.2)
    assert s.items["pb_rel"][0] == pytest.approx(0.8 / 1.2)          # rẻ hơn lịch sử 33%
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
    assert "Vùng mua" in out and "LN kỳ vọng" in out and "Thời điểm" in out
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


def test_partial_reversion_to_historical_multiples():
    assert fv.revert(10.0, 20.0, 0.5) == 15.0                        # thu hẹp nửa khoảng cách
    assert fv.revert(10.0, 20.0, 1.0) == 20.0                        # cách cũ: quay hẳn về trung vị
    assert fv.revert(float("nan"), 20.0, 0.5) == 20.0                # EPS âm: dùng trung vị
    fin = _fundamentals("X")
    px = _prices(price=40_000)
    price = 20_000.0                                                  # rẻ một nửa so với lịch sử
    full = fv.combine(price, fin, px)
    half = fv.combine(price, fin, px, reversion=0.5)
    assert half.pe_median == pytest.approx(full.pe_target)
    eps = full.methods["pe_hist"] / full.pe_target
    pe_now = price / eps
    assert half.pe_target == pytest.approx(pe_now + 0.5 * (full.pe_median - pe_now))
    assert price < half.value < full.value                           # upside nhỏ hơn, vẫn dương


def test_peer_multiples_add_independent_method():
    fin, px = _fundamentals("X"), _prices(price=40_000)
    hist_only = fv.combine(40_000, fin, px)
    assert not hist_only.independent                              # chỉ bội số quá khứ
    peer = fv.combine(40_000, fin, px, peers=(12.0, 1.5, 4))
    assert {"pe_peer", "pb_peer"} <= set(peer.weights) and peer.independent
    eps = hist_only.methods["pe_hist"] / hist_only.pe_target
    assert peer.methods["pe_peer"] == pytest.approx(12.0 * eps)   # reversion mặc định 1
    half = fv.combine(40_000, fin, px, peers=(12.0, 1.5, 4), reversion=0.5)
    pe_now = 40_000 / eps
    assert half.methods["pe_peer"] == pytest.approx((pe_now + 0.5 * (12.0 - pe_now)) * eps)
    assert any("4 mã cùng ngành" in n for n in peer.notes)


def _declining(fin: pd.DataFrame) -> pd.DataFrame:
    """Lợi nhuận đỉnh rồi giảm (kiểu DGC sau chu kỳ giá hàng hoá): ROE 44% -> 12,5%."""
    return fin.assign(net_income_parent=[400 * B, 250 * B, 180 * B, 150 * B],
                      equity=[900 * B, 1000 * B, 1100 * B, 1230 * B], minority_interest=0.0)


def test_eps_trend_and_roe_factor():
    from types import SimpleNamespace

    fin = _declining(_fundamentals("X"))
    g = fv.eps_trend(fin)
    assert g == pytest.approx((150 / 400) ** (1 / 3) - 1)               # −27%/năm
    ttm = SimpleNamespace(basis="TTM Q2/2026", net_income_parent=120 * B)
    assert fv.eps_trend(fin, ttm) == pytest.approx(min(g, 120 / 150 - 1))
    year = SimpleNamespace(basis="năm 2025", net_income_parent=150 * B)
    assert fv.eps_trend(fin, year) == pytest.approx(g)                  # số năm không tính là TTM
    assert fv.eps_trend_detail(fin, ttm)[1] == "−27,9%/năm 2022–2025"     # 3 năm tệ hơn TTM
    worse = SimpleNamespace(basis="TTM Q2/2026", net_income_parent=90 * B)
    assert fv.eps_trend_detail(fin, worse)[1] == "TTM Q2/2026 −40,0% so với năm 2025"
    hist = pd.DataFrame({"roe": [0.44, 0.25, 0.16, 0.125]})
    f, med = fv.roe_factor(hist, 0.125)
    assert med == pytest.approx(0.205) and f == pytest.approx(0.125 / 0.205)
    assert fv.roe_factor(hist, 0.30)[0] == 1.0                           # ROE cao hơn: không lạc quan thêm


def test_quality_adjust_lowers_pb_and_blocks_pe_rerating():
    fin, px = _declining(_fundamentals("X")), _prices(price=60_000)
    price = 20_000.0                                                     # "rẻ" so với lịch sử
    plain = fv.combine(price, fin, px, reversion=0.5)
    q = fv.combine(price, fin, px, reversion=0.5,
                   quality={"roe_adjust": True, "no_rerate_below": -0.05})
    assert q.no_rerate and q.roe_factor < 0.7
    eps = plain.methods["pe_hist"] / plain.pe_target
    assert q.pe_target == pytest.approx(price / eps)                    # P/E giữ ở mức hiện tại
    assert q.methods["pb_hist"] < plain.methods["pb_hist"]
    assert q.value < plain.value
    assert any("ROE hiện tại" in n for n in q.notes)
    assert any("lợi nhuận đang giảm" in n for n in q.notes)
    roe_note = next(n for n in q.notes if "ROE hiện tại" in n)
    assert "," in roe_note and "." not in roe_note                       # 12,5% chứ không 12.5%
    growing = fv.combine(price, _fundamentals("X"), px, reversion=0.5,
                         quality={"roe_adjust": True, "no_rerate_below": -0.05})
    plain_g = fv.combine(price, _fundamentals("X"), px, reversion=0.5)
    assert growing.value == pytest.approx(plain_g.value)                 # DN tốt: không đổi


def test_cap_by_consensus():
    fair = fv.FairValue(price=33_800, methods={}, value=77_353, p10=70_000, p90=130_000)
    fv.cap_by_consensus(fair, 52_100, 0.3)
    assert fair.value == pytest.approx(52_100 * 1.3) and fair.uncapped == 77_353
    assert fair.p10 <= fair.value and "giá mục tiêu CTCK 52.100đ" in fair.notes[-1]
    ok = fv.FairValue(price=33_800, methods={}, value=60_000)
    fv.cap_by_consensus(ok, 52_100, 0.3)                                # trong biên: giữ nguyên
    assert ok.value == 60_000 and np.isnan(ok.uncapped)
    off = fv.FairValue(price=33_800, methods={}, value=77_353)
    fv.cap_by_consensus(off, float("nan"), 0.3)                          # không có giá CTCK
    assert off.value == 77_353


def test_valuation_group_weights(warehouse):
    """Bước 2.1: bất động sản bỏ DCF, nhóm chu kỳ giảm DCF; còn lại trọng số chung."""
    from vnsa import analysis
    from vnsa.data.store import Store

    with Store(warehouse) as store:
        store.write_companies(pd.DataFrame([
            {"symbol": "DEMO_A", "name": "A", "exchange": "HOSE", "industry": "Bất động sản",
             "model": "general"}]), "test")
    assert analysis.valuation_group("DEMO_A") == ("real_estate", "Bất động sản")
    assert analysis.valuation_group("DEMO_B")[0] == "general"         # chưa có ngành
    d = analysis.load("DEMO_A")
    _, v, _ = analysis.valuation(d)
    assert v.value_per_share > 0                                     # DCF tính được nhưng không dùng
    fair = analysis.general_fair_value(d, v)
    assert "dcf" not in fair.weights and fair.methods["dcf"] > 0
    assert any("quỹ đất" in n for n in fair.notes)


LADDER = {**ENTRY, "ladder": True, "ladder_weights": [0.3, 0.4, 0.3], "l2_range_atr": 4.0,
          "l3_range_atr": 7.0, "ladder_gap_atr": 0.75, "ladder_fallback_atr": 1.5,
          "ladder_stop_atr": 1.0}


def test_ladder_uses_strong_supports_deeper_than_old_zone():
    """v2.1: lớp 1 = neo cũ; lớp 2/3 = hỗ trợ chạm nhiều lần sâu hơn; cắt lỗ dưới lớp 3."""
    ind = _ind(close=30_000, sma20=29_500, atr=500).assign(sma50=28_800.0, sma200=26_000.0)
    sup = [Level(29_600, 1, pd.Timestamp("2026-01-05")),     # gần nhưng mới chạm 1 lần
           Level(28_400, 3, pd.Timestamp("2026-01-06")),     # mạnh: lớp 2
           Level(27_000, 2, pd.Timestamp("2026-01-07"))]     # lớp 3
    old = entry_zone(ind, sup, ENTRY, TICKS)
    z = entry_zone(ind, sup, LADDER, TICKS)
    assert [t.price for t in z.tiers] == [30_000, 28_400, 27_000]
    assert "3 lần chạm" in z.tiers[1].source and z.tiers[2].depth == pytest.approx(-0.10)
    assert z.low == 27_000 and z.high == z.tiers[0].price and z.low < old.low
    assert z.mid == pytest.approx(round((30_000 * .3 + 28_400 * .4 + 27_000 * .3) / 50) * 50)
    assert z.stop_cap == pytest.approx(26_500)                    # lớp 3 − 1 ATR


def test_ladder_fallback_and_value_cap():
    ind = _ind(close=30_000, sma20=29_500, atr=500)
    z = entry_zone(ind, [], LADDER, TICKS)                        # không có hỗ trợ nào
    p = [t.price for t in z.tiers]
    assert p[1] == pytest.approx(p[0] - 750) and p[2] == pytest.approx(p[1] - 750)
    # giá cao hơn trần giá trị (25.500): thang mua bắt đầu từ mức đỡ dưới trần, không phải một số
    sup = [Level(25_200, 2, pd.Timestamp("2026-01-05")), Level(23_500, 3, pd.Timestamp("2026-01-06"))]
    wait = entry_zone(ind, sup, LADDER, TICKS, fair_value=30_000)
    assert wait.status == "CHỜ GIÁ"
    assert [t.price for t in wait.tiers][:2] == [25_200, 23_500]
    assert all(t.price <= 25_500 for t in wait.tiers)


def test_trade_plan_stop_below_deepest_tier():
    from vnsa.ta.plan import trade_plan

    cfg = {"atr_stop": 2.0, "support_buffer_atr": 0.5, "max_stop_atr": 3.0, "min_stop_atr": 1.0,
           "reward_multiples": [2.0, 3.0], "capital": 100_000_000, "risk_per_trade": 0.01,
           "lot_size": 100, "max_position_pct": 0.15}
    p = trade_plan(28_500, 500, [], [], cfg, TICKS, max_stop=26_500)
    assert p.stop <= 26_500 and "lớp" in p.stop_source
