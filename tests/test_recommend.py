import numpy as np
import pandas as pd
import pytest

from vnsa.recommend.backtest import backtest
from vnsa.recommend.engine import Inputs, recommend

CFG = {"rating_thresholds": [0.15, 0.05, -0.05, -0.15], "bear_market_extra": 0.05,
       "min_fa_quality": 50, "timing_wait": 40, "timing_go": 55, "weight_fa": 0.6,
       "weight_ta": 0.4, "min_reward_risk": 2.0, "high_vol": 0.40}
NAN = float("nan")


def _inputs(**kw):
    base = {"symbol": "X", "price": 50_000, "date": pd.Timestamp("2026-10-09"), "fa_score": 80,
            "ta_score": 70, "z_em": 3.0, "dcf_upside": 0.2, "pe": 10, "pb": 1.5,
            "vol_annual": 0.3, "prob_up": 0.55, "first_target_rr": 2.5,
            "fair_value": 60_000, "fair_upside": 0.20, "dividend_yield": 0.0,
            "zone_status": "MUA TRONG VÙNG"}
    return Inputs(**{**base, **kw})


@pytest.mark.parametrize("upside, expected", [(0.20, "MUA"), (0.10, "KHẢ QUAN"), (0.0, "NẮM GIỮ"),
                                              (-0.10, "KÉM KHẢ QUAN"), (-0.30, "BÁN")])
def test_rating_by_expected_return(upside, expected):
    assert recommend(_inputs(fair_upside=upside), CFG).action == expected


def test_dividend_adds_to_expected_return_and_bear_market_raises_bar():
    r = recommend(_inputs(fair_upside=0.12, dividend_yield=0.04), CFG)
    assert r.action == "MUA" and r.expected_return == pytest.approx(0.16)
    bear = recommend(_inputs(fair_upside=0.12, dividend_yield=0.04, regime="giảm"), CFG)
    assert bear.action == "KHẢ QUAN"                            # cần ≥ 20% khi thị trường giảm
    assert any("thị trường chung đang giảm" in s for s in bear.reasons)


def test_quality_gates_and_score_is_only_for_sorting():
    weak = recommend(_inputs(fa_score=40), CFG)
    assert weak.action == "KHẢ QUAN" and any("hạ một bậc" in s for s in weak.reasons)
    trap = recommend(_inputs(fair_upside=0.5, z_em=0.9), CFG)
    assert trap.action == "NẮM GIỮ"                             # rẻ nhưng nguy cơ tài chính
    r = recommend(_inputs(fa_score=80, ta_score=10), CFG)       # FA tốt, TA xấu: vẫn MUA ...
    assert r.action == "MUA" and r.timing == "Chờ xu hướng cải thiện"   # ... nhưng chờ thời điểm
    assert r.score == pytest.approx(0.6 * 80 + 0.4 * 10)


def test_timing_axis_and_not_rated_without_fair_value():
    assert recommend(_inputs(ta_score=70), CFG).timing == "Vào được trong vùng mua"
    assert recommend(_inputs(ta_score=50), CFG).timing == "Mua dần trong vùng mua"
    assert recommend(_inputs(zone_status="CHỜ GIÁ"), CFG).timing == "Chờ giá về vùng mua"
    r = recommend(_inputs(fair_upside=NAN, fa_score=NAN), CFG)
    assert r.action == "KHÔNG XẾP HẠNG" and r.score == 70
    r = recommend(_inputs(first_target_rr=1.2, vol_annual=0.5), CFG)
    assert any("1.2R" in s for s in r.reasons) and any("biến động cao" in s for s in r.reasons)


BT = {"entry_score": 65, "exit_score": 45, "atr_stop": 2.0, "fee": 0.0, "sell_tax": 0.0}


def _frame(opens, lows=None, atr=1_000.0):
    opens = np.asarray(opens, dtype=float)
    return pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=len(opens)),
        "open": opens, "high": opens + 500, "low": opens - 500 if lows is None else lows,
        "close": opens, "atr": atr,
    })


def test_backtest_trades_next_open_and_respects_t2():
    # Cần ≥ 30 phiên điểm TA; tín hiệu bán ở phiên 4, sau đó điểm thấp nên không mua lại
    df = _frame([10_000, 10_000, 11_000, 12_000, 13_000, 14_000] + [15_000] * 30)
    score = pd.Series([70, 70, 70, 70, 30] + [30] * 31)
    res = backtest(df, score, BT)
    t = res.trades[0]
    assert (t.entry, t.exit, t.reason) == (10_000, 14_000, "điểm TA")   # mua mở cửa p1, bán p5
    assert res.stats["total"] == pytest.approx(0.4)
    assert res.benchmark_stats["total"] == pytest.approx(0.5)


def test_backtest_stop_waits_for_t2_and_includes_fees():
    # Mua ở phiên 1 (T), cắt lỗ 8.000. Giá thủng cắt lỗ ngay ở T và T+1 nhưng chưa bán được;
    # phiên 3 (T+2) vẫn dưới cắt lỗ nên bán ở 8.000.
    lows = [9_500, 7_000, 7_500, 7_800] + [9_500] * 30
    df = _frame([10_000] * 34, lows=np.array(lows, dtype=float))
    score = pd.Series([70] + [50] * 33)                    # mua 1 lần, sau đó giữ (≥ 45)
    res = backtest(df, score, {**BT, "fee": 0.001, "sell_tax": 0.001})
    t = res.trades[0]
    assert t.reason == "cắt lỗ" and t.exit_date == df["date"].iloc[3]   # bán sớm nhất ở T+2
    assert t.exit == 8_000
    expected = (1 - 0.001) / 10_000 * 8_000 * (1 - 0.002) - 1
    assert res.stats["total"] == pytest.approx(expected)
    with pytest.raises(ValueError):
        backtest(df.head(10), score.head(10), BT)


def test_backtest_regime_filter_and_liquidity_cap():
    df = _frame([10_000] * 3 + [12_000] * 33).assign(volume=1_000)
    score = pd.Series([70] * 36)                           # luôn đủ điểm mua, không bán
    bear = pd.Series("giảm", index=df["date"])
    # VNINDEX giảm: cần điểm ≥ 75 -> không mua
    assert backtest(df, score, {**BT, "bear_entry_extra": 10}, bear).trades == []
    # VNINDEX giảm, điểm đủ: chỉ dùng 50% vốn -> lãi 20% × 50%
    res = backtest(df, score, {**BT, "bear_entry_extra": 0, "bear_position": 0.5}, bear)
    assert res.stats["total"] == pytest.approx(0.5 * 0.2)
    # Thanh khoản: GTGD 20 phiên ~10-12 triệu, vốn mỗi mã 50 triệu, lệnh ≤ 10% GTGD -> ~2% vốn < 10%
    df = _frame([10_000] * 23 + [12_000] * 40).assign(volume=1_000)
    score = pd.Series([np.nan] * 20 + [70] * 43)           # điểm TA có sau khi đủ 20 phiên GTGD
    liq = {**BT, "capital": 50_000_000, "max_adv_pct": 0.1, "min_position": 0.1}
    assert backtest(df, score, liq).trades == []
    loose = backtest(df, score, {**liq, "min_position": 0.0})
    assert loose.stats["total"] == pytest.approx(0.1 * 10_000_000 / 50_000_000 * 0.2)


def test_backtests_compares_two_formulas(warehouse):
    from vnsa import analysis

    d = analysis.load("DEMO_A", need_fin=False)
    res = analysis.backtests(d)
    assert set(res) == {"Điểm TA (hiện tại)", "Điểm TA + VNINDEX"}
    a, b = res.values()
    assert len(a.equity) == len(b.equity) and a.equity.index[0] == b.equity.index[0]


def test_timing_rule_modes():
    from vnsa.recommend.engine import timing

    x = _inputs(zone_status="MUA TRONG VÙNG")
    assert timing(x, {**CFG, "timing_rule": "none"}).startswith("Trong vùng mua")
    x.rule_state = {"holding": True, "stop": 52_300.0, "since": None, "signal_today": False,
                    "rule": "trend"}
    assert timing(x, {**CFG, "timing_rule": "trend"}) == ("Vào được — xu hướng tăng xác nhận, "
                                                         "cắt lỗ động 52,300")
    x.rule_state = {**x.rule_state, "holding": False}
    assert timing(x, {**CFG, "timing_rule": "trend"}) == "Chờ tín hiệu xu hướng"
    x.zone_status = "CHỜ GIÁ"                                    # định giá vẫn chặn trước
    assert timing(x, {**CFG, "timing_rule": "trend"}) == "Chờ giá về vùng mua"


def test_rule_state_reports_open_position():
    from vnsa.recommend.strategies import rule_state

    n = 400
    close = pd.Series(np.linspace(10_000, 20_000, n))            # xu hướng tăng đều
    ind = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=n), "open": close,
                        "high": close + 50, "low": close - 50, "close": close,
                        "volume": 1_000.0, "atr": 100.0})
    ind["sma50"] = ind["close"].rolling(50).mean()
    ind["sma200"] = ind["close"].rolling(200).mean()
    ind["vol_ma20"] = 1_000.0
    st = rule_state(ind, pd.Series(np.nan, index=ind.index), None, "trend", 3.0,
                    {**BT, "fee": 0, "sell_tax": 0}, 0.07)
    assert st["holding"] and st["stop"] == pytest.approx(20_000 - 300)
