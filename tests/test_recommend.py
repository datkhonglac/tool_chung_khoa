import numpy as np
import pandas as pd
import pytest

from vnsa.recommend.backtest import backtest
from vnsa.recommend.engine import Inputs, recommend

CFG = {"weight_fa": 0.6, "weight_ta": 0.4, "buy_above": 65, "min_ta_buy": 45,
       "min_upside_buy": -0.10, "sell_below": 40, "min_reward_risk": 2.0, "high_vol": 0.40}
NAN = float("nan")


def _inputs(**kw):
    base = {"symbol": "X", "price": 50_000, "date": pd.Timestamp("2026-10-09"), "fa_score": 80,
            "ta_score": 70, "z_em": 3.0, "dcf_upside": 0.2, "pe": 10, "pb": 1.5,
            "vol_annual": 0.3, "prob_up": 0.55, "first_target_rr": 2.5}
    return Inputs(**{**base, **kw})


def test_buy_when_fa_and_ta_agree():
    r = recommend(_inputs(), CFG)
    assert r.action == "MUA"
    assert r.score == pytest.approx(0.6 * 80 + 0.4 * 70)


def test_strong_fa_but_weak_trend_is_watch_with_reason():
    r = recommend(_inputs(fa_score=95, ta_score=30), CFG)     # tổng 69 nhưng TA < 45
    assert r.action == "THEO DÕI"
    assert any("xu hướng" in s for s in r.reasons)
    r = recommend(_inputs(dcf_upside=-0.3), CFG)              # giá cao hơn DCF 30%
    assert r.action == "THEO DÕI"


def test_avoid_on_low_score_or_distress_and_ta_only_fallback():
    assert recommend(_inputs(fa_score=30, ta_score=30), CFG).action == "TRÁNH"
    assert recommend(_inputs(z_em=0.9), CFG).action == "TRÁNH"
    r = recommend(_inputs(fa_score=NAN, ta_score=70), CFG)
    assert r.score == 70 and r.action == "MUA"
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
