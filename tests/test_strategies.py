"""Đợt 4 bước 3: máy mô phỏng chung (trần/sàn, T+2, cắt lỗ động) và kiểm định walk-forward."""

import numpy as np
import pandas as pd
import pytest

from vnsa.recommend.strategies import Rule, simulate


def _df(closes, opens=None, highs=None, lows=None, atr=100.0):
    c = np.asarray(closes, dtype=float)
    o = c if opens is None else np.asarray(opens, dtype=float)
    return pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=len(c)), "open": o,
                         "high": np.maximum(o, c) + 50 if highs is None else highs,
                         "low": np.minimum(o, c) - 50 if lows is None else lows,
                         "close": c, "atr": atr})


def _rule(n, entry_at=(0,), exit_at=(), stop="fixed", k=50.0):
    e = pd.Series(False, index=range(n))
    x = pd.Series(False, index=range(n))
    e[list(entry_at)] = True
    if exit_at:
        x[list(exit_at)] = True
    return Rule("t", e, x, pd.Series(1.0, index=range(n)), stop, k)


def test_cannot_buy_at_ceiling_open():
    closes = [10_000] + [10_000] * 40
    opens = [10_000, 10_700] + [10_000] * 39          # phiên 1 mở cửa ở trần (+7%)
    res = simulate(_df(closes, opens), _rule(41, entry_at=(0, 1)), 0.07, 0, 0)
    assert res.trades[0].entry_date == pd.Timestamp("2024-01-03")   # mua được ở phiên 2


def test_sell_waits_t2_and_floor_lock():
    n = 40
    closes = [10_000.0] * n
    opens = list(closes)
    highs = [c + 50 for c in closes]
    lows = [c - 50 for c in closes]
    # phiên 3: cả phiên nằm sàn (cao nhất = sàn) -> không bán được, dời sang phiên 4
    closes[3] = opens[3] = highs[3] = lows[3] = 9_300.0
    res = simulate(_df(closes, opens, highs, lows), _rule(n, entry_at=(0,), exit_at=(1, 2)),
                   0.07, 0, 0)
    t = res.trades[0]
    assert t.entry_date == pd.Timestamp("2024-01-02")            # mua mở cửa phiên 1
    assert t.exit_date == pd.Timestamp("2024-01-05")             # phiên 3 bị khoá sàn -> phiên 4


def test_trailing_stop_moves_up():
    closes = [10_000, 10_000, 10_500, 11_000, 11_500, 12_000, 11_000] + [11_000] * 30
    lows = [c - 50 for c in closes]
    lows[6] = 10_900                                             # rơi về dưới 12.000 − 2 × 500
    res = simulate(_df(closes, lows=lows, atr=500.0), _rule(len(closes), stop="trail", k=2.0),
                   0.07, 0, 0)
    t = res.trades[0]
    assert t.reason == "cắt lỗ" and t.exit == pytest.approx(11_000)   # min(mở cửa 11.000, stop 11.000)
    assert t.exit > t.entry                                      # cắt lỗ động đã khoá lãi


def test_timing_study_runs_on_sample(warehouse):
    from vnsa import analysis
    from vnsa.recommend.study import format_study

    res = analysis.timing_study(["DEMO_A", "DEMO_B"], split="2024-06-01", start="2022-06-01")
    assert set(res.table["period"]) == {"Chọn tham số", "Kiểm tra độc lập"}
    assert {"ta", "trend", "breakout", "hold"} <= set(res.table["rule"])
    assert res.chosen_k["trend"] in (2.5, 3.0, 4.0) and res.chosen_k["ta"] == 2.0
    text = format_study(res)
    assert "Mua & giữ" in text and "Kiểm tra độc lập" in text
