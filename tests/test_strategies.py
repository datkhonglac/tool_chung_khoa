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


def _ind(n=5, **cols):
    base = {"date": pd.bdate_range("2024-01-01", periods=n), "open": 100.0, "high": 104.0,
            "low": 99.0, "close": 103.0, "volume": 1_500.0, "vol_ma20": 1_000.0, "atr": 2.0,
            "sma20": 102.0, "sma50": 100.0, "sma200": 90.0, "rsi": 55.0, "bb_width_pct": 0.2}
    base.update(cols)
    return pd.DataFrame(base)


def _vcp(ind):
    from vnsa.recommend.strategies import rule_set

    cfg = {"entry_score": 65, "exit_score": 45, "atr_stop": 2.0}
    return rule_set(ind, pd.Series(50.0, index=ind.index), None, 2.5, cfg)["vcp_pullback"]


def test_vcp_pullback_entry_conditions():
    assert _vcp(_ind()).entry.all()                                  # đủ 4 điều kiện
    assert not _vcp(_ind(rsi=70.0)).entry.any()                      # RSI quá cao: mua đuổi
    assert not _vcp(_ind(sma50=85.0)).entry.any()                    # MA50 dưới MA200
    assert not _vcp(_ind(volume=1_100.0)).entry.any()                # chưa có phiên xác nhận
    assert not _vcp(_ind(close=99.5)).entry.any()                    # nến giảm (đóng < mở)
    # nền chưa chặt nhưng giá sát MA20 (≤ 1 ATR) vẫn đạt; xa MA20 thì không
    assert _vcp(_ind(bb_width_pct=0.8, sma20=102.0)).entry.all()
    assert not _vcp(_ind(bb_width_pct=0.8, sma20=99.0)).entry.any()


def test_vcp_pullback_exit_needs_volume():
    assert _vcp(_ind(close=95.0, volume=2_000.0)).exit.all()         # thủng MA50, KL lớn
    assert not _vcp(_ind(close=95.0, volume=1_200.0)).exit.any()     # thủng MA50, KL thường


def test_squeeze_percentile_rank():
    from vnsa.ta.indicators import percentile_rank

    s = pd.Series(np.r_[np.linspace(1, 2, 119), 0.5])                # phiên cuối hẹp nhất
    r = percentile_rank(s, 120)
    assert np.isnan(r.iloc[118]) and r.iloc[-1] == pytest.approx(1 / 120)
