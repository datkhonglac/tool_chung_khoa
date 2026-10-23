"""Xoay vòng ngành RRG: chỉ số nhóm theo tuần, RS-Ratio, RS-Momentum, góc phần tư."""

import numpy as np
import pandas as pd
import pytest

from vnsa.screen import rrg

DAYS = pd.bdate_range("2025-01-06", periods=300)


def _px(growth, sym, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    r = np.full(len(DAYS), growth) + rng.normal(0, noise, len(DAYS))
    return pd.DataFrame({"symbol": sym, "date": DAYS, "close": 10_000 * np.cumprod(1 + r)})


def test_weekly_close_and_group_index():
    w = rrg.weekly_close(_px(0.001, "A"))
    assert (w.index.dayofweek == 4).all() and len(w) == 60
    a = pd.Series([100.0, 110.0, 121.0], index=pd.date_range("2026-01-02", periods=3, freq="W-FRI"))
    b = pd.Series([100.0, 90.0, 99.0], index=a.index)
    idx = rrg.group_index({"A": a, "B": b}, min_members=2)
    assert idx.iloc[-1] == pytest.approx(100 * 1.0 * 1.10)       # (+10% −10%)/2 = 0; (+10%+10%)/2
    assert rrg.group_index({"A": a, "B": b}, min_members=3).empty


def test_quadrants():
    assert rrg.quadrant(101, 101) == rrg.LEADING and rrg.quadrant(101, 99) == rrg.WEAKENING
    assert rrg.quadrant(99, 99) == rrg.LAGGING and rrg.quadrant(99, 101) == rrg.IMPROVING


def test_outperforming_group_leads_and_laggard_lags():
    bench = _px(0.0005, "VNINDEX")
    prices, groups = {}, {}
    # nhóm mạnh: tăng nhanh dần (đang tăng tốc so với chỉ số); nhóm yếu: giảm nhanh dần
    for i in range(3):
        g = np.linspace(0.0, 0.004, len(DAYS)) + 0.0005
        prices[f"S{i}"] = pd.DataFrame({"symbol": f"S{i}", "date": DAYS,
                                        "close": 10_000 * np.cumprod(1 + g)})
        groups[f"S{i}"] = "Mạnh"
        w = 0.0005 - np.linspace(0.0, 0.004, len(DAYS))
        prices[f"W{i}"] = pd.DataFrame({"symbol": f"W{i}", "date": DAYS,
                                        "close": 10_000 * np.cumprod(1 + w)})
        groups[f"W{i}"] = "Yếu"
    prices["X0"], groups["X0"] = _px(0.001, "X0"), "Ít mã"           # < 3 mã: bỏ
    tails = rrg.compute(prices, groups, bench, {"tail": 6})
    assert set(tails["group"]) == {"Mạnh", "Yếu"}
    assert tails.groupby("group").size().eq(6).all()
    last = rrg.latest(tails).set_index("group")
    assert last.loc["Mạnh", "quadrant"] == rrg.LEADING
    assert last.loc["Yếu", "quadrant"] == rrg.LAGGING
    text = rrg.format_rrg(tails)
    assert "Dẫn dắt" in text and "Mạnh" in text
    assert "Chưa đủ dữ liệu" in rrg.format_rrg(tails.iloc[0:0])
