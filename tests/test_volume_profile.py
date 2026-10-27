"""Hồ sơ khối lượng: rải khối lượng theo giá, POC, vùng giá trị 70%, điểm neo vùng mua."""

import numpy as np
import pandas as pd
import pytest

from vnsa.ta import volume_profile as vp


def test_distribute_uniform_and_flat_day():
    edges = np.array([0.0, 10.0, 20.0, 30.0])
    v = vp.distribute(np.array([5.0, 25.0]), np.array([25.0, 25.0]), np.array([20.0, 25.0]),
                      np.array([100.0, 40.0]), edges)
    # phiên 1: 100 cp rải trên 5–25 (20 đồng): 25 / 50 / 25; phiên 2 không biên độ: 40 vào ô 20–30
    assert v == pytest.approx([25.0, 50.0, 65.0])
    assert v.sum() == pytest.approx(140.0)


def test_value_area_expands_to_larger_side():
    vol = np.array([1, 2, 10, 30, 20, 5, 1, 1], dtype=float)       # tổng 70, 70% = 49
    a, b = vp.value_area(vol, 3, 0.70)
    assert (a, b) == (3, 4) and vol[a:b + 1].sum() >= 49           # 30 → +20 = 50 ≥ 49
    assert vp.value_area(vol, 3, 1.0) == (0, 7)


def _days(n=120):
    rng = np.random.default_rng(1)
    close = np.r_[np.full(n // 2, 100.0), np.full(n - n // 2, 120.0)] + rng.normal(0, 0.5, n)
    vol = np.r_[np.full(n // 2, 1_000.0), np.full(n - n // 2, 5_000.0)]  # giao dịch dày quanh 120
    return pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=n), "close": close,
                         "high": close + 1, "low": close - 1, "volume": vol})


def test_profile_finds_heavy_zone():
    p = vp.volume_profile(_days())
    assert 118 <= p.poc <= 122 and p.va_low > 110 and p.va_high <= 122
    assert p.position(130) == "trên vùng giá trị" and p.position(100) == "dưới vùng giá trị"
    assert "POC" in vp.describe(p, 121) and vp.volume_profile(_days(10)) is None


def test_entry_zone_va_anchor():
    from vnsa.ta.entry import entry_zone

    df = _days()
    df.loc[df.index[-1], ["close", "high", "low"]] = [124.0, 125.0, 123.0]
    df["atr"], df["sma20"] = 2.0, 121.0
    cfg = {"max_pullback_atr": 3.0, "zone_atr": 0.5, "chase_atr": 2.0, "vwap_sessions": 20,
           "margin_of_safety": 0.15}
    classic = entry_zone(df, [], cfg)
    va = entry_zone(df, [], {**cfg, "anchor_mode": "va"})
    assert va.anchor_source in ("cận trên vùng giá trị", "POC")
    assert classic.anchor_source in ("VWAP20", "SMA20")
