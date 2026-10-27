"""v2.1: kiểm định điểm mua — vùng cũ vs thang 3 lớp vs mua ngay, không nhìn trước."""

import numpy as np
import pandas as pd
import pytest

from vnsa.recommend.entry_study import evaluate_symbol, format_entry_study, study, summarize
from vnsa.ta.indicators import add_indicators

TA = {"swing_window": 5, "lookback": 250, "level_tolerance": 0.02}
ENTRY = {"max_pullback_atr": 2.0, "zone_atr": 0.5, "chase_atr": 2.0, "vwap_sessions": 20,
         "margin_of_safety": 0.15, "weak_fa_extra": 0.10, "ladder_weights": [0.3, 0.4, 0.3],
         "l2_range_atr": 4.0, "l3_range_atr": 7.0, "ladder_gap_atr": 0.75,
         "ladder_fallback_atr": 1.5, "ladder_stop_atr": 1.0}


def _prices(n=700, seed=5):
    rng = np.random.default_rng(seed)
    close = 20_000 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n)))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    return pd.DataFrame({"date": pd.bdate_range("2019-01-01", periods=n), "open": open_,
                         "high": np.maximum(open_, close) * 1.01,
                         "low": np.minimum(open_, close) * 0.99, "close": close,
                         "volume": 1_000_000.0})


def test_evaluate_symbol_fills_and_no_lookahead():
    ind = add_indicators(_prices())
    rows = evaluate_symbol(ind, ENTRY, TA, step=20)
    assert set(rows["variant"]) == {"Mua ngay", "Vùng cũ", "Thang 3 lớp", "Vùng VA"}
    lad = rows[rows["variant"] == "Thang 3 lớp"]
    assert len(lad) % 3 == 0 and (lad.groupby("date")["weight"].sum().round(6) == 1).all()
    # lệnh đã khớp có giá khớp ≤ giá đặt; lệnh không khớp không có lãi
    f = rows[rows["filled"]]
    assert (f["fill_px"] <= f["price"] + 1e-6).all()
    assert rows.loc[~rows["filled"], "ret"].isna().all()
    # đổi giá TƯƠNG LAI không làm đổi giá đặt lệnh ở phiên trước đó
    p2 = _prices()
    p2.loc[600:, ["open", "high", "low", "close"]] *= 1.5
    rows2 = evaluate_symbol(add_indicators(p2), ENTRY, TA, step=20)
    early = rows["date"] < pd.Timestamp(p2.at[600, "date"]) - pd.Timedelta(days=120)
    early2 = rows2["date"] < pd.Timestamp(p2.at[600, "date"]) - pd.Timedelta(days=120)
    assert np.allclose(rows.loc[early, "price"].to_numpy(), rows2.loc[early2, "price"].to_numpy())


def test_ladder_is_deeper_and_study_reports():
    data = {s: {"ind": add_indicators(_prices(seed=i))} for i, s in enumerate(["AAA", "BBB"])}
    res = study(data, ENTRY, TA, split="2020-06-01", start="2019-01-01", step=10)
    tiers = res.tiers
    deep = tiers.loc[("Thang 3 lớp", "Lớp 3 sâu"), "depth"]
    assert deep < tiers.loc[("Vùng cũ", "đáy vùng"), "depth"]
    assert tiers.loc[("Thang 3 lớp", "Lớp 3 sâu"), "fill"] <= tiers.loc[("Thang 3 lớp",
                                                                           "Lớp 1 thăm dò"), "fill"]
    tot = res.summary["Toàn kỳ"]
    assert tot.loc["Mua ngay", "fill"] == pytest.approx(1.0)
    assert 0 < tot.loc["Thang 3 lớp", "fill"] < 1
    text = format_entry_study(res)
    assert "Lãi/tổng vốn" in text and "Lớp 3 sâu" in text


def test_summarize_counts_unfilled_as_cash():
    rows = pd.DataFrame([
        {"variant": "X", "symbol": "A", "date": 1, "close": 100, "weight": 0.5, "filled": True,
         "fill_px": 90, "ret": 0.2, "price": 90, "order": "a"},
        {"variant": "X", "symbol": "A", "date": 1, "close": 100, "weight": 0.5, "filled": False,
         "fill_px": np.nan, "ret": np.nan, "price": 80, "order": "b"}])
    r = summarize(rows).loc["X"]
    assert r["fill"] == pytest.approx(0.5) and r["ret_total"] == pytest.approx(0.1)
    assert r["ret_deployed"] == pytest.approx(0.2) and r["discount"] == pytest.approx(-0.1)
