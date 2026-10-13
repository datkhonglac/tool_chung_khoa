import numpy as np
import pandas as pd
import pytest

from vnsa.ta import indicators as ti
from vnsa.ta.levels import Level, cluster, support_resistance, swing_points
from vnsa.ta.plan import round_tick, tick_size, trade_plan
from vnsa.ta.score import rsi_points, ta_score


def _candles(close, spread=100.0, volume=1000):
    close = pd.Series(close, dtype=float)
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=len(close)),
        "open": close, "high": close + spread, "low": close - spread, "close": close,
        "volume": volume,
    })


def test_moving_averages_and_macd_on_simple_series():
    s = pd.Series([1.0, 2, 3, 4, 5])
    assert ti.sma(s, 3).tolist()[2:] == [2, 3, 4]
    assert ti.ema(s, 3).iloc[1] == pytest.approx(1.5)          # 0,5 × 2 + 0,5 × 1
    flat = ti.macd(pd.Series([10.0] * 60))
    assert flat.abs().max().max() == pytest.approx(0)


def test_rsi_extremes_and_balance():
    up = pd.Series(np.arange(1.0, 40))
    assert ti.rsi(up).iloc[-1] == 100
    zigzag = pd.Series([100.0, 101] * 50)
    assert ti.rsi(zigzag).iloc[-1] == pytest.approx(50, abs=3)
    assert ti.rsi(pd.Series(np.arange(40.0, 1, -1))).iloc[-1] == pytest.approx(0)


def test_atr_bollinger_adx():
    df = _candles([10_000.0] * 30, spread=150)
    assert ti.atr(df).iloc[-1] == pytest.approx(300)            # cao − thấp không đổi
    assert np.isnan(ti.bollinger(df["close"])["bb_pctb"].iloc[-1])   # dải bằng 0 -> không chia
    trend = _candles(10_000 + 100 * np.arange(80.0))
    a = ti.adx(trend).iloc[-1]
    assert a["adx"] > 25 and a["plus_di"] > a["minus_di"]


def test_obv_adds_up_days_and_subtracts_down_days():
    df = _candles([10.0, 11, 10, 10, 12], volume=100)
    assert ti.obv(df).tolist() == [0, 100, 0, 0, 100]


def _zigzag():
    """Dao động giữa đáy ~100 và đỉnh ~120 (nghìn đồng), 4 chu kỳ, kết thúc ở 110."""
    leg = list(np.linspace(100, 120, 11)) + list(np.linspace(120, 100, 11))[1:-1]
    close = (leg * 4) + list(np.linspace(100, 110, 6))
    close = [c * 1000 for c in close]
    return _candles(close, spread=0)


def test_support_and_resistance_from_swings():
    df = _zigzag()
    pts = swing_points(df, window=3)
    assert set(pts["kind"]) == {"high", "low"}
    sup, res = support_resistance(df, window=3, lookback=500, tolerance=0.02)
    assert sup[0].price == pytest.approx(100_000) and sup[0].touches >= 3
    assert res[0].price == pytest.approx(120_000) and res[0].touches >= 3


def test_cluster_merges_close_levels():
    d = pd.Timestamp("2025-01-01")
    pts = pd.DataFrame({"price": [100.0, 101.0, 130.0], "date": [d, d, d]})
    levels = cluster(pts, tolerance=0.02)
    assert [(lv.price, lv.touches) for lv in levels] == [(100.5, 2), (130.0, 1)]


def test_tick_rounding_hose():
    assert tick_size(9_990) == 10 and tick_size(20_000) == 50 and tick_size(57_900) == 100
    assert round_tick(57_930) == 57_900 and round_tick(20_120) == 20_100
    assert round_tick(48_520, "down") == 48_500 and round_tick(48_520, "up") == 48_550


CFG = {"capital": 100_000_000, "risk_per_trade": 0.01, "atr_stop": 2.0, "max_stop_atr": 3.0,
       "min_stop_atr": 1.0,
       "support_buffer_atr": 0.5, "reward_multiples": [2.0, 3.0], "min_reward_risk": 2.0,
       "lot_size": 100}
D = pd.Timestamp("2025-01-01")


def test_trade_plan_atr_stop_targets_and_size_by_hand():
    p = trade_plan(50_000, 1_000, [], [Level(56_000, 3, D)], CFG)
    assert (p.entry, p.stop) == (50_000, 48_000)                 # 2 × ATR
    assert [(t.price, t.source) for t in p.targets] == [(54_000, "2R"), (56_000, "kháng cự")]
    assert p.targets[1].reward_risk == pytest.approx(3)
    assert p.shares == 500                                       # 1 triệu / 2.000 đ
    assert p.max_loss == 1_000_000 and p.position_value == 25_000_000


def test_trade_plan_stop_below_nearby_support_and_cash_cap():
    p = trade_plan(50_000, 1_000, [Level(49_000, 2, D)], [], CFG)
    assert p.stop == 48_500 and "hỗ trợ" in p.stop_source          # 49.000 − 0,5 ATR
    assert [t.price for t in p.targets] == [53_000, 54_500]        # 2R, 3R với R = 1.500
    # Hỗ trợ sát giá (cắt lỗ dưới nó chỉ cách 0,6 ATR) bị bỏ qua, dùng hỗ trợ kế tiếp
    p = trade_plan(50_000, 1_000, [Level(49_900, 1, D), Level(48_000, 2, D)], [], CFG)
    assert p.stop == 47_500 and "48,000" in p.stop_source
    p = trade_plan(50_000, 1_000, [Level(49_900, 1, D)], [], CFG)
    assert p.stop == 48_000 and p.stop_source == "2 × ATR"
    tight = trade_plan(50_000, 50, [], [], CFG)                    # R = 100 đ -> 10.000 cp
    assert tight.shares == 2_000                                   # giới hạn bởi vốn 100 triệu
    with pytest.raises(ValueError):
        trade_plan(50_000, float("nan"), [], [], CFG)


def test_ta_score_uptrend_vs_downtrend():
    weights = {"xu_huong": 40, "dong_luong": 30, "suc_manh": 15, "dong_tien": 15}
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 150, 300)
    up = ti.add_indicators(_candles(20_000 + 60 * np.arange(300.0) + noise))
    down = ti.add_indicators(_candles(40_000 - 60 * np.arange(300.0) + noise))
    su, sd = ta_score(up, weights), ta_score(down, weights)
    assert su.groups["xu_huong"] == 100 and sd.groups["xu_huong"] == 0
    assert su.total > 60 and sd.total < 40
    assert rsi_points(65) == 100 and rsi_points(80) == 50 and rsi_points(30) == 0


def test_technical_chart_draws_configured_smas():
    """Biểu đồ kỹ thuật (dashboard) vẽ mọi đường SMA trong [ta] sma, gồm SMA10."""
    pytest.importorskip("plotly")
    from vnsa.config import settings
    from vnsa.report.charts import technical_chart

    assert 10 in settings()["ta"]["sma"]
    ind = ti.add_indicators(_candles(20_000 + 60 * np.arange(300.0)), settings()["ta"])
    names = [t.name for t in technical_chart(ind).data]
    assert [n for n in names if n.startswith("SMA")] == ["SMA10", "SMA20", "SMA50", "SMA200"]
