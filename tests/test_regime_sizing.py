"""Đợt 2: trạng thái thị trường (ta/regime.py), trần tỷ trọng và giới hạn thanh khoản (ta/plan.py)."""

import numpy as np
import pandas as pd
import pytest

from vnsa.ta.levels import Level
from vnsa.ta.plan import trade_plan
from vnsa.ta.regime import regime, regime_series

RCFG = {"sma_long": 200, "sma_short": 50, "slope_sessions": 20}
PLAN = {"capital": 100_000_000, "risk_per_trade": 0.01, "atr_stop": 2.0, "min_stop_atr": 1.0,
        "max_stop_atr": 3.0, "support_buffer_atr": 0.5, "reward_multiples": [2.0, 3.0],
        "min_reward_risk": 2.0, "lot_size": 100}


def _index(closes):
    return pd.DataFrame({"date": pd.bdate_range("2025-01-01", periods=len(closes)),
                         "close": np.asarray(closes, dtype=float)})


def test_regime_up_down_and_neutral():
    up = _index(np.linspace(1000, 1400, 300))
    down = _index(np.linspace(1400, 1000, 300))
    assert regime(up, RCFG).state == "tăng" and regime(down, RCFG).state == "giảm"
    assert regime(_index(np.linspace(1000, 1400, 100)), RCFG).state == "trung tính"   # < 200 phiên
    s = regime_series(down, RCFG)
    assert s.iloc[0] == "trung tính" and s.iloc[-1] == "giảm"
    assert "MA200" in str(regime(up, RCFG))


def test_position_caps_by_nav_liquidity_and_bear_market():
    entry, atr = 20_000.0, 200.0                       # cắt lỗ 400đ -> 1% vốn = 2.500 cp
    base = trade_plan(entry, atr, [], [Level(25_000, 2, pd.Timestamp("2026-01-01"))], PLAN)
    assert base.shares == 2_500
    capped = trade_plan(entry, atr, [], [], {**PLAN, "max_position_pct": 0.15})
    assert capped.shares == 700 and any("trần tỷ trọng 15%" in n for n in capped.notes)
    thin = trade_plan(entry, atr, [], [], {**PLAN, "max_adv_pct": 0.10}, adv_value=200_000_000)
    assert thin.shares == 1_000 and any("thanh khoản" in n for n in thin.notes)   # 20 triệu / 20k
    bear = trade_plan(entry, atr, [], [], PLAN, risk_scale=0.5)
    assert bear.shares == 1_200 and bear.risk_budget == pytest.approx(500_000)


def test_assess_reports_regime_and_dividend_yield(warehouse):
    from vnsa import analysis
    from vnsa.data.store import Store

    with Store(warehouse) as store:
        last = store.prices("DEMO_A")["date"].iloc[-1]
        store.write_dividends(pd.DataFrame({"symbol": ["DEMO_A"],
                                            "exright_date": [last - pd.Timedelta(days=30)],
                                            "cash_per_share": [1_000.0], "stock_ratio": [None],
                                            "title": ["Cổ tức"]}), "DEMO_A")
    a = analysis.assess("DEMO_A")
    x = a.rec.inputs
    assert x.regime in {"tăng", "trung tính", "giảm"}
    assert x.dividend_yield == pytest.approx(1_000 / x.price)
    assert a.rec.expected_return == pytest.approx(x.fair_upside + x.dividend_yield)
    assert a.rec.timing
