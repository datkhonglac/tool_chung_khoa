import math

import numpy as np
import pandas as pd
import pytest

from vnsa.fa import valuation as val

B = 1_000_000_000


def _fin():
    return pd.DataFrame({
        "year": [2023, 2024, 2025],
        "net_income_parent": [100 * B, 110 * B, 121 * B],   # CAGR đúng 10%
        "equity": [1000 * B, 1100 * B, 1200 * B],
        "minority_interest": [0, 0, 200 * B],              # VCSH cty mẹ cuối kỳ: 1.000 tỷ
        "short_debt": [300 * B, 300 * B, 300 * B],
        "long_debt": [100 * B, 100 * B, 100 * B],
        "interest_expense": [40 * B, 40 * B, 40 * B],       # k_d = 40 / 400 = 10%
        "pbt": [125 * B, 137.5 * B, 150 * B],
        "tax_expense": [25 * B, 27.5 * B, 30 * B],          # thuế suất 20%
        "cfo": [150 * B, 160 * B, 170 * B],
        "capex": [50 * B, 60 * B, 70 * B],
        "cash": [80 * B, 90 * B, 100 * B],
        "short_term_investments": [0, 0, 50 * B],
        "paid_in_capital": [1000 * B, 1000 * B, 1000 * B],  # 100 triệu cp
    })


def test_fcff_and_growth_by_hand():
    h = val.fcff(_fin())
    # 150 + 40×0,8 − 50 = 132 ; 160 + 32 − 60 = 132 ; 170 + 32 − 70 = 132
    assert h.tolist() == pytest.approx([132 * B] * 3)
    assert val.historical_growth(_fin(), cap=0.20) == pytest.approx(0.10)
    assert val.historical_growth(_fin(), cap=0.05) == 0.05


def test_dcf_two_stage_by_hand():
    d = val.dcf_enterprise_value(100.0, g1=0.10, years=2, g=0.03, wacc=0.12)
    f1, f2 = 110.0, 121.0
    tv = f2 * 1.03 / (0.12 - 0.03)
    assert d["pv_flows"] == pytest.approx(f1 / 1.12 + f2 / 1.12**2)
    assert d["terminal_value"] == pytest.approx(tv)
    assert d["ev"] == pytest.approx(f1 / 1.12 + f2 / 1.12**2 + tv / 1.12**2)
    with pytest.raises(ValueError):
        val.dcf_enterprise_value(100.0, 0.1, 2, g=0.12, wacc=0.12)


def test_full_valuation_wacc_bridge_and_multiples():
    a = val.Assumptions(risk_free=0.03, equity_risk_premium=0.08, terminal_growth=0.03,
                        forecast_years=5, adjust_beta=False)
    v = val.value("abc", _fin(), price=20_000, a=a, beta_value=1.25, beta_weeks=150)
    assert v.shares == 100_000_000
    assert v.cost_of_equity == pytest.approx(0.03 + 1.25 * 0.08)       # 13%
    assert v.cost_of_debt == pytest.approx(0.10)
    assert v.tax_rate == pytest.approx(0.20)
    e, d = 20_000 * 1e8, 400 * B                                        # 2.000 tỷ vs 400 tỷ
    assert v.weight_equity == pytest.approx(e / (e + d))
    assert v.wacc == pytest.approx(e / (e + d) * 0.13 + d / (e + d) * 0.10 * 0.8)
    assert v.fcff_base == pytest.approx(132 * B)
    ev = val.dcf_enterprise_value(132 * B, 0.10, 5, 0.03, v.wacc)["ev"]
    # Cầu nối: − vay 400 + tiền 100 + ĐTNH 50 − thiểu số 200
    assert v.equity_value == pytest.approx(ev - 400 * B + 150 * B - 200 * B)
    assert v.value_per_share == pytest.approx(v.equity_value / 1e8)
    assert v.pe == pytest.approx(20_000 / (121 * B / 1e8))              # EPS 1.210 đ
    assert v.pb == pytest.approx(20_000 / (1000 * B / 1e8))            # BVPS 10.000 đ -> 2,0 lần


def test_blume_adjusted_beta():
    v = val.value("abc", _fin(), price=20_000, a=val.Assumptions(), beta_value=1.3)
    assert v.beta == pytest.approx(0.67 * 1.3 + 0.33)
    assert v.cost_of_equity == pytest.approx(0.035 + v.beta * 0.08)


def test_default_beta_and_negative_fcff_are_flagged():
    fin = _fin()
    fin["capex"] = 500 * B                                              # FCFF âm
    v = val.value("abc", fin, price=20_000, a=val.Assumptions())
    assert v.beta_is_default and v.beta == 1.0
    assert math.isnan(v.value_per_share)
    assert any("FCFF" in n for n in v.notes)


def test_beta_recovers_known_slope():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2022-01-03", periods=900)
    m = rng.normal(0, 0.01, len(days))
    s = 1.5 * m + rng.normal(0, 0.002, len(days))
    index = pd.DataFrame({"date": days, "close": 1000 * np.cumprod(1 + m)})
    stock = pd.DataFrame({"date": days, "close": 50 * np.cumprod(1 + s)})
    b, weeks = val.beta(stock, index, years=3)
    assert weeks > 150
    assert b == pytest.approx(1.5, abs=0.1)


def test_sensitivity_grid_center_equals_base():
    a = val.Assumptions(risk_free=0.03, terminal_growth=0.03)
    v = val.value("abc", _fin(), price=20_000, a=a, beta_value=1.25)
    t = val.sensitivity(v, a)
    assert t.shape == (5, 5)
    assert t.loc[t.index[2], 0.03] == pytest.approx(v.value_per_share)
    assert t.iloc[0, 2] > t.iloc[4, 2]          # WACC thấp hơn -> giá trị cao hơn
