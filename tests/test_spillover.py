"""Lan truyền Diebold–Yilmaz: mã dẫn có ròng dương, GFEVD chuẩn hoá, cân bằng tổng lan truyền."""

import numpy as np
import pandas as pd
import pytest

from vnsa.ta import spillover as sp


def _leader_data(T=800, seed=0):
    rng = np.random.default_rng(seed)
    lead = rng.normal(0, 1, T)
    a = 0.6 * np.r_[0, lead[:-1]] + rng.normal(0, 0.6, T)       # A, B đi theo L với độ trễ 1 phiên
    b = 0.5 * np.r_[0, lead[:-1]] + rng.normal(0, 0.7, T)
    c = rng.normal(0, 1, T)                                    # C độc lập
    idx = pd.bdate_range("2023-01-01", periods=T)
    return pd.DataFrame({"L": lead, "A": a, "B": b, "C": c}, index=idx)


def test_var_ols_recovers_lag_coefficients():
    phis, sigma = sp.fit_var(_leader_data().to_numpy(), p=1)
    assert phis[0][1, 0] == pytest.approx(0.6, abs=0.08)       # phương trình A, hệ số trễ của L
    assert sigma.shape == (4, 4)


def test_gfevd_rows_and_leader():
    res = sp.spillover(_leader_data(), p=1, H=10)
    assert np.allclose(res.theta.sum(axis=1), 1.0)
    assert res.net.index[0] == "L" and res.net["L"] > 0.3
    assert abs(res.net["C"]) < 0.1
    assert res.net.sum() == pytest.approx(0.0, abs=1e-9)        # lan ra = nhận vào trên toàn mạng
    assert res.edges(1)[0][0] == "L"
    assert "Dẫn biến động" in sp.format_spillover(res)


def test_ma_coefficients_var1():
    phi = np.array([[0.5, 0.0], [0.2, 0.1]])
    A = sp.ma_coefficients([phi], 3)
    assert np.allclose(A[2], phi @ phi)


def test_rolling_total_and_guards():
    r = _leader_data(600)
    s = sp.rolling_total(r, window=250, step=100)
    assert len(s) >= 3 and ((s > 0) & (s < 1)).all()
    with pytest.raises(ValueError):
        sp.spillover(r[["L", "A"]])
