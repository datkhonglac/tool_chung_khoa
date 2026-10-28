"""Bước nhảy Merton trên nền GARCH, VaR chặn biên độ, kiểm định Kupiec, kiểm định VaR quá khứ."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from vnsa.risk import jump as jp
from vnsa.risk import var_study as vs


def test_fit_jumps_counts_large_moves():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 1, 2000)
    z[::100] = -6.0                                  # 20 cú sốc giảm
    fit = jp.fit_jumps(z * 2.0, np.full(2000, 2.0), k=3.0)
    assert fit.n_jumps >= 20 and fit.lam == pytest.approx(fit.n_jumps / 2000)
    assert fit.down_share > 0.8 and 0.9 < fit.diffusion_sd < 1.1


def test_cdf_without_jumps_is_normal_and_quantile():
    fit = jp.JumpFit(lam=0.0, diffusion_sd=1.0)
    assert jp.cdf(0.0, fit)[0] == pytest.approx(0.5)
    assert jp.z_quantile(0.01, fit) == pytest.approx(norm.ppf(0.01), abs=1e-4)
    with_jumps = jp.JumpFit(lam=0.02, diffusion_sd=1.0, jumps=np.array([-6.0, -5.0]))
    assert jp.z_quantile(0.01, with_jumps) < norm.ppf(0.01)        # đuôi trái dày hơn


def test_var_is_capped_by_price_band_and_floor_probability():
    fit = jp.JumpFit(lam=0.05, diffusion_sd=1.0, jumps=np.array([-8.0, -9.0]))
    assert jp.jump_var(fit, sigma_next_pct=3.0, q=0.99, band=0.07) == pytest.approx(0.07)
    assert jp.jump_var(fit, 3.0, 0.99, band=None) > 0.07
    p1, p20 = jp.floor_probability(fit, 3.0, band=0.07, days=20)
    assert 0.05 < p1 < 0.1 and p20 == pytest.approx(1 - (1 - p1) ** 20)


def test_kupiec():
    assert vs.kupiec(1000, 10, 0.01) == pytest.approx(1.0)            # đúng tần suất
    assert vs.kupiec(1000, 30, 0.01) < 0.001                          # vượt quá nhiều
    assert vs.kupiec(1000, 0, 0.01) < 0.001 and np.isnan(vs.kupiec(0, 0, 0.01))


def test_backtest_on_gaussian_returns_has_right_coverage():
    rng = np.random.default_rng(3)
    n = 1300
    r = rng.normal(0, 0.015, n)
    close = pd.Series(10_000 * np.exp(np.cumsum(r)))
    dates = pd.Series(pd.bdate_range("2020-01-01", periods=n))
    rows = vs.backtest_symbol(close, dates, qs=(0.95,), window=500, refit=100)
    assert len(rows) == n - 1 - 500
    summ = vs.summarize(rows).set_index("method")
    assert 0.03 < summ.loc["GARCH-N", "exceed"] < 0.075
    res = vs.VarStudy(["X"], {"Toàn kỳ": vs.summarize(rows)}, rows)
    assert "Kupiec" in vs.format_var_study(res)
