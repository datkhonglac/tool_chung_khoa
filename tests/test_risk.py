import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from vnsa.risk.garch import fit_garch, log_returns_pct
from vnsa.risk.simulate import Simulation, garch_var, historical_var, simulate


def _simulate_garch(n=4000, mu=0.05, omega=0.05, alpha=0.10, beta=0.85, seed=1):
    rng = np.random.default_rng(seed)
    s2 = omega / (1 - alpha - beta)
    r = np.empty(n)
    for t in range(n):
        e = np.sqrt(s2) * rng.standard_normal()
        r[t] = mu + e
        s2 = omega + alpha * e ** 2 + beta * s2
    return r


def test_garch_recovers_known_parameters():
    fit = fit_garch(_simulate_garch())
    assert fit.converged
    assert fit.alpha == pytest.approx(0.10, abs=0.04)
    assert fit.beta == pytest.approx(0.85, abs=0.06)
    assert fit.long_run_var == pytest.approx(1.0, rel=0.35)      # 0,05 / (1 − 0,95)
    assert len(fit.resid) == 4000 and np.std(fit.resid) == pytest.approx(1, abs=0.05)


def test_garch_forecast_mean_reverts_and_needs_enough_data():
    fit = fit_garch(_simulate_garch())
    f = fit.forecast_var(2000)
    assert f[0] == pytest.approx(fit.next_sigma ** 2)
    assert f[-1] == pytest.approx(fit.long_run_var, rel=1e-3)
    with pytest.raises(ValueError):
        fit_garch(np.zeros(100))


def test_log_returns_in_percent():
    r = log_returns_pct(pd.Series([100.0, 110.0, 99.0]))
    assert r == pytest.approx([100 * np.log(1.1), 100 * np.log(0.9)])


def test_monte_carlo_respects_band_and_is_reproducible():
    fit = fit_garch(_simulate_garch(omega=0.5, alpha=0.15, beta=0.8))   # biến động rất cao
    a = simulate(fit, 50_000, 20, n_paths=2000, band=0.07, seed=7)
    b = simulate(fit, 50_000, 20, n_paths=2000, band=0.07, seed=7)
    assert np.array_equal(a.paths, b.paths)
    daily = a.paths[:, 1:] / a.paths[:, :-1] - 1
    assert daily.max() <= 0.07 + 1e-9 and daily.min() >= -0.07 - 1e-9
    pc = a.percentiles()
    assert pc[10] < pc[25] < pc[50] < pc[75] < pc[90]
    assert 0 < a.prob_above() < 1


def test_first_touch_and_var_on_known_paths():
    paths = np.array([[100.0, 95, 90, 120],     # chạm cắt lỗ 90 trước
                      [100.0, 105, 111, 80],    # chạm mục tiêu 110 trước
                      [100.0, 101, 102, 103]])  # chưa chạm
    sim = Simulation(100.0, paths)
    assert sim.first_touch(stop=90, target=110) == pytest.approx(
        {"stop": 1 / 3, "target": 1 / 3, "neither": 1 / 3})
    assert sim.var(0.5) == pytest.approx(-0.03)     # trung vị lợi suất +3% -> "lỗ" −3%


def test_historical_and_garch_var_by_hand():
    returns = np.linspace(-0.05, 0.05, 101)          # −5% … +5%, bước 0,1%
    close = pd.Series(100 * np.cumprod(np.r_[1, 1 + returns]))
    v, cv = historical_var(close, q=0.95, window=1000)
    assert v == pytest.approx(0.045, abs=1e-6)       # phân vị 5% của 101 giá trị
    assert cv == pytest.approx(0.0475, abs=1e-6)     # trung bình −5,0% … −4,5%
    fit = fit_garch(_simulate_garch())
    expected = -(np.exp((fit.mu + fit.next_sigma * norm.ppf(0.05)) / 100) - 1)
    assert garch_var(fit, 0.95) == pytest.approx(expected)
