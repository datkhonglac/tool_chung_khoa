"""GARCH(1,1) cho lợi suất ngày, ước lượng bằng hợp lý cực đại (phân phối chuẩn).

Mô hình, với r_t = lợi suất log ngày tính bằng % (nhân 100 để tối ưu ổn định):
    r_t = mu + e_t,   e_t = sigma_t × z_t,   z_t ~ N(0, 1)
    sigma²_t = omega + alpha × e²_{t−1} + beta × sigma²_{t−1}
Điều kiện: omega > 0, alpha ≥ 0, beta ≥ 0, alpha + beta < 1 (biến động quay về mức dài hạn
V_L = omega / (1 − alpha − beta)). sigma²_1 khởi tạo bằng phương sai mẫu.
Dự báo k ngày tới: sigma²_{t+k} = V_L + (alpha + beta)^{k−1} × (sigma²_{t+1} − V_L).
Năm giao dịch = 252 phiên khi quy đổi biến động năm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


def log_returns_pct(close: pd.Series) -> np.ndarray:
    return (np.log(close.astype(float)).diff().dropna() * 100).to_numpy()


def _variance_path(e: np.ndarray, omega: float, alpha: float, beta: float, s0: float) -> np.ndarray:
    s2 = np.empty(len(e) + 1)        # phần tử cuối là dự báo cho phiên kế tiếp
    s2[0] = s0
    for t in range(len(e)):
        s2[t + 1] = omega + alpha * e[t] ** 2 + beta * s2[t]
    return s2


def _neg_loglik(params: np.ndarray, r: np.ndarray, s0: float) -> float:
    mu, omega, alpha, beta = params
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
        return 1e10
    e = r - mu
    s2 = _variance_path(e, omega, alpha, beta, s0)[:-1]
    if np.any(s2 <= 0):
        return 1e10
    return float(0.5 * np.sum(np.log(2 * np.pi) + np.log(s2) + e ** 2 / s2))


@dataclass
class GarchFit:
    mu: float           # %/ngày
    omega: float
    alpha: float
    beta: float
    sigma: np.ndarray   # độ lệch chuẩn có điều kiện từng phiên (%)
    resid: np.ndarray   # phần dư chuẩn hoá z_t
    next_sigma: float   # dự báo sigma phiên kế tiếp (%)
    loglik: float
    converged: bool

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_var(self) -> float:
        return self.omega / (1 - self.persistence)

    @property
    def long_run_vol_annual(self) -> float:
        return float(np.sqrt(self.long_run_var * TRADING_DAYS) / 100)

    @property
    def current_vol_annual(self) -> float:
        return float(self.next_sigma * np.sqrt(TRADING_DAYS) / 100)

    @property
    def half_life(self) -> float:
        """Số phiên để cú sốc biến động giảm một nửa."""
        p = self.persistence
        return float(np.log(0.5) / np.log(p)) if 0 < p < 1 else float("inf")

    def forecast_var(self, horizon: int) -> np.ndarray:
        """Phương sai dự báo cho phiên t+1 … t+horizon (%²)."""
        k = np.arange(horizon)
        return self.long_run_var + self.persistence ** k * (self.next_sigma ** 2 - self.long_run_var)


def fit_garch(r: np.ndarray) -> GarchFit:
    """Ước lượng GARCH(1,1) cho chuỗi lợi suất % (cần ít nhất 250 phiên)."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 250:
        raise ValueError(f"cần ít nhất 250 phiên để ước lượng GARCH, mới có {len(r)}")
    var = float(r.var())
    x0 = np.array([r.mean(), var * 0.05, 0.08, 0.87])
    bounds = [(-abs(r).max(), abs(r).max()), (1e-6 * var, 10 * var), (0.0, 0.5), (0.0, 0.9999)]
    cons = [{"type": "ineq", "fun": lambda p: 0.9999 - p[2] - p[3]}]
    res = minimize(_neg_loglik, x0, args=(r, var), method="SLSQP", bounds=bounds,
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
    mu, omega, alpha, beta = res.x
    e = r - mu
    s2 = _variance_path(e, omega, alpha, beta, var)
    sigma = np.sqrt(s2[:-1])
    return GarchFit(mu=float(mu), omega=float(omega), alpha=float(alpha), beta=float(beta),
                    sigma=sigma, resid=e / sigma, next_sigma=float(np.sqrt(s2[-1])),
                    loglik=-float(res.fun), converged=bool(res.success))
