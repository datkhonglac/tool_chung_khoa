"""Khuếch tán có bước nhảy (Merton jump-diffusion) trên nền GARCH — nâng cấp v4.

    dS/S = μ dt + σ dW + J dN      (N: quá trình Poisson cường độ λ, J: cỡ bước nhảy)

Ước lượng (đơn giản, kiểm chứng được — lệnh study-var):
1. σ_t = độ lệch chuẩn có điều kiện của GARCH(1,1) (risk/garch.py); z_t = lợi suất / σ_t.
2. Phiên có |z_t| > k (mặc định 3) là "bước nhảy": λ = tỷ lệ phiên nhảy, cỡ nhảy = các z_t đó (giữ
   nguyên phân phối thực nghiệm, không ép về chuẩn). Phần khuếch tán: độ lệch chuẩn s của z ở các
   phiên còn lại.
3. Lợi suất phiên tới (đơn vị z):  X = s·ε + J·1{có nhảy},  ε ~ N(0,1), P(có nhảy) = 1 − e^(−λ).
   Hàm phân phối: F(x) = e^(−λ)·Φ(x/s) + (1 − e^(−λ))·trung bình_j Φ((x − z_j)/s).
   VaR_q = −(e^{(μ + σ_{t+1}·x_q)/100} − 1) với F(x_q) = 1 − q, chặn ở biên độ sàn (HOSE 7%): lỗ một
   phiên không thể vượt biên độ.
4. Xác suất "phiên giảm sàn": P(lợi suất ≤ ln(1 − biên độ + 0,5%)) mỗi phiên; trong h phiên:
   1 − (1 − p)^h.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


@dataclass
class JumpFit:
    lam: float                   # xác suất nhảy mỗi phiên (λ)
    diffusion_sd: float          # độ lệch chuẩn của z ở phiên không nhảy
    jumps: np.ndarray = field(default_factory=lambda: np.array([]))   # cỡ nhảy (đơn vị z)
    k: float = 3.0
    n: int = 0

    @property
    def n_jumps(self) -> int:
        return len(self.jumps)

    @property
    def down_share(self) -> float:
        return float(np.mean(self.jumps < 0)) if len(self.jumps) else float("nan")


def fit_jumps(r_pct: np.ndarray, sigma_pct: np.ndarray, k: float = 3.0, mu: float = 0.0) -> JumpFit:
    """r_pct: lợi suất log %, sigma_pct: σ GARCH cùng độ dài."""
    r, s = np.asarray(r_pct, float), np.asarray(sigma_pct, float)
    ok = np.isfinite(r) & np.isfinite(s) & (s > 0)
    z = (r[ok] - mu) / s[ok]
    if len(z) < 50:
        raise ValueError("cần ít nhất 50 phiên để ước lượng bước nhảy")
    jump = np.abs(z) > k
    rest = z[~jump]
    return JumpFit(lam=float(jump.mean()), diffusion_sd=float(rest.std()) if len(rest) > 2 else 1.0,
                   jumps=z[jump], k=k, n=len(z))


def cdf(x: float | np.ndarray, fit: JumpFit) -> np.ndarray:
    """F(x) của lợi suất phiên tới theo đơn vị z (vector hoá theo x và các bước nhảy)."""
    x = np.atleast_1d(np.asarray(x, float))
    s = fit.diffusion_sd
    p_jump = 1 - np.exp(-fit.lam) if fit.n_jumps else 0.0
    base = norm.cdf(x / s)
    if not p_jump:
        return base
    mix = norm.cdf((x[:, None] - fit.jumps[None, :]) / s).mean(axis=1)
    return (1 - p_jump) * base + p_jump * mix


def z_quantile(p: float, fit: JumpFit) -> float:
    return float(brentq(lambda x: cdf(x, fit)[0] - p, -60.0, 60.0, xtol=1e-6))


def jump_var(fit: JumpFit, sigma_next_pct: float, q: float = 0.99, mu: float = 0.0,
             band: float | None = 0.07) -> float:
    """VaR 1 ngày (tỷ lệ, lỗ dương), chặn ở biên độ sàn."""
    x = z_quantile(1 - q, fit)
    loss = -(np.exp((mu + sigma_next_pct * x) / 100) - 1)
    return float(min(loss, band)) if band else float(loss)


def floor_probability(fit: JumpFit, sigma_next_pct: float, band: float = 0.07, days: int = 20,
                      mu: float = 0.0, margin: float = 0.005) -> tuple[float, float]:
    """(xác suất một phiên giảm gần sàn, xác suất có ít nhất một phiên như vậy trong `days` phiên).
    Giả định σ giữ nguyên trong kỳ — chỉ tham khảo."""
    level = np.log(1 - band + margin) * 100
    p = float(cdf((level - mu) / sigma_next_pct, fit)[0])
    return p, float(1 - (1 - p) ** days)
