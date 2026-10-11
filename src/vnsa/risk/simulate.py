"""Mô phỏng Monte Carlo giá tương lai và các thước đo rủi ro (VaR, CVaR).

Monte Carlo theo "mô phỏng lịch sử có lọc" (filtered historical simulation):
- Mỗi phiên mô phỏng: r = mu + sigma_t × z, z rút ngẫu nhiên (có hoàn lại) từ phần dư chuẩn hoá
  của GARCH — giữ được đuôi dày của dữ liệu thật thay vì giả định phân phối chuẩn.
- sigma_t cập nhật theo đúng phương trình GARCH(1,1), nên giai đoạn biến động cao kéo dài như thực tế.
- Lợi suất mỗi phiên bị chặn trong biên độ sàn (HOSE ±7%): giá không thể vượt trần/sàn trong ngày.
- Kết quả: các phân vị P10…P90 của giá sau h phiên, xác suất giá cao hơn hiện tại, và xác suất chạm
  mức cắt lỗ / mục tiêu (theo giá đóng cửa) trước khi hết h phiên.

VaR / CVaR (lỗ tính dương, mức tin cậy q, ví dụ 95%):
- VaR lịch sử 1 ngày = − phân vị (1 − q) của lợi suất ngày trong `window` phiên gần nhất.
- CVaR (Expected Shortfall) = − trung bình các lợi suất tệ hơn mức VaR.
- VaR GARCH 1 ngày = − (mu + sigma_{t+1} × z_{1−q}). Mặc định z là phân vị (1 − q) của phần dư
  chuẩn hoá thật (đuôi dày như dữ liệu); `empirical=False` dùng phân vị phân phối chuẩn (thường
  đánh giá thấp lỗ ở mức 99%).
- VaR Monte Carlo h ngày = − phân vị (1 − q) của (giá sau h phiên / giá hiện tại − 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from vnsa.risk.garch import GarchFit

PERCENTILES = (10, 25, 50, 75, 90)


@dataclass
class Simulation:
    start_price: float
    paths: np.ndarray        # (số đường, h + 1) giá đóng cửa, cột 0 là giá hiện tại

    def percentiles(self, day: int | None = None) -> dict[int, float]:
        final = self.paths[:, day if day is not None else -1]
        return {p: float(np.percentile(final, p)) for p in PERCENTILES}

    def prob_above(self, level: float | None = None, day: int | None = None) -> float:
        final = self.paths[:, day if day is not None else -1]
        return float(np.mean(final > (level if level is not None else self.start_price)))

    def var(self, q: float = 0.95, day: int | None = None) -> float:
        ret = self.paths[:, day if day is not None else -1] / self.start_price - 1
        return float(-np.percentile(ret, (1 - q) * 100))

    def first_touch(self, stop: float, target: float) -> dict[str, float]:
        """Xác suất chạm cắt lỗ trước, chạm mục tiêu trước, hoặc chưa chạm mức nào khi hết kỳ."""
        p = self.paths[:, 1:]
        n = p.shape[1]
        hit_stop = np.where((p <= stop).any(axis=1), (p <= stop).argmax(axis=1), n)
        hit_target = np.where((p >= target).any(axis=1), (p >= target).argmax(axis=1), n)
        return {"stop": float(np.mean(hit_stop < hit_target)),
                "target": float(np.mean(hit_target < hit_stop)),
                "neither": float(np.mean((hit_stop == n) & (hit_target == n)))}


def simulate(fit: GarchFit, start_price: float, horizon: int, n_paths: int = 10_000,
             band: float | None = 0.07, bootstrap: bool = True, seed: int | None = 42) -> Simulation:
    rng = np.random.default_rng(seed)
    resid = fit.resid[np.isfinite(fit.resid)]
    lo, hi = (np.log(1 - band) * 100, np.log(1 + band) * 100) if band else (-np.inf, np.inf)
    s2 = np.full(n_paths, fit.next_sigma ** 2)
    log_price = np.zeros((n_paths, horizon + 1))
    for t in range(1, horizon + 1):
        z = rng.choice(resid, n_paths) if bootstrap else rng.standard_normal(n_paths)
        e = np.sqrt(s2) * z
        r = np.clip(fit.mu + e, lo, hi)
        log_price[:, t] = log_price[:, t - 1] + r / 100
        s2 = fit.omega + fit.alpha * (r - fit.mu) ** 2 + fit.beta * s2
    return Simulation(start_price=start_price, paths=start_price * np.exp(log_price))


def historical_var(close: pd.Series, q: float = 0.95, window: int = 750) -> tuple[float, float]:
    """(VaR, CVaR) 1 ngày từ lợi suất ngày đơn giản trong `window` phiên gần nhất."""
    r = close.astype(float).pct_change().dropna().tail(window).to_numpy()
    cut = np.percentile(r, (1 - q) * 100)
    return float(-cut), float(-r[r <= cut].mean())


def garch_var(fit: GarchFit, q: float = 0.95, empirical: bool = True) -> float:
    """VaR 1 ngày (tỷ lệ, lỗ tính dương) theo sigma GARCH phiên kế tiếp."""
    resid = fit.resid[np.isfinite(fit.resid)]
    z = np.percentile(resid, (1 - q) * 100) if empirical else norm.ppf(1 - q)
    log_r = (fit.mu + fit.next_sigma * z) / 100
    return float(-(np.exp(log_r) - 1))
