"""Độ nhạy thực nghiệm: giá cổ phiếu có thật sự đi theo yếu tố vĩ mô như bảng ánh xạ giả định không?

Với mỗi yếu tố của ngành (config/macro_map.toml), hồi quy theo THÁNG trên lịch sử giá có trong kho:

    lợi nhuận tháng của mã = a + b_thị trường × lợi nhuận VNINDEX + b × biến động tháng của yếu tố

- Biến động tháng của yếu tố cùng cách đọc như macro/signals.py: % thay đổi (change), chênh lệch điểm %
  (diff, level, ytd) so với tháng trước.
- Có VNINDEX trong hồi quy để tách phần "cả thị trường cùng lên/xuống": b là ảnh hưởng RIÊNG của
  yếu tố. b quy về "1 độ lệch chuẩn biến động yếu tố" -> lợi nhuận tháng của mã (%), dễ so sánh.
- t = b / sai số chuẩn (OLS). Kết luận so với dấu cấu hình:
    |t| ≥ 2 cùng dấu  -> "đúng dấu"         |t| ≥ 2 ngược dấu -> "NGƯỢC dấu cấu hình"
    1 ≤ |t| < 2       -> "yếu"              |t| < 1           -> "chưa thấy liên hệ"
- Cần ≥ `min_months` tháng có cả giá và yếu tố (mặc định 36); ít hơn -> "thiếu lịch sử".
Hồi quy trên quá khứ, mẫu nhỏ (5–10 năm): chỉ để kiểm tra giả thuyết, không dùng để dự báo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vnsa.macro.catalog import Series


@dataclass
class Sensitivity:
    series: str
    name: str
    sign: int               # dấu giả định trong cấu hình
    beta_sd: float          # % lợi nhuận tháng của mã khi yếu tố biến động 1 độ lệch chuẩn
    t: float
    n: int
    verdict: str            # đúng dấu | NGƯỢC dấu cấu hình | yếu | chưa thấy liên hệ | thiếu lịch sử

    @property
    def conflict(self) -> bool:
        return self.verdict == "NGƯỢC dấu cấu hình"


def monthly_close(prices: pd.DataFrame) -> pd.Series:
    s = pd.Series(prices["close"].to_numpy(float), index=pd.to_datetime(prices["date"]))
    return s.groupby(s.index.to_period("M")).last()


def monthly_change(df: pd.DataFrame, spec: Series) -> pd.Series:
    """Biến động tháng của một chuỗi vĩ mô (PeriodIndex tháng)."""
    s = pd.Series(df["value"].to_numpy(float), index=pd.to_datetime(df["date"])).sort_index()
    m = s.groupby(s.index.to_period("M")).last()
    if spec.kind == "change":
        prev = m.shift(1)
        return ((m / prev - 1) * 100).where(prev > 0)
    return m - m.shift(1)


def ols_last(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """(hệ số, t) của cột cuối cùng trong x (x đã có cột hằng số)."""
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    dof = len(y) - x.shape[1]
    if dof <= 0:
        return float("nan"), float("nan")
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(x.T @ x)
    se = float(np.sqrt(cov[-1, -1]))
    return float(coef[-1]), float(coef[-1] / se) if se > 0 else float("nan")


def verdict(sign: int, beta: float, t: float) -> str:
    if np.isnan(t):
        return "thiếu lịch sử"
    if abs(t) >= 2:
        return "đúng dấu" if np.sign(beta) == np.sign(sign) else "NGƯỢC dấu cấu hình"
    return "yếu" if abs(t) >= 1 else "chưa thấy liên hệ"


def sensitivity(prices: pd.DataFrame, index: pd.DataFrame | None, drivers: list[dict],
                data: pd.DataFrame, cat: dict[str, Series], min_months: int = 36) -> list[Sensitivity]:
    """Độ nhạy của mã với từng yếu tố trong `drivers` ({series, sign}); data = bảng macro_series."""
    stock = monthly_close(prices)
    r = (stock / stock.shift(1) - 1) * 100
    mkt = None
    if index is not None and len(index):
        im = monthly_close(index)
        mkt = (im / im.shift(1) - 1) * 100
    groups = {k: g for k, g in data.groupby("series")} if not data.empty else {}
    out = []
    for d in drivers:
        sid, sign = d["series"], int(np.sign(d.get("sign", 1))) or 1
        spec = cat.get(sid)
        name = spec.name if spec else sid
        g = groups.get(sid)
        if spec is None or g is None or g.empty:
            out.append(Sensitivity(sid, name, sign, np.nan, np.nan, 0, "thiếu lịch sử"))
            continue
        x = monthly_change(g, spec)
        cols = {"r": r, "x": x} if mkt is None else {"r": r, "m": mkt, "x": x}
        df = pd.concat(cols, axis=1).dropna()
        df = df[np.isfinite(df).all(axis=1)]
        if len(df) < min_months or df["x"].std() == 0:
            out.append(Sensitivity(sid, name, sign, np.nan, np.nan, len(df), "thiếu lịch sử"))
            continue
        sd = float(df["x"].std())
        xs = df["x"] / sd
        design = np.column_stack([np.ones(len(df)), *([df["m"].to_numpy()] if mkt is not None
                                                      else []), xs.to_numpy()])
        beta, t = ols_last(df["r"].to_numpy(), design)
        out.append(Sensitivity(sid, name, sign, beta, t, len(df), verdict(sign, beta, t)))
    return out
