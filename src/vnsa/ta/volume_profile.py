"""Hồ sơ khối lượng theo giá — Volume Profile (nâng cấp v4).

Nguồn miễn phí chỉ có giá NGÀY (không có khớp lệnh từng giá trong phiên), nên khối lượng mỗi phiên
được rải ĐỀU trên khoảng [thấp nhất, cao nhất] của phiên đó (giả định phổ biến khi không có dữ liệu
tick); phiên không biên độ (cao = thấp) dồn hết vào ô chứa giá đóng cửa.

  POC  (Point of Control)  = mức giá có khối lượng lớn nhất:  argmax_p V(p)
  VA   (Value Area)        = khoảng liên tục quanh POC chứa `value_area` (70%) tổng khối lượng: bắt
                             đầu từ ô POC, mỗi bước nới sang phía (trên/dưới) có khối lượng lớn hơn.

Giá dùng là giá ĐÃ ĐIỀU CHỈNH (như mọi chỉ báo khác). "Mất cân bằng lệnh" (order flow imbalance) trong
blueprint cần dữ liệu khớp lệnh mua/bán chủ động từng lệnh — nguồn miễn phí không có lịch sử nên
không làm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULTS = {"sessions": 120, "bins": 50, "value_area": 0.70}


@dataclass
class Profile:
    poc: float
    va_low: float
    va_high: float
    centers: np.ndarray
    volume: np.ndarray            # khối lượng theo ô giá
    sessions: int

    def position(self, price: float) -> str:
        if price > self.va_high:
            return "trên vùng giá trị"
        if price < self.va_low:
            return "dưới vùng giá trị"
        return "trong vùng giá trị"


def distribute(low: np.ndarray, high: np.ndarray, close: np.ndarray, volume: np.ndarray,
               edges: np.ndarray) -> np.ndarray:
    """Khối lượng theo ô giá: mỗi phiên rải đều trên [low, high] (ma trận phiên × ô, không vòng lặp)."""
    lo, hi = edges[:-1][None, :], edges[1:][None, :]
    l, h = low[:, None], high[:, None]
    span = h - l
    overlap = np.clip(np.minimum(hi, h) - np.maximum(lo, l), 0.0, None)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(span > 0, overlap / np.where(span > 0, span, 1.0), 0.0)
    flat = span[:, 0] <= 0                     # phiên không biên độ: cả khối lượng vào ô của giá đóng cửa
    if flat.any():
        idx = np.clip(np.searchsorted(edges, close[flat], side="right") - 1, 0, len(edges) - 2)
        share[flat] = 0.0
        share[np.flatnonzero(flat), idx] = 1.0
    return (share * volume[:, None]).sum(axis=0)


def value_area(volume: np.ndarray, poc_idx: int, share: float = 0.70) -> tuple[int, int]:
    """Chỉ số ô [a, b] của vùng giá trị: nới từ POC sang phía có khối lượng lớn hơn tới khi đủ `share`."""
    total = volume.sum()
    a = b = poc_idx
    got = volume[poc_idx]
    n = len(volume)
    while got < share * total and (a > 0 or b < n - 1):
        up = volume[b + 1] if b < n - 1 else -1.0
        down = volume[a - 1] if a > 0 else -1.0
        if up >= down:
            b += 1
            got += up
        else:
            a -= 1
            got += down
    return a, b


def volume_profile(df: pd.DataFrame, sessions: int = 120, bins: int = 50,
                   share: float = 0.70) -> Profile | None:
    """df: giá ngày có high, low, close, volume. None nếu thiếu dữ liệu."""
    d = df.tail(sessions).dropna(subset=["high", "low", "close", "volume"])
    if len(d) < 20:
        return None
    low, high = d["low"].to_numpy(float), d["high"].to_numpy(float)
    close, vol = d["close"].to_numpy(float), d["volume"].to_numpy(float)
    lo, hi = float(low.min()), float(high.max())
    if not np.isfinite(lo) or hi <= lo or vol.sum() <= 0:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    v = distribute(low, high, close, vol, edges)
    centers = (edges[:-1] + edges[1:]) / 2
    poc = int(np.argmax(v))
    a, b = value_area(v, poc, share)
    return Profile(float(centers[poc]), float(edges[a]), float(edges[b + 1]), centers, v, len(d))


def describe(p: Profile | None, price: float) -> str:
    if p is None:
        return ""
    fmt = lambda x: f"{x:,.0f}".replace(",", ".")  # noqa: E731
    return (f"Hồ sơ khối lượng {p.sessions} phiên: POC {fmt(p.poc)}đ · vùng giá trị 70% "
            f"{fmt(p.va_low)}–{fmt(p.va_high)}đ · giá {p.position(price)} (tham khảo)")
