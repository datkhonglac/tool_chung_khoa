"""Vùng hỗ trợ / kháng cự từ các đỉnh, đáy cục bộ (swing high / swing low).

Cách làm:
1. Trong `lookback` phiên gần nhất, phiên i là đỉnh cục bộ nếu giá cao nhất của nó là cao nhất trong
   cửa sổ ±`window` phiên; đáy cục bộ tương tự với giá thấp nhất.
2. Gom các đỉnh/đáy có giá gần nhau (chênh không quá `tolerance`, ví dụ 2%) thành một vùng. Mức của
   vùng = trung bình các giá trong vùng; số lần chạm = số đỉnh/đáy trong vùng (càng nhiều càng mạnh).
3. Vùng nằm dưới giá hiện tại là hỗ trợ (gần nhất trước), trên giá hiện tại là kháng cự.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Level:
    price: float
    touches: int
    last_date: pd.Timestamp

    def __str__(self) -> str:
        return f"{self.price:,.0f} ({self.touches} lần chạm, gần nhất {self.last_date:%d/%m/%Y})"


def swing_points(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Các đỉnh/đáy cục bộ: cột date, price, kind ('high' | 'low')."""
    span = 2 * window + 1
    hi = df["high"] == df["high"].rolling(span, center=True, min_periods=span).max()
    lo = df["low"] == df["low"].rolling(span, center=True, min_periods=span).min()
    highs = pd.DataFrame({"date": df.loc[hi, "date"], "price": df.loc[hi, "high"], "kind": "high"})
    lows = pd.DataFrame({"date": df.loc[lo, "date"], "price": df.loc[lo, "low"], "kind": "low"})
    return pd.concat([highs, lows]).sort_values("date").reset_index(drop=True)


def cluster(points: pd.DataFrame, tolerance: float = 0.02) -> list[Level]:
    """Gom các mức giá gần nhau (so với mức trung bình của cụm đang gom) thành vùng."""
    levels: list[Level] = []
    group: list[tuple[float, pd.Timestamp]] = []

    def flush() -> None:
        if group:
            prices = [p for p, _ in group]
            levels.append(Level(sum(prices) / len(prices), len(group), max(d for _, d in group)))

    for price, day in sorted(zip(points["price"], points["date"], strict=True)):
        if group and price > (sum(p for p, _ in group) / len(group)) * (1 + tolerance):
            flush()
            group = []
        group.append((price, day))
    flush()
    return levels


def support_resistance(df: pd.DataFrame, window: int = 5, lookback: int = 250,
                       tolerance: float = 0.02) -> tuple[list[Level], list[Level]]:
    """(hỗ trợ gần nhất trước, kháng cự gần nhất trước) so với giá đóng cửa phiên cuối."""
    recent = df.sort_values("date").tail(lookback).reset_index(drop=True)
    close = float(recent["close"].iloc[-1])
    levels = cluster(swing_points(recent, window), tolerance)
    supports = sorted((lv for lv in levels if lv.price < close), key=lambda lv: -lv.price)
    resistances = sorted((lv for lv in levels if lv.price > close), key=lambda lv: lv.price)
    return supports, resistances
