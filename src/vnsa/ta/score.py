"""Điểm kỹ thuật (TA) 0–100 từ phiên gần nhất. Trọng số các nhóm ở mục [ta_score] trong settings.

Nhóm và cách chấm (mỗi nhóm 0–100):
- Xu hướng: 4 điều kiện, mỗi điều kiện đạt 25 điểm: giá > SMA50, giá > SMA200, SMA50 > SMA200,
  SMA50 đi lên so với 20 phiên trước.
- Động lượng: trung bình của (a) RSI: 0 điểm ở RSI ≤ 30, 100 điểm ở RSI 60–70, giảm dần về 50 điểm
  khi RSI lên 80 (quá mua); và (b) MACD: histogram > 0 được 50, histogram tăng so với phiên trước
  được thêm 50.
- Sức mạnh xu hướng (ADX): 50 ± 50 × mức ADX quy về 0–1 giữa 15 và 35; cộng nếu +DI > −DI, trừ
  nếu ngược lại. ADX thấp (đi ngang) cho điểm quanh 50.
- Dòng tiền (khối lượng): so giá với 20 phiên trước và KL trung bình 20 phiên với 50 phiên:
  tăng giá + KL tăng = 100, tăng giá + KL giảm = 60, giảm giá + KL giảm = 40, giảm giá + KL tăng = 0.
Xếp loại: ≥ 65 tích cực, 45–65 trung tính, < 45 tiêu cực.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vnsa.fa.scores import scale


@dataclass
class TAScore:
    total: float
    groups: dict[str, float]
    facts: dict[str, float]

    @property
    def rating(self) -> str:
        if pd.isna(self.total):
            return "—"
        return "Tích cực" if self.total >= 65 else ("Trung tính" if self.total >= 45 else "Tiêu cực")


def rsi_points(r: float) -> float:
    if pd.isna(r):
        return float("nan")
    if r <= 60:
        return scale(r, 30, 60)
    if r <= 70:
        return 100.0
    return 100 - scale(r, 70, 80) / 2


def ta_score(ind: pd.DataFrame, weights: dict[str, float]) -> TAScore:
    """ind = bảng đã qua add_indicators (cần sma50, sma200, rsi, macd_hist, adx, plus_di,
    minus_di, vol_ma20, vol_ma50)."""
    last, prev = ind.iloc[-1], ind.iloc[-2]
    close = last["close"]
    sma50_then = ind["sma50"].iloc[-21] if len(ind) > 21 else np.nan

    trend_checks = [close > last["sma50"], close > last["sma200"], last["sma50"] > last["sma200"],
                    last["sma50"] > sma50_then]
    trend_inputs = [last["sma50"], last["sma200"], sma50_then]
    trend = (25.0 * sum(bool(c) for c in trend_checks)
             if not any(pd.isna(x) for x in trend_inputs) else float("nan"))

    macd_pts = (50.0 * (last["macd_hist"] > 0) + 50.0 * (last["macd_hist"] > prev["macd_hist"])
                if not pd.isna(prev["macd_hist"]) else float("nan"))
    mom_parts = [x for x in (rsi_points(last["rsi"]), macd_pts) if not pd.isna(x)]
    momentum = float(np.mean(mom_parts)) if mom_parts else float("nan")

    if pd.isna(last["adx"]):
        strength = float("nan")
    else:
        sign = 1 if last["plus_di"] > last["minus_di"] else -1
        strength = 50 + sign * scale(last["adx"], 15, 35) / 2

    close_then = ind["close"].iloc[-21] if len(ind) > 21 else np.nan
    if pd.isna(close_then) or pd.isna(last["vol_ma50"]):
        flow = float("nan")
    else:
        up, vol_up = close > close_then, last["vol_ma20"] > last["vol_ma50"]
        flow = {(True, True): 100.0, (True, False): 60.0,
                (False, False): 40.0, (False, True): 0.0}[(bool(up), bool(vol_up))]

    groups = {"xu_huong": trend, "dong_luong": momentum, "suc_manh": strength, "dong_tien": flow}
    w = {g: weights[g] for g in groups if not pd.isna(groups[g])}
    total = sum(groups[g] * x for g, x in w.items()) / sum(w.values()) if w else float("nan")
    facts = {k: float(last[k]) for k in ("close", "sma20", "sma50", "sma200", "rsi", "macd",
                                         "macd_signal", "macd_hist", "bb_pctb", "atr", "adx",
                                         "plus_di", "minus_di") if k in last}
    return TAScore(total=total, groups=groups, facts=facts)
