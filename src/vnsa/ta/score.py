"""Điểm kỹ thuật (TA) 0–100. Trọng số các nhóm ở mục [ta_score] trong settings.

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

`ta_score_series` tính điểm cho MỌI phiên (chỉ dùng dữ liệu đến phiên đó, không nhìn trước),
dùng cho backtest; `ta_score` là điểm của phiên cuối.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

GROUP_NAMES = {"xu_huong": "Xu hướng", "dong_luong": "Động lượng", "suc_manh": "Sức mạnh (ADX)",
               "dong_tien": "Dòng tiền (KL)"}


@dataclass
class TAScore:
    total: float
    groups: dict[str, float]
    facts: dict[str, float]

    @property
    def rating(self) -> str:
        return rating(self.total)


def rating(total: float) -> str:
    if pd.isna(total):
        return "—"
    return "Tích cực" if total >= 65 else ("Trung tính" if total >= 45 else "Tiêu cực")


def _scale_series(x: pd.Series, bad: float, good: float) -> pd.Series:
    return ((x - bad) / (good - bad)).clip(0, 1) * 100


def rsi_points(r):
    """Điểm RSI; nhận một số hoặc một Series."""
    s = pd.Series(r, dtype=float) if not isinstance(r, pd.Series) else r.astype(float)
    pts = np.where(s <= 60, _scale_series(s, 30, 60),
                   np.where(s <= 70, 100.0, 100 - _scale_series(s, 70, 80) / 2))
    out = pd.Series(pts, index=s.index).where(s.notna())
    return out if isinstance(r, pd.Series) else float(out.iloc[0])


def _known(*cols: pd.Series) -> pd.Series:
    return pd.concat(cols, axis=1).notna().all(axis=1)


def group_scores(ind: pd.DataFrame) -> pd.DataFrame:
    """Điểm 4 nhóm cho từng phiên. ind = bảng đã qua add_indicators."""
    close, sma50, sma200 = ind["close"], ind["sma50"], ind["sma200"]
    sma50_then = sma50.shift(20)
    trend = 25.0 * ((close > sma50).astype(int) + (close > sma200).astype(int)
                    + (sma50 > sma200).astype(int) + (sma50 > sma50_then).astype(int))
    trend = trend.where(_known(sma50, sma200, sma50_then))

    hist, hist_prev = ind["macd_hist"], ind["macd_hist"].shift(1)
    macd_pts = (50.0 * (hist > 0) + 50.0 * (hist > hist_prev)).where(_known(hist, hist_prev))
    momentum = pd.concat([rsi_points(ind["rsi"]), macd_pts], axis=1).mean(axis=1)

    sign = np.where(ind["plus_di"] > ind["minus_di"], 1, -1)
    strength = (50 + sign * _scale_series(ind["adx"], 15, 35) / 2).where(ind["adx"].notna())

    up = close > close.shift(20)
    vol_up = ind["vol_ma20"] > ind["vol_ma50"]
    flow = pd.Series(np.select([up & vol_up, up & ~vol_up, ~up & ~vol_up], [100.0, 60.0, 40.0], 0.0),
                     index=ind.index).where(_known(close.shift(20), ind["vol_ma50"]))
    return pd.DataFrame({"xu_huong": trend, "dong_luong": momentum, "suc_manh": strength,
                         "dong_tien": flow})


def ta_score_series(ind: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Điểm TA tổng cho từng phiên; nhóm thiếu số thì chia lại trọng số cho nhóm còn lại."""
    g = group_scores(ind)
    w = pd.Series({k: float(weights[k]) for k in g.columns})
    num = (g * w).sum(axis=1, min_count=1)
    den = g.notna().mul(w).sum(axis=1)
    return (num / den.where(den > 0)).rename("ta_score")


def ta_score(ind: pd.DataFrame, weights: dict[str, float]) -> TAScore:
    """Điểm TA của phiên cuối cùng."""
    g = group_scores(ind).iloc[-1]
    total = ta_score_series(ind, weights).iloc[-1]
    last = ind.iloc[-1]
    facts = {k: float(last[k]) for k in ("close", "sma20", "sma50", "sma200", "rsi", "macd",
                                         "macd_signal", "macd_hist", "bb_pctb", "atr", "adx",
                                         "plus_di", "minus_di") if k in last}
    return TAScore(total=float(total), groups={k: float(v) for k, v in g.items()}, facts=facts)
