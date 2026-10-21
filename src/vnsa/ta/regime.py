"""Trạng thái thị trường chung theo VNINDEX — phần lớn cổ phiếu VN đi cùng thị trường, nên khi chỉ số
đang xu hướng giảm thì cần biên lợi nhuận lớn hơn và vị thế nhỏ hơn (tham số ở mục [regime]).

- "tăng": VNINDEX trên MA200 và MA50 đi lên (so với `slope_sessions` phiên trước).
- "giảm": VNINDEX dưới MA200 và MA50 đi xuống.
- "trung tính": các trường hợp còn lại (hoặc chưa đủ dữ liệu).
`regime_series` tính cho mọi phiên (chỉ dùng dữ liệu đến phiên đó) để backtest.

Cảnh báo sớm (nâng cấp v2.0, bước 3.1 — TẮT mặc định, `[regime] early = true` để bật sau khi đã
kiểm định): chuyển sang "giảm" ngay khi một trong hai điều kiện, không chờ thủng MA200:
  1. Phiên phân phối: trong 20 phiên gần nhất có ≥ 5 phiên VNINDEX giảm > 0,5% với khối lượng lớn
     hơn trung bình 20 phiên, đồng thời đóng cửa dưới MA50.
  2. Biến động tăng vọt: ATR14 / giá vượt phân vị 85% của 250 phiên gần nhất, đóng cửa dưới MA20.
Ngưỡng là của đề xuất, CHƯA kiểm định với VNINDEX — bật lên sẽ cắt khối lượng và nâng ngưỡng xếp
hạng thường xuyên hơn.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Regime:
    state: str              # tăng | trung tính | giảm
    close: float
    sma_long: float
    sma_short: float
    slope: float            # MA50 hiện tại / MA50 slope_sessions phiên trước − 1

    def __str__(self) -> str:
        from vnsa.report.card import vn_num, vn_pct

        if pd.isna(self.sma_long):
            return f"VNINDEX {self.state} (chưa đủ dữ liệu MA200)"
        return (f"VNINDEX {self.state} ({vn_num(self.close)} so với MA200 {vn_num(self.sma_long)}; "
                f"MA50 {vn_pct(self.slope, 1, sign=True)} trong 20 phiên)")


def regime_series(index: pd.DataFrame, cfg: dict) -> pd.Series:
    """Trạng thái từng phiên (index = date)."""
    if index is None or index.empty:
        return pd.Series(dtype=object)
    s = index.sort_values("date").set_index("date")["close"]
    long_ = s.rolling(cfg.get("sma_long", 200)).mean()
    short = s.rolling(cfg.get("sma_short", 50)).mean()
    slope = short / short.shift(cfg.get("slope_sessions", 20)) - 1
    state = pd.Series("trung tính", index=s.index)
    state[(s > long_) & (slope > 0)] = "tăng"
    state[(s < long_) & (slope < 0)] = "giảm"
    if cfg.get("early", False):
        state[early_warning(index, cfg).reindex(s.index, fill_value=False)] = "giảm"
    return state


def early_warning(index: pd.DataFrame, cfg: dict) -> pd.Series:
    """True ở các phiên có cảnh báo sớm thị trường giảm (xem docstring module)."""
    d = index.sort_values("date").set_index("date")
    c = d["close"].astype(float)
    sma50, sma20 = c.rolling(50).mean(), c.rolling(20).mean()
    out = pd.Series(False, index=d.index)
    if "volume" in d and d["volume"].fillna(0).gt(0).any():
        v = d["volume"].astype(float)
        dist = (c.pct_change() < -cfg.get("dist_drop", 0.005)) & (v > v.rolling(20).mean())
        out |= (dist.rolling(20).sum() >= cfg.get("dist_days", 5)) & (c < sma50)
    if {"high", "low"} <= set(d.columns):
        prev = c.shift(1)
        tr = pd.concat([d["high"] - d["low"], (d["high"] - prev).abs(), (d["low"] - prev).abs()],
                       axis=1).max(axis=1)
        atr_pct = tr.ewm(alpha=1 / 14, adjust=False).mean() / c
        cut = atr_pct.rolling(250).quantile(cfg.get("vol_quantile", 0.85))
        out |= (atr_pct > cut) & (c < sma20)
    return out.fillna(False)


def regime(index: pd.DataFrame, cfg: dict) -> Regime:
    nan = float("nan")
    if index is None or index.empty:
        return Regime("trung tính", nan, nan, nan, nan)
    s = index.sort_values("date")["close"].reset_index(drop=True)
    long_ = s.rolling(cfg.get("sma_long", 200)).mean().iloc[-1]
    short_series = s.rolling(cfg.get("sma_short", 50)).mean()
    n = cfg.get("slope_sessions", 20)
    slope = (short_series.iloc[-1] / short_series.iloc[-1 - n] - 1) if len(s) > n else nan
    state = regime_series(index, cfg).iloc[-1]
    return Regime(state, float(s.iloc[-1]), float(long_), float(short_series.iloc[-1]),
                  float(slope))
