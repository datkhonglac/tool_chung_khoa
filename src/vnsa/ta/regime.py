"""Trạng thái thị trường chung theo VNINDEX — phần lớn cổ phiếu VN đi cùng thị trường, nên khi chỉ số
đang xu hướng giảm thì cần biên lợi nhuận lớn hơn và vị thế nhỏ hơn (tham số ở mục [regime]).

- "tăng": VNINDEX trên MA200 và MA50 đi lên (so với `slope_sessions` phiên trước).
- "giảm": VNINDEX dưới MA200 và MA50 đi xuống.
- "trung tính": các trường hợp còn lại (hoặc chưa đủ dữ liệu).
`regime_series` tính cho mọi phiên (chỉ dùng dữ liệu đến phiên đó) để backtest.
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
        if pd.isna(self.sma_long):
            return f"VNINDEX {self.state} (chưa đủ dữ liệu MA200)"
        return (f"VNINDEX {self.state} ({self.close:,.0f} so với MA200 {self.sma_long:,.0f}; "
                f"MA50 {self.slope:+.1%} trong 20 phiên)")


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
    return state


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
