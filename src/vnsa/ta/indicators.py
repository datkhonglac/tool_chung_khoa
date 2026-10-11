"""Chỉ báo kỹ thuật trên giá ngày (đồng, đã điều chỉnh). Đầu vào: bảng có date, open, high,
low, close, volume, sắp xếp theo ngày.

Công thức (để đối chiếu với phần mềm giao dịch):
- SMA n: trung bình cộng n phiên. EMA n: hệ số 2/(n+1), bắt đầu từ phiên đầu tiên.
- RSI 14 (Wilder): trung bình tăng / trung bình giảm làm mượt hệ số 1/n. RSI = 100 − 100/(1+RS).
- MACD 12-26-9: EMA12 − EMA26; đường tín hiệu = EMA9 của MACD; histogram = MACD − tín hiệu.
- Bollinger 20, 2: SMA20 ± 2 độ lệch chuẩn (tổng thể, ddof=0); %B = (giá − dải dưới)/(độ rộng dải).
- True range = max(cao − thấp, |cao − đóng cửa hôm trước|, |thấp − đóng cửa hôm trước|).
  ATR 14 = TR làm mượt Wilder (1/n).
- ADX 14 (Wilder): +DM/−DM, +DI/−DI = 100 × DM làm mượt / ATR, DX = 100 × |+DI − −DI| / (+DI + −DI),
  ADX = DX làm mượt. ADX > 25: có xu hướng rõ; < 20: đi ngang.
- OBV: cộng khối lượng phiên tăng, trừ khối lượng phiên giảm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    diff = close.diff()
    gain, loss = wilder(diff.clip(lower=0), n), wilder(-diff.clip(upper=0), n)
    rs = gain / loss.where(loss != 0)
    out = 100 - 100 / (1 + rs)
    return out.where(loss != 0, 100.0).where(gain.notna())   # không có phiên giảm -> 100


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    width = (upper - lower).where(upper != lower)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
                         "bb_pctb": (close - lower) / width})


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(),
                      (df["low"] - prev).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return wilder(true_range(df), n)


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up, down = df["high"].diff(), -df["low"].diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    tr = wilder(true_range(df), n)
    plus_di = 100 * wilder(plus_dm, n) / tr
    minus_di = 100 * wilder(minus_dm, n) / tr
    total = (plus_di + minus_di).where(lambda x: x != 0)
    dx = 100 * (plus_di - minus_di).abs() / total
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": wilder(dx, n)})


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def add_indicators(prices: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Thêm mọi chỉ báo vào bảng giá. cfg = mục [ta] trong settings (độ dài các chỉ báo)."""
    c = {"sma": [20, 50, 200], "rsi": 14, "macd": [12, 26, 9], "bollinger": [20, 2.0],
         "atr": 14, "adx": 14, "volume_ma": [20, 50], **(cfg or {})}
    df = prices.sort_values("date").reset_index(drop=True).copy()
    for n in c["sma"]:
        df[f"sma{n}"] = sma(df["close"], n)
    df["rsi"] = rsi(df["close"], c["rsi"])
    df = df.join(macd(df["close"], *c["macd"]))
    df = df.join(bollinger(df["close"], int(c["bollinger"][0]), float(c["bollinger"][1])))
    df["atr"] = atr(df, c["atr"])
    df = df.join(adx(df, c["adx"]))
    df["obv"] = obv(df)
    for n in c["volume_ma"]:
        df[f"vol_ma{n}"] = sma(df["volume"].astype(float), n)
    return df
