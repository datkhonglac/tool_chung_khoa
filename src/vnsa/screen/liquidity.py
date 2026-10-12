"""Tầng 1 của sàng lọc: số liệu giá từng mã và lọc thanh khoản.

Chỉ dùng giá (đồng, đã điều chỉnh) và khối lượng khớp lệnh trong kho:
- GTGD bình quân 20 phiên = trung bình (giá khớp thật × khối lượng) 20 phiên gần nhất; giá thật là
  `close_raw` (chưa điều chỉnh cổ tức) nếu nguồn có, không thì `close`.
- Số phiên có giao dịch trong 365 ngày gần nhất; phiên cuối cũ hơn `max_stale_days` = ngừng giao dịch.
- Lợi nhuận 1/3/6 tháng (21/63/126 phiên, giá điều chỉnh) và sức mạnh tương đối so với VNINDEX cùng
  khoảng: (1 + r mã) / (1 + r chỉ số) − 1. Giá trên MA50/MA200 (1/0). Dùng ở tầng chấm ngành.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SESSIONS = {"ret_1m": 21, "ret_3m": 63, "ret_6m": 126}


def price_stats(px: pd.DataFrame, index_close: pd.Series | None, as_of: pd.Timestamp) -> dict:
    """Số liệu giá của một mã. px: các phiên (date, close, close_raw, volume), index_close: giá
    đóng cửa VNINDEX theo ngày. Thiếu phiên thì để NaN."""
    nan = float("nan")
    out = {"last_date": pd.NaT, "price": nan, "value_20d": nan, "sessions": 0,
           **{k: nan for k in SESSIONS}, "rs_3m": nan, "rs_6m": nan,
           "above_ma50": nan, "above_ma200": nan}
    if px is None or px.empty:
        return out
    px = px.sort_values("date")
    close = px["close"].to_numpy(float)
    raw = px["close_raw"].where(px["close_raw"] > 0, px["close"]).to_numpy(float)
    vol = px["volume"].fillna(0).to_numpy(float)
    dates = pd.to_datetime(px["date"]).reset_index(drop=True)
    out["last_date"] = dates.iloc[-1]
    out["price"] = float(raw[-1])
    out["value_20d"] = float(np.mean(raw[-20:] * vol[-20:]))
    out["sessions"] = int(((dates > as_of - pd.Timedelta(days=365)).to_numpy() & (vol > 0)).sum())
    for key, n in SESSIONS.items():
        if len(close) > n and close[-1 - n] > 0:
            out[key] = float(close[-1] / close[-1 - n] - 1)
    if index_close is not None and len(index_close):
        idx = index_close.sort_index()
        for key, n in (("rs_3m", 63), ("rs_6m", 126)):
            r = out[f"ret_{key[3:]}"]
            if np.isnan(r):
                continue
            i0, i1 = idx.asof(dates.iloc[-1 - n]), idx.asof(dates.iloc[-1])
            if pd.notna(i0) and pd.notna(i1) and i0 > 0:
                out[key] = float((1 + r) / (i1 / i0) - 1)
    for key, n in (("above_ma50", 50), ("above_ma200", 200)):
        if len(close) >= n:
            out[key] = float(close[-1] > close[-n:].mean())
    return out


def _bn(x: float) -> str:
    return f"{x / 1e9:,.1f}".replace(",", "_").replace(".", ",").replace("_", ".")


def liquidity_check(st: dict, cfg: dict, as_of: pd.Timestamp) -> str:
    """Lý do mã bị loại ở tầng 1 ("" = qua). Dạng "nhóm: chi tiết" để đếm theo nhóm."""
    if pd.isna(st["last_date"]):
        return "không có giá: nguồn không trả dữ liệu"
    stale = (as_of - st["last_date"]).days
    if stale > cfg.get("max_stale_days", 10):
        return f"ngừng giao dịch: phiên cuối {st['last_date']:%d/%m/%Y}"
    need = cfg.get("min_sessions", 200)
    if st["sessions"] < need:
        return f"ít phiên: {st['sessions']}/{need} phiên có giao dịch trong 1 năm"
    if st["price"] < cfg.get("min_price", 5000):
        return f"giá thấp: {st['price']:,.0f}đ < {cfg.get('min_price', 5000):,.0f}đ".replace(",", ".")
    floor = cfg.get("min_value_bn", 3.0) * 1e9
    if not st["value_20d"] >= floor:
        return f"thanh khoản thấp: GTGD 20 phiên {_bn(st['value_20d'])} tỷ < {_bn(floor)} tỷ"
    return ""


def reason_group(reason: str) -> str:
    return reason.split(":", 1)[0] if reason else ""
