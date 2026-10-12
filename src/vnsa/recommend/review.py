"""Chấm lại các khuyến nghị đã lưu (bảng signals) sau N phiên — thước đo thật của tool.

Với mỗi khuyến nghị (ngày D, mã, giá đóng cửa P) và mỗi kỳ h phiên (mục [signals].horizons):
- Lãi/lỗ = giá đóng cửa phiên D+h / P − 1; VNINDEX cùng kỳ; vượt trội = lãi/lỗ − VNINDEX.
- Chạm trước (chỉ khi có cắt lỗ và mục tiêu): duyệt từng phiên D+1..D+h, phiên nào giá thấp nhất ≤ cắt
  lỗ thì "cắt lỗ", giá cao nhất ≥ mục tiêu 1 thì "mục tiêu"; cùng một phiên chạm cả hai thì tính
  "cắt lỗ" (thận trọng); không chạm thì "chưa chạm".
- Khớp vùng mua: trong D+1..D+h có phiên giá thấp nhất ≤ đỉnh vùng mua không (lệnh LO đặt ở đỉnh vùng
  có khớp được không); nếu khớp, lãi/lỗ tính từ đỉnh vùng (cho thấy mua theo vùng tốt hơn hay kém
  hơn mua ngay giá đóng cửa).
Khuyến nghị chưa đủ h phiên sau đó thì chưa chấm. MUA/KHẢ QUAN tốt khi vượt trội bình quân > 0,
KÉM KHẢ QUAN/BÁN tốt khi < 0. Ít mẫu (dưới ~30) thì kết quả chỉ mang tính tham khảo.
Lớp vĩ mô (summarize_macro): chia theo điểm vĩ mô ngành lúc khuyến nghị (Thuận lợi / Trung tính /
Bất lợi) để xem vĩ mô có giúp phân biệt mã vượt trội không — điều kiện trước khi cho vĩ mô vào điểm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _first_touch(window: pd.DataFrame, stop: float, target: float) -> str:
    if pd.isna(stop) or pd.isna(target):
        return "—"
    for low, high in zip(window["low"], window["high"], strict=True):
        if low <= stop:
            return "cắt lỗ"
        if high >= target:
            return "mục tiêu"
    return "chưa chạm"


def evaluate(signals: pd.DataFrame, prices: dict[str, pd.DataFrame], index: pd.DataFrame,
             horizons: list[int]) -> pd.DataFrame:
    """Mỗi dòng = một khuyến nghị × một kỳ đã đủ phiên để chấm."""
    idx = index.set_index("date")["close"].sort_index() if len(index) else pd.Series(dtype=float)
    rows = []
    for s in signals.itertuples(index=False):
        px = prices.get(s.symbol)
        if px is None or px.empty:
            continue
        px = px.sort_values("date").reset_index(drop=True)
        pos = px["date"].searchsorted(pd.Timestamp(s.as_of), side="right") - 1
        if pos < 0:
            continue
        for h in horizons:
            if pos + h >= len(px):
                continue
            end = px.iloc[pos + h]
            window = px.iloc[pos + 1: pos + h + 1]
            ret = end["close"] / s.price - 1
            idx_ret = np.nan
            if len(idx):
                i0, i1 = idx.asof(px["date"].iloc[pos]), idx.asof(end["date"])
                idx_ret = i1 / i0 - 1 if i0 > 0 else np.nan
            filled = (bool((window["low"] <= s.buy_high).any())
                      if not pd.isna(s.buy_high) else False)
            rows.append({
                "as_of": s.as_of, "symbol": s.symbol, "action": s.action, "horizon": h,
                "macro_industry": getattr(s, "macro_industry", np.nan),
                "ret": ret, "index_ret": idx_ret, "excess": ret - idx_ret,
                "touch": _first_touch(window, s.stop, s.target),
                "zone_filled": filled,
                "ret_from_zone": end["close"] / s.buy_high - 1 if filled else np.nan,
            })
    return pd.DataFrame(rows)


def summarize(ev: pd.DataFrame) -> pd.DataFrame:
    """Theo kỳ × khuyến nghị: số mẫu, lãi/lỗ và vượt trội bình quân, tỷ lệ thắng, chạm mục tiêu
    trước, khớp vùng mua."""
    if ev.empty:
        return pd.DataFrame()
    out = []
    for (h, action), g in ev.groupby(["horizon", "action"], sort=True):
        touched = g[g["touch"].isin(["mục tiêu", "cắt lỗ"])]
        out.append({
            "horizon": h, "action": action, "n": len(g),
            "avg_ret": g["ret"].mean(), "avg_excess": g["excess"].mean(),
            "win_rate": (g["ret"] > 0).mean(),
            "target_first": ((touched["touch"] == "mục tiêu").mean() if len(touched)
                             else np.nan),
            "zone_fill": g["zone_filled"].mean(),
            "avg_ret_from_zone": g["ret_from_zone"].mean(),
        })
    return pd.DataFrame(out)


def macro_label(score: float, favorable: float = 0.3, unfavorable: float = -0.3) -> str:
    if pd.isna(score):
        return "chưa có điểm"
    return "Thuận lợi" if score >= favorable else "Bất lợi" if score <= unfavorable else "Trung tính"


def summarize_macro(ev: pd.DataFrame, favorable: float = 0.3,
                    unfavorable: float = -0.3) -> pd.DataFrame:
    """Lớp vĩ mô có ích không? Theo kỳ × nhóm vĩ mô ngành lúc khuyến nghị: số mẫu, vượt trội VNINDEX
    bình quân, tỷ lệ thắng. Có ích khi nhóm Thuận lợi vượt trội cao hơn rõ nhóm Bất lợi, đủ mẫu."""
    if ev.empty or "macro_industry" not in ev:
        return pd.DataFrame()
    ev = ev.assign(macro=[macro_label(x, favorable, unfavorable) for x in ev["macro_industry"]])
    out = [{"horizon": h, "macro": m, "n": len(g), "avg_excess": g["excess"].mean(),
            "win_rate": (g["ret"] > 0).mean()}
           for (h, m), g in ev.groupby(["horizon", "macro"], sort=True)]
    return pd.DataFrame(out)
