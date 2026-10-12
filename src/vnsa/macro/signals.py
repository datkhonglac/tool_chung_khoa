"""Đọc từng chuỗi vĩ mô: giá trị mới nhất, biến động và mức độ bất thường (z).

Cách đọc theo `kind` của chuỗi (config/macro_map.toml):
- change: biến động = % thay đổi sau `months` tháng            (giá hàng hoá, tỷ giá)
- diff  : biến động = chênh lệch điểm % sau `months` tháng       (lãi suất, lợi suất)
- ytd   : biến động = số luỹ kế từ đầu năm − cùng kỳ năm trước  (tăng trưởng tín dụng)
- level : biến động = mức hiện tại − `neutral`                   (tăng trưởng so cùng kỳ, PMI)

z = biến động hiện tại so với phân phối biến động của chính chuỗi đó trong `z_years` năm gần nhất
((x − trung bình) / độ lệch chuẩn), chặn trong ±`z_cap`. Nhờ vậy tỷ giá +3% và dầu +3% không bị coi
là ngang nhau. Chuỗi `level` (và chuỗi chưa đủ lịch sử nhưng có `scale`) dùng z = biến động / scale
theo ngưỡng cấu hình. Không đủ số thì z = NaN, không ước lượng.

Chuỗi ngày được gộp về tháng (giá trị phiên cuối tháng); kỳ cuối cùng có thể là tháng chưa hết.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vnsa.macro.catalog import Series

STALE_DAYS = {"D": 10, "M": 75, "Q": 150}   # số cũ hơn (tính từ cuối kỳ) -> đánh dấu "cũ"


@dataclass
class Reading:
    series: str
    name: str
    unit: str
    kind: str
    date: pd.Timestamp | None       # ngày của số mới nhất (chuỗi ngày) hoặc đầu kỳ (tháng/quý)
    period_end: pd.Timestamp | None
    value: float
    change: float                   # biến động theo kind (xem đầu file)
    z: float                        # trong ±z_cap; NaN nếu thiếu số / thiếu lịch sử
    basis: str                      # "z 5 năm" | "ngưỡng cấu hình" | "thiếu lịch sử" | "chưa có số"
    age_days: int | None
    stale: bool
    freq: str = "M"

    @property
    def ok(self) -> bool:
        return not np.isnan(self.z)


def _periodic(df: pd.DataFrame, freq: str) -> pd.Series:
    """Chuỗi trong kho -> giá trị theo kỳ (PeriodIndex tháng, hoặc quý với chuỗi quý)."""
    s = pd.Series(df["value"].to_numpy(float), index=pd.to_datetime(df["date"])).sort_index()
    p = "Q" if freq == "Q" else "M"
    return s.groupby(s.index.to_period(p)).last()


def _changes(s: pd.Series, spec: Series) -> pd.Series:
    step = max(1, spec.months // 3) if spec.freq == "Q" else spec.months
    if spec.kind == "change":
        prev = s.shift(step)
        return (s / prev - 1).where(prev > 0) * 100      # %, cùng đơn vị với điểm %
    if spec.kind == "diff":
        return s - s.shift(step)
    if spec.kind == "ytd":
        return s - s.shift(4 if spec.freq == "Q" else 12)
    return s - spec.neutral                               # level


def reading(df: pd.DataFrame, spec: Series, today: pd.Timestamp | None = None,
            z_years: int = 5, z_cap: float = 2.0, min_history: int = 24,
            stale_days: dict | None = None) -> Reading:
    """Đọc một chuỗi (df: các dòng của chuỗi trong bảng macro_series)."""
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    stale_days = {**STALE_DAYS, **(stale_days or {})}
    nan = float("nan")
    if df is None or df.empty:
        return Reading(spec.id, spec.name, spec.unit, spec.kind, None, None, nan, nan, nan,
                       "chưa có số", None, True, spec.freq)
    last_date = pd.Timestamp(df["date"].max())
    s = _periodic(df, spec.freq)
    period_end = (last_date if spec.freq == "D"
                  else s.index[-1].to_timestamp(how="end").normalize())
    age = int((today - period_end).days)
    ch = _changes(s, spec).dropna()
    value = float(s.iloc[-1])
    change = float(ch.iloc[-1]) if len(ch) and ch.index[-1] == s.index[-1] else nan
    z, basis = nan, "thiếu lịch sử"
    if spec.kind == "level":
        z, basis = change / spec.scale, "ngưỡng cấu hình"
    elif not np.isnan(change):
        per_year = 4 if spec.freq == "Q" else 12
        hist = ch.iloc[-z_years * per_year:]
        need = max(8, min_history // 3) if spec.freq == "Q" else min_history
        sd = float(hist.std())
        if len(hist) >= need and sd > 0:
            z, basis = (change - float(hist.mean())) / sd, f"z {z_years} năm"
        elif spec.scale:
            z, basis = change / spec.scale, "ngưỡng cấu hình"
    if not np.isnan(z):
        z = float(np.clip(z, -z_cap, z_cap))
    return Reading(spec.id, spec.name, spec.unit, spec.kind, last_date, period_end, value, change,
                   z, basis, age, age > stale_days.get(spec.freq, 75), spec.freq)


def fmt_period(r: Reading) -> str:
    """Kỳ của số mới nhất: 09/10/2026 (ngày) · T9/2026 (tháng) · Q3/2026 (quý)."""
    if r.date is None:
        return "—"
    if r.freq == "D":
        return f"{r.date:%d/%m/%Y}"
    if r.freq == "Q":
        return f"Q{r.date.quarter}/{r.date.year}"
    return f"T{r.date.month}/{r.date.year}"


def readings(data: pd.DataFrame, cat: dict[str, Series], today: pd.Timestamp | None = None,
             **kw) -> dict[str, Reading]:
    """Đọc mọi chuỗi trong danh mục từ bảng macro_series (dạng dài)."""
    groups = {k: g for k, g in data.groupby("series")} if not data.empty else {}
    return {sid: reading(groups.get(sid), spec, today, **kw) for sid, spec in cat.items()}


def fmt_value(r: Reading) -> str:
    if np.isnan(r.value):
        return "—"
    v = f"{r.value:,.2f}" if abs(r.value) < 1000 else f"{r.value:,.0f}"
    v = v.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{v}{'%' if r.unit == '%' else ''}"


def fmt_change(r: Reading) -> str:
    """Biến động kèm đơn vị: % với change; điểm % với diff/ytd; so ngưỡng với level."""
    if np.isnan(r.change):
        return "—"
    x = f"{r.change:+.1f}".replace(".", ",")
    if r.kind == "change":
        return f"{x}%"
    if r.kind == "level":
        return f"{x} so mốc"
    return f"{x} điểm %"
