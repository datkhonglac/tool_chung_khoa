"""Kiểm tra chất lượng bảng giá trước khi đưa vào kho.

Trả về danh sách vấn đề thay vì tự sửa: dữ liệu sai cần người xem lại,
nhất là các phiên giá nhảy vượt biên độ (thường do chưa điều chỉnh cổ tức/chia tách).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Issue:
    level: str  # "LOI" (phải sửa) | "CANH_BAO" (nên xem)
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.rule}: {self.detail}"


def _dates(rows: pd.DataFrame, limit: int = 5) -> str:
    shown = ", ".join(d.strftime("%d/%m/%Y") for d in rows["date"].head(limit))
    more = f" (+{len(rows) - limit} phiên khác)" if len(rows) > limit else ""
    return shown + more


def validate_prices(df: pd.DataFrame, band: float = 0.07, max_gap_days: int = 10) -> list[Issue]:
    """band: biên độ sàn (HOSE 0.07). max_gap_days: số ngày lịch tối đa giữa 2 phiên
    (Tết có thể nghỉ ~9 ngày, nên mặc định 10)."""
    issues: list[Issue] = []
    if df.empty:
        return [Issue("LOI", "rong", "Không có dòng dữ liệu nào")]

    bad = df[(df[["open", "high", "low", "close"]] <= 0).any(axis=1)]
    if len(bad):
        issues.append(Issue("LOI", "gia_khong_duong", f"{len(bad)} phiên: {_dates(bad)}"))

    hi_bad = df[df["high"] < df[["open", "close"]].max(axis=1)]
    lo_bad = df[df["low"] > df[["open", "close"]].min(axis=1)]
    if len(hi_bad):
        issues.append(Issue("LOI", "high_nho_hon_open_close", f"{len(hi_bad)} phiên: {_dates(hi_bad)}"))
    if len(lo_bad):
        issues.append(Issue("LOI", "low_lon_hon_open_close", f"{len(lo_bad)} phiên: {_dates(lo_bad)}"))

    dup = df[df["date"].duplicated(keep=False)]
    if len(dup):
        issues.append(Issue("LOI", "trung_ngay", f"{len(dup)} dòng: {_dates(dup)}"))

    if not df["date"].is_monotonic_increasing:
        issues.append(Issue("LOI", "chua_sap_xep", "Ngày không tăng dần"))

    gaps = df["date"].diff().dt.days
    big = df[gaps > max_gap_days]
    if len(big):
        issues.append(Issue("CANH_BAO", "thieu_phien", f"Khoảng trống > {max_gap_days} ngày trước: {_dates(big)}"))

    weekend = df[df["date"].dt.dayofweek >= 5]
    if len(weekend):
        issues.append(Issue("CANH_BAO", "ngay_cuoi_tuan", f"{len(weekend)} phiên: {_dates(weekend)}"))

    # Biến động vượt biên độ (+0,5% sai số làm tròn bước giá)
    ret = df["close"].pct_change()
    jumps = df[ret.abs() > band + 0.005]
    if len(jumps):
        issues.append(Issue(
            "CANH_BAO", "vuot_bien_do",
            f"{len(jumps)} phiên thay đổi > {band:.0%}: {_dates(jumps)}. "
            "Thường do ngày GDKHQ (cổ tức/chia tách) nếu giá chưa điều chỉnh, "
            "hoặc phiên đầu sau niêm yết/chuyển sàn.",
        ))

    zero_vol = df[df["volume"] <= 0]
    if len(zero_vol) > 0.05 * len(df):
        issues.append(Issue("CANH_BAO", "thanh_khoan_thap", f"{len(zero_vol)} phiên khối lượng = 0"))

    return issues


def summarize(df: pd.DataFrame) -> dict:
    return {
        "so_phien": len(df),
        "tu_ngay": df["date"].min().date().isoformat() if len(df) else None,
        "den_ngay": df["date"].max().date().isoformat() if len(df) else None,
        "gia_dong_cua_cuoi": float(df["close"].iloc[-1]) if len(df) else None,
    }
