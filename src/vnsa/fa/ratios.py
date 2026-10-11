"""Bộ chỉ số tài chính năm, tính từ bảng BCTC chuẩn hoá (mỗi dòng một năm).

Bảng đầu vào: cột `year` và các khoản mục trong FIELDS, đơn vị **đồng**. Khoản nào nguồn không
có thì để trống (NaN); chỉ số phụ thuộc vào khoản đó sẽ là NaN, không đoán.

Quy ước (ghi rõ để đối chiếu với số công bố):
- ROE = LNST của cổ đông công ty mẹ / vốn chủ sở hữu **bình quân** (đầu kỳ + cuối kỳ) / 2.
  Năm đầu tiên không có số đầu kỳ thì dùng vốn chủ cuối kỳ.
- ROA = LNST (toàn bộ) / tổng tài sản bình quân, cùng quy ước như trên.
- Biên lợi nhuận gộp = lợi nhuận gộp / doanh thu thuần; biên ròng = LNST / doanh thu thuần.
- D/E = (vay ngắn hạn + vay dài hạn) / vốn chủ sở hữu cuối kỳ (chỉ nợ vay, không gồm nợ phải trả
  khác). Nợ phải trả / vốn chủ để ở cột riêng.
- CFO/LNST = lưu chuyển tiền thuần từ hoạt động kinh doanh / LNST: > 1 là lợi nhuận "có tiền về".
- Tăng trưởng EPS = EPS năm nay / EPS năm trước − 1, chỉ tính khi EPS năm trước > 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Khoản mục chuẩn (đồng). Nguồn dữ liệu nào cũng phải đổi về các tên này.
FIELDS = [
    "revenue",            # doanh thu thuần
    "gross_profit",       # lợi nhuận gộp
    "net_income",         # LNST
    "net_income_parent",  # LNST của cổ đông công ty mẹ
    "eps",                # lãi cơ bản trên cổ phiếu (đồng/cp)
    "total_assets",       # tổng tài sản
    "total_liabilities",  # nợ phải trả
    "equity",             # vốn chủ sở hữu
    "short_debt",         # vay và nợ thuê tài chính ngắn hạn
    "long_debt",          # vay và nợ thuê tài chính dài hạn
    "cfo",                # lưu chuyển tiền thuần từ HĐKD
]

RATIO_COLUMNS = [
    "roe", "roa", "gross_margin", "net_margin", "debt_to_equity",
    "liabilities_to_equity", "cfo_to_net_income", "eps_growth",
]


def _div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Chia an toàn: mẫu bằng 0 hoặc thiếu -> NaN."""
    return a / b.where(b != 0)


def compute_ratios(fin: pd.DataFrame) -> pd.DataFrame:
    """Trả về bảng: year + RATIO_COLUMNS (tỷ lệ dạng số thập phân, 0.25 = 25%)."""
    df = fin.sort_values("year").reset_index(drop=True).copy()
    for col in FIELDS:
        if col not in df:
            df[col] = np.nan
    # Năm phải liên tiếp thì mới lấy được số đầu kỳ / năm trước
    gap = df["year"].diff() != 1

    def prev_ok(s: pd.Series) -> pd.Series:
        return s.shift(1).where(~gap)

    avg_equity = ((df["equity"] + prev_ok(df["equity"])) / 2).fillna(df["equity"])
    avg_assets = ((df["total_assets"] + prev_ok(df["total_assets"])) / 2).fillna(df["total_assets"])
    debt = df["short_debt"].fillna(0) + df["long_debt"].fillna(0)
    debt = debt.where(df["short_debt"].notna() | df["long_debt"].notna())
    eps_prev = prev_ok(df["eps"])

    out = pd.DataFrame({"year": df["year"].astype(int)})
    out["roe"] = _div(df["net_income_parent"], avg_equity)
    out["roa"] = _div(df["net_income"], avg_assets)
    out["gross_margin"] = _div(df["gross_profit"], df["revenue"])
    out["net_margin"] = _div(df["net_income"], df["revenue"])
    out["debt_to_equity"] = _div(debt, df["equity"])
    out["liabilities_to_equity"] = _div(df["total_liabilities"], df["equity"])
    out["cfo_to_net_income"] = _div(df["cfo"], df["net_income"])
    out["eps_growth"] = (df["eps"] / eps_prev.where(eps_prev > 0) - 1)
    return out


def format_ratios(ratios: pd.DataFrame) -> pd.DataFrame:
    """Bảng dễ đọc: hàng là chỉ số, cột là năm, đơn vị % (D/E và CFO/LNST để dạng lần)."""
    labels = {
        "roe": "ROE (%)", "roa": "ROA (%)", "gross_margin": "Biên gộp (%)",
        "net_margin": "Biên ròng (%)", "debt_to_equity": "Vay/VCSH (lần)",
        "liabilities_to_equity": "Nợ phải trả/VCSH (lần)", "cfo_to_net_income": "CFO/LNST (lần)",
        "eps_growth": "Tăng trưởng EPS (%)",
    }
    pct = {"roe", "roa", "gross_margin", "net_margin", "eps_growth"}
    t = ratios.set_index("year")[RATIO_COLUMNS].copy()
    for c in RATIO_COLUMNS:
        t[c] = (t[c] * 100).round(1) if c in pct else t[c].round(2)
    return t.rename(columns=labels).T
