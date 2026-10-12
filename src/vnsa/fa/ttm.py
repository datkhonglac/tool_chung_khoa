"""Số liệu 12 tháng gần nhất (TTM) từ BCTC quý — thị trường VN định giá theo TTM, không theo năm
tài chính đã cũ 6–9 tháng.

Quy ước (đơn vị đồng):
- Khoản dòng chảy (doanh thu, lợi nhuận, dòng tiền, capex...) = TỔNG 4 quý liên tiếp gần nhất; thiếu
  một quý nào thì khoản đó là NaN.
- Khoản số dư (tài sản, nợ, vốn chủ, tiền...) = số cuối quý gần nhất.
- ROE TTM = LNST cty mẹ TTM / vốn chủ cty mẹ bình quân (đầu và cuối cửa sổ 4 quý: cuối quý đầu
  tiên và cuối quý gần nhất — bản miễn phí chỉ có 4 quý nên không có số đầu kỳ thật).
- EPS, BVPS chia cho số cổ phiếu lưu hành HIỆN TẠI (hồ sơ doanh nghiệp, đã trừ cổ phiếu quỹ và
  tính cổ phiếu thưởng mới nhất); thiếu thì dùng vốn góp / mệnh giá.
Chỉ dùng TTM khi nó mới hơn BCTC năm gần nhất (quý cuối sau năm tài chính đó).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FLOW = ["revenue", "gross_profit", "net_income", "net_income_parent", "cfo", "pbt",
        "tax_expense", "interest_expense", "capex", "equity_issued",
        "nii", "toi", "bank_opex", "credit_provision", "brokerage_revenue", "operating_revenue"]


def ttm(quarterly: pd.DataFrame) -> pd.Series | None:
    """Dòng TTM (FIELDS + year, quarter, equity_parent_start) hoặc None nếu chưa đủ 4 quý liên tiếp."""
    if quarterly is None or len(quarterly) < 4:
        return None
    q = quarterly.sort_values(["year", "quarter"]).tail(4).reset_index(drop=True)
    idx = q["year"] * 4 + q["quarter"]
    if not (idx.diff().dropna() == 1).all():
        return None
    row = q.iloc[-1].copy()
    for f in FLOW:
        if f in q:
            row[f] = q[f].sum(min_count=4)
    row["eps"] = np.nan                       # tính lại theo số cổ phiếu hiện tại
    first = q.iloc[0]
    mi = first.get("minority_interest")
    row["equity_parent_start"] = first["equity"] - (0 if pd.isna(mi) else mi)
    return row


@dataclass
class Current:
    """Số liệu "hiện tại" dùng để định giá: TTM nếu có và mới hơn BCTC năm, ngược lại năm gần nhất."""
    basis: str            # "TTM Q2/2026" | "năm 2025"
    net_income_parent: float
    equity_parent: float
    shares: float
    roe: float
    eps: float
    bvps: float
    row: pd.Series        # dòng BCTC (TTM hoặc năm) dùng cho các khoản khác (vay, tiền...)


def _parent_equity(row: pd.Series) -> float:
    mi = row.get("minority_interest")
    return float(row["equity"] - (0 if pd.isna(mi) else mi))


def current(fin: pd.DataFrame, quarterly: pd.DataFrame | None, shares_out: float = np.nan,
            par_value: float = 10_000) -> Current | None:
    """Số liệu hiện tại của doanh nghiệp (xem đầu file). None nếu không có BCTC nào."""
    annual = fin.sort_values("year").reset_index(drop=True) if fin is not None else pd.DataFrame()
    t = ttm(quarterly) if quarterly is not None else None
    use_ttm = t is not None and (annual.empty or int(t["year"]) > int(annual["year"].iloc[-1]))
    if use_ttm:
        row = t
        eq = _parent_equity(t)
        avg = (eq + t["equity_parent_start"]) / 2
        roe = t["net_income_parent"] / avg if avg > 0 else np.nan
        basis = f"TTM Q{int(t['quarter'])}/{int(t['year'])}"
    elif not annual.empty:
        row = annual.iloc[-1]
        eq = _parent_equity(row)
        prev = _parent_equity(annual.iloc[-2]) if len(annual) > 1 else eq
        avg = (eq + prev) / 2
        roe = row["net_income_parent"] / avg if avg > 0 else np.nan
        basis = f"năm {int(row['year'])}"
    else:
        return None
    shares = shares_out
    if pd.isna(shares) or shares <= 0:
        cap = row.get("paid_in_capital")
        shares = cap / par_value if not pd.isna(cap) and cap > 0 else np.nan
    ni = float(row["net_income_parent"])
    return Current(basis=basis, net_income_parent=ni, equity_parent=eq, shares=float(shares),
                   roe=float(roe), eps=ni / shares if shares > 0 else np.nan,
                   bvps=eq / shares if shares > 0 else np.nan, row=row)
