"""Chỉ số riêng của ngân hàng và công ty chứng khoán, tính từ khoản mục BCTC (SECTOR_FIELDS).

Kỳ số liệu: 4 quý gần nhất (TTM) nếu có, ngược lại năm gần nhất. Khoản dòng chảy (thu nhập lãi
thuần, chi phí, dự phòng, doanh thu) = tổng 4 quý / số năm; khoản số dư dùng BÌNH QUÂN đầu và cuối
kỳ (quý đầu và quý cuối của cửa sổ TTM, hoặc hai năm gần nhất) khi tính trên tài sản, cuối kỳ khi
là tỷ lệ cơ cấu.

Ngân hàng (đơn vị: tỷ lệ, 0.035 = 3,5%):
- NIM (xấp xỉ) = thu nhập lãi thuần / tài sản sinh lãi bình quân; tài sản sinh lãi = tiền gửi NHNN
  + tiền gửi và cho vay TCTD khác + chứng khoán kinh doanh + chứng khoán đầu tư + cho vay khách hàng.
- CIR = chi phí hoạt động / tổng thu nhập hoạt động.
- Chi phí tín dụng = chi phí dự phòng rủi ro tín dụng / dư nợ cho vay bình quân.
- Dự phòng / dư nợ = số dư dự phòng cho vay / dư nợ cho vay (cuối kỳ).
- LDR (đơn giản) = dư nợ cho vay / tiền gửi khách hàng — KHÔNG phải công thức LDR của NHNN.
- Tăng trưởng tín dụng = dư nợ cuối năm gần nhất / năm trước − 1 (BCTC năm).
KHÔNG có trong nguồn miễn phí VCI (đã dò thật): nợ xấu (NPL), tỷ lệ bao phủ nợ xấu, CASA, CAR —
không tự ước lượng, báo rõ là thiếu.

Công ty chứng khoán:
- Cho vay ký quỹ / vốn chủ (pháp luật giới hạn tổng dư nợ ký quỹ ≤ 2 lần vốn chủ).
- Tự doanh FVTPL / vốn chủ (độ nhạy với giá cổ phiếu, trái phiếu).
- Tỷ trọng doanh thu môi giới.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BANK_MISSING = ["nợ xấu (NPL)", "bao phủ nợ xấu", "CASA", "CAR"]
EARNING_ASSETS = ["sbv_balances", "interbank_placements", "trading_securities",
                  "investment_securities", "loans_gross"]


@dataclass
class SectorMetrics:
    model: str                  # bank | securities
    basis: str                  # "TTM Q2/2026" | "năm 2025"
    values: dict[str, float] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)   # chỉ tiêu nguồn không có


def _window(fin: pd.DataFrame, quarterly: pd.DataFrame | None, need: str):
    """(nhãn kỳ, khoản dòng chảy, dòng cuối kỳ, dòng đầu kỳ). Ưu tiên 4 quý liên tiếp có `need`
    và mới hơn BCTC năm; ngược lại hai năm gần nhất."""
    if quarterly is not None and len(quarterly) >= 4 and need in quarterly:
        q = quarterly.sort_values(["year", "quarter"]).tail(4).reset_index(drop=True)
        idx = q["year"] * 4 + q["quarter"]
        last_year = int(fin["year"].max()) if not fin.empty else -1
        if ((idx.diff().dropna() == 1).all() and q[need].notna().all()
                and int(q["year"].iloc[-1]) * 4 + int(q["quarter"].iloc[-1]) > last_year * 4 + 4):
            flows = q.sum(numeric_only=True, min_count=4)
            return (f"TTM Q{int(q['quarter'].iloc[-1])}/{int(q['year'].iloc[-1])}", flows,
                    q.iloc[-1], q.iloc[0])
    if fin.empty or need not in fin:
        return None
    f = fin.sort_values("year").reset_index(drop=True)
    last = f.iloc[-1]
    first = f.iloc[-2] if len(f) >= 2 and f["year"].iloc[-2] == last["year"] - 1 else last
    return f"năm {int(last['year'])}", last, last, first


def _div(a: float, b: float) -> float:
    return float(a / b) if not pd.isna(a) and not pd.isna(b) and b > 0 else float("nan")


def _avg(end: pd.Series, start: pd.Series, col: str) -> float:
    vals = [v for v in (end.get(col), start.get(col)) if not pd.isna(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _earning_assets(row: pd.Series) -> float:
    if pd.isna(row.get("loans_gross")):
        return float("nan")
    return float(sum(0.0 if pd.isna(row.get(c)) else row[c] for c in EARNING_ASSETS))


def bank(fin: pd.DataFrame, quarterly: pd.DataFrame | None = None) -> SectorMetrics | None:
    w = _window(fin, quarterly, "nii")
    if w is None or pd.isna(w[1].get("nii")):
        return None
    basis, flows, end, start = w
    ea = [v for v in (_earning_assets(end), _earning_assets(start)) if not pd.isna(v)]
    ea = float(np.mean(ea)) if ea else float("nan")
    values = {
        "nim": _div(flows["nii"], ea),
        "cir": _div(flows.get("bank_opex"), flows.get("toi")),
        "credit_cost": _div(flows.get("credit_provision"), _avg(end, start, "loans_gross")),
        "llr": _div(end.get("loan_loss_reserve"), end.get("loans_gross")),
        "ldr": _div(end.get("loans_gross"), end.get("customer_deposits")),
    }
    f = fin.sort_values("year")
    if "loans_gross" in f and len(f) >= 2 and f["year"].iloc[-1] - f["year"].iloc[-2] == 1:
        values["loan_growth"] = _div(f["loans_gross"].iloc[-1], f["loans_gross"].iloc[-2]) - 1
    return SectorMetrics("bank", basis, values, list(BANK_MISSING))


def securities(fin: pd.DataFrame, quarterly: pd.DataFrame | None = None) -> SectorMetrics | None:
    w = _window(fin, quarterly, "operating_revenue")
    if w is None:
        return None
    basis, flows, end, _ = w
    equity = end.get("equity")
    values = {
        "margin_to_equity": _div(end.get("margin_loans"), equity),
        "fvtpl_to_equity": _div(end.get("fvtpl_assets"), equity),
        "brokerage_share": _div(flows.get("brokerage_revenue"), flows.get("operating_revenue")),
    }
    if all(pd.isna(v) for v in values.values()):
        return None
    return SectorMetrics("securities", basis, values, ["thị phần môi giới"])


def metrics(model: str, fin: pd.DataFrame,
            quarterly: pd.DataFrame | None = None) -> SectorMetrics | None:
    if model == "bank":
        return bank(fin, quarterly)
    if model == "securities":
        return securities(fin, quarterly)
    return None


LABELS = {
    "nim": "NIM (xấp xỉ)", "cir": "CIR", "credit_cost": "Chi phí tín dụng",
    "llr": "Dự phòng/dư nợ", "ldr": "Cho vay/tiền gửi KH", "loan_growth": "Tăng trưởng tín dụng",
    "margin_to_equity": "Ký quỹ/vốn chủ", "fvtpl_to_equity": "Tự doanh FVTPL/vốn chủ",
    "brokerage_share": "Tỷ trọng DT môi giới",
}
