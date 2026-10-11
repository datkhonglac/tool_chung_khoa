"""BCTC năm qua thư viện vnstock (nguồn VCI - Vietcap).

vnstock không có trên PyPI, cài riêng:
    pip install -U --extra-index-url https://vnstocks.com/api/simple vnstock vnai

Bản miễn phí (Cộng đồng) chỉ trả 4 kỳ gần nhất. Đã kiểm tra: nguồn KBS gắn nhãn năm bị đảo
ngược (cột "2025" chứa số năm 2022), nên dùng VCI.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable

import numpy as np
import pandas as pd

from vnsa.data.providers.base import DataProvider
from vnsa.fa.ratios import FIELDS

# Khoản mục chuẩn -> (báo cáo, các item_id có thể gặp, mẫu tên tiếng Anh dự phòng).
# Lấy dòng đầu tiên khớp và có số liệu.
MAPPING: dict[str, tuple[str, list[str], str]] = {
    "revenue": ("income_statement", ["net_sales", "revenue"], r"^net sales$|^net revenue"),
    "gross_profit": ("income_statement", ["gross_profit"], r"^gross profit$"),
    "net_income": ("income_statement", ["net_profit_loss_after_tax", "net_profit"],
                   r"^net profit.*after tax"),
    "net_income_parent": ("income_statement", ["attributable_to_parent_company"],
                          r"^attributable to parent"),
    "eps": ("income_statement", ["eps_basic_vnd", "eps_basic"], r"^eps basic"),
    "total_assets": ("balance_sheet", ["total_assets"], r"^total assets$"),
    "total_liabilities": ("balance_sheet", ["liabilities", "total_liabilities"], r"^liabilities$"),
    "equity": ("balance_sheet", ["owners_equity", "owner_s_equity", "equity"],
               r"^owner'?s'? equity$|^owners' equity$|^equity$"),
    "short_debt": ("balance_sheet", ["short_term_borrowings"], r"^short-term borrowings$"),
    "long_debt": ("balance_sheet", ["long_term_borrowings"], r"^long-term borrowings$"),
    "cfo": ("cash_flow", ["net_cash_inflows_outflows_from_operating_activities"],
            r"^net cash inflows/\(outflows\) from operating"),
}
STATEMENTS = ["income_statement", "balance_sheet", "cash_flow"]


def _year_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]


def _pick(df: pd.DataFrame, ids: list[str], pattern: str) -> pd.Series | None:
    """Dòng khớp item_id (theo thứ tự ưu tiên) hoặc tên tiếng Anh, có ít nhất một số liệu."""
    years = _year_columns(df)
    candidates = [df[df["item_id"] == i] for i in ids]
    if "item_en" in df:
        en = df["item_en"].astype(str).str.strip()
        candidates.append(df[en.str.contains(pattern, case=False, regex=True)])
    for rows in candidates:
        for _, row in rows.iterrows():
            values = pd.to_numeric(row[years], errors="coerce")
            if values.notna().any():
                return values
    return None


def standardize(statements: dict[str, pd.DataFrame], symbol: str) -> tuple[pd.DataFrame, list[str]]:
    """Ba báo cáo dạng vnstock (dòng = khoản mục, cột = năm) -> bảng year + FIELDS (đồng).

    Trả về (bảng, danh sách khoản mục không tìm thấy)."""
    years = sorted({int(y) for df in statements.values() for y in _year_columns(df)})
    out = pd.DataFrame({"year": years})
    missing = []
    for field in FIELDS:
        stmt, ids, pattern = MAPPING[field]
        values = _pick(statements[stmt], ids, pattern) if stmt in statements else None
        if values is None:
            missing.append(field)
            out[field] = np.nan
            continue
        values.index = values.index.astype(int)
        out[field] = out["year"].map(values)
    out.insert(0, "symbol", symbol.upper())
    return out, missing


def _vnstock_statements(symbol: str, period: str = "year") -> dict[str, pd.DataFrame]:
    os.environ.setdefault("VNSTOCK_TELEMETRY", "off")
    try:
        from vnstock import Finance
    except ImportError as e:
        raise RuntimeError(
            "Chưa cài vnstock. Chạy: pip install -U --extra-index-url "
            "https://vnstocks.com/api/simple vnstock vnai"
        ) from e
    fin = Finance(source="VCI", symbol=symbol.upper())
    return {name: getattr(fin, name)(period=period) for name in STATEMENTS}


class VnstockProvider(DataProvider):
    name = "vnstock-vci"

    def __init__(self, get_statements: Callable[[str], dict[str, pd.DataFrame]] = _vnstock_statements):
        self.get_statements = get_statements   # thay được trong test, không cần mạng
        self.missing: list[str] = []

    def symbols(self) -> list[str]:
        return []

    def ohlcv(self, symbol, start=None, end=None):
        raise NotImplementedError("Giá lấy bằng 'fetch' (CafeF); vnstock chỉ dùng cho BCTC")

    def fundamentals(self, symbol: str) -> pd.DataFrame:
        """BCTC năm chuẩn hoá: symbol, year + FIELDS. Khoản không có thì NaN (xem self.missing)."""
        df, self.missing = standardize(self.get_statements(symbol), symbol)
        return df
