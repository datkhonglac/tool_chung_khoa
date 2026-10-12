"""BCTC năm, danh sách niêm yết (sàn, ngành) và giá dự phòng qua thư viện vnstock (nguồn VCI).

Giá: CafeF là nguồn chính, nhưng API của CafeF không trả dữ liệu cho một số mã HNX/UPCoM (đã thấy
khi chạy thật với SHS, ACV); khi đó dùng lịch sử giá VCI qua vnstock (có đủ 3 sàn).

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

from vnsa.data.cleaning import repair_ohlc
from vnsa.data.providers.base import DataProvider
from vnsa.data.providers.csv_provider import normalize_prices
from vnsa.fa.ratios import FIELDS, SECTOR_FIELDS

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
    "minority_interest": ("balance_sheet", ["minority_interests", "non_controlling_interests",
                                            "minority_interest"],
                          r"^minority interests?$|non-controlling interest"),
    "short_debt": ("balance_sheet", ["short_term_borrowings"], r"^short-term borrowings$"),
    "long_debt": ("balance_sheet", ["long_term_borrowings"], r"^long-term borrowings$"),
    "cfo": ("cash_flow", ["net_cash_inflows_outflows_from_operating_activities"],
            r"^net cash inflows/\(outflows\) from operating"),
    "pbt": ("income_statement", ["net_accounting_profit_loss_before_tax", "profit_before_tax"],
            r"profit/\(loss\) before tax$|^profit before tax"),
    "tax_expense": ("income_statement", ["corporate_income_tax_expenses"],
                    r"^corporate income tax expenses?$"),
    "interest_expense": ("income_statement", ["interest_expenses", "interest_expense"],
                         r"^interest expenses?$"),
    "capex": ("cash_flow", ["purchases_of_fixed_assets_and_other_long_term_assets"],
              r"^purchases of fixed assets"),
    "cash": ("balance_sheet", ["cash_and_cash_equivalents"], r"^cash and cash equivalents$"),
    "short_term_investments": ("balance_sheet", ["short_term_investments"],
                               r"^short-term investments$"),
    "paid_in_capital": ("balance_sheet", ["paid_in_capital", "owner_s_capital", "common_shares",
                                          "charter_capital"],
                        r"^paid-in capital$|^common shares|^owner'?s'? capital$|charter capital"),
    "current_assets": ("balance_sheet", ["current_assets", "short_term_assets"],
                       r"^current assets$|^short-term assets$"),
    "current_liabilities": ("balance_sheet", ["current_liabilities", "short_term_liabilities"],
                            r"^current liabilities$|^short-term liabilities$"),
    "retained_earnings": ("balance_sheet", ["undistributed_earnings", "retained_earnings",
                                            "undistributed_earnings_after_tax"],
                          r"^undistributed (?:earnings|profit)|^retained earnings"),
    "equity_issued": ("cash_flow", ["proceeds_from_issue_of_shares",
                                    "proceeds_from_issuing_shares_capital_contributions"],
                      (r"^proceeds from (?:the )?issu\w* (?:of )?(?:shares|stocks|equity)"
                       r"|capital contribution")),
}
# Khoản mục ngân hàng / công ty chứng khoán: chỉ khớp item_id (đã dò thật VCB, SSI qua
# scripts/probe_vnstock.py --items). Không có trong nguồn miễn phí: nợ xấu (NPL), CASA, CAR.
SECTOR_MAPPING: dict[str, tuple[str, list[str]]] = {
    "nii": ("income_statement", ["net_interest_income"]),
    "toi": ("income_statement", ["total_operating_income"]),
    "bank_opex": ("income_statement", ["general_and_admin_expenses"]),
    "credit_provision": ("income_statement", ["provision_for_credit_losses"]),
    "loans_gross": ("balance_sheet", ["loans_and_advances_to_customers"]),
    "loan_loss_reserve": ("balance_sheet",
                          ["less_provision_for_losses_on_loans_and_advances_to_customers"]),
    "customer_deposits": ("balance_sheet", ["deposits_from_customers"]),
    "sbv_balances": ("balance_sheet", ["balances_with_the_sbv"]),
    "interbank_placements": ("balance_sheet",
                             ["placements_with_and_loans_to_other_credit_institutions"]),
    "trading_securities": ("balance_sheet", ["trading_securities"]),
    "investment_securities": ("balance_sheet", ["investment_securities"]),
    "margin_loans": ("balance_sheet", ["loans"]),
    "fvtpl_assets": ("balance_sheet",
                     ["financial_assets_at_fair_value_through_profit_or_loss_fvtpl"]),
    "brokerage_revenue": ("income_statement", ["revenue_in_brokerage_services"]),
    "operating_revenue": ("income_statement", ["operating_sales"]),
}
# VCI ghi chi phí/khoản chi bằng số âm; lưu số dương cho dễ dùng.
POSITIVE = {"tax_expense", "interest_expense", "capex", "equity_issued", "bank_opex",
            "credit_provision", "loan_loss_reserve"}
STATEMENTS = ["income_statement", "balance_sheet", "cash_flow"]


def _year_columns(df: pd.DataFrame, quarterly: bool = False) -> list[str]:
    """Cột kỳ báo cáo: năm "2025" hoặc quý "2026-Q2" (định dạng VCI, đã dò thật)."""
    pattern = r"\d{4}-Q[1-4]" if quarterly else r"\d{4}"
    return [c for c in df.columns if re.fullmatch(pattern, str(c))]


def _pick(df: pd.DataFrame, ids: list[str], pattern: str | None,
          quarterly: bool = False) -> pd.Series | None:
    """Dòng khớp item_id (theo thứ tự ưu tiên) hoặc tên tiếng Anh. Ưu tiên dòng có số khác 0
    (VCI có các dòng cũ "trước 2015" toàn số 0), sau đó mới đến dòng có số liệu bất kỳ."""
    years = _year_columns(df, quarterly)
    candidates = [df[df["item_id"] == i] for i in ids]
    if pattern and "item_en" in df:
        en = df["item_en"].astype(str).str.strip()
        candidates.append(df[en.str.contains(pattern, case=False, regex=True)])
    rows = [pd.to_numeric(r[years], errors="coerce") for c in candidates for _, r in c.iterrows()]
    for values in rows:
        if values.fillna(0).ne(0).any():
            return values
    for values in rows:
        if values.notna().any():
            return values
    return None


def standardize(statements: dict[str, pd.DataFrame], symbol: str,
                quarterly: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Ba báo cáo dạng vnstock (dòng = khoản mục, cột = kỳ) -> bảng year (+ quarter) + FIELDS
    (đồng). Trả về (bảng, danh sách khoản mục không tìm thấy)."""
    periods = sorted({c for df in statements.values() for c in _year_columns(df, quarterly)})
    out = pd.DataFrame({"period": periods})
    out["year"] = out["period"].str[:4].astype(int)
    if quarterly:
        out["quarter"] = out["period"].str[-1].astype(int)
    missing = []
    for field in FIELDS:
        stmt, ids, pattern = MAPPING[field]
        values = _pick(statements[stmt], ids, pattern, quarterly) if stmt in statements else None
        if values is None:
            missing.append(field)
            out[field] = np.nan
            continue
        values.index = values.index.astype(str)
        out[field] = out["period"].map(values.abs() if field in POSITIVE else values)
    for field in SECTOR_FIELDS:                     # không báo thiếu (doanh nghiệp thường không có)
        stmt, ids = SECTOR_MAPPING[field]
        values = _pick(statements[stmt], ids, None, quarterly) if stmt in statements else None
        if values is None:
            out[field] = np.nan
            continue
        values.index = values.index.astype(str)
        out[field] = out["period"].map(values.abs() if field in POSITIVE else values)
    out = out.drop(columns="period")
    out.insert(0, "symbol", symbol.upper())
    return out, missing


def standardize_profile(overview: pd.DataFrame) -> dict:
    """Hồ sơ doanh nghiệp (Company.overview của VCI) -> các số dùng cho phân tích.
    Số cổ phiếu lưu hành = vốn hoá / giá hiện tại (VCI trả cột issue_share trùng tên, có cả số trước
    và sau khi phát hành thêm; vốn hoá luôn theo số cổ phiếu hiện tại)."""
    if overview is None or len(overview) == 0:
        return {}
    row = overview.loc[:, ~overview.columns.duplicated()].iloc[0]

    def num(key):
        v = pd.to_numeric(row.get(key), errors="coerce")
        return float(v) if not pd.isna(v) else float("nan")

    cap, price = num("market_cap"), num("current_price")
    shares = cap / price if cap > 0 and price > 0 else float("nan")
    if pd.isna(shares):
        dup = pd.to_numeric(overview.filter(like="issue_share").iloc[0], errors="coerce")
        shares = float(dup.max()) if dup.notna().any() else float("nan")
    return {
        "shares_out": shares,
        "foreign_pct": num("foreigner_percentage"),
        "foreign_max_pct": num("maximum_foreign_percentage"),
        "state_pct": num("state_percentage"),
        "free_float_pct": num("free_float_percentage"),
        "avg_value_1m": num("average_match_value1_month"),
        "analyst_target": num("target_price"),
        "analyst_rating": row.get("rating") if isinstance(row.get("rating"), str) else None,
    }


def standardize_dividends(events: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Sự kiện quyền (Company.events của VCI, category DIVIDEND) -> symbol, exright_date,
    cash_per_share (đồng/cp, cổ tức tiền), stock_ratio (tỷ lệ cổ tức/thưởng cổ phiếu), title."""
    cols = ["symbol", "exright_date", "cash_per_share", "stock_ratio", "title"]
    if events is None or len(events) == 0 or "category" not in events:
        return pd.DataFrame(columns=cols)
    ev = events[events["category"].astype(str).str.upper() == "DIVIDEND"].copy()
    date = pd.to_datetime(ev.get("exright_date"), errors="coerce")
    cash = pd.to_numeric(ev.get("value_per_share"), errors="coerce")
    ratio = pd.to_numeric(ev.get("exercise_ratio"), errors="coerce")
    out = pd.DataFrame({"symbol": symbol.upper(), "exright_date": date,
                        "cash_per_share": cash.where(cash > 0),
                        "stock_ratio": ratio.where(cash.isna() | (cash <= 0)),
                        "title": ev.get("event_title_vi")})
    out = out.dropna(subset=["exright_date"])
    # nguồn có thể trả cùng một đợt chia nhiều lần (thông báo, điều chỉnh): bỏ dòng trùng
    out = out.drop_duplicates(subset=["exright_date", "cash_per_share", "stock_ratio"])
    return out.sort_values("exright_date").reset_index(drop=True)


def _vnstock_company(symbol: str, method: str) -> pd.DataFrame:
    os.environ.setdefault("VNSTOCK_TELEMETRY", "off")
    from vnstock import Company

    return getattr(Company(source="VCI", symbol=symbol.upper()), method)()


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


# Sàn trong danh sách niêm yết VCI -> tên chuẩn của tool
EXCHANGE_NAMES = {"HSX": "HOSE", "HOSE": "HOSE", "HNX": "HNX", "UPCOM": "UPCOM"}
# Mã loại doanh nghiệp của VCI (com_type_code) -> mô hình phân tích
COM_TYPES = {"NH": "bank", "CK": "securities", "BH": "insurance"}
# Dự phòng khi thiếu com_type_code: tên ngành ICB cấp 2
INDUSTRY_MODELS = [(r"ngân hàng|^banks?$", "bank"),
                   (r"dịch vụ tài chính|financial services", "securities"),
                   (r"bảo hiểm|insurance", "insurance")]


def _col(df: pd.DataFrame, *names: str) -> pd.Series | None:
    for n in names:
        if n in df.columns:
            return df[n]
    return None


def standardize_listing(by_exchange: pd.DataFrame,
                        by_industry: pd.DataFrame | None = None) -> pd.DataFrame:
    """Hai bảng của vnstock Listing -> symbol, name, exchange, industry, model (chỉ cổ phiếu)."""
    df = by_exchange.copy()
    kind = _col(df, "type")
    if kind is not None:
        df = df[kind.astype(str).str.upper() == "STOCK"]
    out = pd.DataFrame({
        "symbol": df["symbol"].astype(str).str.upper().str.strip(),
        "exchange": _col(df, "exchange", "board", "comGroupCode").astype(str).str.upper()
                    .map(EXCHANGE_NAMES),
        "name": _col(df, "organ_name", "organ_short_name", "company_name"),
    }).dropna(subset=["exchange"])
    out["industry"] = None
    out["model"] = "general"
    if by_industry is not None and len(by_industry):
        ind = by_industry.assign(symbol=by_industry["symbol"].astype(str).str.upper())
        ind = ind.drop_duplicates("symbol").set_index("symbol")
        name2 = _col(ind, "icb_name2", "industry", "icb_name")
        if name2 is not None:
            out["industry"] = out["symbol"].map(name2)
            for pattern, model in INDUSTRY_MODELS:
                hit = out["industry"].fillna("").str.contains(pattern, case=False, regex=True)
                out.loc[hit, "model"] = model
        com = _col(ind, "com_type_code")
        if com is not None:
            mapped = out["symbol"].map(com.astype(str).str.upper().map(COM_TYPES))
            out["model"] = mapped.fillna(out["model"])
        if out["name"].isna().all():
            out["name"] = out["symbol"].map(_col(ind, "organ_name"))
    return out.reset_index(drop=True)


def _vnstock_listing() -> pd.DataFrame:
    os.environ.setdefault("VNSTOCK_TELEMETRY", "off")
    try:
        from vnstock import Listing
    except ImportError as e:
        raise RuntimeError(
            "Chưa cài vnstock. Chạy: pip install -U --extra-index-url "
            "https://vnstocks.com/api/simple vnstock vnai"
        ) from e
    ls = Listing(source="VCI")
    try:
        by_industry = ls.symbols_by_industries()
    except Exception:  # noqa: BLE001 - thiếu ngành vẫn dùng được sàn
        by_industry = None
    return standardize_listing(ls.symbols_by_exchange(), by_industry)


def _vnstock_history(symbol: str, start: str, end: str) -> pd.DataFrame:
    os.environ.setdefault("VNSTOCK_TELEMETRY", "off")
    try:
        from vnstock import Quote
    except ImportError as e:
        raise RuntimeError(
            "Chưa cài vnstock. Chạy: pip install -U --extra-index-url "
            "https://vnstocks.com/api/simple vnstock vnai"
        ) from e
    return Quote(symbol=symbol.upper(), source="VCI").history(start=start, end=end, interval="1D")


INDEX_SYMBOLS = {"VNINDEX", "VN30", "HNXINDEX", "HNX30", "UPCOMINDEX"}


class VnstockProvider(DataProvider):
    name = "vnstock-vci"

    def __init__(self, get_statements: Callable[[str], dict[str, pd.DataFrame]] = _vnstock_statements,
                 get_listing: Callable[[], pd.DataFrame] = _vnstock_listing,
                 get_history: Callable[[str, str, str], pd.DataFrame] = _vnstock_history,
                 get_quarterly: Callable[[str], dict[str, pd.DataFrame]] | None = None,
                 get_company: Callable[[str, str], pd.DataFrame] = _vnstock_company):
        self.get_statements = get_statements   # thay được trong test, không cần mạng
        self.get_listing = get_listing
        self.get_history = get_history
        self.get_quarterly = get_quarterly or (lambda s: _vnstock_statements(s, "quarter"))
        self.get_company = get_company
        self.repaired: list[str] = []
        self.missing: list[str] = []

    def symbols(self) -> list[str]:
        return self.listing()["symbol"].tolist()

    def listing(self) -> pd.DataFrame:
        """Mọi cổ phiếu trên HOSE, HNX, UPCoM: symbol, name, exchange, industry, model."""
        return self.get_listing()

    def ohlcv(self, symbol, start=None, end=None):
        """Giá ngày (VCI) theo PRICE_COLUMNS, đồng. VCI ghi cổ phiếu bằng nghìn đồng, chỉ số bằng
        điểm. Nguồn dự phòng khi CafeF không có mã (lệnh analyze tự chọn)."""
        from datetime import date, timedelta

        end = end or date.today()
        start = start or end - timedelta(days=365 * 5)
        raw = self.get_history(symbol, str(start), str(end))
        if raw is None or len(raw) == 0:
            raise ValueError(f"vnstock không có giá {symbol.upper()} từ {start} đến {end}")
        unit = "dong" if symbol.upper() in INDEX_SYMBOLS else "auto"
        df = normalize_prices(raw, symbol, price_unit=unit, adjust=False)
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        df, self.repaired = repair_ohlc(df)
        return df

    def fundamentals(self, symbol: str) -> pd.DataFrame:
        """BCTC năm chuẩn hoá: symbol, year + FIELDS. Khoản không có thì NaN (xem self.missing)."""
        df, self.missing = standardize(self.get_statements(symbol), symbol)
        return df

    def quarterly(self, symbol: str) -> pd.DataFrame:
        """BCTC quý chuẩn hoá: symbol, year, quarter + FIELDS (bản miễn phí: 4 quý gần nhất)."""
        df, self.missing = standardize(self.get_quarterly(symbol), symbol, quarterly=True)
        return df

    def profile(self, symbol: str) -> dict:
        """Số cổ phiếu lưu hành, sở hữu nước ngoài/nhà nước, free float, GTGD bình quân 1 tháng."""
        return standardize_profile(self.get_company(symbol, "overview"))

    def dividends(self, symbol: str) -> pd.DataFrame:
        return standardize_dividends(self.get_company(symbol, "events"), symbol)
