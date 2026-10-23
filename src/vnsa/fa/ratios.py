"""Bộ chỉ số tài chính năm, tính từ bảng BCTC chuẩn hoá (mỗi dòng một năm).

Bảng đầu vào: cột `year` và các khoản mục trong FIELDS, đơn vị **đồng**. Khoản nào nguồn không
có thì để trống (NaN); chỉ số phụ thuộc vào khoản đó sẽ là NaN, không đoán.

Quy ước (ghi rõ để đối chiếu với số công bố):
- ROE = LNST của cổ đông công ty mẹ / vốn chủ của cổ đông công ty mẹ **bình quân** (đầu kỳ +
  cuối kỳ) / 2. Vốn chủ công ty mẹ = vốn chủ sở hữu − lợi ích cổ đông không kiểm soát (theo VAS,
  khoản này nằm trong vốn chủ). Thiếu số cổ đông thiểu số thì dùng toàn bộ vốn chủ.
  Năm đầu tiên không có số đầu kỳ thì dùng vốn chủ cuối kỳ.
- ROA = LNST (toàn bộ) / tổng tài sản bình quân, cùng quy ước như trên.
- Biên lợi nhuận gộp = lợi nhuận gộp / doanh thu thuần; biên ròng = LNST / doanh thu thuần.
- D/E = (vay ngắn hạn + vay dài hạn) / vốn chủ sở hữu cuối kỳ (chỉ nợ vay, không gồm nợ phải trả
  khác). Nợ phải trả / vốn chủ để ở cột riêng.
- CFO/LNST = lưu chuyển tiền thuần từ hoạt động kinh doanh / LNST: > 1 là lợi nhuận "có tiền về".
- Tăng trưởng EPS = EPS năm nay / EPS năm trước − 1, chỉ tính khi EPS năm trước > 0. Dùng EPS
  công bố từng năm, KHÔNG điều chỉnh khi có cổ phiếu thưởng/chia tách, nên năm phát hành thêm
  cổ phiếu sẽ thấp đi. Xem thêm tăng trưởng LNST công ty mẹ (không bị ảnh hưởng).
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
    "equity",             # vốn chủ sở hữu (gồm lợi ích cổ đông không kiểm soát)
    "minority_interest",  # lợi ích cổ đông không kiểm soát (trong vốn chủ)
    "short_debt",         # vay và nợ thuê tài chính ngắn hạn
    "long_debt",          # vay và nợ thuê tài chính dài hạn
    "cfo",                # lưu chuyển tiền thuần từ HĐKD
    # Dùng cho định giá (ngày 3). Chi phí/khoản chi lưu dạng số DƯƠNG.
    "pbt",                # lợi nhuận trước thuế
    "tax_expense",        # chi phí thuế TNDN
    "interest_expense",   # chi phí lãi vay
    "capex",              # tiền chi mua sắm TSCĐ và tài sản dài hạn
    "cash",               # tiền và tương đương tiền
    "short_term_investments",  # đầu tư tài chính ngắn hạn (chủ yếu tiền gửi có kỳ hạn)
    "paid_in_capital",    # vốn góp của chủ sở hữu (mệnh giá 10.000 đ/cp -> số cổ phiếu)
    # Dùng cho F-Score, Altman Z (ngày 4).
    "current_assets",     # tài sản ngắn hạn
    "current_liabilities",  # nợ ngắn hạn
    "retained_earnings",  # lợi nhuận sau thuế chưa phân phối
    "equity_issued",      # tiền thu từ phát hành cổ phiếu, góp vốn (lưu chuyển tiền tài chính)
]

# Khoản mục riêng của định chế tài chính (fa/sector.py). Doanh nghiệp thường để trống, không báo
# thiếu. Chi phí/dự phòng lưu số DƯƠNG.
SECTOR_FIELDS = [
    # Ngân hàng
    "nii",                    # thu nhập lãi thuần
    "toi",                    # tổng thu nhập hoạt động
    "bank_opex",              # chi phí hoạt động (quản lý)
    "credit_provision",       # chi phí dự phòng rủi ro tín dụng
    "loans_gross",            # cho vay khách hàng (trước dự phòng)
    "loan_loss_reserve",      # dự phòng rủi ro cho vay khách hàng (số dư)
    "customer_deposits",      # tiền gửi của khách hàng
    "sbv_balances",           # tiền gửi tại NHNN
    "interbank_placements",   # tiền gửi và cho vay các TCTD khác
    "trading_securities",     # chứng khoán kinh doanh
    "investment_securities",  # chứng khoán đầu tư
    # Công ty chứng khoán
    "margin_loans",           # các khoản cho vay (chủ yếu ký quỹ)
    "fvtpl_assets",           # tài sản tài chính FVTPL (tự doanh)
    "brokerage_revenue",      # doanh thu môi giới
    "operating_revenue",      # doanh thu hoạt động
]
# Kiểm tra chất lượng BCTC (v3, fa/forensic.py): Beneish M-Score, người mua trả trước, phải thu
# khác. Doanh nghiệp thường; không báo thiếu với ngân hàng/CK. Đã dò item_id thật (FPT, DIG).
FORENSIC_FIELDS = [
    "receivables",            # phải thu khách hàng ngắn hạn
    "other_receivables",      # phải thu khác ngắn hạn
    "other_receivables_lt",   # phải thu dài hạn khác
    "inventory",              # hàng tồn kho (ròng)
    "ppe",                    # giá trị còn lại TSCĐ hữu hình
    "depreciation",           # khấu hao TSCĐ và BĐSĐT (lưu chuyển tiền)
    "selling_expenses",       # chi phí bán hàng (số dương)
    "admin_expenses",         # chi phí quản lý doanh nghiệp (số dương)
    "customer_advances",      # người mua trả tiền trước ngắn hạn
    "customer_advances_lt",   # người mua trả tiền trước dài hạn
]
STORED_FIELDS = FIELDS + SECTOR_FIELDS + FORENSIC_FIELDS

RATIO_COLUMNS = [
    "roe", "roa", "gross_margin", "net_margin", "debt_to_equity",
    "liabilities_to_equity", "cfo_to_net_income", "eps_growth", "net_income_parent_growth",
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

    parent_equity = df["equity"] - df["minority_interest"].fillna(0)
    avg_equity = ((parent_equity + prev_ok(parent_equity)) / 2).fillna(parent_equity)
    avg_assets = ((df["total_assets"] + prev_ok(df["total_assets"])) / 2).fillna(df["total_assets"])
    debt = df["short_debt"].fillna(0) + df["long_debt"].fillna(0)
    debt = debt.where(df["short_debt"].notna() | df["long_debt"].notna())
    eps_prev = prev_ok(df["eps"])
    ni_prev = prev_ok(df["net_income_parent"])

    out = pd.DataFrame({"year": df["year"].astype(int)})
    out["roe"] = _div(df["net_income_parent"], avg_equity)
    out["roa"] = _div(df["net_income"], avg_assets)
    out["gross_margin"] = _div(df["gross_profit"], df["revenue"])
    out["net_margin"] = _div(df["net_income"], df["revenue"])
    out["debt_to_equity"] = _div(debt, df["equity"])
    out["liabilities_to_equity"] = _div(df["total_liabilities"], df["equity"])
    out["cfo_to_net_income"] = _div(df["cfo"], df["net_income"])
    out["eps_growth"] = df["eps"] / eps_prev.where(eps_prev > 0) - 1
    out["net_income_parent_growth"] = df["net_income_parent"] / ni_prev.where(ni_prev > 0) - 1
    return out


def format_ratios(ratios: pd.DataFrame) -> pd.DataFrame:
    """Bảng dễ đọc: hàng là chỉ số, cột là năm, đơn vị % (D/E và CFO/LNST để dạng lần)."""
    labels = {
        "roe": "ROE (%)", "roa": "ROA (%)", "gross_margin": "Biên gộp (%)",
        "net_margin": "Biên ròng (%)", "debt_to_equity": "Vay/VCSH (lần)",
        "liabilities_to_equity": "Nợ phải trả/VCSH (lần)", "cfo_to_net_income": "CFO/LNST (lần)",
        "eps_growth": "Tăng trưởng EPS công bố (%)",
        "net_income_parent_growth": "Tăng trưởng LNST cty mẹ (%)",
    }
    pct = {"roe", "roa", "gross_margin", "net_margin", "eps_growth", "net_income_parent_growth"}
    t = ratios.set_index("year")[RATIO_COLUMNS].copy()
    for c in RATIO_COLUMNS:
        t[c] = (t[c] * 100).round(1) if c in pct else t[c].round(2)
    return t.rename(columns=labels).T
