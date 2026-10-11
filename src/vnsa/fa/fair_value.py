"""Giá trị hợp lý mỗi cổ phiếu: gộp nhiều phương pháp, kể cả cho ngân hàng / chứng khoán.

Vì sao không chỉ dùng DCF: ở Việt Nam nhiều doanh nghiệp đang đầu tư mạnh (HPG, ACV...) có FCFF
âm nhiều năm nên DCF không ra số; ngân hàng/chứng khoán thì FCFF không có nghĩa. Thị trường VN
định giá phổ biến bằng P/E, P/B so với lịch sử của chính mã đó, nên gộp như sau:

Doanh nghiệp thường — giá trị hợp lý = TRUNG VỊ các phương pháp ra số dương:
  1. DCF-FCFF (fa/valuation.py), nếu FCFF bình quân dương.
  2. P/E lịch sử: P/E mục tiêu = trung vị P/E từng năm (giá bình quân năm / EPS năm) × EPS năm gần
     nhất. Lấy giá bình quân cả năm thay vì giá một ngày cuối năm để bớt nhiễu.
  3. P/B lịch sử: tương tự với giá trị sổ sách mỗi cổ phiếu (BVPS) cuối năm.
  EPS và BVPS các năm đều chia cho số cổ phiếu HIỆN TẠI, vì giá đã điều chỉnh về cơ sở cổ phiếu
  hiện tại (sau cổ tức cổ phiếu/thưởng) — chia số cổ phiếu cũ sẽ làm P/E quá khứ bị méo.

Ngân hàng / chứng khoán / bảo hiểm — giá trị hợp lý = trung vị của:
  1. Thu nhập thặng dư (Residual Income, mô hình chuẩn cho định chế tài chính):
       V0 = B0 + Σ_{t=1..N} (ROE_t − k_e) × B_{t−1} / (1 + k_e)^t
              + (ROE_T − k_e) × B_N / ((k_e − g) × (1 + k_e)^N)
     B0 = vốn chủ công ty mẹ / số cổ phiếu; ROE_0 = ROE bình quân `roe_years` năm gần nhất;
     ROE_t giảm tuyến tính từ ROE_0 về ROE_T = k_e + `roe_persistence` × (ROE_0 − k_e)
     (lợi thế cạnh tranh mòn dần, chỉ giữ một phần chênh lệch so với chi phí vốn);
     B_t = B_{t−1} × (1 + ROE_t × `retention`) (ngân hàng VN giữ lại phần lớn lợi nhuận).
  2. P/B hợp lý theo ROE (Gordon): P/B* = (ROE_0 − g) / (k_e − g); giá trị = P/B* × B0.
  3. P/B lịch sử (như trên).
k_e = r_f + beta × ERP (cùng giả định [valuation]); tham số riêng ở [valuation.financials].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vnsa.fa.ratios import compute_ratios

METHOD_NAMES = {"dcf": "DCF-FCFF", "pe_hist": "P/E lịch sử", "pb_hist": "P/B lịch sử",
                "ri": "Thu nhập thặng dư", "pb_roe": "P/B hợp lý theo ROE"}


def shares_outstanding(fin: pd.DataFrame, par_value: float = 10_000) -> float:
    cap = fin.sort_values("year")["paid_in_capital"].iloc[-1]
    return float(cap / par_value) if not pd.isna(cap) and cap > 0 else float("nan")


def parent_equity(row: pd.Series) -> float:
    mi = row.get("minority_interest")
    return float(row["equity"] - (0 if pd.isna(mi) else mi))


def historical_multiples(fin: pd.DataFrame, prices: pd.DataFrame,
                         par_value: float = 10_000) -> pd.DataFrame:
    """Mỗi năm BCTC: giá bình quân năm, EPS và BVPS theo số cổ phiếu hiện tại, P/E, P/B."""
    shares = shares_outstanding(fin, par_value)
    close = prices.set_index("date")["close"]
    rows = []
    for _, r in fin.sort_values("year").iterrows():
        year = int(r["year"])
        in_year = close[close.index.year == year]
        avg = float(in_year.mean()) if len(in_year) >= 60 else float("nan")   # đủ ~1 quý giao dịch
        eps = r["net_income_parent"] / shares
        bvps = parent_equity(r) / shares
        rows.append({"year": year, "avg_price": avg, "eps": eps, "bvps": bvps,
                     "pe": avg / eps if eps > 0 else np.nan,
                     "pb": avg / bvps if bvps > 0 else np.nan})
    return pd.DataFrame(rows)


def _median_positive(s: pd.Series) -> float:
    s = s[(s > 0) & s.notna()]
    return float(s.median()) if len(s) else float("nan")


@dataclass
class ResidualIncome:
    cost_of_equity: float
    roe0: float
    roe_terminal: float
    book0: float                 # B0, đồng/cp
    roe_path: list[float]
    pv_ri: float                 # tổng PV thu nhập thặng dư giai đoạn dự phóng
    pv_terminal: float
    value_per_share: float
    justified_pb: float          # P/B* = (ROE0 − g) / (ke − g)
    notes: list[str] = field(default_factory=list)


def residual_income(fin: pd.DataFrame, ke: float, g: float, cfg: dict,
                    par_value: float = 10_000) -> ResidualIncome:
    """Định giá thu nhập thặng dư cho ngân hàng / chứng khoán / bảo hiểm (xem đầu file)."""
    if ke <= g:
        raise ValueError(f"chi phí vốn chủ ({ke:.1%}) phải lớn hơn tăng trưởng dài hạn ({g:.1%})")
    df = fin.sort_values("year").reset_index(drop=True)
    shares = shares_outstanding(df, par_value)
    if pd.isna(shares):
        raise ValueError("thiếu vốn góp chủ sở hữu, không tính được số cổ phiếu")
    b0 = parent_equity(df.iloc[-1]) / shares
    roe_hist = compute_ratios(df)["roe"].dropna().tail(int(cfg.get("roe_years", 3)))
    if b0 <= 0 or roe_hist.empty:
        raise ValueError("thiếu vốn chủ hoặc ROE, không định giá thu nhập thặng dư được")
    roe0 = float(roe_hist.mean())
    notes = []
    if roe0 <= g:
        notes.append(f"ROE bình quân {roe0:.1%} ≤ tăng trưởng dài hạn: P/B hợp lý theo ROE không "
                     "có ý nghĩa")
    n = int(cfg.get("forecast_years", 5))
    roe_t = ke + float(cfg.get("roe_persistence", 0.5)) * (roe0 - ke)
    path = [roe0 + (roe_t - roe0) * (t - 1) / (n - 1) if n > 1 else roe_t for t in range(1, n + 1)]
    retention = float(cfg.get("retention", 0.8))
    book, pv = b0, 0.0
    for t, roe in enumerate(path, start=1):
        pv += (roe - ke) * book / (1 + ke) ** t
        book *= 1 + roe * retention
    pv_tv = (roe_t - ke) * book / ((ke - g) * (1 + ke) ** n)
    pb_star = (roe0 - g) / (ke - g) if roe0 > g else float("nan")
    return ResidualIncome(cost_of_equity=ke, roe0=roe0, roe_terminal=roe_t, book0=b0,
                          roe_path=path, pv_ri=pv, pv_terminal=pv_tv,
                          value_per_share=b0 + pv + pv_tv, justified_pb=pb_star, notes=notes)


@dataclass
class FairValue:
    price: float
    methods: dict[str, float]    # tên phương pháp -> giá trị/cp (đồng); NaN = không dùng được
    value: float                 # trung vị các phương pháp dùng được
    pe_target: float = float("nan")
    pb_target: float = float("nan")
    ri: ResidualIncome | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def upside(self) -> float:
        return self.value / self.price - 1 if self.value > 0 else float("nan")

    @property
    def used(self) -> dict[str, float]:
        return {k: v for k, v in self.methods.items() if not pd.isna(v) and v > 0}


def combine(price: float, fin: pd.DataFrame, prices: pd.DataFrame, par_value: float = 10_000,
            dcf: float = float("nan"), ri: ResidualIncome | None = None) -> FairValue:
    """Giá trị hợp lý = trung vị các phương pháp ra số dương (doanh nghiệp thường: DCF, P/E, P/B
    lịch sử; định chế tài chính khi truyền `ri`: RI, P/B theo ROE, P/B lịch sử)."""
    hist = historical_multiples(fin, prices, par_value)
    last = hist.iloc[-1]
    pe_t, pb_t = _median_positive(hist["pe"]), _median_positive(hist["pb"])
    notes = []
    if ri is None:
        methods = {"dcf": dcf,
                   "pe_hist": pe_t * last["eps"] if last["eps"] > 0 else np.nan,
                   "pb_hist": pb_t * last["bvps"] if last["bvps"] > 0 else np.nan}
        if last["eps"] <= 0:
            notes.append("EPS năm gần nhất ≤ 0: bỏ phương pháp P/E")
    else:
        methods = {"ri": ri.value_per_share, "pb_roe": ri.justified_pb * ri.book0,
                   "pb_hist": pb_t * ri.book0}
        notes += ri.notes
    if hist["pe"].notna().sum() + hist["pb"].notna().sum() < 4:
        notes.append("ít năm có đủ giá để tính bội số lịch sử: P/E, P/B lịch sử kém tin cậy")
    used = [v for v in methods.values() if not pd.isna(v) and v > 0]
    value = float(np.median(used)) if used else float("nan")
    if not used:
        notes.append("không phương pháp nào ra giá trị dương")
    return FairValue(price=price, methods=methods, value=value, pe_target=pe_t, pb_target=pb_t,
                     ri=ri, notes=notes)
