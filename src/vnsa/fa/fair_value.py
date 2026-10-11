"""Giá trị hợp lý mỗi cổ phiếu: gộp nhiều phương pháp, kể cả cho ngân hàng / chứng khoán.

Vì sao không chỉ dùng DCF: ở Việt Nam nhiều doanh nghiệp đang đầu tư mạnh (HPG, ACV...) có FCFF
âm nhiều năm nên DCF không ra số; ngân hàng/chứng khoán thì FCFF không có nghĩa. Thị trường VN
định giá phổ biến bằng P/E, P/B so với lịch sử của chính mã đó, nên gộp như sau:

Giá trị hợp lý = bình quân CÓ TRỌNG SỐ các phương pháp ra số dương (trọng số ở [valuation.weights];
có từ 3 phương pháp thì loại phương pháp lệch quá 2,5 lần trung vị, chia lại trọng số).

Doanh nghiệp thường (mặc định DCF 40% · P/E lịch sử 30% · P/B lịch sử 30%):
  1. DCF-FCFF (fa/valuation.py), nếu FCFF bình quân dương.
  2. P/E lịch sử: P/E mục tiêu = trung vị P/E từng năm × EPS hiện tại (TTM nếu có, fa/ttm.py).
     P/E năm = vốn hoá bình quân năm / LNST cty mẹ năm, vốn hoá = giá CHƯA điều chỉnh (giá khớp
     thật) × số cổ phiếu năm đó (vốn góp / mệnh giá). Giá đã điều chỉnh bị trừ cả cổ tức tiền mặt
     nên làm P/E quá khứ thấp đi; chỉ dùng cách cũ (giá điều chỉnh / EPS theo số cp hiện tại) khi
     nguồn không có giá chưa điều chỉnh (giá lấy từ vnstock).
  3. P/B lịch sử: tương tự với vốn chủ công ty mẹ cuối năm; × BVPS hiện tại (quý gần nhất).
  Bội số MỤC TIÊU 12 tháng không giả định quay hẳn về trung vị lịch sử (thị trường có thể đã định
  giá lại mã): mục tiêu = bội số hiện tại + `multiple_reversion` × (trung vị − hiện tại), mặc định
  0,5 ([valuation]); 1 = quay hẳn về trung vị như cách cũ. Áp dụng cả cho kịch bản P10/P90.

Ngân hàng / chứng khoán / bảo hiểm (mặc định RI 25% · P/B theo ROE 25% · P/B lịch sử 50%): hai
phương pháp đầu cùng dựa vào ROE và k_e nên tổng chỉ bằng P/B lịch sử — mức thị trường vẫn trả
cho mã (chạy thật: SSI ROE tăng nhanh, mô hình ROE thuần cho −29%):
  1. Thu nhập thặng dư (Residual Income, mô hình chuẩn cho định chế tài chính):
       V0 = B0 + Σ_{t=1..N} (ROE_t − k_e) × B_{t−1} / (1 + k_e)^t
              + (ROE_T − k_e) × B_N / ((k_e − g) × (1 + k_e)^N)
     B0 = vốn chủ công ty mẹ / số cổ phiếu; ROE_0 = ROE bình quân có trọng số `roe_years` năm gần
     nhất, năm gần hơn nặng hơn (3 năm: 1:2:3) để bắt xu hướng ROE đang tăng/giảm;
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
    """Mỗi năm BCTC: P/E, P/B theo vốn hoá thật (giá chưa điều chỉnh × số cp năm đó); thiếu giá chưa
    điều chỉnh thì theo giá điều chỉnh / EPS, BVPS tính bằng số cp hiện tại. Cột basis cho biết."""
    shares_now = shares_outstanding(fin, par_value)
    px = prices.set_index("date")
    raw = px["close_raw"] if "close_raw" in px else pd.Series(np.nan, index=px.index)
    rows = []
    for _, r in fin.sort_values("year").iterrows():
        year = int(r["year"])
        in_year = px.index.year == year
        ni, eq = r["net_income_parent"], parent_equity(r)
        cap = r.get("paid_in_capital")
        shares_y = cap / par_value if not pd.isna(cap) and cap > 0 else np.nan
        raw_y = raw[in_year].dropna()
        if len(raw_y) >= 60 and shares_y > 0:            # đủ ~1 quý giao dịch có giá thật
            mcap, basis = float(raw_y.mean()) * shares_y, "giá thật"
        else:
            adj = px["close"][in_year]
            mcap = float(adj.mean()) * shares_now if len(adj) >= 60 else np.nan
            basis = "giá điều chỉnh"
        rows.append({"year": year, "market_cap": mcap, "basis": basis,
                     "eps": ni / shares_now, "bvps": eq / shares_now,
                     "pe": mcap / ni if ni > 0 else np.nan,
                     "pb": mcap / eq if eq > 0 else np.nan})
    return pd.DataFrame(rows)


def history_targets(fin: pd.DataFrame, prices: pd.DataFrame,
                    par_value: float = 10_000) -> tuple[float, float]:
    """(P/E, P/B) trung vị lịch sử của chính mã — mốc để chấm "rẻ hay đắt so với chính nó"."""
    hist = historical_multiples(fin, prices, par_value)
    return _median_positive(hist["pe"]), _median_positive(hist["pb"])


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
                    par_value: float = 10_000, current=None) -> ResidualIncome:
    """Định giá thu nhập thặng dư cho ngân hàng / chứng khoán / bảo hiểm (xem đầu file).
    current = fa.ttm.Current: B0 theo vốn chủ quý gần nhất / số cp lưu hành hiện tại, ROE TTM là
    điểm ROE mới nhất (nặng nhất) khi nó mới hơn BCTC năm."""
    if ke <= g:
        raise ValueError(f"chi phí vốn chủ ({ke:.1%}) phải lớn hơn tăng trưởng dài hạn ({g:.1%})")
    df = fin.sort_values("year").reset_index(drop=True)
    shares = current.shares if current is not None else shares_outstanding(df, par_value)
    if pd.isna(shares):
        raise ValueError("thiếu vốn góp chủ sở hữu, không tính được số cổ phiếu")
    b0 = (current.bvps if current is not None else parent_equity(df.iloc[-1]) / shares)
    n_roe = int(cfg.get("roe_years", 3))
    roe_hist = compute_ratios(df)["roe"].dropna()
    if current is not None and current.basis.startswith("TTM") and not pd.isna(current.roe):
        roe_hist = pd.concat([roe_hist, pd.Series([current.roe])], ignore_index=True)
    roe_hist = roe_hist.tail(n_roe)
    if b0 <= 0 or roe_hist.empty:
        raise ValueError("thiếu vốn chủ hoặc ROE, không định giá thu nhập thặng dư được")
    weights = np.arange(1, len(roe_hist) + 1, dtype=float)       # năm gần nhất nặng nhất
    roe0 = float(np.average(roe_hist, weights=weights))
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


DEFAULT_WEIGHTS = {"general": {"dcf": 0.4, "pe_hist": 0.3, "pb_hist": 0.3},
                   "financial": {"ri": 0.25, "pb_roe": 0.25, "pb_hist": 0.5}}
OUTLIER_RATIO = 2.5     # phương pháp lệch quá 2,5 lần trung vị các phương pháp bị loại


@dataclass
class FairValue:
    price: float
    methods: dict[str, float]    # tên phương pháp -> giá trị/cp (đồng); NaN = không dùng được
    value: float                 # bình quân có trọng số các phương pháp dùng được
    pe_target: float = float("nan")    # P/E mục tiêu (đã tính mức hồi quy về lịch sử)
    pb_target: float = float("nan")
    ri: ResidualIncome | None = None
    pe_median: float = float("nan")    # trung vị P/E, P/B lịch sử của mã
    pb_median: float = float("nan")
    weights: dict[str, float] = field(default_factory=dict)   # trọng số thực dùng (tổng = 1)
    p10: float = float("nan")    # dải giá trị nội tại theo kịch bản (scenario_range)
    p90: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def upside(self) -> float:
        return self.value / self.price - 1 if self.value > 0 else float("nan")

    @property
    def used(self) -> dict[str, float]:
        return {k: v for k, v in self.methods.items() if k in self.weights}


def weighted_value(methods: dict[str, float], weights: dict[str, float]
                   ) -> tuple[float, dict[str, float], list[str]]:
    """Bình quân có trọng số các phương pháp ra số dương. Có từ 3 phương pháp trở lên thì loại
    phương pháp lệch quá OUTLIER_RATIO lần so với trung vị; trọng số chia lại cho phần còn lại.
    Trả về (giá trị, trọng số thực dùng, ghi chú)."""
    ok = {k: v for k, v in methods.items() if not pd.isna(v) and v > 0 and weights.get(k, 0) > 0}
    notes = []
    if len(ok) >= 3:
        mid = float(np.median(list(ok.values())))
        for k in [k for k, v in ok.items() if v > mid * OUTLIER_RATIO or v < mid / OUTLIER_RATIO]:
            notes.append(f"bỏ {METHOD_NAMES[k]} ({ok.pop(k):,.0f}) vì lệch quá xa các phương pháp khác")
    if not ok:
        return float("nan"), {}, notes + ["không phương pháp nào ra giá trị dương"]
    total = sum(weights[k] for k in ok)
    used = {k: weights[k] / total for k in ok}
    return float(sum(ok[k] * w for k, w in used.items())), used, notes


def revert(now: float, median: float, reversion: float = 1.0) -> float:
    """Bội số mục tiêu = hiện tại + reversion × (trung vị lịch sử − hiện tại). Thiếu bội số hiện
    tại (EPS âm...) thì dùng trung vị."""
    if pd.isna(now) or now <= 0:
        return median
    return now + reversion * (median - now)


def scenario_range(values: list[float], center: float) -> tuple[float, float]:
    """(P10, P90) của các giá trị kịch bản dương; luôn bao lấy giá trị trung tâm."""
    v = np.array([x for x in values if not pd.isna(x) and x > 0], dtype=float)
    if len(v) < 3 or pd.isna(center):
        return float("nan"), float("nan")
    lo, hi = np.percentile(v, [10, 90])
    return float(min(lo, center)), float(max(hi, center))


def combine(price: float, fin: pd.DataFrame, prices: pd.DataFrame, par_value: float = 10_000,
            dcf: float = float("nan"), ri: ResidualIncome | None = None,
            weights: dict[str, float] | None = None, current=None,
            reversion: float = 1.0) -> FairValue:
    """Giá trị hợp lý = bình quân có trọng số (mục [valuation.weights]) các phương pháp:
    doanh nghiệp thường DCF, P/E lịch sử, P/B lịch sử; định chế tài chính (truyền `ri`) thu nhập
    thặng dư, P/B theo ROE, P/B lịch sử. reversion: phần chênh lệch bội số hiện tại so với trung vị
    lịch sử được giả định thu hẹp trong 12 tháng (revert)."""
    hist = historical_multiples(fin, prices, par_value)
    last = hist.iloc[-1].copy()
    if current is not None:                     # EPS TTM, BVPS quý gần nhất, số cp hiện tại
        last["eps"], last["bvps"] = current.eps, current.bvps
    pe_med, pb_med = _median_positive(hist["pe"]), _median_positive(hist["pb"])
    bvps = last["bvps"] if ri is None else ri.book0
    pe_t = revert(price / last["eps"] if last["eps"] > 0 else np.nan, pe_med, reversion)
    pb_t = revert(price / bvps if bvps > 0 else np.nan, pb_med, reversion)
    notes = []
    if ri is None:
        methods = {"dcf": dcf,
                   "pe_hist": pe_t * last["eps"] if last["eps"] > 0 else np.nan,
                   "pb_hist": pb_t * last["bvps"] if last["bvps"] > 0 else np.nan}
        if last["eps"] <= 0:
            notes.append("EPS năm gần nhất ≤ 0: bỏ phương pháp P/E")
        weights = weights or DEFAULT_WEIGHTS["general"]
    else:
        methods = {"ri": ri.value_per_share, "pb_roe": ri.justified_pb * ri.book0,
                   "pb_hist": pb_t * ri.book0}
        notes += ri.notes
        weights = weights or DEFAULT_WEIGHTS["financial"]
    if hist["pe"].notna().sum() + hist["pb"].notna().sum() < 4:
        notes.append("ít năm có đủ giá để tính bội số lịch sử: P/E, P/B lịch sử kém tin cậy")
    value, used, more = weighted_value(methods, weights)
    return FairValue(price=price, methods=methods, value=value, pe_target=pe_t, pb_target=pb_t,
                     ri=ri, weights=used, notes=notes + more, pe_median=pe_med, pb_median=pb_med)
