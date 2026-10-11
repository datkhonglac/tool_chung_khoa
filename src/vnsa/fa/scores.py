"""Điểm chất lượng tài chính: Piotroski F-Score, Altman Z và điểm FA tổng hợp (0–100).

Đầu vào là bảng BCTC năm chuẩn hoá (year + FIELDS, đồng) như ratios.py. Năm phải liên tiếp mới
so sánh với năm trước; thiếu số thì tiêu chí đó là NaN (không tính điểm), không đoán.

Piotroski F-Score (9 tiêu chí, mỗi tiêu chí đạt = 1 điểm), so năm t với năm t−1:
  Sinh lời   1. ROA > 0            2. CFO > 0            3. ROA tăng
             4. CFO > LNST (lợi nhuận có tiền về, ít "dồn tích")
  Đòn bẩy,   5. Vay dài hạn / tổng tài sản giảm (không có vay cả hai năm cũng tính đạt)
  thanh khoản 6. Tỷ số thanh toán hiện hành (TSNH / nợ ngắn hạn) tăng
             7. Không phát hành cổ phiếu thu tiền đáng kể (pha loãng cổ đông công ty mẹ):
                - có lịch sử cổ tức/thưởng bằng cổ phiếu (`stock_ratio`): vốn góp công ty mẹ năm
                  t / năm t−1 / (1 + tỷ lệ cổ phiếu thưởng, cổ tức cổ phiếu trong năm t) − 1
                  ≤ max(`issue_materiality`, 2%) (2%: chênh thời điểm đăng ký vốn so với ngày
                  GDKHQ). Không dùng dòng tiền vì "tiền thu phát hành, nhận vốn góp" gồm cả vốn cổ
                  đông thiểu số góp vào công ty con (chạy thật: FPT 1.196 tỷ năm 2025);
                - không có: tiền thu phát hành trong lưu chuyển tiền ≤ `issue_materiality` (1%) vốn
                  chủ; thiếu nữa thì vốn góp không tăng (thận trọng: thưởng cổ phiếu cũng bị tính).
  Hiệu quả   8. Biên lợi nhuận gộp tăng   9. Vòng quay tài sản (doanh thu / tổng tài sản) tăng
  ROA ở đây = LNST / tổng tài sản cuối kỳ (để có F-Score cho cả năm thứ 2 khi chỉ có 4 năm BCTC).
  8–9: mạnh, 0–2: yếu (Piotroski, 2000).

Altman Z (dùng tổng tài sản TA cuối kỳ):
  X1 = (TSNH − nợ ngắn hạn) / TA      X2 = LNST chưa phân phối / TA
  X3 = EBIT / TA, EBIT = LNTT + chi phí lãi vay     X5 = doanh thu thuần / TA
  Z'' (Altman 1995, cho doanh nghiệp ngoài sản xuất và thị trường mới nổi — dùng làm chính):
      Z'' = 6,56 X1 + 3,26 X2 + 6,72 X3 + 1,05 × (vốn chủ sổ sách / nợ phải trả)
      > 2,6 an toàn · 1,1–2,6 vùng xám · < 1,1 nguy cơ.
  Z gốc (1968, cho doanh nghiệp sản xuất niêm yết):
      Z = 1,2 X1 + 1,4 X2 + 3,3 X3 + 0,6 × (vốn hoá / nợ phải trả) + 1,0 X5
      > 2,99 an toàn · 1,81–2,99 vùng xám · < 1,81 nguy cơ.

Điểm FA (0–100): mỗi chỉ tiêu được quy về 0–100 tuyến tính giữa mức "xấu" và mức "tốt" trong
mục [fa_score] của config/settings.toml (định giá chấm TƯƠNG ĐỐI: P/E, P/B hiện tại chia trung vị
lịch sử của chính mã — P/E 15 là rẻ với FPT nhưng đắt với HPG, nên không dùng ngưỡng tuyệt đối), trung bình trong từng nhóm, rồi bình quân gia quyền
theo trọng số nhóm. Chỉ tiêu thiếu số bị bỏ qua và trọng số được chia lại cho phần còn lại.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vnsa.fa.ratios import FIELDS, compute_ratios

FSCORE_CRITERIA = {
    "f_roa": "ROA > 0",
    "f_cfo": "CFO > 0",
    "f_droa": "ROA tăng",
    "f_accrual": "CFO > LNST",
    "f_dlever": "Vay dài hạn/TS giảm",
    "f_dliquid": "Thanh toán hiện hành tăng",
    "f_no_issue": "Không phát hành cp thu tiền",
    "f_dmargin": "Biên gộp tăng",
    "f_dturn": "Vòng quay tài sản tăng",
}


def _prepare(fin: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = fin.sort_values("year").reset_index(drop=True).copy()
    for col in FIELDS:
        if col not in df:
            df[col] = np.nan
    gap = df["year"].diff() != 1
    return df, gap


def _flag(cond: pd.Series, *inputs: pd.Series) -> pd.Series:
    """1/0 theo điều kiện; NaN nếu thiếu bất kỳ số đầu vào nào."""
    ok = pd.concat(inputs, axis=1).notna().all(axis=1)
    return cond.astype(float).where(ok)


def piotroski(fin: pd.DataFrame, issue_materiality: float = 0.01,
              stock_ratio: dict[int, float] | None = None) -> pd.DataFrame:
    """Bảng: year, 9 cột tiêu chí (1/0/NaN) và f_score (tổng; NaN nếu thiếu tiêu chí nào).
    stock_ratio: {năm: tổng tỷ lệ thưởng/cổ tức bằng cổ phiếu} (bảng dividends) — xem tiêu chí 7."""
    df, gap = _prepare(fin)

    def prev(s: pd.Series) -> pd.Series:
        return s.shift(1).where(~gap)

    ta = df["total_assets"].where(df["total_assets"] > 0)
    roa = df["net_income"] / ta
    lever = df["long_debt"].fillna(0) / ta
    liquid = df["current_assets"] / df["current_liabilities"].where(df["current_liabilities"] > 0)
    margin = df["gross_profit"] / df["revenue"].where(df["revenue"] > 0)
    turn = df["revenue"] / ta

    out = pd.DataFrame({"year": df["year"].astype(int)})
    out["f_roa"] = _flag(roa > 0, roa)
    out["f_cfo"] = _flag(df["cfo"] > 0, df["cfo"])
    out["f_droa"] = _flag(roa > prev(roa), roa, prev(roa))
    out["f_accrual"] = _flag(df["cfo"] > df["net_income"], df["cfo"], df["net_income"])
    lever_prev = prev(lever)
    out["f_dlever"] = _flag((lever < lever_prev) | ((lever == 0) & (lever_prev == 0)),
                            lever, lever_prev)
    out["f_dliquid"] = _flag(liquid > prev(liquid), liquid, prev(liquid))
    # Ưu tiên tiền thu từ phát hành cổ phiếu; thiếu thì so vốn góp với năm trước.
    by_cash = _flag(df["equity_issued"] <= issue_materiality * df["equity"],
                    df["equity_issued"], df["equity"])
    cap_prev = prev(df["paid_in_capital"])
    by_capital = _flag(df["paid_in_capital"] <= cap_prev, df["paid_in_capital"], cap_prev)
    out["f_no_issue"] = by_cash.fillna(by_capital)
    if stock_ratio is not None:
        bonus = df["year"].map(lambda y: stock_ratio.get(int(y), 0.0)).astype(float)
        net_new = df["paid_in_capital"] / cap_prev / (1 + bonus) - 1
        by_parent = _flag(net_new <= max(issue_materiality, 0.02), df["paid_in_capital"], cap_prev)
        out["f_no_issue"] = by_parent.fillna(out["f_no_issue"])
    out["f_dmargin"] = _flag(margin > prev(margin), margin, prev(margin))
    out["f_dturn"] = _flag(turn > prev(turn), turn, prev(turn))
    out["f_score"] = out[list(FSCORE_CRITERIA)].sum(axis=1, min_count=len(FSCORE_CRITERIA))
    return out


def altman(fin: pd.DataFrame, market_caps: dict[int, float] | None = None) -> pd.DataFrame:
    """Bảng: year, x1..x5, z_em (Z''), z_orig (Z gốc, cần vốn hoá từng năm, đồng)."""
    df, _ = _prepare(fin)
    ta = df["total_assets"].where(df["total_assets"] > 0)
    tl = df["total_liabilities"].where(df["total_liabilities"] > 0)
    ebit = df["pbt"] + df["interest_expense"].fillna(0)
    caps = df["year"].map(market_caps or {}).astype(float)

    out = pd.DataFrame({"year": df["year"].astype(int)})
    out["x1"] = (df["current_assets"] - df["current_liabilities"]) / ta
    out["x2"] = df["retained_earnings"] / ta
    out["x3"] = ebit / ta
    out["x4_book"] = df["equity"] / tl
    out["x4_market"] = caps / tl
    out["x5"] = df["revenue"] / ta
    out["z_em"] = 6.56 * out["x1"] + 3.26 * out["x2"] + 6.72 * out["x3"] + 1.05 * out["x4_book"]
    out["z_orig"] = (1.2 * out["x1"] + 1.4 * out["x2"] + 3.3 * out["x3"]
                     + 0.6 * out["x4_market"] + 1.0 * out["x5"])
    return out


def z_zone(z: float, kind: str = "em") -> str:
    if pd.isna(z):
        return "—"
    lo, hi = (1.1, 2.6) if kind == "em" else (1.81, 2.99)
    return "an toàn" if z > hi else ("vùng xám" if z >= lo else "nguy cơ")


def year_end_market_caps(fin: pd.DataFrame, prices: pd.DataFrame, par_value: float = 10_000
                         ) -> dict[int, float]:
    """Vốn hoá cuối mỗi năm BCTC = giá đóng cửa phiên cuối năm × (vốn góp / mệnh giá)."""
    if prices.empty:
        return {}
    close = prices.set_index("date")["close"]
    caps = {}
    for _, row in fin.iterrows():
        year, cap = int(row["year"]), row.get("paid_in_capital")
        in_year = close[close.index.year == year]
        if len(in_year) and not pd.isna(cap) and cap > 0:
            caps[year] = float(in_year.iloc[-1]) * cap / par_value
    return caps


def scale(x: float, bad: float, good: float) -> float:
    """Quy x về 0–100: bằng `bad` -> 0, bằng `good` -> 100, tuyến tính ở giữa, chặn hai đầu.
    Dùng được cả khi số nhỏ là tốt (bad > good, ví dụ P/E)."""
    if x is None or pd.isna(x) or bad == good:
        return float("nan")
    return float(np.clip((x - bad) / (good - bad), 0.0, 1.0) * 100)


@dataclass
class FAScore:
    total: float
    groups: dict[str, float]                       # điểm từng nhóm (0–100)
    items: dict[str, tuple[float, float]]          # chỉ tiêu -> (giá trị, điểm)
    notes: list[str] = field(default_factory=list)

    @property
    def rating(self) -> str:
        if pd.isna(self.total):
            return "—"
        return "Tốt" if self.total >= 70 else ("Trung bình" if self.total >= 50 else "Yếu")


GROUPS = {
    "sinh_loi": ["roe"],
    "tang_truong": ["ni_cagr"],
    "suc_khoe": ["z_em", "debt_to_equity"],
    "chat_luong": ["f_score", "cfo_to_ni"],
    "dinh_gia": ["pe_rel", "pb_rel", "dcf_upside"],
}


def cagr(first: float, last: float, years: int) -> float:
    if years <= 0 or pd.isna(first) or pd.isna(last) or first <= 0 or last <= 0:
        return float("nan")
    return (last / first) ** (1 / years) - 1


def _div_sum(a: pd.Series, b: pd.Series) -> float:
    """Tổng a / tổng b (cộng dồn để một năm lãi rất nhỏ không làm méo tỷ lệ)."""
    if a.isna().any() or b.isna().any() or b.sum() <= 0:
        return float("nan")
    return float(a.sum() / b.sum())


def fa_score(fin: pd.DataFrame, cfg: dict, pe: float = np.nan, pb: float = np.nan,
             dcf_upside: float = np.nan, roe: float | None = None,
             pe_hist: float = np.nan, pb_hist: float = np.nan,
             stock_ratio: dict[int, float] | None = None) -> FAScore:
    """Điểm FA năm BCTC gần nhất. cfg = mục [fa_score] trong settings (trọng số + ngưỡng).
    roe: ROE hiện tại (TTM, fa/ttm.py) thay cho ROE năm gần nhất nếu có."""
    df = fin.sort_values("year").reset_index(drop=True)
    ratios = compute_ratios(df)
    f = piotroski(df, cfg.get("issue_materiality", 0.01), stock_ratio)
    z = altman(df)
    span = int(df["year"].iloc[-1] - df["year"].iloc[0])
    values = {
        "roe": ratios["roe"].iloc[-1] if roe is None or pd.isna(roe) else roe,
        "ni_cagr": cagr(df["net_income_parent"].iloc[0], df["net_income_parent"].iloc[-1], span),
        "z_em": z["z_em"].iloc[-1],
        "debt_to_equity": ratios["debt_to_equity"].iloc[-1],
        "f_score": f["f_score"].iloc[-1],
        "cfo_to_ni": _div_sum(df["cfo"].tail(3), df["net_income"].tail(3)),
        # P/E âm (lỗ) không chấm; thiếu mốc lịch sử thì bỏ chỉ tiêu (chia lại trọng số)
        "pe_rel": pe / pe_hist if pe > 0 and pe_hist > 0 else np.nan,
        "pb_rel": pb / pb_hist if pb > 0 and pb_hist > 0 else np.nan,
        "dcf_upside": dcf_upside,
    }
    thresholds = cfg["thresholds"]
    items = {k: (float(v), scale(v, *thresholds[k])) for k, v in values.items()}
    notes = []
    if pd.isna(values["ni_cagr"]):
        notes.append("LNST cty mẹ đầu hoặc cuối kỳ ≤ 0: không tính tăng trưởng kép")
    groups = {}
    for g, keys in GROUPS.items():
        scores = [items[k][1] for k in keys if not pd.isna(items[k][1])]
        groups[g] = float(np.mean(scores)) if scores else float("nan")
    weights = {g: w for g, w in cfg["weights"].items() if not pd.isna(groups.get(g, np.nan))}
    total = (sum(groups[g] * w for g, w in weights.items()) / sum(weights.values())
             if weights else float("nan"))
    return FAScore(total=total, groups=groups, items=items, notes=notes)


# Định chế tài chính (ngân hàng, chứng khoán, bảo hiểm): không có biên gộp, vòng quay, Altman Z...
# nên chấm bằng bộ chỉ tiêu riêng ở [fa_score.financials]. Vốn chủ / tổng tài sản đo "đệm vốn"
# (ngân hàng thấp tự nhiên, nên ngưỡng tách theo loại doanh nghiệp).
FIN_GROUPS = {
    "sinh_loi": ["roe"],
    "tang_truong": ["ni_cagr"],
    "an_toan": ["equity_to_assets"],
    "hieu_qua": ["nim", "cir"],                         # ngân hàng (fa/sector.py)
    "rui_ro": ["credit_cost", "margin_to_equity"],      # ngân hàng | công ty chứng khoán
    "dinh_gia": ["pb_rel", "fair_upside"],
}


def financial_score(fin: pd.DataFrame, cfg: dict, model: str, pb: float = np.nan,
                    fair_upside: float = np.nan, roe: float | None = None,
                    pb_hist: float = np.nan, sector=None) -> FAScore:
    """Điểm FA 0–100 cho định chế tài chính. cfg = mục [fa_score.financials] trong settings.
    sector = fa.sector.SectorMetrics (NIM, CIR, chi phí tín dụng, ký quỹ/vốn chủ) nếu có; chỉ tiêu
    thiếu thì nhóm đó bị bỏ, trọng số chia lại cho các nhóm còn lại."""
    df = fin.sort_values("year").reset_index(drop=True)
    ratios = compute_ratios(df)
    span = int(df["year"].iloc[-1] - df["year"].iloc[0])
    last = df.iloc[-1]
    values = {
        "roe": ratios["roe"].iloc[-1] if roe is None or pd.isna(roe) else roe,
        "ni_cagr": cagr(df["net_income_parent"].iloc[0], df["net_income_parent"].iloc[-1], span),
        "equity_to_assets": (last["equity"] / last["total_assets"]
                             if last["total_assets"] > 0 else np.nan),
        "pb_rel": pb / pb_hist if pb > 0 and pb_hist > 0 else np.nan,
        "fair_upside": fair_upside,
    }
    sv = sector.values if sector is not None else {}
    for k in ("nim", "cir", "credit_cost", "margin_to_equity"):
        values[k] = sv.get(k, np.nan)
    thresholds = dict(cfg["thresholds"])
    thresholds["equity_to_assets"] = cfg["equity_to_assets"][model]
    items = {k: (float(v), scale(v, *thresholds[k]) if k in thresholds else np.nan)
             for k, v in values.items()}
    groups = {}
    for g, keys in FIN_GROUPS.items():
        scores = [items[k][1] for k in keys if not pd.isna(items[k][1])]
        groups[g] = float(np.mean(scores)) if scores else float("nan")
    weights = {g: w for g, w in cfg["weights"].items() if not pd.isna(groups.get(g, np.nan))}
    total = (sum(groups[g] * w for g, w in weights.items()) / sum(weights.values())
             if weights else float("nan"))
    notes = []
    if model == "bank":
        missing = sector.missing if sector else ["NPL", "CASA", "CAR"]
        notes.append(f"nguồn miễn phí không có {', '.join(missing)}: chưa chấm chất lượng tài sản "
                     "theo nợ xấu — cần xem BCTC gốc")
        if sector is None:
            notes.append("chưa có khoản mục ngân hàng (NIM, CIR, chi phí tín dụng): chạy lại "
                         "fetch-bctc để tải")
    return FAScore(total=total, groups=groups, items=items, notes=notes)
