"""Kiểm tra chất lượng BCTC (nâng cấp v3) — CHỈ CẢNH BÁO: hiện trên thẻ báo cáo và giảm độ tin cậy
một bậc, KHÔNG đổi xếp hạng (chưa kiểm định trên dữ liệu Việt Nam).

1. Beneish M-Score 8 biến (Beneish 1999), năm t so với t−1:
     DSRI = (Phải thu_t/DT_t) ÷ (Phải thu_{t−1}/DT_{t−1})       phải thu tăng nhanh hơn doanh thu
     GMI  = Biên gộp_{t−1} ÷ Biên gộp_t                          biên gộp xấu đi
     AQI  = [1 − (TSNH_t + TSCĐ_t)/TTS_t] ÷ [cùng công thức năm t−1]   tài sản "mềm" tăng
     SGI  = DT_t ÷ DT_{t−1}                                      tăng trưởng doanh thu
     DEPI = Tỷ lệ KH_{t−1} ÷ Tỷ lệ KH_t, tỷ lệ KH = KH/(KH + TSCĐ)   giảm tốc độ khấu hao
     SGAI = (CPBH + QLDN)/DT năm t ÷ năm t−1
     LVGI = (Nợ ngắn hạn + Vay dài hạn)/TTS năm t ÷ năm t−1
     TATA = (LNST − CFO) / TTS năm t                             lợi nhuận dồn tích
     M = −4,84 + 0,920·DSRI + 0,528·GMI + 0,404·AQI + 0,892·SGI + 0,115·DEPI − 0,172·SGAI
         + 4,679·TATA − 0,327·LVGI
   M > −1,78 (`[forensic] mscore_threshold`): có dấu hiệu "làm đẹp" lợi nhuận. DEPI/SGAI/LVGI thiếu
   số thì lấy 1 (trung tính); thiếu DSRI/GMI/AQI/SGI/TATA thì không tính. Mô hình xây trên doanh
   nghiệp Mỹ: bất động sản/xây dựng (doanh thu ghi nhận theo đợt bàn giao) dễ báo nhầm — chỉ tham khảo.
2. Phải thu khác (ngắn + dài hạn) / vốn chủ > `other_receivables_max` (30%): rủi ro tiền bị chiếm
   dụng hoặc cho bên liên quan vay.
3. Bất động sản: người mua trả tiền trước / doanh thu năm ("backlog" — doanh thu chờ bàn giao).
4. Tiền ròng (tiền + đầu tư ngắn hạn − vay) / vốn hoá — thông tin, không cảnh báo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COEF = {"const": -4.84, "dsri": 0.920, "gmi": 0.528, "aqi": 0.404, "sgi": 0.892, "depi": 0.115,
        "sgai": -0.172, "tata": 4.679, "lvgi": -0.327}
REQUIRED = ("dsri", "gmi", "aqi", "sgi", "tata")
DEFAULTS = {"mscore_threshold": -1.78, "other_receivables_max": 0.30}


@dataclass
class Forensic:
    year: int | None
    mscore: float = float("nan")
    components: dict[str, float] = field(default_factory=dict)
    neutral: list[str] = field(default_factory=list)       # biến thiếu số, lấy 1
    other_receivables: float = float("nan")                 # / vốn chủ
    backlog: float = float("nan")                           # người mua trả trước / doanh thu
    backlog_prev: float = float("nan")
    net_cash: float = float("nan")                          # / vốn hoá
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)


def _n(x: float, digits: int = 2) -> str:
    """Số kiểu Việt Nam: dấu phẩy thập phân, dấu trừ in."""
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


def _v(row: pd.Series, key: str) -> float:
    x = row.get(key, np.nan)
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _div(a: float, b: float) -> float:
    return a / b if not (pd.isna(a) or pd.isna(b)) and b != 0 else float("nan")


def _sum(*xs: float) -> float:
    vals = [x for x in xs if not pd.isna(x)]
    return float(sum(vals)) if vals else float("nan")


def beneish(cur: pd.Series, prev: pd.Series) -> tuple[float, dict[str, float], list[str]]:
    """(M-Score, các chỉ số, biến lấy trung tính). M = NaN nếu thiếu biến bắt buộc."""
    s1, s0 = _v(cur, "revenue"), _v(prev, "revenue")
    gm1, gm0 = _div(_v(cur, "gross_profit"), s1), _div(_v(prev, "gross_profit"), s0)
    ta1, ta0 = _v(cur, "total_assets"), _v(prev, "total_assets")

    def soft(row, ta):
        return 1 - _div(_sum(_v(row, "current_assets"), _v(row, "ppe")), ta)

    def dep_rate(row):
        d = _v(row, "depreciation")
        return _div(d, d + _v(row, "ppe")) if not pd.isna(d) else float("nan")

    def sga(row, sales):
        return _div(_sum(_v(row, "selling_expenses"), _v(row, "admin_expenses")), sales)

    def lev(row, ta):
        return _div(_sum(_v(row, "current_liabilities"), _v(row, "long_debt")), ta)

    c = {"dsri": _div(_div(_v(cur, "receivables"), s1), _div(_v(prev, "receivables"), s0)),
         "gmi": _div(gm0, gm1) if not pd.isna(gm1) and gm1 > 0 and gm0 > 0 else float("nan"),
         "aqi": _div(soft(cur, ta1), soft(prev, ta0)),
         "sgi": _div(s1, s0),
         "depi": _div(dep_rate(prev), dep_rate(cur)),
         "sgai": _div(sga(cur, s1), sga(prev, s0)),
         "lvgi": _div(lev(cur, ta1), lev(prev, ta0)),
         "tata": _div(_v(cur, "net_income") - _v(cur, "cfo"), ta1)}
    neutral = []
    for k in ("depi", "sgai", "lvgi"):
        if pd.isna(c[k]) or c[k] <= 0:
            c[k], neutral = 1.0, neutral + [k.upper()]
    if any(pd.isna(c[k]) for k in REQUIRED):
        return float("nan"), c, neutral
    m = COEF["const"] + sum(COEF[k] * c[k] for k in COEF if k != "const")
    return float(m), c, neutral


def analyze(fin: pd.DataFrame, market_cap: float = float("nan"), real_estate: bool = False,
            cfg: dict | None = None) -> Forensic:
    """fin: BCTC năm (doanh nghiệp thường) gồm FIELDS + FORENSIC_FIELDS."""
    c = {**DEFAULTS, **(cfg or {})}
    if fin is None or fin.empty:
        return Forensic(None)
    df = fin.sort_values("year").reset_index(drop=True)
    cur = df.iloc[-1]
    out = Forensic(int(cur["year"]))
    if len(df) >= 2:
        prev = df.iloc[-2]
        if int(prev["year"]) == out.year - 1:
            out.mscore, out.components, out.neutral = beneish(cur, prev)
    thr = float(c["mscore_threshold"])
    if not pd.isna(out.mscore):
        top = sorted(((k, v) for k, v in out.components.items() if k in REQUIRED),
                     key=lambda kv: -COEF[kv[0]] * (kv[1] - (0 if kv[0] == "tata" else 1)))[:2]
        why = ", ".join(f"{k.upper()} {_n(v)}" for k, v in top)
        if out.mscore > thr:
            out.warnings.append(
                f"Beneish M-Score {_n(out.mscore)} > {_n(thr)} năm {out.year}: có dấu hiệu làm đẹp "
                f"lợi nhuận (nổi bật: {why})" + ("; BĐS ghi nhận doanh thu theo đợt bàn giao nên "
                                                  "dễ báo nhầm" if real_estate else ""))
        else:
            out.info.append(f"Beneish M-Score {_n(out.mscore)} (ngưỡng {_n(thr)}): chưa thấy dấu "
                            "hiệu làm đẹp lợi nhuận")
    eq = _v(cur, "equity")
    other = _sum(_v(cur, "other_receivables"), _v(cur, "other_receivables_lt"))
    out.other_receivables = _div(other, eq) if eq > 0 else float("nan")
    if not pd.isna(out.other_receivables) and out.other_receivables > float(c["other_receivables_max"]):
        out.warnings.append(f"phải thu khác {out.other_receivables:.0%} vốn chủ (> "
                            f"{float(c['other_receivables_max']):.0%}): rủi ro tiền bị chiếm dụng/"
                            "cho bên liên quan vay — đọc thuyết minh BCTC")
    if real_estate:
        adv = _sum(_v(cur, "customer_advances"), _v(cur, "customer_advances_lt"))
        out.backlog = _div(adv, _v(cur, "revenue"))
        if len(df) >= 2:
            p = df.iloc[-2]
            out.backlog_prev = _div(_sum(_v(p, "customer_advances"), _v(p, "customer_advances_lt")),
                                    _v(p, "revenue"))
        if not pd.isna(out.backlog):
            out.info.append(f"Người mua trả trước = {_n(out.backlog)} lần doanh thu năm"
                            + (f" (năm trước {_n(out.backlog_prev)})"
                               if not pd.isna(out.backlog_prev) else "")
                            + ": doanh thu chờ bàn giao")
    debt = _sum(_v(cur, "short_debt"), _v(cur, "long_debt"))
    cash = _sum(_v(cur, "cash"), _v(cur, "short_term_investments"))
    if not pd.isna(cash) and not pd.isna(market_cap) and market_cap > 0:
        out.net_cash = (cash - (0 if pd.isna(debt) else debt)) / market_cap
        out.info.append(f"Tiền ròng {'(tiền − vay) ' if out.net_cash >= 0 else '(âm: vay > tiền) '}"
                        f"{out.net_cash:+.0%} vốn hoá".replace("-", "−"))
    return out
