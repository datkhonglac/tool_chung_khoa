"""Định giá: beta, WACC, DCF-FCFF hai giai đoạn, P/E, P/B và ma trận độ nhạy.

Mọi giả định thị trường (lãi suất phi rủi ro, phần bù rủi ro, tăng trưởng dài hạn...) nằm ở mục
[valuation] trong config/settings.toml. Đơn vị tiền: đồng.

Công thức (để đối chiếu):
- Beta = Cov(r_cp, r_index) / Var(r_index), lợi suất TUẦN trong `beta_years` năm gần nhất.
  Nếu `adjust_beta` bật: beta điều chỉnh Blume = 0,67 × beta thô + 0,33 (kéo về 1, giảm nhiễu).
- Chi phí vốn chủ k_e = r_f + beta × ERP (CAPM).
- Chi phí nợ k_d = chi phí lãi vay năm gần nhất / vay bình quân đầu-cuối kỳ; sau thuế k_d × (1 − t).
- Thuế suất t = chi phí thuế TNDN / LNTT, giới hạn 0–50%, thiếu số thì dùng `default_tax_rate`.
- WACC = E/(D+E) × k_e + D/(D+E) × k_d × (1 − t); E = giá thị trường × số cổ phiếu, D = vay (sổ sách).
- FCFF = CFO + lãi vay × (1 − t) − capex. Điểm xuất phát = trung bình FCFF 3 năm gần nhất
  (capex từng năm dao động mạnh).
- DCF: FCFF tăng g1 trong `forecast_years` năm, sau đó tăng đều g (Gordon):
  TV = FCFF_N × (1 + g) / (WACC − g). EV = Σ PV(FCFF) + PV(TV).
- Giá trị vốn chủ = EV − vay + tiền + đầu tư ngắn hạn − lợi ích cổ đông không kiểm soát.
- Số cổ phiếu = vốn góp / mệnh giá 10.000 đ (năm BCTC gần nhất).
- P/E = giá / (LNST cty mẹ / số cổ phiếu); P/B = giá / (VCSH cty mẹ / số cổ phiếu).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Assumptions:
    risk_free: float = 0.035           # lợi suất TPCP 10 năm
    equity_risk_premium: float = 0.08  # phần bù rủi ro thị trường VN
    terminal_growth: float = 0.03      # tăng trưởng dài hạn của FCFF
    forecast_years: int = 5
    max_stage1_growth: float = 0.20    # chặn trên cho g1 lấy từ lịch sử
    default_tax_rate: float = 0.20
    default_beta: float = 1.0          # dùng khi không đủ dữ liệu giá để tính beta
    beta_years: int = 3
    adjust_beta: bool = True
    par_value: float = 10_000

    @classmethod
    def from_settings(cls, cfg: dict) -> Assumptions:
        known = {k: v for k, v in cfg.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def beta(stock: pd.DataFrame, index: pd.DataFrame, years: int = 3) -> tuple[float, int]:
    """Beta từ lợi suất tuần. Đầu vào: bảng giá có cột date, close. Trả về (beta, số tuần)."""
    s = stock.set_index("date")["close"]
    m = index.set_index("date")["close"]
    end = min(s.index.max(), m.index.max())
    start = end - pd.DateOffset(years=years)
    weekly = pd.concat([s, m], axis=1, keys=["s", "m"]).loc[start:end].resample("W-FRI").last()
    r = weekly.pct_change().dropna()
    if len(r) < 52 or r["m"].var() == 0:
        return float("nan"), len(r)
    return float(r["s"].cov(r["m"]) / r["m"].var()), len(r)


def tax_rate(row: pd.Series, default: float = 0.20) -> float:
    pbt, tax = row.get("pbt"), row.get("tax_expense")
    if pd.isna(pbt) or pd.isna(tax) or pbt <= 0:
        return default
    return float(min(max(tax / pbt, 0.0), 0.5))


def total_debt(row: pd.Series) -> float:
    return float(np.nansum([row.get("short_debt", np.nan), row.get("long_debt", np.nan)]))


def cost_of_debt(fin: pd.DataFrame) -> float:
    """Lãi vay năm cuối / vay bình quân hai năm cuối. Không có vay -> NaN."""
    last = fin.iloc[-1]
    debts = [total_debt(r) for _, r in fin.tail(2).iterrows()]
    avg = float(np.mean(debts))
    if avg <= 0 or pd.isna(last.get("interest_expense")):
        return float("nan")
    return float(last["interest_expense"] / avg)


def fcff(fin: pd.DataFrame, default_tax: float = 0.20) -> pd.Series:
    """FCFF từng năm = CFO + lãi vay × (1 − t) − capex (index = năm)."""
    rates = fin.apply(lambda r: tax_rate(r, default_tax), axis=1)
    values = fin["cfo"] + fin["interest_expense"].fillna(0) * (1 - rates) - fin["capex"]
    return pd.Series(values.to_numpy(), index=fin["year"].astype(int), name="fcff")


def historical_growth(fin: pd.DataFrame, cap: float) -> float:
    """CAGR của LNST công ty mẹ qua các năm có trong kho, giới hạn trong [0, cap]."""
    ni = fin["net_income_parent"].dropna()
    n = len(ni) - 1
    if n < 1 or ni.iloc[0] <= 0 or ni.iloc[-1] <= 0:
        return 0.0
    g = (ni.iloc[-1] / ni.iloc[0]) ** (1 / n) - 1
    return float(min(max(g, 0.0), cap))


def dcf_enterprise_value(fcff0: float, g1: float, years: int, g: float, wacc: float) -> dict:
    """EV theo DCF hai giai đoạn. fcff0 là FCFF năm gốc (năm 0)."""
    if wacc <= g:
        raise ValueError(f"WACC ({wacc:.1%}) phải lớn hơn tăng trưởng dài hạn ({g:.1%})")
    flows = [fcff0 * (1 + g1) ** t for t in range(1, years + 1)]
    pv_flows = sum(f / (1 + wacc) ** t for t, f in enumerate(flows, start=1))
    tv = flows[-1] * (1 + g) / (wacc - g)
    pv_tv = tv / (1 + wacc) ** years
    return {"flows": flows, "pv_flows": pv_flows, "terminal_value": tv, "pv_terminal": pv_tv,
            "ev": pv_flows + pv_tv}


@dataclass
class Valuation:
    symbol: str
    price: float
    shares: float
    beta: float          # beta dùng trong CAPM (đã điều chỉnh Blume nếu bật)
    beta_weeks: int
    beta_is_default: bool
    cost_of_equity: float
    cost_of_debt: float
    tax_rate: float
    weight_equity: float
    wacc: float
    g1: float
    g: float
    fcff_history: pd.Series
    fcff_base: float
    dcf: dict
    net_debt_bridge: dict
    equity_value: float
    value_per_share: float
    pe: float
    pb: float
    notes: list[str] = field(default_factory=list)

    @property
    def upside(self) -> float:
        return self.value_per_share / self.price - 1


def bridge(last: pd.Series) -> dict:
    """Từ EV sang giá trị vốn chủ (các khoản cộng/trừ, đồng)."""
    return {
        "debt": total_debt(last),
        "cash": float(np.nansum([last.get("cash"), last.get("short_term_investments")])),
        "minority": float(0 if pd.isna(last.get("minority_interest")) else last["minority_interest"]),
    }


def equity_from_ev(ev: float, b: dict) -> float:
    return ev - b["debt"] + b["cash"] - b["minority"]


def value(
    symbol: str,
    fin: pd.DataFrame,
    price: float,
    a: Assumptions,
    beta_value: float | None = None,
    beta_weeks: int = 0,
    g1: float | None = None,
    wacc_override: float | None = None,
) -> Valuation:
    fin = fin.sort_values("year").reset_index(drop=True)
    last = fin.iloc[-1]
    notes: list[str] = []
    shares = last.get("paid_in_capital", np.nan) / a.par_value
    if not shares or pd.isna(shares) or shares <= 0:
        raise ValueError(f"{symbol}: thiếu vốn góp chủ sở hữu, không tính được số cổ phiếu")

    beta_is_default = beta_value is None or pd.isna(beta_value)
    if beta_is_default:
        b = a.default_beta
        notes.append(f"Không đủ dữ liệu giá để tính beta, dùng beta mặc định {a.default_beta}")
    else:
        if not 0.3 <= beta_value <= 2.5:
            notes.append(f"Beta thô {beta_value:.2f} bất thường, nên kiểm tra lại dữ liệu giá")
        b = 0.67 * beta_value + 0.33 if a.adjust_beta else beta_value
    ke = a.risk_free + b * a.equity_risk_premium
    t = tax_rate(last, a.default_tax_rate)
    kd = cost_of_debt(fin)
    debt = total_debt(last)
    e = price * shares
    we = e / (e + debt) if debt > 0 else 1.0
    if debt > 0 and pd.isna(kd):
        kd = a.risk_free + 0.02
        notes.append("Thiếu chi phí lãi vay, giả định chi phí nợ = r_f + 2%")
    w = wacc_override if wacc_override is not None else we * ke + (1 - we) * (kd if debt > 0 else 0) * (1 - t)

    hist = fcff(fin, a.default_tax_rate)
    base = float(hist.tail(3).mean())
    growth = historical_growth(fin, a.max_stage1_growth) if g1 is None else g1
    br = bridge(last)
    if base <= 0 or pd.isna(base):
        notes.append("FCFF bình quân 3 năm ≤ 0: DCF không có ý nghĩa với mã này")
        result = {"flows": [], "pv_flows": np.nan, "terminal_value": np.nan, "pv_terminal": np.nan,
                  "ev": np.nan}
    else:
        result = dcf_enterprise_value(base, growth, a.forecast_years, a.terminal_growth, w)
    eq = equity_from_ev(result["ev"], br)

    parent_equity = last["equity"] - (0 if pd.isna(last.get("minority_interest")) else last["minority_interest"])
    eps = last["net_income_parent"] / shares
    bvps = parent_equity / shares
    return Valuation(
        symbol=symbol.upper(), price=price, shares=shares, beta=b, beta_weeks=beta_weeks,
        beta_is_default=beta_is_default, cost_of_equity=ke, cost_of_debt=kd, tax_rate=t,
        weight_equity=we, wacc=w, g1=growth, g=a.terminal_growth, fcff_history=hist,
        fcff_base=base, dcf=result, net_debt_bridge=br, equity_value=eq,
        value_per_share=eq / shares, pe=price / eps if eps > 0 else np.nan,
        pb=price / bvps if bvps > 0 else np.nan, notes=notes,
    )


def sensitivity(v: Valuation, a: Assumptions, wacc_step: float = 0.01,
                growths: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.05)) -> pd.DataFrame:
    """Giá trị mỗi cổ phiếu (đồng) theo WACC (hàng) × tăng trưởng dài hạn g (cột)."""
    waccs = [v.wacc + k * wacc_step for k in range(-2, 3)]
    table = {}
    for g in growths:
        col = []
        for w in waccs:
            if v.fcff_base <= 0 or w <= g:
                col.append(np.nan)
                continue
            ev = dcf_enterprise_value(v.fcff_base, v.g1, a.forecast_years, g, w)["ev"]
            col.append(equity_from_ev(ev, v.net_debt_bridge) / v.shares)
        table[g] = col
    return pd.DataFrame(table, index=waccs)
