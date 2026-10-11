"""Thẻ báo cáo một màn hình cho một mã — gom kết quả analysis.assess() thành 7 mục:
khuyến nghị, định giá, điểm số, cơ bản, kỹ thuật, dự báo 20 phiên, kế hoạch giao dịch.

Mọi con số lấy từ Assessment (đã tính bằng Python); thẻ chỉ trình bày, không tính lại mô hình.

Quy ước:
- Khuyến nghị: MUA = Khả quan, THEO DÕI = Trung lập, TRÁNH = Kém khả quan. Khung: trung hạn (kế
  hoạch, mô phỏng tính theo 20–60 phiên, định giá theo năm).
- Độ tin cậy = số điều kiện đạt trong 6 điều kiện (ngưỡng ở mục [report]):
  1. có BCTC và điểm FA; 2. có ≥ 2 phương pháp định giá; 3. kịch bản định giá hẹp (P90 / P10 ≤
  `max_value_spread`); 4. giá trị hợp lý không lệch quá `max_abs_upside` so với giá (lệch quá xa
  thường là mô hình không hợp với mã); 5. FA và TA cùng chiều (cùng ≥ 50 hoặc cùng < 50);
  6. có ≥ `min_sessions` phiên giá. ≥ `high_confidence` → Cao, ≥ `medium_confidence` → Trung bình, còn lại Thấp.
- Giá trị nội tại P10 / P50 / P90: P50 = giá trị hợp lý (bình quân có trọng số các phương pháp),
  P10 / P90 = phân vị 10% / 90% các kịch bản (độ nhạy DCF, P/E và P/B từng năm lịch sử; định chế tài
  chính: k_e ± 1%, độ bền ROE). Giá mục tiêu 12 tháng = giá trị hợp lý.
- Tăng trưởng EPS 3 năm = tăng trưởng kép LNST công ty mẹ (bằng EPS điều chỉnh theo số cp hiện tại).
- ROE ngành = trung vị ROE năm gần nhất của các mã cùng ngành ICB đã có BCTC trong kho (cần ≥ 3 mã).
- Cờ đỏ: Altman Z'' < 1,1; CFO/LNST 3 năm < 0,5; vay/VCSH > 2; F-Score ≤ 3; LNST âm; LNST cty mẹ
  giảm 2 năm liên tiếp; tiền thu phát hành cổ phiếu > 10% vốn chủ (pha loãng). Định chế tài chính:
  ROE < 8%, LNST âm hoặc giảm 2 năm liên tiếp.
- Tỷ trọng gợi ý = giá trị vị thế của kế hoạch (rủi ro 1% vốn/lệnh) / vốn trong [plan].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from vnsa.config import settings

LABELS = {"MUA": "Khả quan", "THEO DÕI": "Trung lập", "TRÁNH": "Kém khả quan"}


def vn_num(x: float, digits: int = 0, unit: str = "") -> str:
    """50000 -> '50.000', 0.6 -> '0,6' (kiểu Việt Nam)."""
    if x is None or pd.isna(x):
        return "—"
    s = f"{x:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return s + unit


def vn_pct(x: float, digits: int = 1, sign: bool = False) -> str:
    if x is None or pd.isna(x):
        return "—"
    s = vn_num(abs(x) * 100, digits) + "%"
    return ("+" if x >= 0 else "−") + s if sign else ("−" + s if x < 0 else s)


@dataclass
class Card:
    symbol: str
    action: str
    label: str
    confidence: str
    confidence_checks: dict[str, bool]
    analysis_date: str
    price_date: str
    price: float
    p10: float
    p50: float
    p90: float
    target: float
    upside: float
    methods: str
    fa: float
    ta: float
    quant: float
    total: float
    fundamentals: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    technical: list[str] = field(default_factory=list)
    forecast: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def red_flags(fin: pd.DataFrame, financial: bool, z_em: float, f_score: float) -> list[str]:
    from vnsa.fa.ratios import compute_ratios

    if fin.empty:
        return []
    df = fin.sort_values("year").reset_index(drop=True)
    r = compute_ratios(df)
    last = df.iloc[-1]
    flags = []
    ni = df["net_income_parent"]
    if last["net_income_parent"] < 0:
        flags.append("LNST công ty mẹ âm")
    if len(ni) >= 3 and ni.iloc[-1] < ni.iloc[-2] < ni.iloc[-3]:
        flags.append("LNST công ty mẹ giảm 2 năm liên tiếp")
    if financial:
        if r["roe"].iloc[-1] < 0.08:
            flags.append(f"ROE thấp ({vn_pct(r['roe'].iloc[-1])})")
        return flags
    if not pd.isna(z_em) and z_em < 1.1:
        flags.append(f"Altman Z'' {vn_num(z_em, 2)} vùng nguy cơ")
    cfo3, ni3 = df["cfo"].tail(3).sum(), df["net_income"].tail(3).sum()
    if ni3 > 0 and cfo3 / ni3 < 0.5:
        flags.append(f"lợi nhuận ít tiền về (CFO/LNST 3 năm {vn_num(cfo3 / ni3, 2)})")
    if r["debt_to_equity"].iloc[-1] > 2:
        flags.append(f"đòn bẩy cao (vay/VCSH {vn_num(r['debt_to_equity'].iloc[-1], 1)})")
    if not pd.isna(f_score) and f_score <= 3:
        flags.append(f"F-Score thấp ({f_score:.0f}/9)")
    issued = last.get("equity_issued")
    if not pd.isna(issued) and last["equity"] > 0 and issued > 0.10 * last["equity"]:
        flags.append(f"phát hành cổ phiếu thu tiền {vn_pct(issued / last['equity'], 0)} vốn chủ "
                     "(pha loãng)")
    return flags


def industry_roe(symbol: str, industry: str | None) -> tuple[float, int]:
    """(trung vị ROE năm gần nhất của các mã cùng ngành trong kho, số mã); cần ≥ 3 mã."""
    from vnsa.config import path
    from vnsa.data.store import Store
    from vnsa.fa.ratios import compute_ratios

    if not industry or not path("warehouse").exists():
        return float("nan"), 0
    with Store(path("warehouse")) as store:
        peers = [s for s in store.peers(industry) if s != symbol.upper()]
        roes = [compute_ratios(store.fundamentals(s))["roe"].iloc[-1] for s in peers]
    roes = [x for x in roes if not pd.isna(x)]
    return (float(np.median(roes)), len(roes)) if len(roes) >= 3 else (float("nan"), len(roes))


def _confidence(a, cfg: dict) -> tuple[str, dict[str, bool]]:
    fair = a.fair
    spread = (fair.p90 / fair.p10 if fair is not None and fair.p10 > 0 else float("inf"))
    checks = {
        "có BCTC và điểm FA": not pd.isna(a.fa_score),
        "≥ 2 phương pháp định giá": fair is not None and len(fair.weights) >= 2,
        "kịch bản định giá hẹp": spread <= cfg["max_value_spread"],
        "giá trị hợp lý không quá xa giá": (fair is not None
                                            and abs(fair.upside) <= cfg["max_abs_upside"]),
        "FA và TA cùng chiều": (not pd.isna(a.fa_score)
                                and (a.fa_score >= 50) == (a.ta.total >= 50)),
        "đủ dữ liệu giá": len(a.data.prices) >= cfg["min_sessions"],
    }
    n = sum(checks.values())
    level = ("Cao" if n >= cfg["high_confidence"] else
             "Trung bình" if n >= cfg["medium_confidence"] else "Thấp")
    return level, checks


def build_card(a) -> Card:
    """a = analysis.Assessment."""
    from vnsa import market
    from vnsa.fa import scores as sc
    from vnsa.fa.fair_value import METHOD_NAMES
    from vnsa.fa.ratios import compute_ratios

    d, rec, x, fair, p, r = a.data, a.rec, a.rec.inputs, a.fair, a.plan, a.risk
    prof = market.profile(d.symbol)
    financial = market.financial_note(d.symbol) is not None
    conf, checks = _confidence(a, settings()["report"])
    nan = float("nan")

    # Cơ bản
    fundamentals, flags = [], []
    if not d.fin.empty:
        fin = d.fin.sort_values("year").reset_index(drop=True)
        ratios = compute_ratios(fin)
        roe = ratios["roe"].iloc[-1]
        ind_roe, n_peers = industry_roe(d.symbol, prof["industry"])
        roe_s = f"ROE {vn_pct(roe)}" + (f" (ngành {vn_pct(ind_roe)}, {n_peers} mã)"
                                        if not pd.isna(ind_roe) else "")
        span = min(3, len(fin) - 1)
        g = (sc.cagr(fin["net_income_parent"].iloc[-1 - span], fin["net_income_parent"].iloc[-1],
                     span) if span > 0 else nan)
        fundamentals.append(f"{roe_s} · Tăng trưởng EPS {span} năm {vn_pct(g)}/năm")
        f_score = nan
        if financial:
            eta = fin["equity"].iloc[-1] / fin["total_assets"].iloc[-1]
            fundamentals.append(f"Vốn chủ/tổng tài sản {vn_pct(eta)} · P/B {vn_num(x.pb, 2)} · "
                                f"P/E {vn_num(x.pe, 1)}")
        else:
            f_score = sc.piotroski(fin)["f_score"].iloc[-1]
            cfo3, ni3 = fin["cfo"].tail(3).sum(), fin["net_income"].tail(3).sum()
            fundamentals.append(
                f"D/E {vn_num(ratios['debt_to_equity'].iloc[-1], 2)} · CFO/LNST "
                f"{vn_num(cfo3 / ni3 if ni3 > 0 else nan, 2)} · F-Score "
                f"{'—' if pd.isna(f_score) else f'{f_score:.0f}'}/9 · P/E {vn_num(x.pe, 1)} · "
                f"P/B {vn_num(x.pb, 2)}")
        flags = red_flags(fin, financial, a.z_em, f_score)
    else:
        fundamentals.append("Chưa có BCTC")

    # Kỹ thuật
    last = a.ind.iloc[-1]
    c, s50, s200 = last["close"], last.get("sma50", nan), last.get("sma200", nan)
    if c > s50 > s200:
        trend = "Xu hướng tăng (giá > MA50 > MA200)"
    elif c < s50 < s200:
        trend = "Xu hướng giảm (giá < MA50 < MA200)"
    else:
        trend = "Đi ngang / chưa rõ xu hướng"
    from vnsa.analysis import levels

    sup, res = levels(a.ind)
    technical = [f"{trend}, RSI {vn_num(last['rsi'], 0)}",
                 (f"Hỗ trợ {vn_num(sup[0].price) + 'đ' if sup else '—'} · "
                  f"Kháng cự {vn_num(res[0].price) + 'đ' if res else '— (vùng đỉnh)'}")]

    # Dự báo 20 phiên
    forecast = []
    if r is not None:
        h = 20 if r.sim.paths.shape[1] > 20 else r.sim.paths.shape[1] - 1
        pc = r.sim.percentiles(h)
        var95 = r.var.get(0.95, next(iter(r.var.values())))["hist"]
        forecast = [(f"Dải giá: {vn_num(pc[10])} – {vn_num(pc[50])} – {vn_num(pc[90])}đ · "
                     f"Xác suất tăng {vn_pct(r.sim.prob_above(day=h), 0)}"),
                    f"VaR 95% (1 ngày): {vn_pct(-var95, 1)}"]

    # Kế hoạch
    plan_lines = []
    if p is not None:
        z = p.zone
        zone = (f"Chờ giá ≤ {vn_num(z.high)}đ" if z is not None and z.status == "CHỜ GIÁ" else
                f"Vùng mua {vn_num(z.low)}–{vn_num(z.high)}đ" if z is not None else
                f"Giá vào {vn_num(p.entry)}đ")
        source = p.stop_source.replace(" × ", "×").replace(",", ".")
        plan_lines.append(f"{zone} · Cắt lỗ {vn_num(p.stop)}đ ({source})")
        plan_lines.append("Chốt lời " + " / ".join(f"{vn_num(t.price)}" for t in p.targets) + "đ · "
                          f"Tỷ trọng gợi ý {vn_pct(p.position_value / p.capital, 0)} NAV")

    methods = " · ".join(f"{METHOD_NAMES[k]} {w:.0%}" for k, w in fair.weights.items()) if fair else "—"
    notes = [n for n in rec.reasons if "mục tiêu gần nhất" in n or "biến động cao" in n]
    notes += [n for n in (fair.notes if fair else []) if n.startswith("bỏ ")]
    if p is not None and p.zone is not None:
        notes += [n for n in p.zone.notes if "đuổi giá" in n or "xu hướng yếu" in n]
    if financial:
        notes.append("định chế tài chính: định giá bằng thu nhập thặng dư, P/B theo ROE")
    return Card(
        symbol=d.symbol, action=rec.action, label=LABELS.get(rec.action, ""),
        confidence=conf, confidence_checks=checks,
        analysis_date=datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%d/%m/%Y"),
        price_date=f"{x.date:%d/%m/%Y}", price=x.price,
        p10=fair.p10 if fair else nan, p50=fair.value if fair else nan,
        p90=fair.p90 if fair else nan, target=fair.value if fair else nan,
        upside=fair.upside if fair else nan, methods=methods,
        fa=x.fa_score, ta=x.ta_score, quant=x.quant_score, total=rec.score,
        fundamentals=fundamentals, red_flags=flags, technical=technical, forecast=forecast,
        plan=plan_lines, notes=notes)


def format_card(c: Card) -> str:
    lines = [
        (f"{c.symbol} — Khuyến nghị: {c.action} ({c.label}) · Độ tin cậy: {c.confidence} · "
         "Khung: trung hạn"),
        f"Ngày phân tích: {c.analysis_date} · Giá hiện tại: {vn_num(c.price)}đ (phiên {c.price_date})",
        "",
        "ĐỊNH GIÁ",
        f"  Giá trị nội tại (P10 / P50 / P90): {vn_num(c.p10)} / {vn_num(c.p50)} / {vn_num(c.p90)}đ",
        f"  Giá mục tiêu 12 tháng: {vn_num(c.target)}đ → Upside {vn_pct(c.upside, 0, sign=True)}",
        f"  Phương pháp: {c.methods}",
        "",
        "ĐIỂM SỐ (0-100)",
        f"  FA {vn_num(c.fa)} · TA {vn_num(c.ta)} · Quant {vn_num(c.quant)} → Tổng hợp {vn_num(c.total)}",
        "",
        "CƠ BẢN",
        *[f"  {s}" for s in c.fundamentals],
        "  " + ("Không có cờ đỏ" if not c.red_flags else "Cờ đỏ: " + "; ".join(c.red_flags)),
        "",
        "KỸ THUẬT",
        *[f"  {s}" for s in c.technical],
    ]
    if c.forecast:
        lines += ["", "DỰ BÁO 20 PHIÊN", *[f"  {s}" for s in c.forecast]]
    if c.plan:
        lines += ["", "KẾ HOẠCH GIAO DỊCH", *[f"  {s}" for s in c.plan]]
    if c.notes:
        lines += ["", *[f"  ! {n}" for n in c.notes]]
    lines += ["", "Sàng lọc theo quy tắc trong config/settings.toml, không phải lời khuyên đầu tư."]
    return "\n".join(lines)
