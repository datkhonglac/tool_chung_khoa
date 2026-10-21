"""Thẻ báo cáo một màn hình cho một mã — gom kết quả analysis.assess() thành 7 mục:
khuyến nghị, định giá, điểm số, cơ bản, kỹ thuật, dự báo 20 phiên, kế hoạch giao dịch.

Mọi con số lấy từ Assessment (đã tính bằng Python); thẻ chỉ trình bày, không tính lại mô hình.

Quy ước:
- Khuyến nghị hai trục (recommend/engine.py): XẾP HẠNG theo lợi nhuận kỳ vọng (upside + cổ tức) và
  THỜI ĐIỂM theo kỹ thuật + vùng mua. Khung: trung hạn (định giá 12 tháng, kế hoạch 20–60 phiên).
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
- ROE, P/E, P/B theo số liệu hiện tại (TTM 4 quý gần nhất nếu có — fa/ttm.py), ghi rõ kỳ.
- Cổ tức tiền 12 tháng = tổng cổ tức tiền mặt có ngày GDKHQ trong 365 ngày trước phiên giá; tỷ suất
  = cổ tức / giá hiện tại. Sở hữu nước ngoài / room tối đa, GTGD bình quân 1 tháng: hồ sơ VCI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from vnsa.config import settings

LABELS = {"MUA": "lợi nhuận kỳ vọng ≥ 15%", "KHẢ QUAN": "5–15%", "NẮM GIỮ": "−5% đến 5%",
          "KÉM KHẢ QUAN": "−15% đến −5%", "BÁN": "dưới −15%", "KHÔNG XẾP HẠNG": "thiếu BCTC"}


def vn_num(x: float, digits: int = 0, unit: str = "") -> str:
    """50000 -> '50.000', 0.6 -> '0,6' (kiểu Việt Nam)."""
    if x is None or pd.isna(x):
        return "—"
    s = f"{x:,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return s + unit


def _digits(x: float) -> int:
    """Số lẻ khi in %: số nhỏ (sát ngưỡng xếp hạng) in 1 chữ số để không ra "−0%"."""
    return 1 if not pd.isna(x) and abs(x) < 0.1 else 0


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
    timing: str = ""
    expected_return: float = float("nan")
    dividend_yield: float = 0.0
    regime: str = ""
    fundamentals: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    technical: list[str] = field(default_factory=list)
    forecast: list[str] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    basis: str = ""          # kỳ số liệu tài chính: "TTM Q2/2026" | "năm 2025"
    consensus: str = ""      # dòng đối chiếu giá mục tiêu CTCK (VCI tổng hợp)
    macro: list[str] = field(default_factory=list)   # mục VĨ MÔ & NGÀNH (tham khảo)


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


def consensus(a) -> tuple[float, str, float]:
    """(giá mục tiêu CTCK do VCI tổng hợp, khuyến nghị CTCK, độ lệch giá trị hợp lý của tool so
    với giá mục tiêu đó); NaN nếu nguồn không có."""
    nan = float("nan")
    pr = a.data.profile or {}
    target = pr.get("analyst_target")
    target = float(target) if target is not None and not pd.isna(target) and target > 0 else nan
    rating = str(pr.get("analyst_rating") or "")
    rating = "" if rating in ("None", "nan") else rating
    model = nan
    if a.fair is not None:
        model = a.fair.value if pd.isna(getattr(a.fair, "uncapped", nan)) else a.fair.uncapped
    gap = (model / target - 1) if target > 0 and not pd.isna(model) else nan
    return target, rating, gap


def _confidence(a, cfg: dict) -> tuple[str, dict[str, bool]]:
    fair = a.fair
    spread = (fair.p90 / fair.p10 if fair is not None and fair.p10 > 0 else float("inf"))
    _, _, gap = consensus(a)
    checks = {
        "có BCTC và điểm FA": not pd.isna(a.fa_score),
        # P/E và P/B lịch sử cùng gốc (bội số quá khứ của mã): cần thêm DCF/RI/ngành
        "≥ 2 phương pháp, có phương pháp độc lập": (fair is not None and len(fair.weights) >= 2
                                                     and fair.independent),
        "kịch bản định giá hẹp": spread <= cfg["max_value_spread"],
        "giá trị hợp lý không quá xa giá": (fair is not None
                                            and abs(fair.upside) <= cfg["max_abs_upside"]),
        "FA và TA cùng chiều": (not pd.isna(a.fa_score)
                                and (a.fa_score >= 50) == (a.ta.total >= 50)),
        "đủ dữ liệu giá": len(a.data.prices) >= cfg["min_sessions"],
    }
    n = sum(checks.values())
    # "Cao" đòi hỏi các kịch bản định giá đồng thuận — dải P10–P90 rộng thì tối đa "Trung bình"
    # và giá trị hợp lý không quá xa giá (chênh quá lớn thường do mô hình, không phải cơ hội)
    # và có phương pháp độc lập, không lệch quá xa giá mục tiêu của các CTCK (nếu nguồn có)
    near_consensus = pd.isna(gap) or abs(gap) <= cfg.get("max_consensus_gap", 0.3)
    level = ("Cao" if n >= cfg["high_confidence"] and checks["kịch bản định giá hẹp"]
             and checks["giá trị hợp lý không quá xa giá"]
             and checks["≥ 2 phương pháp, có phương pháp độc lập"] and near_consensus else
             "Trung bình" if n >= cfg["medium_confidence"] else "Thấp")
    return level, checks


def _signed(x: float, digits: int = 2) -> str:
    """+0,42 / −0,42 (dấu trừ kiểu chữ in)."""
    return f"{x:+.{digits}f}".replace(".", ",").replace("-", "−")


def _score_txt(m) -> str:
    """'Bất lợi (−0,42)' hoặc 'chưa đủ số liệu'."""
    if m.label == "Chưa đủ số liệu" or pd.isna(m.score):
        return "chưa đủ số liệu"
    return f"{m.label} ({_signed(m.score)})"


def macro_lines(mv, top: int = 4) -> list[str]:
    """Mục VĨ MÔ & NGÀNH của thẻ (mv = analysis.MacroView). Chỉ trình bày, số do macro/ tính."""
    from vnsa.macro.signals import fmt_change, fmt_period

    if mv is None:
        return ["Chưa có số vĩ mô trong kho (python -m vnsa.cli fetch-macro)"]
    groups = " · ".join(f"{g.scope} {_score_txt(g)}" for g in mv.groups.values())
    lines = [f"Bối cảnh Việt Nam: {_score_txt(mv.market)} — {groups}"]
    ind = mv.industry
    if ind is None:
        lines.append("Ngành: chưa có ánh xạ yếu tố vĩ mô (thêm ở config/macro_map.toml)")
    else:
        lines.append(f"Ngành {ind.scope}: {_score_txt(ind)}"
                     + (f", độ phủ {vn_pct(ind.coverage, 0)}" if ind.coverage < 1 else ""))
        used = sorted((d for d in ind.drivers if d.used), key=lambda d: -abs(d.contribution))
        for d in used[:top]:
            r = d.reading
            mark = "+" if d.contribution > 0 else "−" if d.contribution < 0 else "·"
            lines.append(f"  {mark} {r.name}: {fmt_change(r)} ({fmt_period(r)}, z "
                         f"{_signed(r.z, 1)}) · {d.why}")
        gaps = ([f"{d.reading.name if d.reading else d.series} (chưa có số)" for d in ind.missing]
                + [f"{d.reading.name} (số cũ {fmt_period(d.reading)})" for d in ind.stale])
        if gaps:
            lines.append("  ? Thiếu: " + ", ".join(gaps))
    sens = [x for x in getattr(mv, "sensitivity", []) if x.verdict != "thiếu lịch sử"]
    if sens:
        ok = sum(x.verdict == "đúng dấu" for x in sens)
        bad = [f"{x.name} (t {_signed(x.t, 1)})" for x in sens if x.conflict]
        lines.append(f"Kiểm chứng bằng {max(x.n for x in sens)} tháng giá: {ok}/{len(sens)} yếu tố "
                     "đúng dấu rõ" + (f"; NGƯỢC dấu: {', '.join(bad)} — xem lại macro_map.toml"
                                      if bad else ""))
    lines.append(f"Lãi suất phi rủi ro trong WACC: {vn_pct(mv.risk_free)} ({mv.rf_source})")
    news = getattr(mv, "news", None)
    if news is not None and len(news):
        lines.append("Tin gần đây (tiêu đề, chỉ tham khảo — hỏi Claude Desktop để tóm tắt):")
        for r in news.head(3).itertuples():
            title = r.title if len(r.title) <= 90 else r.title[:87] + "…"
            lines.append(f"  {r.date:%d/%m} {title}")
    return lines


def macro_guard(mv, cfg: dict | None = None) -> str | None:
    """Rào chắn vĩ mô: ngành Bất lợi rõ (điểm ≤ guard, độ phủ ≥ guard_coverage) -> lý do giảm độ
    tin cậy một bậc; None nếu không áp dụng. Không đổi xếp hạng (vĩ mô chưa được kiểm chứng)."""
    cfg = settings().get("macro", {}) if cfg is None else cfg
    ind = mv.industry if mv is not None else None
    if (ind is None or pd.isna(ind.score) or ind.score > cfg.get("guard", -0.5)
            or ind.coverage < cfg.get("guard_coverage", 0.6)):
        return None
    return (f"vĩ mô bất lợi rõ cho ngành ({_signed(ind.score)}): độ tin cậy giảm một bậc "
            "(không đổi xếp hạng)")


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
    guard = macro_guard(getattr(a, "macro", None))
    if guard:
        conf = {"Cao": "Trung bình", "Trung bình": "Thấp"}.get(conf, conf)
    nan = float("nan")

    # Cơ bản
    fundamentals, flags = [], []
    if not d.fin.empty:
        fin = d.fin.sort_values("year").reset_index(drop=True)
        ratios = compute_ratios(fin)
        cur = d.current
        roe = cur.roe if cur is not None else ratios["roe"].iloc[-1]
        ind_roe, n_peers = industry_roe(d.symbol, prof["industry"])
        roe_s = f"ROE {vn_pct(roe)}" + (f" (ngành {vn_pct(ind_roe)}, {n_peers} mã)"
                                        if not pd.isna(ind_roe) else "")
        span = min(3, len(fin) - 1)
        g = (sc.cagr(fin["net_income_parent"].iloc[-1 - span], fin["net_income_parent"].iloc[-1],
                     span) if span > 0 else nan)
        fundamentals.append(f"{roe_s} · Tăng trưởng EPS {span} năm {vn_pct(g)}/năm (BCTC năm)")
        f_score = nan
        if financial:
            eta = fin["equity"].iloc[-1] / fin["total_assets"].iloc[-1]
            fundamentals.append(f"Vốn chủ/tổng tài sản {vn_pct(eta)} · P/B {vn_num(x.pb, 2)} · "
                                f"P/E {vn_num(x.pe, 1)}")
            from vnsa.analysis import sector
            from vnsa.fa.sector import LABELS as SECTOR_LABELS

            sm = sector(d)
            if sm is not None:
                vals = [f"{SECTOR_LABELS[k]} {vn_num(v, 2) + ' lần' if k == 'margin_to_equity' else vn_pct(v)}"
                        for k, v in sm.values.items() if not pd.isna(v)]
                if vals:
                    fundamentals.append(f"{' · '.join(vals)} ({sm.basis}"
                                        + (f"; NPL/CASA/CAR nhập tay kỳ {sm.manual}"
                                           if sm.manual else "") + ")")
                if sm.missing:
                    fundamentals.append(f"Không có trong nguồn miễn phí: {', '.join(sm.missing)}"
                                        + (" — nhập tay ở config/bank_manual.toml"
                                           if sm.model == "bank" else ""))
                v = sm.values
                if v.get("credit_cost", 0) > 0.025:
                    flags.append(f"chi phí tín dụng cao ({vn_pct(v['credit_cost'])} dư nợ)")
                if v.get("margin_to_equity", 0) > 1.8:
                    flags.append(f"cho vay ký quỹ sát trần ({vn_num(v['margin_to_equity'], 2)} "
                                 "lần vốn chủ)")
        else:
            from vnsa.analysis import stock_ratios

            f_score = sc.piotroski(fin, settings()["fa_score"].get("issue_materiality", 0.01),
                                   stock_ratios(d))["f_score"].iloc[-1]
            cfo3, ni3 = fin["cfo"].tail(3).sum(), fin["net_income"].tail(3).sum()
            fundamentals.append(
                f"D/E {vn_num(ratios['debt_to_equity'].iloc[-1], 2)} · CFO/LNST "
                f"{vn_num(cfo3 / ni3 if ni3 > 0 else nan, 2)} · F-Score "
                f"{'—' if pd.isna(f_score) else f'{f_score:.0f}'}/9 · P/E {vn_num(x.pe, 1)} · "
                f"P/B {vn_num(x.pb, 2)}")
        flags = red_flags(fin, financial, a.z_em, f_score) + flags
    else:
        fundamentals.append("Chưa có BCTC")
    extra = []
    divs = d.dividends if d.dividends is not None else pd.DataFrame()
    if len(divs):
        recent = divs[(divs["exright_date"] > x.date - pd.Timedelta(days=365))
                      & (divs["exright_date"] <= x.date)]
        cash = recent["cash_per_share"].sum()
        stock = recent["stock_ratio"].dropna()
        if cash > 0:
            extra.append(f"Cổ tức tiền 12 tháng {vn_num(cash)}đ/cp ({vn_pct(cash / x.price)}"
                         + (f", cao bất thường — lợi nhuận kỳ vọng chỉ tính "
                            f"{vn_pct(x.dividend_yield)} (bình quân 3 năm, tối đa "
                            f"{vn_pct(settings()['recommend'].get('max_dividend_yield', 0.1), 0)})"
                            if x.dividend_yield < cash / x.price - 1e-9 else "") + ")")
        if len(stock):
            extra.append(f"cổ tức/thưởng cổ phiếu {vn_pct(stock.sum(), 0)}")
    pr = d.profile or {}
    if not pd.isna(pr.get("foreign_pct") or nan):
        room = pr.get("foreign_max_pct")
        extra.append(f"NN sở hữu {vn_pct(pr['foreign_pct'])}"
                     + (f"/room {vn_pct(room, 0)}" if room and not pd.isna(room) else ""))
    if not pd.isna(pr.get("avg_value_1m") or nan):
        extra.append(f"GTGD TB 1 tháng {vn_num(pr['avg_value_1m'] / 1e9, 1)} tỷ")
    if extra:
        fundamentals.append(" · ".join(extra))
    from vnsa.fa.ttm import earnings_acceleration

    acc = earnings_acceleration(d.quarterly)
    if acc is not None:
        fundamentals.append(f"Gia tốc LNST quý {acc[3]}: {vn_pct(acc[0], 0, sign=True)} (tăng trưởng cùng kỳ "
                            f"{vn_pct(acc[1], 0, sign=True)} so với quý trước {vn_pct(acc[2], 0, sign=True)}; tham khảo)")

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
    squeeze = last.get("bb_width_pct", nan)
    technical = [f"{trend}, RSI {vn_num(last['rsi'], 0)}"
                 + (f" · Độ nén giá {vn_pct(squeeze, 0)} 6 tháng (thấp = nền chặt)"
                    if not pd.isna(squeeze) else ""),
                 (f"Hỗ trợ {vn_num(sup[0].price) + 'đ' if sup else '—'} · "
                  f"Kháng cự {vn_num(res[0].price) + 'đ' if res else '— (vùng đỉnh)'}")]
    from vnsa.analysis import flow_summary

    fs = flow_summary(d)
    if fs is not None:
        def net(v, pct=float("nan")):
            if pd.isna(v):
                return "—"
            s = f"{'mua' if v >= 0 else 'bán'} ròng {vn_num(abs(v) / 1e9, 1)} tỷ"
            return s + (f" ({vn_pct(pct)} GTGD)" if not pd.isna(pct) else "")

        wins = settings().get("flows", {}).get("windows", [5, 20])
        technical.append("Khối ngoại " + " · ".join(
            f"{n} phiên {net(fs[f'foreign_{n}'], fs[f'foreign_{n}_pct'])}" for n in wins)
            + f" · Tự doanh {wins[-1]} phiên {net(fs[f'prop_{wins[-1]}'])}"
            + (f" · Tổ chức (NN + tự doanh) 10 phiên {vn_pct(fs['smart_10'], 1, sign=True)} GTGD"
               if not pd.isna(fs.get("smart_10", nan)) else "")
            + f" (đến {fs['last_date']:%d/%m}, tham khảo)")

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
                (f"Vùng mua {vn_num(z.low)}–{vn_num(z.high)}đ" if z.high > z.low
                 else f"Mua quanh {vn_num(z.low)}đ") if z is not None else
                f"Giá vào {vn_num(p.entry)}đ")
        source = p.stop_source.replace(" × ", "×").replace(",", ".")
        plan_lines.append(f"{zone} · Cắt lỗ {vn_num(p.stop)}đ ({source})")
        plan_lines.append("Chốt lời " + " / ".join(f"{vn_num(t.price)}" for t in p.targets) + "đ · "
                          f"Tỷ trọng gợi ý {vn_pct(p.position_value / p.capital, 0)} NAV")

    methods = " · ".join(f"{METHOD_NAMES[k]} {w:.0%}" for k, w in fair.weights.items()) if fair else "—"
    notes = [n for n in rec.reasons if any(k in n for k in (
        "mục tiêu gần nhất", "biến động cao", "hạ một bậc", "tối đa NẮM GIỮ",
        "thị trường chung đang giảm"))]
    if p is not None:
        notes += [n for n in p.notes if "giới hạn bởi" in n]
    notes += [n for n in (fair.notes if fair else [])
              if n.startswith("bỏ ") or any(k in n for k in (
                  "giá mục tiêu CTCK", "ROE hiện tại", "lợi nhuận đang giảm"))]
    target, rating, gap = consensus(a)
    cons_line = ""
    if not pd.isna(target):
        cons_line = (f"Giá mục tiêu CTCK (VCI tổng hợp): {vn_num(target)}đ"
                     + (f" · {rating}" if rating else "")
                     + (f" · giá trị mô hình của tool lệch {vn_pct(gap, 0, sign=True)}"
                        if not pd.isna(gap) else ""))
        capped = fair is not None and not pd.isna(getattr(fair, "uncapped", float("nan")))
        if (not capped and not pd.isna(gap)
                and abs(gap) > settings()["report"].get("max_consensus_gap", 0.3)):
            notes.append(f"giá trị hợp lý lệch {vn_pct(gap, 0, sign=True)} so với giá mục tiêu "
                         "CTCK: kiểm tra lại giả định (độ tin cậy không xếp Cao)")
    if fair is not None and not fair.independent:
        notes.append("chỉ có định giá theo bội số quá khứ của chính mã (thiếu DCF/ngành): tham "
                     "khảo thêm P/E ngành")
    if p is not None and p.zone is not None:
        notes += [n for n in p.zone.notes if "đuổi giá" in n or "xu hướng yếu" in n]
    if financial:
        notes.append("định chế tài chính: định giá bằng thu nhập thặng dư, P/B theo ROE")
    if guard:
        notes.append(guard)
    return Card(
        symbol=d.symbol, action=rec.action, label=LABELS.get(rec.action, ""),
        confidence=conf, confidence_checks=checks,
        analysis_date=datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%d/%m/%Y"),
        price_date=f"{x.date:%d/%m/%Y}", price=x.price,
        basis=d.current.basis if d.current is not None else "",
        p10=fair.p10 if fair else nan, p50=fair.value if fair else nan,
        p90=fair.p90 if fair else nan, target=fair.value if fair else nan,
        upside=fair.upside if fair else nan, methods=methods,
        fa=x.fa_score, ta=x.ta_score, quant=x.quant_score, total=rec.score,
        timing=rec.timing, expected_return=rec.expected_return,
        dividend_yield=x.dividend_yield, regime=x.regime,
        fundamentals=fundamentals, red_flags=flags, technical=technical, forecast=forecast,
        plan=plan_lines, notes=notes, consensus=cons_line,
        macro=macro_lines(getattr(a, "macro", None)))


def format_card(c: Card) -> str:
    lines = [
        (f"{c.symbol} — Xếp hạng: {c.action} · Thời điểm: {c.timing} · Độ tin cậy: "
         f"{c.confidence} · Khung: trung hạn"),
        (f"Ngày phân tích: {c.analysis_date} · Giá hiện tại: {vn_num(c.price)}đ (phiên {c.price_date})"
         + (f" · Số liệu tài chính: {c.basis}" if c.basis else "")),
        "",
        "ĐỊNH GIÁ",
        f"  Giá trị nội tại (P10 / P50 / P90): {vn_num(c.p10)} / {vn_num(c.p50)} / {vn_num(c.p90)}đ",
        (f"  Giá mục tiêu 12 tháng: {vn_num(c.target)}đ → Upside {vn_pct(c.upside, _digits(c.upside), sign=True)}"
         + (f" + cổ tức {vn_pct(c.dividend_yield)}" if c.dividend_yield > 0 else "")
         + f" = lợi nhuận kỳ vọng {vn_pct(c.expected_return, _digits(c.expected_return), sign=True)}"),
        f"  Phương pháp: {c.methods}",
        *([f"  {c.consensus}"] if c.consensus else []),
        "",
        "ĐIỂM SỐ (0-100)",
        (f"  FA (chất lượng) {vn_num(c.fa)} · TA (xu hướng) {vn_num(c.ta)} · "
         f"Thị trường: VNINDEX {c.regime}"),
        "",
        "CƠ BẢN",
        *[f"  {s}" for s in c.fundamentals],
        "  " + ("Không có cờ đỏ" if not c.red_flags else "Cờ đỏ: " + "; ".join(c.red_flags)),
        "",
        "KỸ THUẬT",
        *[f"  {s}" for s in c.technical],
    ]
    if c.macro:
        lines += ["", "VĨ MÔ & NGÀNH (tham khảo, chưa vào điểm)", *[f"  {s}" for s in c.macro]]
    if c.forecast:
        lines += ["", "DỰ BÁO 20 PHIÊN", *[f"  {s}" for s in c.forecast]]
    if c.plan:
        lines += ["", "KẾ HOẠCH GIAO DỊCH", *[f"  {s}" for s in c.plan]]
    if c.notes:
        lines += ["", *[f"  ! {n}" for n in c.notes]]
    lines += ["", "Sàng lọc theo quy tắc trong config/settings.toml, không phải lời khuyên đầu tư."]
    return "\n".join(lines)
