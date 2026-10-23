"""MCP server: cho Claude Desktop gọi các phép phân tích của tool như "công cụ".

Chạy:  python -m vnsa.mcp_server        (giao tiếp qua stdio, Claude Desktop tự khởi động)
Cài:   pip install -e ".[mcp]"

Mọi con số do Python tính từ kho DuckDB (data/warehouse.duckdb); Claude chỉ đọc và diễn giải.
Tiền tính bằng đồng, tỷ lệ dạng thập phân (0.25 = 25%). Ô trống (NaN) trả về null.
Hướng dẫn cấu hình Claude Desktop: README, mục "MCP server".
"""

from __future__ import annotations

import contextlib
import functools
import math
import sys
from dataclasses import asdict, is_dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from vnsa import analysis
from vnsa.config import settings

try:                                   # mcp 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:                    # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

INSTRUCTIONS = (
    "Công cụ phân tích cổ phiếu Việt Nam (mọi mã HOSE, HNX, UPCoM). Mọi số liệu do các tool này "
    "tính từ dữ liệu đã tải về máy; không tự đoán hay bịa số liệu thị trường. Giá tính bằng đồng, "
    "đã điều chỉnh cổ tức/chia tách; tỷ lệ là số thập phân (0.25 = 25%). Khi người dùng hỏi về một "
    "mã, gọi analyze_symbol trước: tool tự tải giá (CafeF), BCTC và sàn/ngành (vnstock) nếu kho "
    "chưa có hoặc đã cũ. Ngân hàng, chứng khoán, bảo hiểm không có DCF/điểm FA (mô hình không áp "
    "dụng). Hỏi nhóm ngành hay mã tiềm năng trên toàn thị trường: gọi screen_market (đọc lượt sàng "
    "lọc gần nhất đã lưu). Kết quả là sàng lọc theo quy tắc, không phải lời khuyên đầu tư."
)
mcp = _Server("vn-stock-analyzer", instructions=INSTRUCTIONS)


def clean(x):
    """Đổi kết quả về kiểu JSON: NaN -> None, numpy -> số Python, ngày -> chuỗi ISO."""
    if is_dataclass(x) and not isinstance(x, type):
        return clean(asdict(x))
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, pd.DataFrame):
        return clean(x.to_dict(orient="records"))
    if isinstance(x, pd.Series):
        return clean(x.to_dict())
    if isinstance(x, (pd.Timestamp, datetime, date)):
        return x.strftime("%Y-%m-%d")
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _guard(fn):
    """Lỗi thiếu dữ liệu / không tính được -> {"error": "..."} thay vì làm hỏng phiên chat."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return clean(fn(*args, **kwargs))
        except (analysis.DataMissing, ValueError) as e:
            return {"error": str(e)}
    return wrapper


def _update(symbol: str, refresh: bool = False):
    # Thư viện tải dữ liệu có thể in ra stdout, mà stdout là kênh giao tiếp với Claude Desktop
    with contextlib.redirect_stdout(sys.stderr):
        return analysis.update(symbol, refresh=refresh)


@mcp.tool()
@_guard
def update_symbol(symbol: str, refresh: bool = False) -> dict:
    """Tải/cập nhật dữ liệu một mã bất kỳ (HOSE/HNX/UPCoM): giá 5 năm từ CafeF (lần sau chỉ tải
    phiên mới), BCTC năm và sàn/ngành từ vnstock. refresh=True: tải lại BCTC dù chưa cũ.
    Lần đầu mất khoảng 10–30 giây."""
    from vnsa import market

    rep = _update(symbol, refresh)
    return {"symbol": rep.symbol, "ready": rep.ok, "errors": rep.errors,
            "steps": [{"step": s.what, "status": s.status, "detail": s.detail} for s in rep.steps],
            "profile": market.profile(rep.symbol)}


@mcp.tool()
@_guard
def analyze_symbol(symbol: str) -> dict:
    """Phân tích nhanh MỘT MÃ BẤT KỲ: tự cập nhật dữ liệu nếu thiếu/cũ, rồi trả `report` — thẻ báo
    cáo đã định dạng (khuyến nghị, độ tin cậy, giá trị nội tại P10/P50/P90, điểm FA/TA/Quant, cơ bản,
    cờ đỏ, kỹ thuật, dự báo 20 phiên, vùng mua, cắt lỗ, chốt lời, tỷ trọng). Hãy trình bày nguyên
    văn `report` cho người dùng rồi mới diễn giải. Chi tiết hơn: get_valuation, get_technical..."""
    from vnsa import market
    from vnsa.report.card import build_card, format_card

    rep = _update(symbol)
    if not rep.ok:
        return {"symbol": rep.symbol, "error": "không có giá để phân tích: " + "; ".join(rep.errors)}
    a = analysis.assess(rep.symbol)
    rec, plan = a.rec, a.plan
    analysis.record_signals([(rec, plan)])
    card = build_card(a)
    return {"symbol": rec.symbol, "report": format_card(card),
            "confidence": card.confidence, "confidence_checks": card.confidence_checks,
            "intrinsic_value": {"p10": card.p10, "p50": card.p50, "p90": card.p90},
            "scores": {"fa": card.fa, "ta": card.ta, "quant": card.quant, "total": card.total},
            "red_flags": card.red_flags, "macro": card.macro, "profile": market.profile(rec.symbol),
            "band": market.band(rec.symbol), "action": rec.action, "score": rec.score,
            "reasons": rec.reasons, "inputs": rec.inputs, "plan": plan,
            "data_update": [str(s).strip() for s in rep.steps],
            "disclaimer": "Sàng lọc theo quy tắc trong settings.toml, không phải lời khuyên đầu tư."}


@mcp.tool()
@_guard
def list_symbols() -> dict:
    """Danh sách mã đang có trong kho dữ liệu và danh sách theo dõi trong cấu hình."""
    from vnsa.config import path
    from vnsa.data.store import Store

    with Store(path("warehouse")) as store:
        inv = store.inventory()
    return {"warehouse": inv, "watchlist": settings()["watchlist"]["symbols"]}


@mcp.tool()
@_guard
def get_prices(symbol: str, sessions: int = 60) -> dict:
    """Giá ngày gần nhất của một mã (mở, cao, thấp, đóng cửa bằng đồng; khối lượng khớp lệnh).
    sessions: số phiên gần nhất (tối đa 1500)."""
    d = analysis.load(symbol, need_fin=False)
    df = d.prices.tail(max(1, min(int(sessions), 1500)))
    return {"symbol": d.symbol, "prices": df.drop(columns="symbol")}


@mcp.tool()
@_guard
def get_ratios(symbol: str) -> dict:
    """Chỉ số tài chính năm: ROE, ROA, biên gộp/ròng, vay/VCSH, CFO/LNST, tăng trưởng EPS và
    LNST công ty mẹ (tỷ lệ thập phân), cùng các khoản mục BCTC chuẩn hoá (đồng)."""
    from vnsa.fa.ratios import compute_ratios

    d = analysis.load(symbol, need_prices=False)
    return {"symbol": d.symbol, "ratios": compute_ratios(d.fin),
            "statements": d.fin.drop(columns="symbol")}


@mcp.tool()
@_guard
def get_valuation(symbol: str, growth: float | None = None, wacc: float | None = None) -> dict:
    """Định giá. Doanh nghiệp thường: beta, WACC, FCFF, DCF hai giai đoạn, ma trận độ nhạy (hàng
    WACC, cột tăng trưởng dài hạn), P/E, P/B; growth/wacc ghi đè giả định. Ngân hàng/chứng khoán/
    bảo hiểm: thu nhập thặng dư, P/B hợp lý theo ROE, P/B lịch sử. Mọi mã đều có fair_value = trung
    vị các phương pháp (kể cả P/E, P/B lịch sử của chính mã)."""
    from vnsa.fa import valuation as val
    from vnsa.market import financial_note

    d = analysis.load(symbol)
    if financial_note(d.symbol):
        _, fair = analysis.financial(d)
        ke, b, raw, _ = analysis.cost_of_equity(d)
        return {"symbol": d.symbol, "model": "residual_income", "price": fair.price,
                "cost_of_equity": ke, "beta": b, "raw_beta": raw, "residual_income": fair.ri,
                "fair_value": fair.value, "upside": fair.upside, "methods": fair.methods,
                "pb_hist_target": fair.pb_target, "pb_hist_median": fair.pb_median,
                "notes": fair.notes + [financial_note(d.symbol)]}
    a, v, raw_beta = analysis.valuation(d, growth, wacc)
    fair = analysis.general_fair_value(d, v)
    table = val.sensitivity(v, a)
    return {
        "symbol": d.symbol, "price": v.price, "shares": v.shares, "pe": v.pe, "pb": v.pb,
        "beta": v.beta, "raw_beta": raw_beta, "cost_of_equity": v.cost_of_equity,
        "cost_of_debt": v.cost_of_debt, "tax_rate": v.tax_rate, "wacc": v.wacc,
        "g1": v.g1, "terminal_growth": v.g,
        "growth_path": val.growth_path(v.g1, v.g, a.forecast_years, a.fade_growth),
        "fcff_history": {int(k): x for k, x in v.fcff_history.items()},
        "fcff_base": v.fcff_base, "dcf": {k: x for k, x in v.dcf.items() if k != "flows"},
        "bridge": v.net_debt_bridge, "equity_value": v.equity_value,
        "value_per_share": v.value_per_share, "upside": v.upside, "notes": v.notes + fair.notes,
        "fair_value": fair.value, "fair_upside": fair.upside, "fair_methods": fair.methods,
        "pe_hist_target": fair.pe_target, "pb_hist_target": fair.pb_target,
        "pe_hist_median": fair.pe_median, "pb_hist_median": fair.pb_median,
        "sensitivity": {f"wacc={w:.4f}": {f"g={g:.3f}": table.at[w, g] for g in table.columns}
                        for w in table.index},
        "assumptions": asdict(a),
    }


@mcp.tool()
@_guard
def get_fa_score(symbol: str) -> dict:
    """Piotroski F-Score từng năm (9 tiêu chí), Altman Z'' và Z gốc, điểm FA 0–100 theo nhóm.
    Ngân hàng/chứng khoán/bảo hiểm: điểm FA theo bộ chỉ tiêu riêng (ROE, tăng trưởng, đệm vốn,
    định giá)."""
    from vnsa.fa.scores import FSCORE_CRITERIA, z_zone
    from vnsa.market import financial_note

    d = analysis.load(symbol, need_prices=False)
    if financial_note(d.symbol):
        s, _ = analysis.financial(analysis.load(symbol))
        return {"symbol": d.symbol, "model": "financial", "fa_score": s.total,
                "rating": s.rating, "groups": s.groups,
                "items": {k: {"value": v, "points": p} for k, (v, p) in s.items.items()},
                "notes": s.notes + [financial_note(d.symbol)]}
    f, z, s, _ = analysis.fa(d)
    z = z.assign(zone_em=z["z_em"].map(z_zone), zone_orig=z["z_orig"].map(lambda v: z_zone(v, "orig")))
    return {"symbol": d.symbol, "fscore_criteria": FSCORE_CRITERIA, "piotroski": f,
            "altman": z, "fa_score": s.total, "rating": s.rating, "groups": s.groups,
            "items": {k: {"value": v, "points": p} for k, (v, p) in s.items.items()},
            "notes": s.notes}


@mcp.tool()
@_guard
def get_technical(symbol: str) -> dict:
    """Chỉ báo kỹ thuật phiên gần nhất (SMA, RSI, MACD, Bollinger, ATR, ADX), vùng hỗ trợ/kháng cự
    và điểm TA 0–100 theo nhóm."""
    d = analysis.load(symbol, need_fin=False)
    ind = analysis.indicators(d.prices)
    t = analysis.ta(ind)
    sup, res = analysis.levels(ind)
    last = ind.iloc[-1]
    return {"symbol": d.symbol, "date": last["date"], "indicators": last.drop("symbol"),
            "ta_score": t.total, "rating": t.rating, "groups": t.groups,
            "supports": sup[:5], "resistances": res[:5]}


@mcp.tool()
@_guard
def get_trade_plan(symbol: str) -> dict:
    """Kế hoạch mua tham khảo. plan.zone = VÙNG MUA khuyến nghị (neo hỗ trợ / VWAP20 / SMA20, chặn
    bởi giá trị hợp lý trừ biên an toàn; status "CHỜ GIÁ" nếu giá đang quá cao); giá vào = giữa vùng
    (2 lệnh LO), cắt lỗ theo ATR/hỗ trợ, mục tiêu, khối lượng theo % rủi ro vốn, bước giá theo sàn."""
    d = analysis.load(symbol, need_fin=False)
    p = analysis.plan(analysis.indicators(d.prices))
    return {"symbol": d.symbol, "plan": p, "risk_per_share": p.risk_per_share,
            "position_value": p.position_value, "max_loss": p.max_loss, "stop_pct": p.stop_pct}


@mcp.tool()
@_guard
def get_risk(symbol: str, sessions: int | None = None) -> dict:
    """Rủi ro: GARCH(1,1), biến động năm, VaR/CVaR 1 ngày, Monte Carlo P10–P90 sau các kỳ (mặc
    định 20 và 60 phiên), xác suất chạm cắt lỗ/mục tiêu của kế hoạch. sessions: một kỳ tuỳ chọn."""
    d = analysis.load(symbol, need_fin=False)
    r = analysis.risk(d.prices, [int(sessions)] if sessions else None)
    sim = r.sim
    return {
        "symbol": d.symbol, "price": r.price,
        "garch": {"mu_pct": r.fit.mu, "omega": r.fit.omega, "alpha": r.fit.alpha,
                  "beta": r.fit.beta, "persistence": r.fit.persistence,
                  "half_life_sessions": r.fit.half_life, "converged": r.fit.converged},
        "vol_annual": {"current": r.fit.current_vol_annual, "long_run": r.fit.long_run_vol_annual,
                       "historical_1y": r.hist_vol},
        "var_1d": {f"{q:.0%}": v for q, v in r.var.items()},
        "monte_carlo": {f"{h}_sessions": {"percentiles": sim.percentiles(h),
                                          "prob_above_now": sim.prob_above(day=h),
                                          "var": {f"{q:.0%}": sim.var(q, h) for q in r.var}}
                        for h in r.horizons},
        "plan_first_touch": r.first_touch,
    }


@mcp.tool()
@_guard
def get_recommendations(symbols: list[str] | None = None) -> dict:
    """Khuyến nghị hai trục kèm lý do: xếp hạng MUA / KHẢ QUAN / NẮM GIỮ / KÉM KHẢ QUAN / BÁN theo lợi
    nhuận kỳ vọng (upside so với giá trị hợp lý + cổ tức, có rào chắn chất lượng và thị trường
    chung), và thời điểm mua (kỹ thuật + vùng mua).
    Bỏ trống symbols = danh sách theo dõi."""
    out, errors, recorded = [], [], []
    for sym in symbols or settings()["watchlist"]["symbols"]:
        try:
            rec, plan = analysis.recommendation(sym)
        except (analysis.DataMissing, ValueError) as e:
            errors.append(str(e))
            continue
        out.append({"symbol": rec.symbol, "action": rec.action, "timing": rec.timing,
                    "expected_return": rec.expected_return, "score": rec.score,
                    "reasons": rec.reasons, "inputs": rec.inputs,
                    "buy_zone": plan.zone if plan is not None else None})
        recorded.append((rec, plan))
    analysis.record_signals(recorded)
    return {"recommendations": out, "errors": errors,
            "disclaimer": "Sàng lọc theo quy tắc trong settings.toml, không phải lời khuyên đầu tư."}


@mcp.tool()
@_guard
def run_backtest(symbol: str, entry_score: float | None = None,
                 exit_score: float | None = None) -> dict:
    """Backtest các quy tắc thời điểm (điểm TA ≥ entry_score mua / < exit_score bán; + lọc
    VNINDEX; quy tắc đang dùng cho trục thời điểm) với khớp lệnh sát thực tế VN (mở cửa phiên sau,
    trần/sàn, T+2, phí và thuế), so với mua và nắm giữ."""
    d = analysis.load(symbol, need_fin=False)
    results = analysis.backtests(d, entry_score, exit_score)
    first = next(iter(results.values()))
    last = list(results.values())[-1]
    return {"symbol": d.symbol, "rules": analysis.backtest_config(d.symbol),
            "timing_rule": settings()["recommend"].get("timing_rule", "ta"),
            "strategies": {k: r.stats for k, r in results.items()},
            "buy_and_hold": first.benchmark_stats,
            "trades": [dict(asdict(t), ret=t.ret) for t in last.trades]}


@mcp.tool()
@_guard
def review_signals(horizons: list[int] | None = None) -> dict:
    """Chấm lại các khuyến nghị đã lưu (mỗi lần recommend/analyze tự lưu) sau N phiên (mặc định 5,
    20, 60): lãi/lỗ bình quân, vượt trội so với VNINDEX, tỷ lệ thắng, tỷ lệ chạm mục tiêu trước cắt
    lỗ, tỷ lệ khớp vùng mua — theo từng xếp hạng. MUA/KHẢ QUAN tốt khi vượt trội dương."""
    ev, summary, n = analysis.review(horizons)
    return {"recorded_signals": n, "summary": summary, "evaluated": ev.tail(200),
            "note": "Dưới ~30 mẫu mỗi nhóm chỉ để tham khảo."}


@mcp.tool()
@_guard
def get_portfolio() -> dict:
    """Danh mục đang nắm giữ từ sổ lệnh cá nhân (lệnh `portfolio`): số lượng, số bán được (đã qua
    T+2), giá vốn bình quân, giá hiện tại, lãi/lỗ tạm tính, tỷ trọng so với vốn [plan], nhóm ngành,
    khoảng cách tới cắt lỗ; lãi/lỗ đã chốt. Chỉ đọc — ghi lệnh bằng dòng lệnh."""
    from vnsa import portfolio as pf

    trades = pf.load()
    if trades.empty:
        return {"positions": [], "note": "Sổ lệnh trống (python -m vnsa.cli portfolio buy MÃ SL GIÁ)."}
    cal, prices, groups = pf.market_context(sorted(trades["symbol"].unique()))
    capital = settings()["plan"]["capital"]
    acts = pf.corporate_actions(sorted(trades["symbol"].unique()))
    tab = pf.status(trades, prices, groups, capital, calendar=cal, actions=acts)
    from vnsa import journal

    j = journal.load_journal()
    report = None
    if j is not None:
        report = {"twr_invested": j.twr, "vnindex": j.vni, "trade_stats": j.stats,
                  "discipline_score": j.discipline.score,
                  "discipline_parts": {k: {"rate": v[0], "ok": v[1], "n": v[2]}
                                       for k, v in j.discipline.parts.items()},
                  "discipline_notes": j.discipline.notes[-10:]}
    return {"capital": capital, "positions": tab,
            "realized_and_dividends": pf.realized_total(trades, calendar=cal, actions=acts),
            "journal": report,
            "note": "Lãi/lỗ %, TWR, tỷ lệ dạng thập phân; giá bằng đồng. TWR chỉ đo phần vốn đã "
                    "giải ngân (sổ lệnh không ghi tiền mặt)."}


def _reading(r) -> dict:
    from vnsa.macro.signals import fmt_change, fmt_period

    return {"series": r.series, "name": r.name, "period": fmt_period(r), "value": r.value,
            "unit": r.unit, "change": fmt_change(r), "z": r.z, "basis": r.basis, "stale": r.stale}


def _score(m) -> dict:
    return {"scope": m.scope, "label": m.label, "score": m.score, "coverage": m.coverage,
            "drivers": [{"series": d.series, "sign": d.sign, "weight": d.weight, "why": d.why,
                         "contribution": d.contribution,
                         "reading": _reading(d.reading) if d.reading is not None else None}
                        for d in m.drivers]}


@mcp.tool()
@_guard
def macro_context(symbol: str | None = None) -> dict:
    """Lớp vĩ mô (chỉ THAM KHẢO, chưa vào điểm xếp hạng): bối cảnh Việt Nam theo nhóm (tiền tệ, tỷ
    giá, lạm phát, tăng trưởng, thế giới), và nếu có `symbol`: các yếu tố vĩ mô của ngành mã đó
    (giá hàng hoá, tỷ giá, lãi suất...) kèm biến động, z so với 5 năm, đóng góp vào điểm ngành
    −1..+1, kiểm chứng độ nhạy bằng giá lịch sử, và tiêu đề tin gần đây. Chỉ diễn giải các con số có
    ở đây; số thiếu thì nói là thiếu, không tự đưa số liệu thị trường khác vào. Không có `symbol`:
    bảng mọi chuỗi vĩ mô trong kho."""
    from vnsa.config import path
    from vnsa.data.store import Store
    from vnsa.macro.catalog import catalog
    from vnsa.macro.signals import readings

    if not symbol:
        with Store(path("warehouse")) as store:
            data = store.macro()
        if data.empty:
            return {"error": "Kho chưa có số vĩ mô: chạy python -m vnsa.cli fetch-macro"}
        rs = readings(data, catalog())
        return {"series": [_reading(r) for r in rs.values()],
                "note": "Số Việt Nam (CPI, tín dụng, PMI, TPCP...) nhập tay ở config/macro_manual.toml."}
    sym = symbol.upper().strip()
    mv = analysis.macro_view(sym)
    if mv is None:
        return {"error": "Kho chưa có số vĩ mô: chạy python -m vnsa.cli fetch-macro"}
    sens = []
    try:
        sens = analysis.macro_sensitivity(analysis.load(sym, need_fin=False))
    except analysis.DataMissing:
        pass
    from vnsa.report.card import macro_guard

    return {"symbol": sym, "market": _score(mv.market),
            "groups": {k: _score(g) for k, g in mv.groups.items()},
            "industry": _score(mv.industry) if mv.industry is not None else None,
            "sensitivity": sens, "guard": macro_guard(mv),
            "risk_free": {"value": mv.risk_free, "source": mv.rf_source},
            "news": mv.news.head(10),
            "how_to_read": ("điểm −1..+1: ≥ +0,3 Thuận lợi, ≤ −0,3 Bất lợi; z: biến động hiện tại so "
                            "với 5 năm của chính chuỗi; sensitivity.verdict: dấu giả định có khớp "
                            "lịch sử giá không. Vĩ mô chưa vào điểm xếp hạng (chờ review_signals "
                            "chứng minh).")}


@mcp.tool()
@_guard
def screen_market(top_industries: int | None = None, group: str | None = None) -> dict:
    """Kết quả SÀNG LỌC TOÀN THỊ TRƯỜNG gần nhất trong kho (lệnh `python -m vnsa.cli screen`, chạy
    1–3 giờ nên công cụ này chỉ ĐỌC kết quả đã lưu, không chạy lại): các nhóm ngành xếp hạng theo
    điểm 0–100 (động lượng giá, độ rộng, tăng trưởng lợi nhuận, định giá, chất lượng, vĩ mô — mỗi phần
    là thứ hạng giữa các nhóm) và các mã được chọn trong những nhóm dẫn đầu (MUA/KHẢ QUAN, FA đạt,
    không cờ đỏ) kèm giá trị hợp lý, lợi nhuận kỳ vọng, vùng mua, cắt lỗ; thêm mã MUA ngoài nhóm dẫn
    đầu; kèm xoay vòng ngành RRG theo tuần (`rrg`, tham khảo). `group`: chỉ trả các mã của một nhóm
    (tên nhóm như trong bảng). Chỉ diễn giải các số có ở đây, nêu ngày phiên (`as_of`); đây là sàng
    lọc theo quy tắc, chưa kiểm định quá khứ."""
    from vnsa.screen.run import latest, outside_picks, screen_cfg

    table, ind = latest()
    if table.empty:
        return {"error": "Chưa có lượt sàng lọc nào: chạy python -m vnsa.cli screen "
                         "(hoặc workflow 'Sàng lọc thị trường' trên GitHub)"}
    cols = ["symbol", "group_name", "action", "price", "fair_value", "expected_return",
            "dividend_yield", "fa_score", "ta_score", "score", "confidence", "rs_6m", "eps_growth",
            "growth_basis", "zone_status", "buy_low", "buy_high", "stop", "pick_rank", "reason"]
    n = top_industries or screen_cfg().get("top_industries", 5)
    out = {"as_of": table["as_of"].iloc[0],
           "funnel": {"universe": len(table), "liquid": int((table["stage"] >= 1).sum()),
                      "analysed": int((table["stage"] >= 2).sum()),
                      "picked": int((table["stage"] >= 4).sum())},
           "industries": ind.drop(columns=["as_of"]),
           "how_to_read": ("industries.score 0–100, pt_*: điểm thành phần (thứ hạng giữa nhóm); rs, "
                           "growth, expected: trung vị các mã. picks: mã được chọn trong "
                           f"{n} nhóm dẫn đầu, pick_rank = thứ tự trong nhóm.")}
    if group:
        g = table[table["group_name"].str.lower() == group.strip().lower()]
        out["group"] = g[cols].sort_values(["pick_rank", "score"], na_position="last")
        return out
    out["picks"] = table[table["stage"] >= 4][cols].sort_values(["group_name", "pick_rank"])
    out["outside_top_industries"] = outside_picks(table)[cols]
    try:
        from vnsa.screen import rrg

        tails = rrg.load()
        if len(tails):
            out["rrg"] = rrg.latest(tails)[["group", "members", "date", "ratio", "momentum",
                                            "quadrant", "prev_quadrant"]]
            out["rrg_note"] = ("Xoay vòng ngành theo tuần so với VNINDEX (RS-Ratio, RS-Momentum quanh "
                               "100; prev_quadrant = 4 tuần trước) — chỉ tham khảo, chưa kiểm định.")
    except Exception:  # noqa: BLE001 - RRG chỉ tham khảo
        pass
    return out


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
