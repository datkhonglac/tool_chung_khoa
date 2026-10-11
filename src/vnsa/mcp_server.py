"""MCP server: cho Claude Desktop gọi các phép phân tích của tool như "công cụ".

Chạy:  python -m vnsa.mcp_server        (giao tiếp qua stdio, Claude Desktop tự khởi động)
Cài:   pip install -e ".[mcp]"

Mọi con số do Python tính từ kho DuckDB (data/warehouse.duckdb); Claude chỉ đọc và diễn giải.
Tiền tính bằng đồng, tỷ lệ dạng thập phân (0.25 = 25%). Ô trống (NaN) trả về null.
Hướng dẫn cấu hình Claude Desktop: README, mục "MCP server".
"""

from __future__ import annotations

import functools
import math
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
    "Công cụ phân tích cổ phiếu Việt Nam (HOSE). Mọi số liệu do các tool này tính từ dữ liệu đã tải "
    "về máy; không tự đoán hay bịa số liệu thị trường. Giá tính bằng đồng, đã điều chỉnh cổ tức/"
    "chia tách; tỷ lệ là số thập phân (0.25 = 25%). Nếu tool báo thiếu dữ liệu, hướng dẫn người "
    "dùng chạy lệnh được nêu trong thông báo. Kết quả là sàng lọc theo quy tắc, không phải lời "
    "khuyên đầu tư."
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
    """Định giá: beta, WACC, FCFF, DCF hai giai đoạn, giá trị/cp, chênh lệch so với giá, P/E, P/B
    và ma trận độ nhạy (hàng WACC, cột tăng trưởng dài hạn). growth/wacc: ghi đè giả định."""
    from vnsa.fa import valuation as val

    d = analysis.load(symbol)
    a, v, raw_beta = analysis.valuation(d, growth, wacc)
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
        "value_per_share": v.value_per_share, "upside": v.upside, "notes": v.notes,
        "sensitivity": {f"wacc={w:.4f}": {f"g={g:.3f}": table.at[w, g] for g in table.columns}
                        for w in table.index},
        "assumptions": asdict(a),
    }


@mcp.tool()
@_guard
def get_fa_score(symbol: str) -> dict:
    """Piotroski F-Score từng năm (9 tiêu chí), Altman Z'' và Z gốc, điểm FA 0–100 theo nhóm."""
    from vnsa.fa.scores import FSCORE_CRITERIA, z_zone

    d = analysis.load(symbol, need_prices=False)
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
    """Kế hoạch mua tham khảo: giá vào, cắt lỗ theo ATR/hỗ trợ, mục tiêu, khối lượng theo % rủi ro
    vốn trong cấu hình (tôn trọng bước giá HOSE và lô 100 cp)."""
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
    """Khuyến nghị MUA / THEO DÕI / TRÁNH theo quy tắc (điểm FA, TA, định giá, rủi ro) kèm lý do.
    Bỏ trống symbols = danh sách theo dõi."""
    out, errors = [], []
    for sym in symbols or settings()["watchlist"]["symbols"]:
        try:
            rec, _ = analysis.recommendation(sym)
        except (analysis.DataMissing, ValueError) as e:
            errors.append(str(e))
            continue
        out.append({"symbol": rec.symbol, "action": rec.action, "score": rec.score,
                    "reasons": rec.reasons, "inputs": rec.inputs})
    return {"recommendations": out, "errors": errors,
            "disclaimer": "Sàng lọc theo quy tắc trong settings.toml, không phải lời khuyên đầu tư."}


@mcp.tool()
@_guard
def run_backtest(symbol: str, entry_score: float | None = None,
                 exit_score: float | None = None) -> dict:
    """Backtest chiến lược theo điểm TA (mua khi điểm ≥ entry_score, bán khi < exit_score, cắt lỗ
    ATR, T+2, phí và thuế), so với mua và nắm giữ."""
    from vnsa.recommend.backtest import backtest
    from vnsa.ta.score import ta_score_series

    d = analysis.load(symbol, need_fin=False)
    cfg = dict(settings()["backtest"], ticks=settings()["plan"].get("ticks"))
    if entry_score is not None:
        cfg["entry_score"] = entry_score
    if exit_score is not None:
        cfg["exit_score"] = exit_score
    ind = analysis.indicators(d.prices)
    res = backtest(ind, ta_score_series(ind, settings()["ta_score"]["weights"], require_all=True), cfg)
    return {"symbol": d.symbol, "rules": cfg, "strategy": res.stats,
            "buy_and_hold": res.benchmark_stats,
            "trades": [dict(asdict(t), ret=t.ret) for t in res.trades]}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
