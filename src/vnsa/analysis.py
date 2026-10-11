"""Lớp dịch vụ: gom các bước phân tích của một mã thành hàm trả về dữ liệu (không in ra).

Dòng lệnh (cli.py), MCP server (mcp_server.py) và dashboard đều gọi qua đây, nên cùng một con số
ở mọi nơi. Mọi dữ liệu đọc từ kho DuckDB qua Store; tham số lấy từ config/settings.toml.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from vnsa.config import path, settings


class DataMissing(Exception):
    """Kho chưa có dữ liệu cần thiết; thông điệp nói rõ lệnh cần chạy."""


@dataclass
class Data:
    symbol: str
    fin: pd.DataFrame      # BCTC năm (year + FIELDS, đồng)
    prices: pd.DataFrame   # giá ngày (đồng, đã điều chỉnh)
    index: pd.DataFrame    # giá chỉ số thị trường (để tính beta)


def load(symbol: str, need_fin: bool = True, need_prices: bool = True) -> Data:
    from vnsa.data.store import Store

    sym = symbol.upper()
    with Store(path("warehouse")) as store:
        fin = store.fundamentals(sym)
        prices = store.prices(sym)
        index = store.prices(settings()["valuation"]["index"])
    missing = [c for c, miss in (("fetch", need_prices and prices.empty),
                                 ("fetch-bctc", need_fin and fin.empty)) if miss]
    if missing:
        steps = " và ".join(f"'{c} {sym}'" for c in missing)
        raise DataMissing(f"{sym}: kho chưa có đủ dữ liệu. Chạy {steps} trước.")
    return Data(sym, fin, prices, index)


def symbols() -> list[str]:
    """Các mã đang có giá trong kho (trừ chỉ số thị trường)."""
    from vnsa.data.store import Store

    with Store(path("warehouse")) as store:
        inv = store.inventory()
    index = settings()["valuation"]["index"]
    return [s for s in inv["symbol"].tolist() if s != index] if len(inv) else []


def valuation(d: Data, growth: float | None = None, wacc: float | None = None):
    """(Assumptions, Valuation, beta thô). ValueError nếu không định giá được."""
    from vnsa.fa import valuation as val

    a = val.Assumptions.from_settings(settings()["valuation"])
    b, weeks = (val.beta(d.prices, d.index, a.beta_years) if not d.index.empty
                else (float("nan"), 0))
    price = float(d.prices["close"].iloc[-1])
    v = val.value(d.symbol, d.fin, price, a, beta_value=b, beta_weeks=weeks, g1=growth,
                  wacc_override=wacc)
    return a, v, b


def fa(d: Data):
    """(bảng F-Score, bảng Altman, FAScore, Valuation hoặc None)."""
    from vnsa.fa import scores as sc

    cfg = settings()["fa_score"]
    v = None
    if not d.prices.empty:
        try:
            _, v, _ = valuation(d)
        except ValueError:
            v = None
    caps = sc.year_end_market_caps(d.fin, d.prices, settings()["valuation"]["par_value"])
    f = sc.piotroski(d.fin, cfg.get("issue_materiality", 0.01))
    z = sc.altman(d.fin, caps)
    nan = float("nan")
    score = sc.fa_score(d.fin, cfg, pe=v.pe if v else nan, pb=v.pb if v else nan,
                        dcf_upside=v.upside if v else nan)
    return f, z, score, v


def indicators(prices: pd.DataFrame) -> pd.DataFrame:
    from vnsa.ta.indicators import add_indicators

    return add_indicators(prices, settings()["ta"])


def levels(ind: pd.DataFrame):
    from vnsa.ta.levels import support_resistance

    cfg = settings()["ta"]
    return support_resistance(ind, cfg["swing_window"], cfg["lookback"], cfg["level_tolerance"])


def ta(ind: pd.DataFrame):
    from vnsa.ta.score import ta_score

    return ta_score(ind, settings()["ta_score"]["weights"])


def plan(ind: pd.DataFrame):
    """Kế hoạch mua theo giá đóng cửa phiên cuối. ValueError nếu chưa đủ dữ liệu tính ATR."""
    from vnsa.ta.plan import trade_plan

    sup, res = levels(ind)
    last = ind.iloc[-1]
    cfg = settings()["plan"]
    return trade_plan(float(last["close"]), float(last["atr"]), sup, res, cfg, cfg.get("ticks"))


@dataclass
class RiskReport:
    price: float
    fit: object                    # GarchFit
    hist_vol: float                # biến động năm từ 250 phiên gần nhất
    var: dict                      # q -> {"hist", "cvar", "garch", "garch_normal"} (tỷ lệ lỗ)
    sim: object                    # Simulation
    horizons: list[int]
    plan: object | None            # TradePlan nếu tính được
    first_touch: dict | None       # xác suất chạm cắt lỗ / mục tiêu 1 trong kỳ dài nhất


def risk(prices: pd.DataFrame, horizons: list[int] | None = None,
         n_paths: int | None = None) -> RiskReport:
    from vnsa.risk.garch import fit_garch, log_returns_pct
    from vnsa.risk.simulate import Simulation, garch_var, historical_var, simulate

    cfg = settings()["risk"]
    close = prices["close"]
    fit = fit_garch(log_returns_pct(close.tail(cfg["garch_window"] + 1)))
    p0 = float(close.iloc[-1])
    var = {}
    for q in cfg["confidence"]:
        hv, cv = historical_var(close, q, cfg["var_window"])
        var[q] = {"hist": hv, "cvar": cv, "garch": garch_var(fit, q),
                  "garch_normal": garch_var(fit, q, empirical=False)}
    horizons = sorted(horizons or cfg["horizons"])
    sim = simulate(fit, p0, max(horizons), n_paths=n_paths or cfg["n_paths"],
                   band=settings()["market"]["band_hose"], bootstrap=cfg["bootstrap"],
                   seed=cfg["seed"])
    try:
        p = plan(indicators(prices))
    except (ValueError, IndexError):
        p = None
    touch = None
    if p and p.targets:
        h = max(horizons)
        touch = Simulation(p0, sim.paths[:, : h + 1]).first_touch(p.stop, p.targets[0].price)
    hist_vol = float(close.pct_change().tail(250).std() * 252 ** 0.5)
    return RiskReport(p0, fit, hist_vol, var, sim, horizons, p, touch)


def recommendation(symbol: str):
    """Khuyến nghị tổng hợp cho một mã (xem recommend/engine.py)."""
    from vnsa.recommend.engine import Inputs, recommend

    d = load(symbol, need_fin=False)
    fa_score = z_em = upside = pe = pb = float("nan")
    notes = []
    if not d.fin.empty:
        _, z, s, v = fa(d)
        fa_score, z_em = s.total, float(z["z_em"].iloc[-1])
        if v is not None:
            upside, pe, pb = v.upside, v.pe, v.pb
    else:
        notes.append("chưa có BCTC: chỉ dựa vào kỹ thuật")
    ind = indicators(d.prices)
    t = ta(ind)
    try:
        p = plan(ind)
    except ValueError:
        p = None
    r = None
    try:
        r = risk(d.prices)
    except ValueError as e:
        notes.append(str(e))
    inputs = Inputs(
        symbol=d.symbol, price=float(d.prices["close"].iloc[-1]),
        date=d.prices["date"].iloc[-1], fa_score=fa_score, ta_score=t.total, z_em=z_em,
        dcf_upside=upside, pe=pe, pb=pb,
        vol_annual=r.fit.current_vol_annual if r else float("nan"),
        prob_up=r.sim.prob_above(day=max(r.horizons)) if r else float("nan"),
        first_target_rr=p.targets[0].reward_risk if p and p.targets else float("nan"),
        horizon=max(r.horizons) if r else 60,
    )
    return recommend(inputs, settings()["recommend"], notes), p
