"""Lớp dịch vụ: gom các bước phân tích của một mã thành hàm trả về dữ liệu (không in ra).

Dòng lệnh (cli.py), MCP server (mcp_server.py) và dashboard đều gọi qua đây, nên cùng một con số
ở mọi nơi. Mọi dữ liệu đọc từ kho DuckDB qua Store; tham số lấy từ config/settings.toml.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    quarterly: pd.DataFrame = field(default_factory=pd.DataFrame)   # BCTC quý (4 quý gần nhất)
    profile: dict = field(default_factory=dict)   # hồ sơ: shares_out, foreign_pct, avg_value_1m...
    dividends: pd.DataFrame = field(default_factory=pd.DataFrame)
    flows: pd.DataFrame = field(default_factory=pd.DataFrame)       # khối ngoại / tự doanh (CafeF)

    @property
    def current(self):
        """fa.ttm.Current: số liệu hiện tại (TTM nếu có) theo số cổ phiếu lưu hành hiện tại."""
        from vnsa.fa.ttm import current

        if not hasattr(self, "_current"):
            shares = self.profile.get("shares_out") if self.profile else None
            self._current = current(self.fin, self.quarterly,
                                    float("nan") if shares is None else float(shares),
                                    settings()["valuation"]["par_value"])
        return self._current


def load(symbol: str, need_fin: bool = True, need_prices: bool = True) -> Data:
    from vnsa.data.store import Store

    sym = symbol.upper()
    with Store(path("warehouse")) as store:
        fin = store.fundamentals(sym)
        prices = store.prices(sym)
        index = store.prices(settings()["valuation"]["index"])
        quarterly = store.quarterly(sym)
        profile = store.company(sym) or {}
        dividends = store.dividends(sym)
        flows = store.flows(sym)
    missing = [c for c, miss in (("fetch", need_prices and prices.empty),
                                 ("fetch-bctc", need_fin and fin.empty)) if miss]
    if missing:
        steps = " và ".join(f"'{c} {sym}'" for c in missing)
        raise DataMissing(f"{sym}: kho chưa có đủ dữ liệu. Chạy {steps} trước "
                          f"(hoặc 'analyze {sym}' để tự tải).")
    return Data(sym, fin, prices, index, quarterly, profile, dividends, flows)


def update(symbol: str, refresh: bool = False, full: bool = False, log=None):
    """Tự tải/cập nhật giá (CafeF), BCTC và thông tin sàn (vnstock) cho một mã bất kỳ.
    Trả về UpdateReport (từng bước đã làm gì); xem data/autofetch.py."""
    from vnsa.data.autofetch import ensure

    return ensure(symbol, refresh, full, log)


def _not_financial(symbol: str) -> None:
    from vnsa.market import financial_note

    note = financial_note(symbol)
    if note:
        raise ValueError(note)


def multiples(d: Data) -> tuple[float, float]:
    """(P/E, P/B) hiện tại: giá / EPS TTM (hoặc năm gần nhất), giá / BVPS quý gần nhất, theo số cổ
    phiếu lưu hành hiện tại (fa/ttm.py). Dùng được cho cả ngân hàng."""
    nan = float("nan")
    if d.fin.empty or d.prices.empty or d.current is None:
        return nan, nan
    c, price = d.current, float(d.prices["close"].iloc[-1])
    return (float(price / c.eps) if c.eps > 0 else nan,
            float(price / c.bvps) if c.bvps > 0 else nan)


def symbols() -> list[str]:
    """Các mã đang có giá trong kho (trừ chỉ số thị trường)."""
    from vnsa.data.store import Store

    with Store(path("warehouse")) as store:
        inv = store.inventory()
    index = settings()["valuation"]["index"]
    return [s for s in inv["symbol"].tolist() if s != index] if len(inv) else []


def cost_of_equity(d: Data) -> tuple[float, float, float, int]:
    """(k_e, beta dùng trong CAPM, beta thô, số tuần) — beta tuần so với VNINDEX, Blume nếu bật."""
    from vnsa.fa import valuation as val

    a = val.Assumptions.from_settings(settings()["valuation"])
    raw, weeks = (val.beta(d.prices, d.index, a.beta_years) if not d.index.empty
                  else (float("nan"), 0))
    b = a.default_beta if pd.isna(raw) else (0.67 * raw + 0.33 if a.adjust_beta else raw)
    return a.risk_free + b * a.equity_risk_premium, b, raw, weeks


def _weights(kind: str) -> dict:
    return settings()["valuation"].get("weights", {}).get(kind)


def sector(d: Data):
    """Chỉ số riêng ngân hàng / công ty chứng khoán (fa/sector.py) hoặc None."""
    from vnsa import market
    from vnsa.fa import sector as sec

    if d.fin.empty:
        return None
    return sec.metrics(market.profile(d.symbol)["model"], d.fin, d.quarterly)


def financial(d: Data):
    """Định chế tài chính: (FAScore, FairValue) bằng thu nhập thặng dư + P/B theo ROE + P/B lịch
    sử (fa/fair_value.py) và điểm FA riêng (scores.financial_score). ValueError nếu thiếu số.
    FairValue có dải P10/P90 từ các kịch bản k_e ± 1%, độ bền ROE 0,25/0,75, P/B từng năm."""
    from vnsa import market
    from vnsa.fa import fair_value as fv
    from vnsa.fa.scores import financial_score

    if d.fin.empty or d.prices.empty:
        raise ValueError(f"{d.symbol}: thiếu BCTC hoặc giá")
    cfg = settings()["valuation"]
    g, par = cfg["terminal_growth"], cfg["par_value"]
    ke, _, _, _ = cost_of_equity(d)
    ri = fv.residual_income(d.fin, ke, g, cfg["financials"], par, current=d.current)
    price = float(d.prices["close"].iloc[-1])
    rev = cfg.get("multiple_reversion", 1.0)
    fair = fv.combine(price, d.fin, d.prices, par, ri=ri, weights=_weights("financial"),
                      current=d.current, reversion=rev)
    scenarios = []
    for k in (ke - 0.01, ke, ke + 0.01):
        for keep in (0.25, 0.75):
            if k > g:
                alt = fv.residual_income(d.fin, k, g, {**cfg["financials"], "roe_persistence": keep},
                                         par, current=d.current)
                scenarios += [alt.value_per_share, alt.justified_pb * alt.book0]
    hist = fv.historical_multiples(d.fin, d.prices, par)
    pb = price / ri.book0
    scenarios += [fv.revert(pb, x, rev) * ri.book0 for x in hist["pb"]]
    fair.p10, fair.p90 = fv.scenario_range(scenarios, fair.value)
    score = financial_score(d.fin, settings()["fa_score"]["financials"],
                            market.profile(d.symbol)["model"], pb=pb, fair_upside=fair.upside,
                            roe=d.current.roe if d.current is not None else None,
                            pb_hist=fair.pb_median, sector=sector(d))
    return score, fair


def general_fair_value(d: Data, v=None):
    """Doanh nghiệp thường: FairValue (DCF, P/E, P/B lịch sử) kèm dải P10/P90 từ ma trận độ nhạy
    DCF và P/E, P/B của từng năm lịch sử. v = Valuation đã tính (nếu có)."""
    from vnsa.fa import fair_value as fv
    from vnsa.fa import valuation as val

    nan = float("nan")
    par = settings()["valuation"]["par_value"]
    rev = settings()["valuation"].get("multiple_reversion", 1.0)
    price = float(d.prices["close"].iloc[-1])
    fair = fv.combine(price, d.fin, d.prices, par, dcf=v.value_per_share if v else nan,
                      weights=_weights("general"), current=d.current, reversion=rev)
    hist = fv.historical_multiples(d.fin, d.prices, par)
    c = d.current
    eps, bvps = (c.eps, c.bvps) if c is not None else (hist["eps"].iloc[-1], hist["bvps"].iloc[-1])
    pe_now = price / eps if eps > 0 else nan
    pb_now = price / bvps if bvps > 0 else nan
    scenarios = ([fv.revert(pe_now, x, rev) * eps for x in hist["pe"]]
                 + [fv.revert(pb_now, x, rev) * bvps for x in hist["pb"]])
    if v is not None and v.fcff_base > 0:
        a = val.Assumptions.from_settings(settings()["valuation"])
        scenarios += [x for x in val.sensitivity(v, a).to_numpy().ravel()]
    fair.p10, fair.p90 = fv.scenario_range(scenarios, fair.value)
    return fair


def fundamentals_view(d: Data):
    """(điểm FA 0–100, FairValue hoặc None, P/E, P/B) cho mọi loại doanh nghiệp; thiếu BCTC ->
    (NaN, None, NaN, NaN). Dùng cho khuyến nghị và vùng mua."""
    from vnsa.market import financial_note

    nan = float("nan")
    if d.fin.empty or d.prices.empty:
        return nan, None, nan, nan
    if financial_note(d.symbol):
        pe, pb = multiples(d)
        try:
            score, fair = financial(d)
        except ValueError:
            return nan, None, pe, pb
        return score.total, fair, pe, pb
    _, _, score, v = fa(d)
    return score.total, general_fair_value(d, v), (v.pe if v else nan), (v.pb if v else nan)


def valuation(d: Data, growth: float | None = None, wacc: float | None = None):
    """(Assumptions, Valuation, beta thô). ValueError nếu không định giá được."""
    from vnsa.fa import valuation as val

    _not_financial(d.symbol)
    a = val.Assumptions.from_settings(settings()["valuation"])
    b, weeks = (val.beta(d.prices, d.index, a.beta_years) if not d.index.empty
                else (float("nan"), 0))
    price = float(d.prices["close"].iloc[-1])
    v = val.value(d.symbol, d.fin, price, a, beta_value=b, beta_weeks=weeks, g1=growth,
                  wacc_override=wacc, current=d.current)
    return a, v, b


def fa(d: Data):
    """(bảng F-Score, bảng Altman, FAScore, Valuation hoặc None)."""
    from vnsa.fa import scores as sc

    _not_financial(d.symbol)
    cfg = settings()["fa_score"]
    v = None
    if not d.prices.empty:
        try:
            _, v, _ = valuation(d)
        except ValueError:
            v = None
    from vnsa.fa.fair_value import history_targets

    par = settings()["valuation"]["par_value"]
    caps = sc.year_end_market_caps(d.fin, d.prices, par)
    f = sc.piotroski(d.fin, cfg.get("issue_materiality", 0.01))
    z = sc.altman(d.fin, caps)
    nan = float("nan")
    pe_hist, pb_hist = history_targets(d.fin, d.prices, par) if not d.prices.empty else (nan, nan)
    score = sc.fa_score(d.fin, cfg, pe=v.pe if v else nan, pb=v.pb if v else nan,
                        dcf_upside=v.upside if v else nan,
                        roe=d.current.roe if d.current is not None else None,
                        pe_hist=pe_hist, pb_hist=pb_hist)
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


def _value_anchor(symbol: str) -> tuple[float, float]:
    """(giá trị hợp lý/cp, điểm FA) của mã trong kho; NaN nếu không tính được."""
    nan = float("nan")
    try:
        score, fair, _, _ = fundamentals_view(load(symbol, need_fin=False))
    except (DataMissing, ValueError, KeyError, IndexError):
        return nan, nan
    return (fair.value if fair else nan), score


def adv_value(prices: pd.DataFrame, n: int = 20) -> float:
    """Giá trị giao dịch bình quân n phiên gần nhất (đồng): giá khớp thật × khối lượng."""
    tail = prices.tail(n)
    px = tail["close_raw"].fillna(tail["close"]) if "close_raw" in tail else tail["close"]
    return float((px * tail["volume"]).mean()) if len(tail) else float("nan")


def flow_summary(d: Data) -> dict | None:
    """Khối ngoại / tự doanh mua(+)/bán(−) ròng cộng dồn N phiên gần nhất ([flows] windows), đồng,
    kèm tỷ lệ so với GTGD cùng kỳ. Chỉ để tham khảo, không vào điểm. None nếu chưa có dữ liệu."""
    f = d.flows
    if f is None or f.empty:
        return None
    f = f.sort_values("date")
    px = d.prices.assign(value=d.prices["close"] * d.prices["volume"]).set_index("date")["value"]
    out = {"last_date": f["date"].iloc[-1], "foreign_pct": f["foreign_pct"].dropna().iloc[-1]
           if f["foreign_pct"].notna().any() else float("nan")}
    for n in settings().get("flows", {}).get("windows", [5, 20]):
        tail = f.tail(n)
        traded = px[px.index.isin(tail["date"])].sum()
        for who in ("foreign", "prop"):
            net = tail[f"{who}_net"].sum(min_count=1)
            out[f"{who}_{n}"] = net
            out[f"{who}_{n}_pct"] = net / traded if traded > 0 and not pd.isna(net) else float("nan")
    return out


def market_regime(index: pd.DataFrame | None = None):
    """Trạng thái VNINDEX (ta/regime.py); không truyền thì đọc từ kho."""
    from vnsa.ta.regime import regime

    if index is None:
        from vnsa.data.store import Store

        with Store(path("warehouse")) as store:
            index = store.prices(settings()["valuation"]["index"])
    return regime(index, settings().get("regime", {}))


def dividend_yield(d: Data) -> float:
    """Cổ tức tiền 12 tháng (GDKHQ trong 365 ngày trước phiên giá cuối) / giá."""
    if d.dividends is None or d.dividends.empty or d.prices.empty:
        return 0.0
    last = d.prices["date"].iloc[-1]
    div = d.dividends
    recent = div[(div["exright_date"] > last - pd.Timedelta(days=365))
                 & (div["exright_date"] <= last)]
    return float(recent["cash_per_share"].sum() / d.prices["close"].iloc[-1])


def plan(ind: pd.DataFrame, fair_value: float | None = None, fa_score: float | None = None,
         risk_scale: float = 1.0):
    """Kế hoạch mua với giá vào = đỉnh VÙNG MUA khuyến nghị (ta/entry.py: hỗ trợ / VWAP20 / SMA20,
    chặn bởi giá trị hợp lý trừ biên an toàn), không phải giá đóng cửa. Không truyền fair_value thì
    tự tính từ kho. ValueError nếu chưa đủ dữ liệu tính ATR."""
    from vnsa import market
    from vnsa.ta.entry import entry_zone
    from vnsa.ta.plan import trade_plan

    sup, res = levels(ind)
    last = ind.iloc[-1]
    sym = last["symbol"] if "symbol" in ind else None
    ticks = market.ticks(sym) if sym else market.ticks()
    if fair_value is None:
        fair_value, fa_score = _value_anchor(sym) if sym else (float("nan"), float("nan"))
    zone = entry_zone(ind, sup, settings()["entry"], ticks, fair_value,
                      float("nan") if fa_score is None else fa_score)
    p = trade_plan(zone.entry, float(last["atr"]), sup, res, settings()["plan"], ticks,
                   adv_value=adv_value(ind), risk_scale=risk_scale)
    p.zone = zone
    return p


def backtest_config(symbol: str) -> dict:
    """Tham số [backtest] kèm bước giá theo sàn của mã."""
    from vnsa import market

    return dict(settings()["backtest"], ticks=market.ticks(symbol))


def backtests(d: Data, entry: float | None = None, exit_: float | None = None) -> dict:
    """So sánh hai công thức trên cùng dữ liệu: {"TA thuần": kết quả, "TA + VNINDEX + thanh khoản":
    kết quả}. Công thức mới dùng trạng thái VNINDEX từng phiên (ta/regime.regime_series) và chặn
    giá trị lệnh ≤ max_adv_pct × GTGD 20 phiên với vốn mỗi mã = capital × max_position_pct.
    ValueError nếu chưa đủ dữ liệu."""
    from vnsa.recommend.backtest import backtest
    from vnsa.ta.regime import regime_series
    from vnsa.ta.score import ta_score_series

    cfg = backtest_config(d.symbol)
    if entry is not None:
        cfg["entry_score"] = entry
    if exit_ is not None:
        cfg["exit_score"] = exit_
    ind = indicators(d.prices)
    score = ta_score_series(ind, settings()["ta_score"]["weights"], require_all=True)
    plan_cfg = settings()["plan"]
    new_cfg = dict(cfg, capital=plan_cfg["capital"] * plan_cfg.get("max_position_pct", 1.0),
                   max_adv_pct=plan_cfg.get("max_adv_pct", 0.1))
    return {"TA thuần": backtest(ind, score, cfg),
            "TA + VNINDEX + thanh khoản": backtest(ind, score, new_cfg,
                                                   regime_series(d.index, settings().get("regime", {})))}


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
         n_paths: int | None = None, trade_plan=None) -> RiskReport:
    from vnsa import market
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
    band = market.band(prices["symbol"].iloc[-1]) if "symbol" in prices else market.band()
    sim = simulate(fit, p0, max(horizons), n_paths=n_paths or cfg["n_paths"], band=band,
                   bootstrap=cfg["bootstrap"], seed=cfg["seed"])
    p = trade_plan
    if p is None:
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


@dataclass
class Assessment:
    """Mọi kết quả phân tích của một mã, tính một lần (khuyến nghị, báo cáo, dashboard dùng chung)."""
    data: Data
    fa_score: float
    fair: object             # FairValue | None
    pe: float
    pb: float
    z_em: float
    dcf_upside: float
    ind: pd.DataFrame
    ta: object               # TAScore
    plan: object             # TradePlan | None
    risk: RiskReport | None
    quant: float
    rec: object              # Recommendation


def assess(symbol: str) -> Assessment:
    """Phân tích đầy đủ một mã: FA, giá trị hợp lý, TA, vùng mua, rủi ro, Quant, khuyến nghị."""
    from vnsa.market import financial_note
    from vnsa.recommend.engine import Inputs, quant_score, recommend

    d = load(symbol, need_fin=False)
    nan = float("nan")
    z_em = upside = nan
    notes = []
    fa_score, fair, pe, pb = fundamentals_view(d)
    if financial_note(d.symbol):
        notes.append(financial_note(d.symbol))
    elif not d.fin.empty:
        _, z, _, v = fa(d)
        z_em = float(z["z_em"].iloc[-1])
        upside = v.upside if v is not None else nan
    else:
        notes.append("chưa có BCTC: chỉ dựa vào kỹ thuật")
    ind = indicators(d.prices)
    t = ta(ind)
    reg = market_regime(d.index)
    scale = settings().get("regime", {}).get("bear_risk_factor", 1.0) if reg.state == "giảm" else 1.0
    notes.append(str(reg))
    try:
        p = plan(ind, fair.value if fair else nan, fa_score, risk_scale=scale)
    except ValueError:
        p = None
    if p is not None and p.zone is not None:
        zn = p.zone
        notes.insert(0, f"{zn.status.lower()} {zn.low:,.0f}–{zn.high:,.0f} "
                        f"(neo {zn.anchor_source})" if zn.status == "MUA TRONG VÙNG" else
                        f"chờ giá về ≤ {zn.high:,.0f} mới mua")
    r = None
    try:
        r = risk(d.prices, trade_plan=p)
    except ValueError as e:
        notes.append(str(e))
    vol = r.fit.current_vol_annual if r else nan
    prob_up = r.sim.prob_above(day=max(r.horizons)) if r else nan
    rr = p.targets[0].reward_risk if p and p.targets else nan
    q = quant_score(prob_up, vol, rr, settings()["recommend"]["quant"])
    inputs = Inputs(
        symbol=d.symbol, price=float(d.prices["close"].iloc[-1]),
        date=d.prices["date"].iloc[-1], fa_score=fa_score, ta_score=t.total, z_em=z_em,
        dcf_upside=upside, pe=pe, pb=pb, vol_annual=vol, prob_up=prob_up, first_target_rr=rr,
        horizon=max(r.horizons) if r else 60,
        fair_value=fair.value if fair else nan, fair_upside=fair.upside if fair else nan,
        quant_score=q, dividend_yield=dividend_yield(d), regime=reg.state,
        zone_status=p.zone.status if p is not None and p.zone is not None else "",
    )
    rec = recommend(inputs, settings()["recommend"], notes)
    return Assessment(d, fa_score, fair, pe, pb, z_em, upside, ind, t, p, r, q, rec)


def recommendation(symbol: str):
    """(Recommendation, TradePlan | None) cho một mã (xem recommend/engine.py)."""
    a = assess(symbol)
    return a.rec, a.plan


def report_card(symbol: str):
    """Thẻ báo cáo gọn một màn hình (report/card.py)."""
    from vnsa.report.card import build_card

    return build_card(assess(symbol))


def _params_hash() -> str:
    """Mã băm phần cấu hình quyết định khuyến nghị: đổi quy tắc thì đổi mã, để chấm riêng từng bộ."""
    import hashlib
    import json

    cfg = settings()
    keys = ["recommend", "fa_score", "ta_score", "entry", "plan", "valuation"]
    blob = json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def record_signals(items) -> int:
    """Lưu khuyến nghị [(Recommendation, TradePlan | None), ...] vào bảng signals và xuất toàn bộ
    lịch sử ra data/signals.csv (file này không bị .gitignore — commit để giữ khi đổi Codespace)."""
    from vnsa.data.store import Store

    if not settings().get("signals", {}).get("record", True) or not items:
        return 0
    nan = float("nan")
    params = _params_hash()
    rows = []
    for rec, p in items:
        x, zn = rec.inputs, (p.zone if p is not None else None)
        rows.append({
            "as_of": x.date, "symbol": rec.symbol, "action": rec.action, "score": rec.score,
            "fa_score": x.fa_score, "ta_score": x.ta_score, "price": x.price,
            "fair_value": x.fair_value, "zone_status": zn.status if zn else None,
            "buy_low": zn.low if zn else nan, "buy_high": zn.high if zn else nan,
            "stop": p.stop if p else nan,
            "target": p.targets[0].price if p and p.targets else nan, "params": params,
        })
    with Store(path("warehouse")) as store:
        n = store.write_signals(pd.DataFrame(rows))
        hist = store.signals()
    out = path("signals_csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    hist.assign(as_of=hist["as_of"].dt.strftime("%Y-%m-%d")).to_csv(out, index=False)
    return n


def import_signals_csv() -> int:
    """Nạp lại data/signals.csv vào kho (khi mở Codespace mới, kho trống nhưng file CSV còn trong Git)."""
    from vnsa.data.store import Store

    src = path("signals_csv")
    if not src.exists():
        return 0
    df = pd.read_csv(src)
    with Store(path("warehouse")) as store:
        if len(store.signals()) >= len(df):
            return 0
        return store.write_signals(df)


def review(horizons: list[int] | None = None):
    """(bảng chấm từng khuyến nghị, bảng tổng hợp theo kỳ × khuyến nghị, số khuyến nghị đã lưu)."""
    from vnsa.data.store import Store
    from vnsa.recommend.review import evaluate, summarize

    import_signals_csv()
    horizons = horizons or settings()["signals"]["horizons"]
    with Store(path("warehouse")) as store:
        sig = store.signals()
        prices = {s: store.prices(s) for s in sig["symbol"].unique()}
        index = store.prices(settings()["valuation"]["index"])
    ev = evaluate(sig, prices, index, horizons)
    return ev, summarize(ev), len(sig)
