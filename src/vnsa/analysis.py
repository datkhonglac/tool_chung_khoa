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

    a = val.Assumptions.from_settings(valuation_settings())
    raw, weeks = (val.beta(d.prices, d.index, a.beta_years) if not d.index.empty
                  else (float("nan"), 0))
    b = a.default_beta if pd.isna(raw) else (0.67 * raw + 0.33 if a.adjust_beta else raw)
    return a.risk_free + b * a.equity_risk_premium, b, raw, weeks


_RF_CACHE: dict = {}


def risk_free() -> tuple[float, str]:
    """(lãi suất phi rủi ro cho k_e/WACC, nguồn). Lợi suất TPCP Việt Nam 10 năm mới nhất trong kho
    (chuỗi `[macro] rf_series`, nhập tay) nếu kỳ đó không cũ quá `rf_max_age_days` ngày và nằm trong
    0,5–15% (chặn gõ nhầm); ngược lại `[valuation] risk_free`. Nhớ 60 giây (một lượt phân tích)."""
    import time

    from vnsa.data.store import Store

    key = str(path("warehouse"))
    hit = _RF_CACHE.get(key)
    if hit and time.monotonic() - hit[0] < 60:
        return hit[1]
    base = float(settings()["valuation"]["risk_free"])
    mc = settings().get("macro", {})
    out = (base, "settings.toml")
    try:
        with Store(path("warehouse")) as store:
            df = store.macro(mc.get("rf_series", "vn_gov_10y"))
    except Exception:  # noqa: BLE001 - kho đang bận/thiếu: dùng giả định trong settings
        df = pd.DataFrame()
    if not df.empty:
        last = df.iloc[-1]
        end = last["date"] + pd.offsets.MonthEnd(0)
        age = (pd.Timestamp.today().normalize() - end).days
        value = float(last["value"]) / 100
        if age <= mc.get("rf_max_age_days", 45) and 0.005 <= value <= 0.15:
            out = (value, f"lợi suất TPCP 10 năm {last['date']:%m/%Y}")
        else:
            out = (base, f"settings.toml (lợi suất TPCP nhập tay kỳ {last['date']:%m/%Y} đã cũ)")
    _RF_CACHE[key] = (time.monotonic(), out)
    return out


def valuation_settings() -> dict:
    """Mục [valuation] với risk_free lấy theo risk_free() (TPCP 10 năm nếu có số mới)."""
    return {**settings()["valuation"], "risk_free": risk_free()[0]}


def _weights(kind: str) -> dict:
    w = settings()["valuation"].get("weights", {})
    return w.get(kind) or w.get("general")


def valuation_group(symbol: str) -> tuple[str, str | None]:
    """(nhóm trọng số định giá, tên nhóm ngành) cho doanh nghiệp thường (nâng cấp v2.0, bước 2.1):
    "real_estate" / "cyclical" theo `[valuation.groups]` (tên nhóm ngành trong macro_map.toml),
    còn lại "general". Không đọc được ngành thì "general"."""
    from vnsa.data.store import Store
    from vnsa.screen.groups import industry_group

    try:
        with Store(path("warehouse")) as store:
            company = store.company(symbol)
        group = industry_group(company, macro_mapping()) if company else None
    except Exception:  # noqa: BLE001 - thiếu ngành: dùng trọng số chung
        return "general", None
    for kind, names in settings()["valuation"].get("groups", {}).items():
        if group in names:
            return kind, group
    return "general", group


def sector(d: Data):
    """Chỉ số riêng ngân hàng / công ty chứng khoán (fa/sector.py) hoặc None."""
    from vnsa import market
    from vnsa.fa import sector as sec

    if d.fin.empty:
        return None
    return sec.metrics(market.profile(d.symbol)["model"], d.fin, d.quarterly, d.symbol)


def bank_peers(d: Data) -> dict[str, float] | None:
    """Trung vị NIM, CIR, chi phí tín dụng của các ngân hàng khác trong kho (cùng cách tính xấp xỉ
    của fa/sector.py) và số ngân hàng `n`; None nếu ít hơn `min_bank_peers`."""
    import numpy as np

    from vnsa import market
    from vnsa.data.store import Store
    from vnsa.fa import sector as sec

    prof = market.profile(d.symbol)
    if prof["model"] != "bank" or not prof.get("industry"):
        return None
    with Store(path("warehouse")) as store:
        peers = [s for s in store.peers(prof["industry"]) if s != d.symbol]
    vals: dict[str, list[float]] = {"nim": [], "cir": [], "credit_cost": []}
    for s in peers:
        if market.profile(s)["model"] != "bank":
            continue
        try:
            p = load(s, need_prices=False)
        except DataMissing:
            continue
        m = sec.bank(p.fin, p.quarterly)
        if m is None:
            continue
        for k, lst in vals.items():
            if not pd.isna(m.values.get(k, float("nan"))):
                lst.append(m.values[k])
    n = len(vals["nim"])
    if n < settings()["fa_score"]["financials"].get("min_bank_peers", 4):
        return None
    return {**{k: float(np.median(v)) for k, v in vals.items() if v}, "n": n}


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
                      current=d.current, reversion=rev, peers=peer_multiples(d),
                      quality={"roe_adjust": _quality_cfg()["roe_adjust"]})
    scenarios = []
    for k in (ke - 0.01, ke, ke + 0.01):
        for keep in (0.25, 0.75):
            if k > g:
                alt = fv.residual_income(d.fin, k, g, {**cfg["financials"], "roe_persistence": keep},
                                         par, current=d.current)
                scenarios += [alt.value_per_share, alt.justified_pb * alt.book0]
    hist = fv.historical_multiples(d.fin, d.prices, par)
    pb = price / ri.book0
    scenarios += [fv.revert(pb, x * fair.roe_factor, rev) * ri.book0 for x in hist["pb"]]
    fair.p10, fair.p90 = fv.scenario_range(scenarios, fair.value)
    fair = _cap_consensus(d, fair)
    score = financial_score(d.fin, settings()["fa_score"]["financials"],
                            market.profile(d.symbol)["model"], pb=pb, fair_upside=fair.upside,
                            roe=d.current.roe if d.current is not None else None,
                            pb_hist=fair.pb_median, sector=sector(d), peer=bank_peers(d))
    return score, fair


def peer_multiples(d: Data) -> tuple[float, float, int] | None:
    """(P/E, P/B trung vị hiện tại, số mã) của các mã CÙNG NGÀNH và cùng loại doanh nghiệp đã có
    trong kho; None nếu ít hơn `[valuation] min_peers` mã. Kho chỉ có những mã đã phân tích, nên
    càng phân tích nhiều mã cùng ngành (hoặc workflow hằng ngày) thì so sánh càng đầy đủ."""
    import numpy as np

    from vnsa import market
    from vnsa.data.store import Store

    cfg = settings()["valuation"]
    prof = market.profile(d.symbol)
    if not prof.get("industry"):
        return None
    with Store(path("warehouse")) as store:
        peers = [s for s in store.peers(prof["industry"]) if s != d.symbol]
    pes, pbs = [], []
    for s in peers:
        if market.profile(s)["model"] != prof["model"]:
            continue
        try:
            p = load(s)
        except DataMissing:
            continue
        pe, pb = multiples(p)
        if 0 < pe <= cfg.get("max_peer_pe", 60):
            pes.append(pe)
        if pb > 0:
            pbs.append(pb)
    n = len(pbs)
    if n < cfg.get("min_peers", 3):
        return None
    pe_med = float(np.median(pes)) if len(pes) >= cfg.get("min_peers", 3) else float("nan")
    return pe_med, float(np.median(pbs)), n


def _quality_cfg() -> dict:
    cfg = settings()["valuation"]
    return {"roe_adjust": cfg.get("roe_adjust", False),
            "no_rerate_below": cfg.get("no_rerate_below")}


def _cap_consensus(d: Data, fair):
    """Chặn giá trị hợp lý ở giá mục tiêu CTCK × (1 + `[valuation] consensus_cap`)."""
    from vnsa.fa import fair_value as fv

    target = (d.profile or {}).get("analyst_target")
    target = float(target) if target is not None and not pd.isna(target) else float("nan")
    return fv.cap_by_consensus(fair, target, settings()["valuation"].get("consensus_cap", 0.0))


def general_fair_value(d: Data, v=None):
    """Doanh nghiệp thường: FairValue (DCF, P/E, P/B lịch sử) kèm dải P10/P90 từ ma trận độ nhạy
    DCF và P/E, P/B của từng năm lịch sử. v = Valuation đã tính (nếu có)."""
    from vnsa.fa import fair_value as fv
    from vnsa.fa import valuation as val

    nan = float("nan")
    par = settings()["valuation"]["par_value"]
    rev = settings()["valuation"].get("multiple_reversion", 1.0)
    price = float(d.prices["close"].iloc[-1])
    kind, group = valuation_group(d.symbol)
    weights = _weights(kind)
    fair = fv.combine(price, d.fin, d.prices, par, dcf=v.value_per_share if v else nan,
                      weights=weights, current=d.current, reversion=rev,
                      peers=peer_multiples(d), quality=_quality_cfg())
    if kind != "general":
        fair.notes.append(f"nhóm ngành {group}: trọng số định giá riêng [valuation.weights] {kind}"
                          + (" (bỏ DCF)" if not weights.get("dcf") else ""))
    if kind == "real_estate":
        fair.notes.append("P/B theo giá sổ sách thường đánh giá thấp quỹ đất (ghi giá gốc); RNAV "
                          "không có trong nguồn miễn phí — đối chiếu báo cáo CTCK")
    hist = fv.historical_multiples(d.fin, d.prices, par)
    c = d.current
    eps, bvps = (c.eps, c.bvps) if c is not None else (hist["eps"].iloc[-1], hist["bvps"].iloc[-1])
    pe_now = price / eps if eps > 0 else nan
    pb_now = price / bvps if bvps > 0 else nan
    pe_s = [fv.revert(pe_now, x, rev) for x in hist["pe"]]
    if fair.no_rerate and not pd.isna(pe_now):
        pe_s = [min(x, pe_now) for x in pe_s]
    scenarios = ([x * eps for x in pe_s]
                 + [fv.revert(pb_now, x * fair.roe_factor, rev) * bvps for x in hist["pb"]])
    if v is not None and v.fcff_base > 0 and weights.get("dcf", 0) > 0:
        a = val.Assumptions.from_settings(valuation_settings())
        scenarios += [x for x in val.sensitivity(v, a).to_numpy().ravel()]
    fair.p10, fair.p90 = fv.scenario_range(scenarios, fair.value)
    return _cap_consensus(d, fair)


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
    a = val.Assumptions.from_settings(valuation_settings())
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
    f = sc.piotroski(d.fin, cfg.get("issue_materiality", 0.01), stock_ratios(d))
    z = sc.altman(d.fin, caps)
    nan = float("nan")
    pe_hist, pb_hist = history_targets(d.fin, d.prices, par) if not d.prices.empty else (nan, nan)
    score = sc.fa_score(d.fin, cfg, pe=v.pe if v else nan, pb=v.pb if v else nan,
                        dcf_upside=v.upside if v else nan,
                        roe=d.current.roe if d.current is not None else None,
                        pe_hist=pe_hist, pb_hist=pb_hist, stock_ratio=stock_ratios(d))
    return f, z, score, v


def stock_ratios(d: Data) -> dict[int, float] | None:
    """{năm: tổng tỷ lệ thưởng/cổ tức bằng cổ phiếu theo ngày GDKHQ} hoặc None nếu chưa có lịch sử
    sự kiện quyền (khi đó F-Score dùng dòng tiền phát hành)."""
    div = d.dividends
    if div is None or div.empty:
        return None
    s = div.dropna(subset=["stock_ratio"])
    return {int(y): float(v) for y, v in
            s.groupby(pd.to_datetime(s["exright_date"]).dt.year)["stock_ratio"].sum().items()}


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
    # dòng tiền tổ chức 10 phiên (bước 3.2, tham khảo): (khối ngoại + tự doanh ròng) / GTGD cùng
    # 10 phiên — cùng cách tính "% GTGD" với các cửa sổ khối ngoại bên dưới để so sánh được
    t10 = f.tail(10)
    traded10 = px[px.index.isin(t10["date"])].sum()
    inst = (t10["foreign_net"].fillna(0) + t10["prop_net"].fillna(0)).sum()
    out["smart_10"] = (inst / traded10 if len(t10) == 10 and traded10 > 0
                       and t10["foreign_net"].notna().any() else float("nan"))
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


def dividend_yield(d: Data, raw: bool = False) -> float:
    """Tỷ suất cổ tức tiền dùng cho lợi nhuận kỳ vọng.

    Cổ tức 12 tháng (GDKHQ trong 365 ngày trước phiên giá cuối) / giá. Nếu vượt
    `[recommend] max_dividend_yield` (10%) — thường do hai đợt của hai năm tài chính rơi vào cùng
    12 tháng hoặc cổ tức đặc biệt một lần — thì dùng bình quân năm của 3 năm gần nhất, và không quá
    ngưỡng đó (thận trọng). raw=True: trả số 12 tháng chưa điều chỉnh (để hiển thị)."""
    if d.dividends is None or d.dividends.empty or d.prices.empty:
        return 0.0
    last, price = d.prices["date"].iloc[-1], float(d.prices["close"].iloc[-1])
    div = d.dividends
    paid = div[div["exright_date"] <= last]
    ttm = paid[paid["exright_date"] > last - pd.Timedelta(days=365)]["cash_per_share"].sum() / price
    cap = settings()["recommend"].get("max_dividend_yield", 0.10)
    if raw or ttm <= cap:
        return float(ttm)
    avg3 = (paid[paid["exright_date"] > last - pd.Timedelta(days=3 * 365)]["cash_per_share"].sum()
            / 3 / price)
    return float(min(ttm, avg3, cap))


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
    adjust, caps = _portfolio_limits(sym, ind)
    p = trade_plan(zone.entry, float(last["atr"]), sup, res, settings()["plan"], ticks,
                   adv_value=adv_value(ind), risk_scale=risk_scale, risk_adjust=adjust,
                   extra_caps=caps, max_stop=zone.stop_cap if zone.tiers else None)
    p.zone = zone
    if zone.tiers or zone.ref_tiers:
        _fill_probs(ind, zone)
    return p


def _fill_probs(ind: pd.DataFrame, zone, horizon: int = 20, n_paths: int = 2000) -> None:
    """Xác suất khớp từng lớp thang mua trong `horizon` phiên (Monte Carlo GARCH, drift như [risk]).
    analysis.risk tính lại với nhiều đường hơn khi chạy đủ phân tích. Lỗi thì để trống."""
    from vnsa import market
    from vnsa.risk.garch import fit_garch, log_returns_pct
    from vnsa.risk.simulate import simulate

    try:
        cfg = settings()["risk"]
        close = ind["close"]
        fit = fit_garch(log_returns_pct(close.tail(cfg["garch_window"] + 1)))
        sym = ind["symbol"].iloc[-1] if "symbol" in ind else None
        drift = None if cfg.get("drift", "zero") == "historical" else 0.0
        sim = simulate(fit, float(close.iloc[-1]), horizon, n_paths=n_paths,
                       band=market.band(sym) if sym else market.band(), seed=cfg["seed"],
                       drift=drift)
        for t in zone.tiers or zone.ref_tiers:
            t.fill_prob = sim.prob_touch(t.price, horizon)
    except Exception:  # noqa: BLE001 - xác suất khớp chỉ để tham khảo
        pass


def _portfolio_limits(sym: str | None, ind: pd.DataFrame):
    """Tương quan với mã đang giữ và trần nhóm ngành (portfolio.sizing_limits); sổ lệnh trống hoặc
    lỗi đọc thì không giới hạn thêm (kế hoạch vẫn tính được)."""
    if not sym:
        return None, []
    from vnsa import portfolio

    try:
        return portfolio.sizing_limits(sym, ind[["date", "close"]], settings()["plan"]["capital"])
    except Exception:  # noqa: BLE001 - sổ lệnh lỗi không được chặn phân tích
        return None, []


def timing_study(symbols: list[str], split: str = "2022-01-01", start: str = "2016-01-01"):
    """Kiểm định các quy tắc thời điểm trên danh mục `symbols` (recommend/study.py). Dùng giá trong
    kho (cần đủ năm: lệnh study-timing --years 10 tự tải)."""
    from vnsa import market
    from vnsa.data.store import Store
    from vnsa.recommend.study import study
    from vnsa.ta.regime import regime_series
    from vnsa.ta.score import ta_score_series

    data, skipped = {}, []
    weights = settings()["ta_score"]["weights"]
    with Store(path("warehouse")) as store:
        prices = {s: store.prices(s) for s in symbols}
        index = store.prices(settings()["valuation"]["index"])
    for s in symbols:
        if len(prices[s]) < 300:
            skipped.append(s)
            continue
        ind = indicators(prices[s])
        data[s] = {"ind": ind, "score": ta_score_series(ind, weights, require_all=True),
                   "band": market.band(s)}
    if not data:
        raise ValueError("không mã nào đủ dữ liệu giá")
    data["__regime__"] = {"series": regime_series(index, settings().get("regime", {}))}
    data["__regime_early__"] = {"series": regime_series(index, {**settings().get("regime", {}),
                                                                "early": True})}
    grid = tuple(settings().get("study", {}).get("atr_grid", (2.5, 3.0, 4.0)))
    res = study(data, settings()["backtest"], split, start, grid)
    res.skipped = skipped
    return res


def entry_study(symbols: list[str], split: str = "2022-01-01", start: str = "2016-01-01"):
    """Kiểm định điểm mua: vùng cũ vs thang 3 lớp vs mua ngay (recommend/entry_study.py)."""
    from vnsa import market
    from vnsa.data.store import Store
    from vnsa.recommend.entry_study import study

    data, skipped = {}, []
    with Store(path("warehouse")) as store:
        prices = {s: store.prices(s) for s in symbols}
    for s in symbols:
        if len(prices[s]) < 400:
            skipped.append(s)
            continue
        data[s] = {"ind": indicators(prices[s]), "ticks": market.ticks(s)}
    res = study(data, settings()["entry"], settings()["ta"], split, start)
    res.skipped = skipped
    return res


_HMM_CACHE: dict = {}


def hmm_view(index: pd.DataFrame | None = None):
    """Trạng thái HMM của VNINDEX (ta/hmm_regime.py) — tính một lần cho mỗi phiên cuối của chỉ số
    (sàng lọc gọi assess hàng trăm lần). None nếu thiếu dữ liệu hoặc tắt `[hmm] show`."""
    from vnsa.ta.hmm_regime import current

    cfg = settings().get("hmm", {})
    if not cfg.get("show", True):
        return None
    if index is None:
        from vnsa.data.store import Store

        with Store(path("warehouse")) as store:
            index = store.prices(settings()["valuation"]["index"])
    if index is None or index.empty:
        return None
    key = (pd.Timestamp(index["date"].max()), len(index))
    if key not in _HMM_CACHE:
        try:
            _HMM_CACHE[key] = current(index, cfg)
        except Exception:  # noqa: BLE001 - chỉ tham khảo
            _HMM_CACHE[key] = None
    return _HMM_CACHE[key]


def hmm_study(split: str = "2022-01-01", start: str = "2016-01-01"):
    """Kiểm định siết rủi ro theo HMM so với MA200 trên VNINDEX (recommend/hmm_study.py)."""
    from vnsa.data.store import Store
    from vnsa.recommend.hmm_study import study
    from vnsa.ta.regime import regime_series

    with Store(path("warehouse")) as store:
        index = store.prices(settings()["valuation"]["index"])
    if len(index) < 1000:
        raise ValueError("VNINDEX chưa đủ dữ liệu (cần ≥ 1.000 phiên; chạy study-var/study-timing để "
                         "tải 10 năm)")
    ma = regime_series(index, settings().get("regime", {}))
    ma.index = pd.to_datetime(ma.index)
    return study(index, ma, settings().get("hmm", {}), settings()["backtest"], split, start)


def hrp_view(symbols: list[str] | None = None, sessions: int = 250):
    """Tỷ trọng HRP + phụ thuộc đuôi cho các mã đang giữ (sổ lệnh) và `symbols` thêm vào."""
    from vnsa import portfolio as pf
    from vnsa.data.store import Store
    from vnsa.risk.hrp import HrpView, hrp_weights, tail_dependence, top_pairs
    from vnsa.risk.hrp_study import returns_from_prices

    trades = pf.load()
    held = {}
    if len(trades):
        cal, last_px, _ = pf.market_context(sorted(trades["symbol"].unique()))
        for s, p in pf.positions(trades, calendar=cal).items():
            if p.quantity > 0:
                held[s] = p.quantity * last_px.get(s, p.avg_cost)
    syms = list(dict.fromkeys([*held, *[x.upper() for x in (symbols or [])]]))
    if len(syms) < 2:
        raise ValueError("cần ít nhất 2 mã (mã đang giữ trong sổ lệnh và/hoặc mã nhập thêm)")
    with Store(path("warehouse")) as store:
        prices = {s: store.prices(s) for s in syms}
    missing = [s for s, p in prices.items() if len(p) < sessions // 2]
    if missing:
        raise DataMissing(f"thiếu giá: {', '.join(missing)} — chạy analyze {' '.join(missing)} trước")
    rets = returns_from_prices(prices).tail(sessions)
    total = sum(held.values())
    current = pd.Series({s: v / total for s, v in held.items()}) if total > 0 else pd.Series(dtype=float)
    q = float(settings().get("hrp", {}).get("tail_q", 0.05))
    return HrpView(hrp_weights(rets), current, top_pairs(tail_dependence(rets, q), 5), len(rets), q)


def spillover_view(symbols: list[str] | None = None):
    """Lan truyền biến động Diebold–Yilmaz giữa các mã (ta/spillover.py); mặc định [spillover]
    symbols hoặc [signals] daily."""
    from vnsa.data.store import Store
    from vnsa.risk.hrp_study import returns_from_prices
    from vnsa.ta.spillover import spillover

    cfg = settings().get("spillover", {})
    syms = [x.upper() for x in (symbols or cfg.get("symbols") or settings()["signals"]["daily"])]
    n = int(cfg.get("sessions", 250))
    with Store(path("warehouse")) as store:
        prices = {s: store.prices(s) for s in syms}
    have = {s: p for s, p in prices.items() if len(p) >= n // 2}
    missing = sorted(set(syms) - set(have))
    if len(have) < 3:
        raise DataMissing("cần giá của ít nhất 3 mã trong kho — chạy analyze/fetch các mã trước")
    res = spillover(returns_from_prices(have).tail(n), int(cfg.get("lags", 1)),
                    int(cfg.get("horizon", 10)))
    return res, missing


def hrp_study(symbols: list[str], split: str = "2022-01-01", start: str = "2016-01-01"):
    from vnsa.data.store import Store
    from vnsa.risk.hrp_study import HrpStudy, returns_from_prices, study

    with Store(path("warehouse")) as store:
        prices = {s: store.prices(s) for s in symbols}
    prices = {s: p for s, p in prices.items() if len(p) >= 300}
    if len(prices) < 5:
        raise ValueError("cần ít nhất 5 mã đủ dữ liệu giá")
    table, _ = study(returns_from_prices(prices), settings()["backtest"], split, start,
                     int(settings().get("hrp", {}).get("sessions", 250)))
    return HrpStudy(table, sorted(prices))


def var_study(symbols: list[str], split: str = "2022-01-01", start: str = "2016-01-01"):
    """Kiểm định VaR 1 ngày: lịch sử, GARCH chuẩn/thực nghiệm, bước nhảy (risk/var_study.py)."""
    from vnsa import market
    from vnsa.data.store import Store
    from vnsa.risk.var_study import study

    cfg = settings()["risk"]
    data, skipped = {}, []
    with Store(path("warehouse")) as store:
        prices = {s: store.prices(s) for s in symbols}
    for s in symbols:
        if len(prices[s]) < cfg["var_window"] + 100:
            skipped.append(s)
            continue
        data[s] = prices[s]
    res = study(data, split, start, window=int(cfg["var_window"]), refit=21,
                k=float(cfg.get("jump_k", 3.0)), bands={s: market.band(s) for s in data})
    res.skipped = skipped
    return res


def backtest_config(symbol: str) -> dict:
    """Tham số [backtest] kèm bước giá theo sàn của mã."""
    from vnsa import market

    return dict(settings()["backtest"], ticks=market.ticks(symbol))


def backtests(d: Data, entry: float | None = None, exit_: float | None = None) -> dict:
    """So sánh các quy tắc thời điểm trên cùng dữ liệu bằng máy mô phỏng chung
    (recommend/strategies.simulate: khớp mở cửa phiên sau, không mua khi mở cửa giá trần, không
    bán khi cả phiên nằm sàn, T+2, phí + thuế): "Điểm TA" (cách cũ), "Phá nền 55 phiên" (quy tắc
    gần đạt nhất trong kiểm định study-timing) và quy tắc đang dùng cho trục thời điểm
    (`[recommend] timing_rule`, nếu là quy tắc khác). Trả về
    {tên: BacktestResult}; mua & giữ nằm ở .benchmark_stats. ValueError nếu chưa đủ dữ liệu."""
    from vnsa import market
    from vnsa.recommend.strategies import NAMES, rule_set, simulate
    from vnsa.ta.regime import regime_series
    from vnsa.ta.score import ta_score_series

    cfg = backtest_config(d.symbol)
    if entry is not None:
        cfg["entry_score"] = entry
    if exit_ is not None:
        cfg["exit_score"] = exit_
    ind = indicators(d.prices)
    score = ta_score_series(ind, settings()["ta_score"]["weights"], require_all=True)
    ok = (score.notna() & ind["atr"].notna()).to_numpy()
    if not ok.any():
        raise ValueError("chưa đủ dữ liệu để backtest (cần điểm TA cho ít nhất 30 phiên)")
    start = pd.Timestamp(ind["date"].iloc[int(ok.argmax())])
    rec = settings()["recommend"]
    rules = rule_set(ind, score, regime_series(d.index, settings().get("regime", {})),
                     rec.get("timing_atr_k", 3.0), cfg)
    names = ["ta", "breakout"]
    if rec.get("timing_rule", "ta") in rules and rec["timing_rule"] not in names:
        names.append(rec["timing_rule"])
    band = market.band(d.symbol)
    try:
        return {NAMES[n]: simulate(ind, rules[n], band, cfg["fee"], cfg["sell_tax"], start)
                for n in names}
    except ValueError as e:
        raise ValueError("chưa đủ dữ liệu để backtest (cần điểm TA cho ít nhất 30 phiên)") from e


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
    jump: object = None            # risk.jump.JumpFit (v4, tham khảo)
    floor: tuple | None = None     # (xác suất 1 phiên giảm sàn, xác suất ≥ 1 phiên trong 20 phiên)


def risk(prices: pd.DataFrame, horizons: list[int] | None = None,
         n_paths: int | None = None, trade_plan=None) -> RiskReport:
    from vnsa import market
    from vnsa.risk.garch import fit_garch, log_returns_pct
    from vnsa.risk.simulate import Simulation, garch_var, historical_var, simulate

    cfg = settings()["risk"]
    close = prices["close"]
    fit = fit_garch(log_returns_pct(close.tail(cfg["garch_window"] + 1)))
    p0 = float(close.iloc[-1])
    drift = None if cfg.get("drift", "zero") == "historical" else 0.0
    from vnsa.risk.jump import fit_jumps, floor_probability, jump_var

    band0 = market.band(prices["symbol"].iloc[-1]) if "symbol" in prices else market.band()
    try:
        jf = fit_jumps(log_returns_pct(close.tail(cfg["garch_window"] + 1)), fit.sigma,
                       float(cfg.get("jump_k", 3.0)), fit.mu)
    except Exception:  # noqa: BLE001 - bước nhảy chỉ tham khảo
        jf = None
    var = {}
    for q in cfg["confidence"]:
        hv, cv = historical_var(close, q, cfg["var_window"])
        var[q] = {"hist": hv, "cvar": cv, "garch": garch_var(fit, q, drift=drift),
                  "garch_normal": garch_var(fit, q, empirical=False, drift=drift)}
        if jf is not None:
            var[q]["jump"] = jump_var(jf, fit.next_sigma, q, drift if drift is not None else fit.mu,
                                      band0)
    floor = floor_probability(jf, fit.next_sigma, band0, 20) if jf is not None else None
    horizons = sorted(horizons or cfg["horizons"])
    band = market.band(prices["symbol"].iloc[-1]) if "symbol" in prices else market.band()
    sim = simulate(fit, p0, max(horizons), n_paths=n_paths or cfg["n_paths"], band=band,
                   bootstrap=cfg["bootstrap"], seed=cfg["seed"], drift=drift)
    p = trade_plan
    if p is None:
        try:
            p = plan(indicators(prices))
        except (ValueError, IndexError):
            p = None
    if p is not None and getattr(p, "zone", None) is not None and (p.zone.tiers or p.zone.ref_tiers):
        h = min(20, max(horizons))
        for t in p.zone.tiers or p.zone.ref_tiers:   # xác suất khớp từng lớp của thang mua
            t.fill_prob = sim.prob_touch(t.price, h)
    touch = None
    if p and p.targets:
        h = max(horizons)
        touch = Simulation(p0, sim.paths[:, : h + 1]).first_touch(p.stop, p.targets[0].price)
    hist_vol = float(close.pct_change().tail(250).std() * 252 ** 0.5)
    return RiskReport(p0, fit, hist_vol, var, sim, horizons, p, touch, jf, floor)


@dataclass
class MacroView:
    """Lớp vĩ mô của một mã (chỉ tham khảo + rào chắn độ tin cậy; chưa vào điểm)."""
    market: object               # macro.score.MacroScore — bối cảnh Việt Nam
    groups: dict                 # nhóm -> MacroScore (tiền tệ, tỷ giá, lạm phát, tăng trưởng, thế giới)
    industry: object | None      # MacroScore của ngành/mã; None nếu ngành chưa có ánh xạ
    readings: dict               # chuỗi -> macro.signals.Reading
    risk_free: float
    rf_source: str
    sensitivity: list = field(default_factory=list)   # macro.sensitivity.Sensitivity theo yếu tố
    news: pd.DataFrame = field(default_factory=pd.DataFrame)   # tiêu đề tin gần đây (tham khảo)


def macro_mapping() -> dict:
    """config/macro_map.toml (danh mục chuỗi, nhóm bối cảnh, ánh xạ ngành)."""
    import tomllib

    f = path("macro_map")
    if not f.exists():
        return {}
    with f.open("rb") as fh:
        return tomllib.load(fh)


def macro_readings() -> tuple[dict, dict] | None:
    """(số đọc từng chuỗi vĩ mô, ngưỡng chấm [macro]) từ bảng macro_series; None nếu kho chưa có
    số vĩ mô. Dùng chung cho thẻ từng mã và điểm nhóm ngành của bộ sàng lọc."""
    from vnsa.data.store import Store
    from vnsa.macro.catalog import catalog
    from vnsa.macro.signals import readings

    with Store(path("warehouse")) as store:
        data = store.macro()
    if data.empty:
        return None
    mc = settings().get("macro", {})
    rs = readings(data, catalog(), z_years=mc.get("z_years", 5), z_cap=mc.get("z_cap", 2.0))
    cfg = {k: mc[k] for k in ("favorable", "unfavorable", "min_coverage", "z_cap") if k in mc}
    return rs, cfg


def macro_view(symbol: str) -> MacroView | None:
    """Bối cảnh vĩ mô Việt Nam và mức thuận lợi của ngành cho `symbol` từ bảng macro_series.
    None nếu kho chưa có số vĩ mô (chạy fetch-macro)."""
    from vnsa.data.store import Store
    from vnsa.macro.score import industry_score, market_context

    got = macro_readings()
    if got is None:
        return None
    rs, cfg = got
    with Store(path("warehouse")) as store:
        company = store.company(symbol)
        news = store.news(symbol, limit=settings().get("macro", {}).get("news_items", 5),
                          days=settings().get("macro", {}).get("news_days", 30))
    mapping = macro_mapping()
    market, groups = market_context(rs, mapping, cfg)
    ind = industry_score(symbol, company, rs, mapping, cfg)
    rf, src = risk_free()
    return MacroView(market, groups, ind, rs, rf, src, news=news)


def macro_sensitivity(d: Data) -> list:
    """Độ nhạy thực nghiệm của mã với từng yếu tố vĩ mô của ngành (macro/sensitivity.py), trên
    lịch sử giá trong kho; [] nếu ngành chưa có ánh xạ hoặc kho chưa có số vĩ mô."""
    from vnsa.data.store import Store
    from vnsa.macro.catalog import catalog
    from vnsa.macro.score import drivers_for
    from vnsa.macro.sensitivity import sensitivity

    with Store(path("warehouse")) as store:
        data = store.macro()
        company = store.company(d.symbol)
    found = drivers_for(d.symbol, company, macro_mapping())
    if data.empty or found is None or d.prices.empty:
        return []
    mc = settings().get("macro", {})
    return sensitivity(d.prices, d.index, found[1], data, catalog(),
                       mc.get("sensitivity_min_months", 36))


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
    macro: MacroView | None = None
    forensic: object = None  # fa.forensic.Forensic | None (doanh nghiệp thường)
    news: object = None      # news.NewsCheck | None


def forensic(d: Data):
    """Kiểm tra chất lượng BCTC (v3): Beneish M-Score, phải thu khác, backlog BĐS, tiền ròng.
    None với định chế tài chính hoặc chưa có BCTC."""
    from vnsa.fa.forensic import analyze
    from vnsa.market import financial_note

    if d.fin.empty or financial_note(d.symbol):
        return None
    shares = (d.profile or {}).get("shares_out")
    price = float(d.prices["close_raw"].iloc[-1]) if "close_raw" in d.prices and \
        not pd.isna(d.prices["close_raw"].iloc[-1]) else float(d.prices["close"].iloc[-1])
    cap = price * float(shares) if shares and not pd.isna(shares) else float("nan")
    return analyze(d.fin, cap, valuation_group(d.symbol)[0] == "real_estate",
                   settings().get("forensic", {}))


def news_check(symbol: str):
    """Tin gần đây gắn nhãn theo từ khoá + trạng thái giao dịch từ bảng giá (news.py)."""
    from vnsa import news as nw
    from vnsa.data.store import Store

    c = nw.cfg()
    with Store(path("warehouse")) as store:
        df = store.news(symbol, limit=100, days=int(c["severe_days"]))
        status = store.trading_status(symbol)
    res = nw.check(df, c=c)
    nw.apply_status(res, status, c)
    return res if len(df) or res.blocked or res.status_note else None


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
    inputs.rule_state = timing_state(d, ind)
    try:
        nc = news_check(d.symbol)
    except Exception:  # noqa: BLE001 - tin chỉ để cảnh báo, lỗi không chặn phân tích
        nc = None
    if nc is not None and nc.blocked:
        inputs.blocked = nc.blocked
    try:
        fx = forensic(d)
    except Exception:  # noqa: BLE001
        fx = None
    rec = recommend(inputs, settings()["recommend"], notes)
    try:
        mv = macro_view(d.symbol)
        if mv is not None:
            mv.sensitivity = macro_sensitivity(d)
            inputs.macro_market = mv.market.score
            inputs.macro_industry = mv.industry.score if mv.industry is not None else nan
    except Exception as e:  # noqa: BLE001 - lớp vĩ mô chỉ tham khảo, lỗi không chặn phân tích
        mv = None
        rec.reasons.append(f"không đọc được số vĩ mô ({e})")
    return Assessment(d, fa_score, fair, pe, pb, z_em, upside, ind, t, p, r, q, rec, mv, fx, nc)


def timing_state(d: Data, ind: pd.DataFrame) -> dict | None:
    """Trạng thái quy tắc thời điểm `[recommend] timing_rule` (khi không phải "ta"/"none")."""
    from vnsa import market
    from vnsa.recommend.strategies import rule_state
    from vnsa.ta.regime import regime_series
    from vnsa.ta.score import ta_score_series

    cfg = settings()["recommend"]
    name = cfg.get("timing_rule", "ta")
    if name in ("ta", "none") or len(ind) < 260:
        return None
    score = ta_score_series(ind, settings()["ta_score"]["weights"], require_all=True)
    reg = regime_series(d.index, settings().get("regime", {}))
    return rule_state(ind, score, reg, name, cfg.get("timing_atr_k", 3.0), settings()["backtest"],
                      market.band(d.symbol))


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
            "macro_industry": x.macro_industry, "macro_market": x.macro_market,
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
