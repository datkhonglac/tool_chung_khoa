"""Dashboard Streamlit cho VN Stock Analyzer.

Chạy:  streamlit run dashboard/app.py
Cài:   pip install -e ".[dashboard]"

Đọc kho DuckDB (data/warehouse.duckdb) qua vnsa.analysis — cùng con số với dòng lệnh.
Ô "Mã bất kỳ" ở thanh bên tự tải giá (CafeF), BCTC và sàn/ngành (vnstock) cho mã chưa có hoặc đã cũ.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from vnsa import analysis, market
from vnsa.config import path, settings
from vnsa.report import charts

st.set_page_config(page_title="VN Stock Analyzer", page_icon="📈", layout="wide")


def _stamp() -> float:
    """Thời điểm sửa kho: đổi khi nạp dữ liệu mới, dùng làm khoá bộ nhớ đệm."""
    p = path("warehouse")
    return p.stat().st_mtime if p.exists() else 0.0


@st.cache_data(show_spinner=False)
def recommendations(symbols: tuple[str, ...], _stamp: float):
    rows, errors = [], []
    for sym in symbols:
        try:
            rec, p = analysis.recommendation(sym)
        except (analysis.DataMissing, ValueError) as e:
            errors.append(str(e))
            continue
        x, z = rec.inputs, (p.zone if p is not None else None)
        zone = ("—" if z is None else f"≤ {z.high:,.0f} (chờ)" if z.status == "CHỜ GIÁ"
                else f"{z.low:,.0f} – {z.high:,.0f}")
        rows.append({"Mã": rec.symbol, "Giá": x.price, "Vùng mua": zone, "Điểm FA": x.fa_score,
                     "Điểm TA": x.ta_score, "LN kỳ vọng": rec.expected_return,
                     "Thời điểm": rec.timing,
                     "P/E": x.pe, "P/B": x.pb, "Biến động/năm": x.vol_annual,
                     "Khuyến nghị": rec.action, "Lý do": "; ".join(rec.reasons)})
    return pd.DataFrame(rows), errors


@st.cache_data(show_spinner=False)
def report_text(symbol: str, _stamp: float) -> str:
    from vnsa.report.card import build_card, format_card

    return format_card(build_card(analysis.assess(symbol)))


@st.cache_data(show_spinner=False)
def load(symbol: str, _stamp: float):
    return analysis.load(symbol, need_fin=False)


def fmt_pct(x) -> str:
    return "—" if pd.isna(x) else f"{x:.1%}"


def fmt_num(x, d: int = 0) -> str:
    return "—" if pd.isna(x) else f"{x:,.{d}f}"


st.title("📈 VN Stock Analyzer")
with st.sidebar:
    with st.form("auto_fetch", clear_on_submit=True):
        new = st.text_input("Mã bất kỳ (HOSE/HNX/UPCoM)", placeholder="VD: SHS, VCB, ACV")
        go = st.form_submit_button("Lấy dữ liệu & phân tích")
    if go and new.strip():
        sym = new.strip().upper()
        with st.spinner(f"Đang tải {sym} từ CafeF và vnstock (lần đầu khoảng 10–30 giây)..."):
            rep = analysis.update(sym)
        st.session_state["update_log"] = (sym, rep.ok, [str(x).strip() for x in rep.steps])
        if rep.ok:
            st.session_state["symbol"] = sym
            st.cache_data.clear()
    if "update_log" in st.session_state:
        sym, ok, lines = st.session_state["update_log"]
        (st.success if ok else st.error)(f"{sym}: " + ("đã sẵn sàng" if ok else "không có giá"))
        st.caption("  \n".join(lines))

stamp = _stamp()
if not stamp:
    st.error("Chưa có kho dữ liệu. Chạy `python -m vnsa.cli fetch` và `python -m vnsa.cli "
             "fetch-bctc` (hoặc `python -m vnsa.cli load --sample` để thử) rồi tải lại trang.")
    st.stop()

available = analysis.symbols()
watch = [s for s in settings()["watchlist"]["symbols"] if s in available] or available
with st.sidebar:
    st.header("Chọn mã")
    if st.session_state.get("symbol") not in available:
        st.session_state["symbol"] = watch[0] if watch else available[0]
    symbol = st.selectbox("Mã cổ phiếu", available, key="symbol")
    prof = market.profile(symbol)
    st.caption(" · ".join(x for x in (prof["name"], prof["exchange"] if prof["known"] else None,
                                      prof["industry"]) if x))
    st.caption(f"Kho: {len(available)} mã. Mã trong kho cập nhật khi nhập lại ở ô phía trên "
               "hoặc chạy lệnh fetch.")
    st.caption("Kết quả là sàng lọc theo quy tắc trong config/settings.toml, "
               "không phải lời khuyên đầu tư.")

tab_all, tab_ta, tab_fa, tab_val, tab_risk, tab_bt, tab_hist = st.tabs(
    ["Tổng quan", "Kỹ thuật & kế hoạch", "Tài chính", "Định giá", "Rủi ro", "Backtest",
     "Lịch sử khuyến nghị"])

with tab_all:
    st.subheader(f"Báo cáo nhanh {symbol}")
    try:
        with st.spinner("Đang phân tích..."):
            st.code(report_text(symbol, stamp), language=None)
    except (analysis.DataMissing, ValueError) as e:
        st.warning(str(e))
    st.subheader("Khuyến nghị danh sách theo dõi")
    with st.spinner("Đang tính..."):
        table, errors = recommendations(tuple(watch), stamp)
    if len(table):
        st.dataframe(
            table, hide_index=True, width="stretch",
            column_config={
                "Giá": st.column_config.NumberColumn(format="localized"),
                "Điểm FA": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                "Điểm TA": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                "LN kỳ vọng": st.column_config.NumberColumn(
                    format="percent", help="Upside so với giá trị hợp lý + tỷ suất cổ tức 12 tháng"),
                "P/E": st.column_config.NumberColumn(format="%.1f"),
                "P/B": st.column_config.NumberColumn(format="%.2f"),
                "Biến động/năm": st.column_config.NumberColumn(format="percent"),
            })
    for e in errors:
        st.warning(e)

try:
    data = load(symbol, stamp)
except analysis.DataMissing as e:
    st.error(str(e))
    st.stop()
ind = analysis.indicators(data.prices)
last = ind.iloc[-1]

with tab_ta:
    t = analysis.ta(ind)
    sup, res = analysis.levels(ind)
    try:
        plan = analysis.plan(ind)
    except ValueError:
        plan = None
    c = st.columns(5)
    c[0].metric(f"Giá đóng cửa {last['date']:%d/%m/%Y}", fmt_num(last["close"]))
    c[1].metric(f"Điểm TA ({t.rating.lower()})", fmt_num(t.total))
    c[2].metric("RSI14", fmt_num(last["rsi"], 1))
    c[3].metric("ADX", fmt_num(last["adx"], 1))
    c[4].metric(f"ATR14 ({fmt_pct(last['atr'] / last['close'])} giá)", fmt_num(last["atr"]))
    st.plotly_chart(charts.technical_chart(ind, sup, res, plan), width="stretch")
    st.caption(" · ".join(f"{k}: {fmt_num(v)}" for k, v in t.groups.items()))
    if plan and plan.zone is not None:
        z = plan.zone
        st.subheader("Vùng mua khuyến nghị")
        c = st.columns(4)
        if z.status == "CHỜ GIÁ":
            c[0].metric("Chờ giá về", f"≤ {z.high:,.0f}", fmt_pct(z.discount))
        else:
            c[0].metric("Vùng mua", f"{z.low:,.0f} – {z.high:,.0f}", fmt_pct(z.discount))
        c[1].metric("Giá bình quân dự kiến (2 lệnh LO)", fmt_num(z.mid))
        c[2].metric(f"Neo kỹ thuật ({z.anchor_source})", fmt_num(z.anchor))
        c[3].metric(f"Trần giá trị (hợp lý − {z.margin_of_safety:.0%})", fmt_num(z.value_cap))
        st.caption(f"Giá đóng cửa {z.close:,.0f} · VWAP20 {z.vwap20:,.0f} · giá trị hợp lý "
                   f"{fmt_num(z.fair_value)}. Vùng mua neo theo hỗ trợ / VWAP20 / SMA20 và không "
                   "vượt giá trị hợp lý trừ biên an toàn (config [entry]).")
        for n in z.notes:
            st.caption(f"! {n}")
    if plan:
        st.subheader("Kế hoạch mua tham khảo")
        rows = [{"Mục": "Giá vào", "Giá": plan.entry, "So với giá vào": 0.0, "Ghi chú": ""},
                {"Mục": "Cắt lỗ", "Giá": plan.stop, "So với giá vào": plan.stop_pct,
                 "Ghi chú": plan.stop_source}]
        rows += [{"Mục": f"Mục tiêu {i}", "Giá": x.price, "So với giá vào": x.price / plan.entry - 1,
                  "Ghi chú": f"{x.reward_risk:.1f}R; {x.source}"} for i, x in enumerate(plan.targets, 1)]
        st.dataframe(pd.DataFrame(rows), hide_index=True,
                     column_config={"Giá": st.column_config.NumberColumn(format="localized"),
                                    "So với giá vào": st.column_config.NumberColumn(format="percent")})
        st.write(f"Khối lượng **{plan.shares:,} cp** ≈ {plan.position_value / 1e6:,.1f} triệu; "
                 f"lỗ tối đa {plan.max_loss / 1e6:,.2f} triệu "
                 f"({settings()['plan']['risk_per_trade']:.0%} vốn {plan.capital / 1e6:,.0f} triệu).")
        for n in plan.notes:
            st.caption(f"! {n}")

with tab_fa:
    if data.fin.empty:
        st.info(f"Chưa có BCTC. Chạy `python -m vnsa.cli fetch-bctc {symbol}`.")
    else:
        from vnsa.fa.ratios import compute_ratios, format_ratios
        from vnsa.fa.scores import FSCORE_CRITERIA, GROUPS, z_zone

        fin_note = market.financial_note(symbol)
        if fin_note:
            from vnsa.fa.scores import FIN_GROUPS

            pe, pb = analysis.multiples(data)
            try:
                fs, _ = analysis.financial(data)
            except ValueError as e:
                fs = None
                st.error(str(e))
            c = st.columns(3)
            c[0].metric(f"Điểm FA ({fs.rating.lower() if fs else '—'})",
                        fmt_num(fs.total if fs else float("nan")))
            c[1].metric("P/E", fmt_num(pe, 1))
            c[2].metric("P/B", fmt_num(pb, 2))
            st.info(fin_note)
            st.subheader("Chỉ số tài chính")
            st.dataframe(format_ratios(compute_ratios(data.fin)), width="stretch")
            if fs:
                st.subheader("Điểm FA theo nhóm (định chế tài chính)")
                st.dataframe(pd.DataFrame([{"Nhóm": g, "Điểm": fs.groups[g],
                                            "Chỉ tiêu": ", ".join(
                                                f"{k} = {fmt_num(fs.items[k][0], 3)}" for k in keys)}
                                           for g, keys in FIN_GROUPS.items()]), hide_index=True)
                for n in fs.notes:
                    st.caption(f"! {n}")
        else:
            f, z, s, _ = analysis.fa(data)
            c = st.columns(3)
            c[0].metric(f"Điểm FA ({s.rating.lower()})", fmt_num(s.total))
            c[1].metric("F-Score năm gần nhất", fmt_num(f["f_score"].iloc[-1]) + "/9")
            c[2].metric(f"Altman Z'' ({z_zone(z['z_em'].iloc[-1])})", fmt_num(z["z_em"].iloc[-1], 2))
            st.subheader("Chỉ số tài chính")
            st.dataframe(format_ratios(compute_ratios(data.fin)), width="stretch")
            st.subheader("Piotroski F-Score")
            ft = f.set_index("year")[[*FSCORE_CRITERIA, "f_score"]].T
            ft.index = [*FSCORE_CRITERIA.values(), "F-Score (/9)"]
            st.dataframe(ft, width="stretch")
            st.subheader("Điểm FA theo nhóm")
            st.dataframe(pd.DataFrame([{"Nhóm": g, "Điểm": s.groups[g],
                                        "Chỉ tiêu": ", ".join(f"{k} = {fmt_num(s.items[k][0], 2)}"
                                                              for k in keys)}
                                       for g, keys in GROUPS.items()]), hide_index=True)

with tab_val:
    if data.fin.empty:
        st.info(f"Chưa có BCTC. Chạy `python -m vnsa.cli fetch-bctc {symbol}`.")
    elif market.financial_note(symbol):
        from vnsa.fa.fair_value import METHOD_NAMES

        try:
            _, fair = analysis.financial(data)
        except ValueError as e:
            st.error(str(e))
        else:
            ri = fair.ri
            c = st.columns(4)
            c[0].metric("Giá trị hợp lý (½ mô hình ROE + ½ P/B lịch sử)", fmt_num(fair.value),
                        fmt_pct(fair.upside))
            c[1].metric("Thu nhập thặng dư", fmt_num(ri.value_per_share))
            c[2].metric("P/B hợp lý theo ROE", fmt_num(ri.justified_pb, 2))
            c[3].metric("Chi phí vốn chủ k_e", fmt_pct(ri.cost_of_equity))
            st.write(f"B0 = {ri.book0:,.0f} đ/cp · ROE bình quân {fmt_pct(ri.roe0)} → dài hạn "
                     f"{fmt_pct(ri.roe_terminal)} · P/B lịch sử {fmt_num(fair.pb_target, 2)}")
            st.dataframe(pd.DataFrame([{"Phương pháp": METHOD_NAMES[k], "Giá trị/cp": v}
                                       for k, v in fair.methods.items()]), hide_index=True,
                         column_config={"Giá trị/cp": st.column_config.NumberColumn(
                             format="localized")})
            for n in fair.notes:
                st.warning(n)
            st.caption(market.financial_note(symbol) + ". Giả định ở [valuation.financials].")
    else:
        from vnsa.fa import valuation as val

        c = st.columns(2)
        g1 = c[0].number_input("Tăng trưởng năm đầu g1 (0 = tự tính)", value=0.0, step=0.01,
                               format="%.2f")
        wacc_in = c[1].number_input("WACC (0 = tự tính)", value=0.0, step=0.005, format="%.3f")
        try:
            a, v, _ = analysis.valuation(data, g1 or None, wacc_in or None)
        except ValueError as e:
            st.error(str(e))
        else:
            c = st.columns(5)
            c[0].metric("Giá trị DCF/cp", fmt_num(v.value_per_share), fmt_pct(v.upside))
            c[1].metric("WACC", fmt_pct(v.wacc))
            c[2].metric("Beta", fmt_num(v.beta, 2))
            c[3].metric("P/E", fmt_num(v.pe, 1))
            c[4].metric("P/B", fmt_num(v.pb, 2))
            for n in v.notes:
                st.warning(n)
            fair = analysis.general_fair_value(data, v)
            st.subheader("Giá trị hợp lý (bình quân có trọng số DCF, P/E lịch sử, P/B lịch sử)")
            c = st.columns(4)
            c[0].metric("Giá trị hợp lý", fmt_num(fair.value), fmt_pct(fair.upside))
            c[1].metric("DCF", fmt_num(fair.methods["dcf"]))
            c[2].metric(f"P/E lịch sử ({fmt_num(fair.pe_target, 1)}x)",
                        fmt_num(fair.methods["pe_hist"]))
            c[3].metric(f"P/B lịch sử ({fmt_num(fair.pb_target, 2)}x)",
                        fmt_num(fair.methods["pb_hist"]))
            for n in fair.notes:
                st.caption(f"! {n}")
            table = val.sensitivity(v, a)
            if table.notna().any().any():
                st.subheader("Độ nhạy giá trị/cp (nghìn đồng)")
                st.plotly_chart(charts.sensitivity_heatmap(table, v.price), width="stretch")
            st.caption("Giả định ở mục [valuation] trong config/settings.toml.")

with tab_risk:
    try:
        r = analysis.risk(data.prices)
    except ValueError as e:
        st.error(str(e))
    else:
        c = st.columns(4)
        c[0].metric("Biến động năm (GARCH)", fmt_pct(r.fit.current_vol_annual))
        q0 = next(iter(r.var))
        c[1].metric(f"VaR {q0:.0%} 1 ngày", fmt_pct(r.var[q0]["hist"]))
        c[2].metric(f"CVaR {q0:.0%}", fmt_pct(r.var[q0]["cvar"]))
        h = max(r.horizons)
        c[3].metric(f"Xác suất tăng sau {h} phiên", fmt_pct(r.sim.prob_above(day=h)))
        st.plotly_chart(charts.fan_chart(r.sim, history=data.prices), width="stretch")
        st.dataframe(pd.DataFrame([{"Kỳ": f"{k} phiên", **{f"P{p}": round(x) for p, x in
                                                            r.sim.percentiles(k).items()}}
                                   for k in r.horizons]), hide_index=True,
                     column_config={f"P{p}": st.column_config.NumberColumn(format="localized")
                                    for p in (10, 25, 50, 75, 90)})
        if r.first_touch:
            ft = r.first_touch
            st.write(f"Theo kế hoạch trong {h} phiên: chạm mục tiêu 1 trước **{ft['target']:.0%}**, "
                     f"chạm cắt lỗ trước **{ft['stop']:.0%}**, chưa chạm {ft['neither']:.0%}.")
        st.caption("Mô phỏng dựa trên biến động quá khứ, không dự báo được sự kiện bất thường.")

with tab_bt:
    from vnsa.recommend.backtest import backtest
    from vnsa.ta.score import ta_score_series

    cfg = analysis.backtest_config(symbol)
    c = st.columns(2)
    cfg["entry_score"] = c[0].slider("Mua khi điểm TA ≥", 40, 90, int(cfg["entry_score"]))
    cfg["exit_score"] = c[1].slider("Bán khi điểm TA <", 10, 70, int(cfg["exit_score"]))
    try:
        res = backtest(ind, ta_score_series(ind, settings()["ta_score"]["weights"],
                                            require_all=True), cfg)
    except ValueError as e:
        st.error(str(e))
    else:
        s, b = res.stats, res.benchmark_stats
        c = st.columns(4)
        c[0].metric(f"Lợi nhuận/năm (mua & giữ {fmt_pct(b['cagr'])})", fmt_pct(s["cagr"]))
        c[1].metric(f"Sụt giảm tối đa (mua & giữ {fmt_pct(b['max_drawdown'])})",
                    fmt_pct(s["max_drawdown"]))
        c[2].metric("Số lệnh / tỷ lệ thắng", f"{s['trades']} / {fmt_pct(s['win_rate'])}")
        c[3].metric("Thời gian cầm cp", fmt_pct(s["exposure"]))
        st.plotly_chart(charts.equity_chart(res), width="stretch")
        st.dataframe(pd.DataFrame([{"Mua": t.entry_date, "Giá mua": t.entry, "Bán": t.exit_date,
                                    "Giá bán": t.exit, "Lãi/lỗ": t.ret, "Lý do": t.reason}
                                   for t in res.trades]), hide_index=True,
                     column_config={"Lãi/lỗ": st.column_config.NumberColumn(format="percent"),
                                    "Mua": st.column_config.DateColumn(format="DD/MM/YYYY"),
                                    "Bán": st.column_config.DateColumn(format="DD/MM/YYYY"),
                                    "Giá mua": st.column_config.NumberColumn(format="localized"),
                                    "Giá bán": st.column_config.NumberColumn(format="localized")})
        st.caption("Kết quả quá khứ không đảm bảo tương lai; chưa tính trượt giá.")

with tab_hist:
    st.subheader("Chấm lại khuyến nghị đã lưu")
    st.caption("Mỗi lần chạy recommend / analyze (dòng lệnh, Claude Desktop) đều được lưu. Sau 5, 20, "
               "60 phiên tool so lãi/lỗ với VNINDEX. MUA tốt khi 'Vượt VNINDEX' dương.")
    try:
        ev, summary, n = analysis.review()
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
    else:
        st.write(f"Đã lưu **{n}** khuyến nghị.")
        if summary.empty:
            st.info("Chưa có khuyến nghị nào đủ số phiên để chấm.")
        else:
            st.dataframe(summary.rename(columns={
                "horizon": "Số phiên", "action": "Khuyến nghị", "n": "Mẫu",
                "avg_ret": "Lãi/lỗ TB", "avg_excess": "Vượt VNINDEX", "win_rate": "Tỷ lệ thắng",
                "target_first": "Chạm mục tiêu trước", "zone_fill": "Khớp vùng mua",
                "avg_ret_from_zone": "Lãi từ vùng mua"}), hide_index=True,
                column_config={c: st.column_config.NumberColumn(format="percent") for c in (
                    "Lãi/lỗ TB", "Vượt VNINDEX", "Tỷ lệ thắng", "Chạm mục tiêu trước",
                    "Khớp vùng mua", "Lãi từ vùng mua")})
        from vnsa.data.store import Store

        with Store(path("warehouse")) as store:
            hist = store.signals(symbol)
        if len(hist):
            st.subheader(f"Các lần khuyến nghị {symbol}")
            st.dataframe(hist.drop(columns=["params", "recorded_at"]), hide_index=True,
                         column_config={"as_of": st.column_config.DateColumn(format="DD/MM/YYYY")})
