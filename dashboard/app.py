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
def macro(symbol: str, stamp: float, warehouse: str):
    """(MacroView | None, độ nhạy, bảng mọi chuỗi vĩ mô trong kho). Khoá đệm gồm thời điểm sửa kho
    và đường dẫn kho (tham số không có dấu _ nên Streamlit đưa vào khoá) — tải số vĩ mô mới là thấy."""
    from vnsa.data.store import Store

    mv = analysis.macro_view(symbol)
    sens = []
    if mv is not None:
        try:
            sens = analysis.macro_sensitivity(analysis.load(symbol, need_fin=False))
        except analysis.DataMissing:
            pass
    with Store(path("warehouse")) as store:
        data = store.macro()
    return mv, sens, data


@st.cache_data(show_spinner=False)
def screen_run(stamp: float, warehouse: str):
    """Lượt sàng lọc toàn thị trường mới nhất trong kho: (bảng mã, điểm nhóm ngành, mã MUA ngoài
    nhóm dẫn đầu). Khoá đệm gồm thời điểm sửa kho và đường dẫn kho."""
    from vnsa.screen.run import latest, outside_picks

    t, ind = latest()
    return t, ind, (outside_picks(t) if not t.empty else t)


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

(tab_all, tab_ta, tab_fa, tab_val, tab_macro, tab_risk, tab_bt, tab_hist,
 tab_screen, tab_pf) = st.tabs(
    ["Tổng quan", "Kỹ thuật & kế hoạch", "Tài chính", "Định giá", "Vĩ mô", "Rủi ro", "Backtest",
     "Lịch sử khuyến nghị", "Sàng lọc thị trường", "Danh mục"])

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
        c[1].metric("Giá bình quân dự kiến" + (" (khớp đủ 3 lớp)" if z.tiers else " (2 lệnh LO)"),
                    fmt_num(z.mid))
        c[2].metric(f"Neo kỹ thuật ({z.anchor_source})", fmt_num(z.anchor))
        c[3].metric(f"Trần giá trị (hợp lý − {z.margin_of_safety:.0%})", fmt_num(z.value_cap))
        st.caption(f"Giá đóng cửa {z.close:,.0f} · VWAP20 {z.vwap20:,.0f} · giá trị hợp lý "
                   f"{fmt_num(z.fair_value)}. Vùng mua neo theo hỗ trợ / VWAP20 / SMA20 và không "
                   "vượt giá trị hợp lý trừ biên an toàn (config [entry]).")
        if z.tiers or z.ref_tiers:
            if not z.tiers:
                st.caption("Mức mua sâu hơn (tham khảo — kế hoạch vẫn dùng vùng mua trên; bật "
                           "[entry] ladder = true để chia vốn theo 3 lớp):")
            st.dataframe(pd.DataFrame([{"Lớp": t.name, "Giá đặt LO": t.price,
                                        "So với giá": t.depth * 100, "Phần vốn": t.weight * 100,
                                        "Khớp trong 20 phiên": t.fill_prob * 100,
                                        "Căn cứ": t.source} for t in (z.tiers or z.ref_tiers)]),
                         hide_index=True, width="stretch",
                         column_config={"Giá đặt LO": st.column_config.NumberColumn(format="localized"),
                                        "So với giá": st.column_config.NumberColumn(format="%+.1f%%"),
                                        "Phần vốn": st.column_config.NumberColumn(format="%.0f%%"),
                                        "Khớp trong 20 phiên": st.column_config.NumberColumn(
                                            format="%.0f%%")})
            st.caption("Xác suất khớp ước tính bằng Monte Carlo theo giá đóng cửa (thận trọng).")
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
                     f"{fmt_pct(ri.roe_terminal)} · P/B lịch sử: trung vị "
                     f"{fmt_num(fair.pb_median, 2)}, mục tiêu {fmt_num(fair.pb_target, 2)}")
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
            c[2].metric(f"P/E mục tiêu ({fmt_num(fair.pe_target, 1)}x, "
                        f"trung vị {fmt_num(fair.pe_median, 1)}x)",
                        fmt_num(fair.methods["pe_hist"]))
            c[3].metric(f"P/B mục tiêu ({fmt_num(fair.pb_target, 2)}x, "
                        f"trung vị {fmt_num(fair.pb_median, 2)}x)",
                        fmt_num(fair.methods["pb_hist"]))
            for n in fair.notes:
                st.caption(f"! {n}")
            table = val.sensitivity(v, a)
            if table.notna().any().any():
                st.subheader("Độ nhạy giá trị/cp (nghìn đồng)")
                st.plotly_chart(charts.sensitivity_heatmap(table, v.price), width="stretch")
            st.caption("Giả định ở mục [valuation] trong config/settings.toml.")

with tab_macro:
    from vnsa.macro.catalog import catalog
    from vnsa.macro.signals import fmt_change, fmt_period, fmt_value

    mv, sens, mdata = macro(symbol, stamp, str(path("warehouse")))
    if mv is None:
        st.info("Kho chưa có số vĩ mô. Chạy `python -m vnsa.cli fetch-macro` (số Việt Nam nhập tay "
                "ở config/macro_manual.toml) rồi tải lại trang.")
    else:
        st.caption("Chỉ THAM KHẢO, chưa vào điểm xếp hạng: chờ review-signals chứng minh vĩ mô giúp "
                   "phân biệt mã vượt VNINDEX. Điểm −1..+1: ≥ +0,3 Thuận lợi, ≤ −0,3 Bất lợi.")
        cols = st.columns(len(mv.groups) + 1)
        for col, g in zip(cols, [mv.market, *mv.groups.values()], strict=True):
            col.metric(g.scope.replace("Bối cảnh Việt Nam", "Bối cảnh VN"),
                       g.label if g.label != "Chưa đủ số liệu" else "Thiếu số",
                       None if pd.isna(g.score) else f"{g.score:+.2f}",
                       delta_color="off" if pd.isna(g.score) else "normal")
        ind_m = mv.industry
        if ind_m is None:
            st.info(f"{symbol}: ngành chưa có ánh xạ yếu tố vĩ mô (thêm ở config/macro_map.toml).")
        else:
            st.subheader(f"{symbol} — {ind_m.scope}")
            st.write(f"**{ind_m.label}**"
                     + ("" if pd.isna(ind_m.score) else f" ({ind_m.score:+.2f})")
                     + f" · độ phủ {ind_m.coverage:.0%}")
            by_sens = {x.series: x for x in sens}
            rows = []
            for d in ind_m.drivers:
                r, x = d.reading, by_sens.get(d.series)
                rows.append({
                    "Yếu tố": r.name if r else d.series, "Dấu giả định": "+" if d.sign > 0 else "−",
                    "Kỳ": fmt_period(r) if r else "—", "Giá trị": fmt_value(r) if r else "—",
                    "Biến động": fmt_change(r) if r else "—",
                    "z": r.z if r is not None else float("nan"),
                    "Đóng góp": d.contribution,
                    "Kiểm chứng": (f"{x.verdict} (t {x.t:+.1f}, {x.n} tháng)"
                                   if x is not None and x.verdict != "thiếu lịch sử"
                                   else "thiếu lịch sử"),
                    "Lý do": d.why})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={"z": st.column_config.NumberColumn(format="%+.1f"),
                                        "Đóng góp": st.column_config.NumberColumn(format="%+.2f")})
        st.caption(f"Lãi suất phi rủi ro trong WACC: {mv.risk_free:.2%} ({mv.rf_source})")
        if len(mv.news):
            st.subheader("Tin gần đây (tiêu đề, chỉ tham khảo)")
            for r in mv.news.itertuples():
                st.markdown(f"- {r.date:%d/%m/%Y} · " + (f"[{r.title}]({r.url})" if r.url else r.title))
        cat = catalog()
        have = [sid for sid in cat if sid in set(mdata["series"])]
        if have:
            st.subheader("Chuỗi vĩ mô")
            default = (next((d.series for d in ind_m.drivers if d.series in have), have[0])
                       if ind_m is not None else have[0])
            sid = st.selectbox("Chuỗi", have, index=have.index(default),
                               format_func=lambda k: f"{cat[k].name} ({k})")
            ser = mdata[mdata["series"] == sid].set_index("date")["value"]
            st.line_chart(ser, height=260)
            st.caption(f"Nguồn: {cat[sid].source} · đơn vị: {cat[sid].unit}"
                       + (f" · {cat[sid].note}" if cat[sid].note else ""))

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
    cfg = analysis.backtest_config(symbol)
    c = st.columns(2)
    entry = c[0].slider("Mua khi điểm TA ≥", 40, 90, int(cfg["entry_score"]))
    exit_ = c[1].slider("Bán khi điểm TA <", 10, 70, int(cfg["exit_score"]))
    try:
        results = analysis.backtests(analysis.load(symbol, need_fin=False), entry, exit_)
    except ValueError as e:
        st.error(str(e))
    else:
        hold = next(iter(results.values())).benchmark_stats
        st.dataframe(pd.DataFrame(
            [{"Công thức": k, "Lợi nhuận/năm": r.stats["cagr"],
              "Sụt giảm tối đa": r.stats["max_drawdown"], "Sharpe": r.stats["sharpe"],
              "Số lệnh": r.stats["trades"], "Thời gian cầm cp": r.stats["exposure"]}
             for k, r in results.items()]
            + [{"Công thức": "Mua & giữ",
                "Lợi nhuận/năm": hold["cagr"], "Sụt giảm tối đa": hold["max_drawdown"],
                "Sharpe": hold["sharpe"]}]),
            hide_index=True,
            column_config={c: st.column_config.NumberColumn(format="percent")
                           for c in ("Lợi nhuận/năm", "Sụt giảm tối đa", "Thời gian cầm cp")})
        rule = settings()["recommend"].get("timing_rule", "ta")
        st.caption("Khớp mở cửa phiên sau, không mua ở giá trần / không bán khi nằm sàn, T+2, phí + "
                   f"thuế. Trục thời điểm đang dùng: {rule}. Bên dưới: {list(results)[-1]}.")
        res = list(results.values())[-1]
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


with tab_screen:
    sc, sc_ind, sc_out = screen_run(stamp, str(path("warehouse")))
    if sc.empty:
        st.info("Chưa có lượt sàng lọc toàn thị trường nào. Chạy `python -m vnsa.cli screen` "
                "(lần đầu 1–3 giờ vì hạn mức vnstock) hoặc workflow **Sàng lọc thị trường** trên "
                "GitHub Actions, rồi tải lại trang.")
    else:
        pct_cols = ["expected_return", "rs_6m", "eps_growth", "rs", "growth", "expected"]

        def _show(df: pd.DataFrame, names: dict) -> pd.DataFrame:
            out = df.copy()
            for c in pct_cols:
                if c in out:
                    out[c] = out[c] * 100
            return out[list(names)].rename(columns=names)

        st.caption(f"Phiên {sc['as_of'].iloc[0]:%d/%m/%Y}. Sàng lọc theo quy tắc trong "
                   "config/settings.toml (kiểm định phần giá: lệnh study-screen) — không phải lời "
                   "khuyên đầu tư.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Vũ trụ", f"{len(sc):,}")
        c2.metric("Qua thanh khoản", f"{int((sc['stage'] >= 1).sum()):,}")
        c3.metric("Phân tích được", f"{int((sc['stage'] >= 2).sum()):,}")
        c4.metric("Được chọn", f"{int((sc['stage'] >= 4).sum()):,}")
        pct = st.column_config.NumberColumn(format="%+.0f%%")
        if not (sc["stage"] >= 2).any():
            # lượt chỉ lọc thanh khoản (--stage liquidity): nói rõ, hiện danh sách mã qua lọc
            st.info("Lượt này chỉ chạy **tầng 1 — lọc thanh khoản** (`--stage liquidity`), "
                    "chưa phân tích từng mã nên chưa có nhóm ngành và mã tiềm năng. Muốn đủ 4 "
                    "tầng: chạy `python -m vnsa.cli screen` (lần đầu 1–3 giờ), hoặc nạp lượt đầy "
                    "đủ từ GitHub: `git fetch origin signals` rồi "
                    "`python -m vnsa.cli history import --git origin/signals`.")
            liq = sc[sc["stage"] >= 1].sort_values("value_20d", ascending=False)
            st.subheader(f"{len(liq):,} mã qua lọc thanh khoản".replace(",", "."))
            liq_cols = {"symbol": "Mã", "exchange": "Sàn", "industry": "Ngành", "price": "Giá",
                        "value_20d": "GTGD 20 phiên (tỷ)", "rs_3m": "RS 3T", "rs_6m": "RS 6T",
                        "above_ma50": "Trên MA50", "above_ma200": "Trên MA200"}
            view = liq.assign(value_20d=liq["value_20d"] / 1e9, rs_3m=liq["rs_3m"] * 100,
                              rs_6m=liq["rs_6m"] * 100)
            st.dataframe(view[[c for c in liq_cols if c in view]].rename(columns=liq_cols),
                         hide_index=True, width="stretch",
                         column_config={"RS 3T": pct, "RS 6T": pct, "GTGD 20 phiên (tỷ)":
                                        st.column_config.NumberColumn(format="%.1f")})
            from vnsa.screen.liquidity import reason_group

            out = sc[sc["stage"] < 1]["reason"].fillna("").map(reason_group).value_counts()
            if len(out):
                st.caption("Bị loại: " + " · ".join(f"{k} {v}" for k, v in out.items() if k))
        else:
            if not sc_ind.empty:
                st.subheader("Nhóm ngành tiềm năng (điểm 0–100)")
                st.dataframe(
                    _show(sc_ind, {"rank": "#", "group_name": "Nhóm ngành", "members": "Số mã",
                                   "score": "Điểm", "pt_momentum": "Động lượng",
                                   "pt_breadth": "Độ rộng", "pt_earnings": "Lợi nhuận",
                                   "pt_valuation": "Định giá", "pt_quality": "Chất lượng",
                                   "pt_macro": "Vĩ mô", "rs": "RS (trung vị)",
                                   "growth": "LN tăng (trung vị)",
                                   "expected": "LN kỳ vọng (trung vị)", "note": "Ghi chú"}),
                    hide_index=True, width="stretch",
                    column_config={"Điểm": st.column_config.ProgressColumn(
                                       min_value=0, max_value=100, format="%.0f"),
                                   "RS (trung vị)": pct, "LN tăng (trung vị)": pct,
                                   "LN kỳ vọng (trung vị)": pct})
                st.caption("Thành phần là thứ hạng giữa các nhóm (tốt nhất 100). Trọng số ở "
                           "[screen.industry_weights].")
            stock_cols = {"symbol": "Mã", "group_name": "Nhóm", "action": "Xếp hạng",
                          "price": "Giá", "fair_value": "Giá trị hợp lý",
                          "expected_return": "LN kỳ vọng", "buy_low": "Vùng mua từ", "buy_high": "đến", "stop": "Cắt lỗ",
                          "fa_score": "FA", "ta_score": "TA", "confidence": "Tin cậy",
                          "rs_6m": "RS 6T", "eps_growth": "LNST tăng", "reason": "Ghi chú"}
            fmt = {"LN kỳ vọng": pct, "RS 6T": pct, "LNST tăng": pct}
            picks = sc[sc["stage"] >= 4].sort_values(["group_rank", "pick_rank"])
            st.subheader("Mã tiềm năng trong các nhóm dẫn đầu")
            if picks.empty:
                st.write("Không mã nào đạt đủ điều kiện ở lượt này.")
            else:
                st.dataframe(_show(picks, stock_cols), hide_index=True, width="stretch",
                             column_config=fmt)
            if not sc_out.empty:
                st.subheader("Mã MUA ngoài các nhóm dẫn đầu")
                st.dataframe(_show(sc_out, stock_cols), hide_index=True, width="stretch",
                             column_config=fmt)
            with st.expander("Toàn bộ mã đã phân tích"):
                groups = ["(tất cả)", *sorted(sc["group_name"].dropna().unique())]
                g = st.selectbox("Nhóm ngành", groups, key="screen_group")
                view = sc[sc["stage"] >= 2]
                if g != "(tất cả)":
                    view = view[view["group_name"] == g]
                st.dataframe(_show(view.sort_values("score", ascending=False), stock_cols),
                             hide_index=True, width="stretch", column_config=fmt)
    with st.expander("Xoay vòng ngành (RRG, tham khảo — chưa kiểm định)", expanded=False):
        try:
            from vnsa.screen import rrg

            tails = rrg.load()
        except Exception as e:  # noqa: BLE001
            st.caption(f"Chưa tính được RRG: {e}")
            tails = pd.DataFrame()
        if tails.empty:
            st.caption("Chưa đủ dữ liệu (cần giá tuần của ≥ 3 mã mỗi nhóm; chạy sàng lọc trước).")
        else:
            st.plotly_chart(charts.rrg_chart(tails), width="stretch")
            st.text(rrg.format_rrg(tails))
    with st.expander("Mã dẫn biến động (lan truyền Diebold–Yilmaz, tham khảo)", expanded=False):
        try:
            from vnsa.ta.spillover import format_spillover

            sp_res, sp_missing = analysis.spillover_view()
            net = sp_res.net.rename("Ròng (lan ra − nhận vào)")
            st.bar_chart(net * 100, height=260)
            st.text(format_spillover(sp_res))
            if sp_missing:
                st.caption("Thiếu giá: " + ", ".join(sp_missing))
        except Exception as e:  # noqa: BLE001
            st.caption(f"Chưa tính được: {e}")

with tab_pf:
    from vnsa import portfolio as pf

    trades = pf.load()
    if trades.empty:
        st.info("Sổ lệnh trống. Ghi lệnh bằng dòng lệnh, ví dụ "
                "`python -m vnsa.cli portfolio buy FPT 500 98500` (giá bằng đồng), rồi tải lại trang. "
                "Sổ lệnh lưu ở data/portfolio/trades.csv và không đưa lên Git.")
    else:
        try:
            cal, last_px, grp = pf.market_context(sorted(trades["symbol"].unique()))
            capital = settings()["plan"]["capital"]
            acts = pf.corporate_actions(sorted(trades["symbol"].unique()))
            tab = pf.status(trades, last_px, grp, capital, calendar=cal, actions=acts)
            realized = pf.realized_total(trades, calendar=cal, actions=acts)
        except ValueError as e:
            st.error(f"Sổ lệnh có lỗi: {e}")
        else:
            c1, c2, c3 = st.columns(3)
            total = float(tab["Giá trị"].sum()) if not tab.empty else 0.0
            c1.metric("Giá trị đang giữ", f"{total:,.0f}đ", f"{total / capital:.0%} vốn")
            c2.metric("Lãi/lỗ tạm tính", f"{tab['Lãi/lỗ'].sum() if not tab.empty else 0:+,.0f}đ")
            c3.metric("Lãi/lỗ đã chốt + cổ tức", f"{realized:+,.0f}đ")
            if not tab.empty:
                heat = float(tab["Rủi ro mở"].sum())
                cap_heat = settings().get("portfolio", {}).get("max_heat", 0.05)
                (st.warning if heat > cap_heat * capital else st.caption)(
                    f"Rủi ro mở (lỗ thêm nếu mọi mã về cắt lỗ): {heat:,.0f}đ = {heat / capital:.1%}"
                    f" vốn, trần {cap_heat:.0%} ([portfolio] max_heat)")
            if not tab.empty:
                view = tab.assign(**{c: tab[c] * 100 for c in ("Lãi/lỗ %", "Tỷ trọng",
                                                                 "Cách cắt lỗ")})
                p1 = st.column_config.NumberColumn(format="%+.1f%%")
                st.dataframe(view, hide_index=True, width="stretch",
                             column_config={"Lãi/lỗ %": p1, "Cách cắt lỗ": p1,
                                            "Tỷ trọng": st.column_config.NumberColumn(
                                                format="%.1f%%")})
                share = tab.groupby("Nhóm ngành")["Giá trị"].sum() / capital
                cap = settings().get("portfolio", {}).get("max_sector_pct", 0.30)
                if (share > cap).any():
                    st.warning("Tập trung ngành vượt {:.0%} vốn: ".format(cap) + ", ".join(
                        f"{g} {w:.0%}" for g, w in share[share > cap].items()))
            st.subheader("Nhật ký giao dịch")
            try:
                from vnsa import journal

                j = journal.load_journal()
            except Exception as e:  # noqa: BLE001
                st.caption(f"Chưa tính được nhật ký: {e}")
                j = None
            if j is not None:
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("TWR phần vốn giải ngân", "—" if pd.isna(j.twr) else f"{j.twr:+.1%}",
                          None if pd.isna(j.twr) else f"{j.twr - j.vni:+.1%} so với VNINDEX")
                s = j.stats
                d2.metric("Tỷ lệ thắng", f"{s['win_rate']:.0%}" if s.get("n") else "—",
                          f"{s['n']} lệnh bán" if s.get("n") else None)
                d3.metric("Payoff", "—" if not s.get("n") or pd.isna(s["payoff"])
                          else f"{s['payoff']:.2f}")
                d4.metric("Điểm kỷ luật", "—" if pd.isna(j.discipline.score)
                          else f"{j.discipline.score:.0f}/100")
                if len(j.curve):
                    st.line_chart(j.curve.set_index("date")[["twr", "vni"]].rename(
                        columns={"twr": "Danh mục (TWR)", "vni": "VNINDEX"}) * 100, height=260)
                with st.expander("Chi tiết nhật ký và kỷ luật"):
                    st.text(journal.format_journal(j))
            with st.expander("Phân bổ cân bằng rủi ro HRP (gợi ý, v4)"):
                try:
                    from vnsa.risk.hrp import format_hrp

                    st.text(format_hrp(analysis.hrp_view()))
                except Exception as e:  # noqa: BLE001
                    st.caption(f"Chưa tính được: {e}")
            with st.expander("Sổ lệnh"):
                st.dataframe(trades.assign(date=trades["date"].dt.strftime("%d/%m/%Y")),
                             hide_index=True, width="stretch")

