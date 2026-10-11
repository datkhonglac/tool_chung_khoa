"""Dashboard Streamlit cho VN Stock Analyzer.

Chạy:  streamlit run dashboard/app.py
Cài:   pip install -e ".[dashboard]"

Chỉ đọc kho DuckDB (data/warehouse.duckdb) qua vnsa.analysis — cùng con số với dòng lệnh.
Cập nhật dữ liệu bằng dòng lệnh: python -m vnsa.cli fetch && python -m vnsa.cli fetch-bctc
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from vnsa import analysis
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
            rec, _ = analysis.recommendation(sym)
        except (analysis.DataMissing, ValueError) as e:
            errors.append(str(e))
            continue
        x = rec.inputs
        rows.append({"Mã": rec.symbol, "Giá": x.price, "Điểm FA": x.fa_score,
                     "Điểm TA": x.ta_score, "Điểm tổng": rec.score, "DCF": x.dcf_upside,
                     "P/E": x.pe, "P/B": x.pb, "Biến động/năm": x.vol_annual,
                     "Khuyến nghị": rec.action, "Lý do": "; ".join(rec.reasons)})
    return pd.DataFrame(rows), errors


@st.cache_data(show_spinner=False)
def load(symbol: str, _stamp: float):
    return analysis.load(symbol, need_fin=False)


def fmt_pct(x) -> str:
    return "—" if pd.isna(x) else f"{x:.1%}"


def fmt_num(x, d: int = 0) -> str:
    return "—" if pd.isna(x) else f"{x:,.{d}f}"


stamp = _stamp()
st.title("📈 VN Stock Analyzer")
if not stamp:
    st.error("Chưa có kho dữ liệu. Chạy `python -m vnsa.cli fetch` và `python -m vnsa.cli "
             "fetch-bctc` (hoặc `python -m vnsa.cli load --sample` để thử) rồi tải lại trang.")
    st.stop()

available = analysis.symbols()
watch = [s for s in settings()["watchlist"]["symbols"] if s in available] or available
with st.sidebar:
    st.header("Chọn mã")
    symbol = st.selectbox("Mã cổ phiếu", available, index=available.index(watch[0]) if watch else 0)
    st.caption(f"Kho: {len(available)} mã. Dữ liệu chỉ cập nhật khi chạy lệnh fetch.")
    st.caption("Kết quả là sàng lọc theo quy tắc trong config/settings.toml, "
               "không phải lời khuyên đầu tư.")

tab_all, tab_ta, tab_fa, tab_val, tab_risk, tab_bt = st.tabs(
    ["Tổng quan", "Kỹ thuật & kế hoạch", "Tài chính", "Định giá", "Rủi ro", "Backtest"])

with tab_all:
    st.subheader("Khuyến nghị danh sách theo dõi")
    with st.spinner("Đang tính..."):
        table, errors = recommendations(tuple(watch), stamp)
    if len(table):
        st.dataframe(
            table, hide_index=True, use_container_width=True,
            column_config={
                "Giá": st.column_config.NumberColumn(format="localized"),
                "Điểm FA": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                "Điểm TA": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                "Điểm tổng": st.column_config.NumberColumn(format="%.0f"),
                "DCF": st.column_config.NumberColumn(format="percent"),
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
    st.plotly_chart(charts.technical_chart(ind, sup, res, plan), use_container_width=True)
    st.caption(" · ".join(f"{k}: {fmt_num(v)}" for k, v in t.groups.items()))
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

        f, z, s, _ = analysis.fa(data)
        c = st.columns(3)
        c[0].metric(f"Điểm FA ({s.rating.lower()})", fmt_num(s.total))
        c[1].metric("F-Score năm gần nhất", fmt_num(f["f_score"].iloc[-1]) + "/9")
        c[2].metric(f"Altman Z'' ({z_zone(z['z_em'].iloc[-1])})", fmt_num(z["z_em"].iloc[-1], 2))
        st.subheader("Chỉ số tài chính")
        st.dataframe(format_ratios(compute_ratios(data.fin)), use_container_width=True)
        st.subheader("Piotroski F-Score")
        ft = f.set_index("year")[[*FSCORE_CRITERIA, "f_score"]].T
        ft.index = [*FSCORE_CRITERIA.values(), "F-Score (/9)"]
        st.dataframe(ft, use_container_width=True)
        st.subheader("Điểm FA theo nhóm")
        st.dataframe(pd.DataFrame([{"Nhóm": g, "Điểm": s.groups[g],
                                    "Chỉ tiêu": ", ".join(f"{k} = {fmt_num(s.items[k][0], 2)}"
                                                          for k in keys)}
                                   for g, keys in GROUPS.items()]), hide_index=True)

with tab_val:
    if data.fin.empty:
        st.info(f"Chưa có BCTC. Chạy `python -m vnsa.cli fetch-bctc {symbol}`.")
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
            table = val.sensitivity(v, a)
            if table.notna().any().any():
                st.subheader("Độ nhạy giá trị/cp (nghìn đồng)")
                st.plotly_chart(charts.sensitivity_heatmap(table, v.price), use_container_width=True)
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
        st.plotly_chart(charts.fan_chart(r.sim, history=data.prices), use_container_width=True)
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

    cfg = dict(settings()["backtest"], ticks=settings()["plan"].get("ticks"))
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
        st.plotly_chart(charts.equity_chart(res), use_container_width=True)
        st.dataframe(pd.DataFrame([{"Mua": t.entry_date, "Giá mua": t.entry, "Bán": t.exit_date,
                                    "Giá bán": t.exit, "Lãi/lỗ": t.ret, "Lý do": t.reason}
                                   for t in res.trades]), hide_index=True,
                     column_config={"Lãi/lỗ": st.column_config.NumberColumn(format="percent"),
                                    "Mua": st.column_config.DateColumn(format="DD/MM/YYYY"),
                                    "Bán": st.column_config.DateColumn(format="DD/MM/YYYY"),
                                    "Giá mua": st.column_config.NumberColumn(format="localized"),
                                    "Giá bán": st.column_config.NumberColumn(format="localized")})
        st.caption("Kết quả quá khứ không đảm bảo tương lai; chưa tính trượt giá.")
