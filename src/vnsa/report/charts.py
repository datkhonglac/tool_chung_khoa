"""Biểu đồ nến + MA + khối lượng để kiểm tra dữ liệu bằng mắt."""

from __future__ import annotations

import pandas as pd


def candlestick(df: pd.DataFrame, title: str | None = None, mas: tuple[int, ...] = (20, 50, 200)):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    symbol = df["symbol"].iloc[0] if len(df) else ""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Giá", increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
    ), row=1, col=1)
    for n in mas:
        if len(df) >= n:
            fig.add_trace(go.Scatter(x=df["date"], y=df["close"].rolling(n).mean(),
                                     name=f"MA{n}", line={"width": 1.2}), row=1, col=1)
    colors = ["#16a34a" if c >= o else "#dc2626" for o, c in zip(df["open"], df["close"])]
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="Khối lượng",
                         marker_color=colors, opacity=0.6), row=2, col=1)
    fig.update_layout(
        title=title or f"{symbol} — giá ngày",
        xaxis_rangeslider_visible=False, height=650, template="plotly_white",
        legend={"orientation": "h", "y": 1.02, "x": 0},
        yaxis_title="Giá (đồng)", yaxis2_title="KL",
    )
    # Ẩn cuối tuần để biểu đồ liền mạch
    fig.update_xaxes(rangebreaks=[{"bounds": ["sat", "mon"]}])
    return fig


def technical_chart(ind: pd.DataFrame, supports=(), resistances=(), plan=None, sessions: int = 250):
    """Nến + SMA + Bollinger + hỗ trợ/kháng cự (+ cắt lỗ/mục tiêu), khối lượng và RSI."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = ind.tail(sessions)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Giá", increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
    ), row=1, col=1)
    for col, color in (("sma20", "#f59e0b"), ("sma50", "#2563eb"), ("sma200", "#7c3aed")):
        if col in df:
            fig.add_trace(go.Scatter(x=df["date"], y=df[col], name=col.upper(),
                                     line={"width": 1.2, "color": color}), row=1, col=1)
    for col in ("bb_upper", "bb_lower"):
        if col in df:
            fig.add_trace(go.Scatter(x=df["date"], y=df[col], name="Bollinger",
                                     line={"width": 0.8, "dash": "dot", "color": "#94a3b8"},
                                     showlegend=col == "bb_upper"), row=1, col=1)
    # Mức giá ngang vẽ thành đường có tên trong chú thích (tránh nhãn chồng lên nhau)
    x = [df["date"].iloc[0], df["date"].iloc[-1]]
    lines = [(f"HT {lv.price:,.0f}", lv.price, "#16a34a", "dash", 1) for lv in list(supports)[:3]]
    lines += [(f"KC {lv.price:,.0f}", lv.price, "#dc2626", "dash", 1)
              for lv in list(resistances)[:3]]
    if plan is not None:
        lines.append((f"Cắt lỗ {plan.stop:,.0f}", plan.stop, "#b91c1c", "solid", 2))
        lines += [(f"Mục tiêu {t.price:,.0f}", t.price, "#15803d", "solid", 2) for t in plan.targets]
    for name, y, color, dash, width in lines:
        fig.add_trace(go.Scatter(x=x, y=[y, y], name=name, mode="lines",
                                 line={"color": color, "dash": dash, "width": width}),
                      row=1, col=1)
    colors = ["#16a34a" if c >= o else "#dc2626" for o, c in zip(df["open"], df["close"])]
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="Khối lượng", marker_color=colors,
                         opacity=0.6, showlegend=False), row=2, col=1)
    if "rsi" in df:
        fig.add_trace(go.Scatter(x=df["date"], y=df["rsi"], name="RSI14",
                                 line={"color": "#0f766e"}), row=3, col=1)
        for y in (30, 70):
            fig.add_hline(y=y, line={"color": "#94a3b8", "width": 1, "dash": "dot"}, row=3, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, height=720, template="plotly_white",
                      legend={"orientation": "h", "y": 1.03, "x": 0}, margin={"t": 40},
                      yaxis_title="Giá (đồng)", yaxis2_title="KL", yaxis3_title="RSI")
    fig.update_xaxes(rangebreaks=[{"bounds": ["sat", "mon"]}])
    return fig


def fan_chart(sim, dates: pd.Series | None = None, history: pd.DataFrame | None = None):
    """Biểu đồ quạt Monte Carlo: dải P10–P90 và P25–P75 quanh trung vị theo từng phiên."""
    import numpy as np
    import plotly.graph_objects as go

    days = np.arange(sim.paths.shape[1])
    q = {p: np.percentile(sim.paths, p, axis=0) for p in (10, 25, 50, 75, 90)}
    fig = go.Figure()
    if history is not None and len(history):
        h = history.tail(120)
        fig.add_trace(go.Scatter(x=np.arange(-len(h) + 1, 1), y=h["close"], name="Giá quá khứ",
                                 line={"color": "#334155"}))
    for lo, hi, alpha in ((10, 90, 0.15), (25, 75, 0.3)):
        fig.add_trace(go.Scatter(x=np.r_[days, days[::-1]], y=np.r_[q[hi], q[lo][::-1]],
                                 fill="toself", fillcolor=f"rgba(37,99,235,{alpha})",
                                 line={"width": 0}, name=f"P{lo}–P{hi}"))
    fig.add_trace(go.Scatter(x=days, y=q[50], name="Trung vị (P50)", line={"color": "#1d4ed8"}))
    fig.update_layout(template="plotly_white", height=420, xaxis_title="Phiên (0 = hôm nay)",
                      yaxis_title="Giá (đồng)", margin={"t": 30})
    return fig


def equity_chart(result):
    """Đường tài sản của chiến lược so với mua và nắm giữ (vốn ban đầu = 1)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity, name="Chiến lược điểm TA"))
    fig.add_trace(go.Scatter(x=result.benchmark.index, y=result.benchmark, name="Mua & giữ",
                             line={"dash": "dot"}))
    fig.update_layout(template="plotly_white", height=380, yaxis_title="Giá trị (vốn đầu = 1)",
                      margin={"t": 30}, legend={"orientation": "h", "y": 1.05})
    return fig


def sensitivity_heatmap(table: pd.DataFrame, price: float):
    """Ma trận độ nhạy giá trị/cp (nghìn đồng); màu theo chênh lệch so với giá hiện tại."""
    import plotly.graph_objects as go

    z = (table / price - 1).to_numpy() * 100
    text = (table / 1000).round(1).astype(str).to_numpy()
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"g = {g:.1%}" for g in table.columns], y=[f"WACC {w:.1%}" for w in table.index],
        text=text, texttemplate="%{text}", colorscale="RdYlGn", zmid=0,
        colorbar={"title": "% so với giá"}))
    fig.update_layout(height=360, margin={"t": 30}, template="plotly_white")
    return fig


QUADRANT_COLORS = {"Dẫn dắt": "#16a34a", "Suy yếu": "#d97706", "Tụt hậu": "#dc2626",
                   "Cải thiện": "#2563eb"}


def rrg_chart(tails: pd.DataFrame):
    """RRG: mỗi nhóm một đuôi (các tuần gần nhất), điểm cuối to hơn, màu theo góc phần tư."""
    import plotly.graph_objects as go

    fig = go.Figure()
    if tails.empty:
        return fig
    lo_x, hi_x = tails["ratio"].min(), tails["ratio"].max()
    lo_y, hi_y = tails["momentum"].min(), tails["momentum"].max()
    pad_x, pad_y = max(1.0, (hi_x - lo_x) * 0.1), max(1.0, (hi_y - lo_y) * 0.1)
    x0, x1 = min(lo_x, 100) - pad_x, max(hi_x, 100) + pad_x
    y0, y1 = min(lo_y, 100) - pad_y, max(hi_y, 100) + pad_y
    for (xa, xb, ya, yb, name) in [(100, x1, 100, y1, "Dẫn dắt"), (100, x1, y0, 100, "Suy yếu"),
                                    (x0, 100, y0, 100, "Tụt hậu"), (x0, 100, 100, y1, "Cải thiện")]:
        fig.add_shape(type="rect", x0=xa, x1=xb, y0=ya, y1=yb, line_width=0,
                      fillcolor=QUADRANT_COLORS[name], opacity=0.06, layer="below")
        fig.add_annotation(x=(xa + xb) / 2, y=yb if yb > 100 else ya, text=name, showarrow=False,
                           font={"color": QUADRANT_COLORS[name]}, yshift=-12 if yb > 100 else 12)
    for g, x in tails.groupby("group"):
        last = x.iloc[-1]
        color = QUADRANT_COLORS.get(last["quadrant"], "#64748b")
        fig.add_trace(go.Scatter(
            x=x["ratio"], y=x["momentum"], mode="lines+markers", name=g,
            line={"color": color, "width": 1.5}, marker={"size": [4] * (len(x) - 1) + [11],
                                                          "color": color},
            text=[f"{g} · {d:%d/%m}" for d in x["date"]],
            hovertemplate="%{text}<br>RS-Ratio %{x:.1f} · RS-Mom %{y:.1f}<extra></extra>"))
        fig.add_annotation(x=last["ratio"], y=last["momentum"], text=g, showarrow=False,
                           yshift=12, font={"size": 11})
    fig.add_hline(y=100, line_color="#94a3b8", line_width=1)
    fig.add_vline(x=100, line_color="#94a3b8", line_width=1)
    fig.update_layout(height=620, template="plotly_white", showlegend=False,
                      xaxis={"title": "RS-Ratio (sức mạnh so với VNINDEX)", "range": [x0, x1]},
                      yaxis={"title": "RS-Momentum (tăng tốc)", "range": [y0, y1]},
                      margin={"t": 30})
    return fig
