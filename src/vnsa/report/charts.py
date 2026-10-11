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
