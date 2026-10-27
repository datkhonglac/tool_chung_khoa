"""Kiểm định HRP (lệnh study-hrp, nâng cấp v4): HRP có giữ danh mục an toàn hơn chia đều không?

Rổ [study] universe, cân lại đầu mỗi tháng chỉ với dữ liệu trước đó (`sessions` phiên):
  Chia đều       mỗi mã có giá 1/N
  Nghịch biến động  w ∝ 1/σ_i
  HRP            risk/hrp.py
Lợi suất danh mục theo ngày, trừ phí khi đổi tỷ trọng ([backtest] fee, tax). Chỉ số: lợi suất/năm,
biến động, Sharpe, sụt giảm tối đa. HRP đáng dùng khi sụt giảm nhỏ hơn và Sharpe không thấp hơn chia
đều ở CẢ HAI giai đoạn. Hạn chế: rổ mã lớn hiện tại (thiên lệch sống sót), không chọn mã.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vnsa.recommend.hmm_study import _p, _stats
from vnsa.risk.hrp import hrp_weights

METHODS = ("Chia đều", "Nghịch biến động", "HRP")


def _weights(hist: pd.DataFrame) -> dict[str, pd.Series]:
    ok = hist.dropna(axis=1, thresh=int(len(hist) * 0.9)).dropna()
    cols = list(ok.columns)
    if len(cols) < 2:
        return {}
    eq = pd.Series(1 / len(cols), index=cols)
    iv = 1 / ok.std()
    return {"Chia đều": eq, "Nghịch biến động": iv / iv.sum(), "HRP": hrp_weights(ok)}


def study(returns: pd.DataFrame, bt: dict, split: str = "2022-01-01", start: str = "2016-01-01",
          sessions: int = 250) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """returns: lợi suất ngày (cột = mã). Trả (bảng chỉ số theo giai đoạn × cách, lợi suất ngày)."""
    r = returns.sort_index()
    fee, tax = float(bt.get("fee", 0.0015)), float(bt.get("tax", 0.001))
    days = r.index[r.index >= pd.Timestamp(start)]
    months = pd.Series(days).dt.to_period("M")
    out = {m: [] for m in METHODS}
    prev = {m: pd.Series(dtype=float) for m in METHODS}
    for per in months.unique():
        block = days[(months == per).to_numpy()]
        hist = r[r.index < block[0]].tail(sessions)
        ws = _weights(hist)
        if not ws:
            continue
        for m, w in ws.items():
            turn = w.subtract(prev[m], fill_value=0.0)
            cost = turn.clip(lower=0).sum() * fee + (-turn).clip(lower=0).sum() * (fee + tax)
            daily = r.loc[block, w.index].fillna(0.0) @ w
            daily.iloc[0] -= cost
            out[m].append(daily)
            prev[m] = w            # (bỏ qua trôi tỷ trọng trong tháng — như nhau ở mọi cách)
    rets = {m: pd.concat(v) if v else pd.Series(dtype=float) for m, v in out.items()}
    t0, t1 = pd.Timestamp(start), pd.Timestamp(split)
    periods = {f"{t0.year}–{t1.year - 1}": (t0, t1 - pd.Timedelta(days=1)),
               f"{t1.year}–nay": (t1, pd.Timestamp.max), "Toàn kỳ": (t0, pd.Timestamp.max)}
    rows = [{"period": p, "method": m, **_stats(rets[m][(rets[m].index >= a) & (rets[m].index <= b)])}
            for p, (a, b) in periods.items() for m in METHODS]
    return pd.DataFrame(rows), rets


@dataclass
class HrpStudy:
    table: pd.DataFrame
    symbols: list[str]


def format_hrp_study(res: HrpStudy) -> str:
    lines = [f"Kiểm định HRP trên {len(res.symbols)} mã (cân lại đầu mỗi tháng, 250 phiên gần nhất)",
             "", f"{'Giai đoạn':<12}{'Cách':<18}{'Lợi suất/năm':>13}{'Biến động':>11}{'Sharpe':>8}"
                 f"{'Sụt giảm max':>14}"]
    for r in res.table.itertuples():
        lines.append(f"{r.period:<12}{r.method:<18}{_p(r.cagr, True):>13}{_p(r.vol):>11}"
                     f"{'—' if pd.isna(r.sharpe) else f'{r.sharpe:.2f}':>8}{_p(r.maxdd):>14}")
    lines += ["", "Đọc kết quả: HRP đáng thay trần tỷ trọng hiện tại khi sụt giảm nhỏ hơn VÀ Sharpe "
              "không thấp hơn Chia đều ở CẢ HAI giai đoạn."]
    return "\n".join(lines)


def returns_from_prices(prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = {s: p.set_index(pd.to_datetime(p["date"]))["close"].astype(float).pct_change()
            for s, p in prices.items() if len(p)}
    df = pd.DataFrame(cols)
    return df.replace([np.inf, -np.inf], np.nan)
