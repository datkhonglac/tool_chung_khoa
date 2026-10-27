"""Kiểm định HMM trạng thái VNINDEX (lệnh study-hmm, nâng cấp v4) — có nên siết rủi ro theo HMM?

Walk-forward không nhìn trước:
1. Đầu mỗi tháng, ước lượng lại HMM trên `train_sessions` phiên GẦN NHẤT TRƯỚC đó.
2. Trong tháng, mỗi phiên t tính xác suất lọc P(S_t | dữ liệu đến t) với tham số cố định, dự báo
   P(ngày mai ∈ Phân phối/Suy thoái) = p_xấu.
3. So sánh 3 cách phân bổ vốn vào VNINDEX (đại diện "ngân sách rủi ro" của cả danh mục), tỷ trọng
   quyết định cuối phiên t áp cho lợi suất phiên t+1, trừ phí khi đổi tỷ trọng ([backtest] fee, tax):
     Mua & giữ          100%
     MA200 (đang dùng)  50% khi VNINDEX ở trạng thái "giảm" (ta/regime.py), còn lại 100%
     HMM                50% khi p_xấu > `threshold` (70% như blueprint), còn lại 100%
Chỉ đưa HMM vào kế hoạch khi HMM tốt hơn MA200 (Sharpe và sụt giảm tối đa) ở CẢ HAI giai đoạn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vnsa.ta.hmm_regime import BAD, features, filtered, fit


def hmm_signal(index: pd.DataFrame, cfg: dict, start: str = "2016-01-01") -> pd.Series:
    """p_xấu theo ngày (chỉ dùng dữ liệu quá khứ), từ `start`."""
    f = features(index, int(cfg.get("ret_days", 5)), int(cfg.get("vol_days", 20)))
    X = f.to_numpy()
    train, k = int(cfg.get("train_sessions", 1500)), int(cfg.get("states", 4))
    months = pd.Series(f.index).dt.to_period("M")
    out = pd.Series(np.nan, index=f.index)
    m = None
    for per in months[f.index >= pd.Timestamp(start)].unique():
        idx = np.flatnonzero((months == per).to_numpy())
        first = idx[0]
        if first < 50 * k:
            continue
        lo = max(0, first - train)
        # tháng đầu: nhiều điểm khởi tạo; các tháng sau: khởi tạo ấm từ mô hình tháng trước + 1 điểm
        m = fit(X[lo:first], k, seeds=int(cfg.get("seeds", 3)) if m is None else 1, init=m)
        # lọc từ đầu cửa sổ huấn luyện tới hết tháng: xác suất ở phiên t chỉ dùng dữ liệu đến t
        p = filtered(m, X[lo: idx[-1] + 1]) @ m.A
        bad = [i for i, n in enumerate(m.names) if n in BAD]
        out.iloc[idx] = p[first - lo:, bad].sum(axis=1)
    return out.dropna()


def _stats(r: pd.Series) -> dict:
    if r.empty:
        return {"cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "maxdd": np.nan}
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    return {"cagr": eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan,
            "vol": r.std() * np.sqrt(252),
            "sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
            "maxdd": (eq / eq.cummax() - 1).min()}


def strategy_returns(index: pd.DataFrame, exposure: pd.Series, fee: float, tax: float) -> pd.Series:
    c = index.sort_values("date").set_index(pd.to_datetime(index["date"]))["close"].astype(float)
    r = c.pct_change().shift(-1)                          # lợi suất phiên sau
    e = exposure.reindex(r.index).ffill().dropna()
    r = r.reindex(e.index)
    de = e.diff().fillna(0)
    cost = de.abs() * fee + (-de).clip(lower=0) * tax
    return (e * r - cost).dropna()


@dataclass
class HmmStudy:
    table: pd.DataFrame          # giai đoạn × cách -> chỉ số
    signal: pd.Series
    share_bad: float             # % phiên HMM siết


def study(index: pd.DataFrame, ma_regime: pd.Series, cfg: dict, bt: dict, split: str = "2022-01-01",
          start: str = "2016-01-01") -> HmmStudy:
    sig = hmm_signal(index, cfg, start)
    thr, cut = float(cfg.get("threshold", 0.7)), float(cfg.get("risk_factor", 0.5))
    days = sig.index
    exp = {"Mua & giữ": pd.Series(1.0, index=days),
           "MA200": pd.Series(np.where(ma_regime.reindex(days).ffill() == "giảm", cut, 1.0), days),
           "HMM": pd.Series(np.where(sig > thr, cut, 1.0), index=days)}
    fee, tax = float(bt.get("fee", 0.0015)), float(bt.get("tax", 0.001))
    rets = {k: strategy_returns(index, v, fee, tax) for k, v in exp.items()}
    t0, t1 = pd.Timestamp(start), pd.Timestamp(split)
    periods = {f"{t0.year}–{t1.year - 1}": (t0, t1 - pd.Timedelta(days=1)),
               f"{t1.year}–nay": (t1, pd.Timestamp.max), "Toàn kỳ": (t0, pd.Timestamp.max)}
    rows = []
    for pname, (a, b) in periods.items():
        for k, r in rets.items():
            sub = r[(r.index >= a) & (r.index <= b)]
            rows.append({"period": pname, "method": k, **_stats(sub),
                         "switches": int((exp[k].diff().abs() > 0)[(days >= a) & (days <= b)].sum())})
    return HmmStudy(pd.DataFrame(rows), sig, float((sig > thr).mean()))


def _p(x, sign=False):
    return "—" if pd.isna(x) else (f"{x:+.1%}" if sign else f"{x:.1%}")


def format_hmm_study(res: HmmStudy, cfg: dict) -> str:
    lines = ["Kiểm định HMM trạng thái VNINDEX (walk-forward, ước lượng lại mỗi tháng)",
             f"HMM: giảm tỷ trọng còn {float(cfg.get('risk_factor', 0.5)):.0%} khi P(ngày mai Phân "
             f"phối/Suy thoái) > {float(cfg.get('threshold', 0.7)):.0%} — xảy ra ở "
             f"{res.share_bad:.0%} số phiên. MA200: giảm còn 50% khi VNINDEX 'giảm' (đang dùng).", "",
             f"{'Giai đoạn':<12}{'Cách':<11}{'Lợi suất/năm':>13}{'Biến động':>11}{'Sharpe':>8}"
             f"{'Sụt giảm max':>14}{'Số lần đổi':>11}"]
    for r in res.table.itertuples():
        lines.append(f"{r.period:<12}{r.method:<11}{_p(r.cagr, True):>13}{_p(r.vol):>11}"
                     f"{'—' if pd.isna(r.sharpe) else f'{r.sharpe:.2f}':>8}{_p(r.maxdd):>14}"
                     f"{r.switches:>11}")
    lines += ["", "Đọc kết quả: HMM đáng dùng để siết rủi ro khi Sharpe cao hơn VÀ sụt giảm nhỏ hơn "
              "MA200 ở CẢ HAI giai đoạn. Đây là VNINDEX (đại diện), chưa tính chọn mã."]
    return "\n".join(lines)
