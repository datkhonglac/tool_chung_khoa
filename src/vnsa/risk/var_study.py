"""Kiểm định VaR quá khứ (lệnh study-var, nâng cấp v4): mô hình nào dự báo đúng tần suất thua lỗ?

Với mỗi mã, mỗi phiên t (sau `window` phiên đầu), CHỈ dùng dữ liệu đến t dự báo VaR 1 ngày của phiên
t+1 theo 4 cách, rồi xem lợi suất thật phiên t+1 có tệ hơn VaR không ("vượt VaR"):
  lịch sử   phân vị (1 − q) lợi suất `window` phiên gần nhất
  GARCH-N   σ_{t+1} GARCH × phân vị phân phối chuẩn
  GARCH-TN  σ_{t+1} GARCH × phân vị phần dư chuẩn hoá thực nghiệm (cách tool đang dùng)
  Bước nhảy σ_{t+1} GARCH + Merton jump-diffusion (risk/jump.py), chặn biên độ sàn
Tham số GARCH/bước nhảy ước lượng lại mỗi `refit` phiên trên `window` phiên gần nhất; giữa hai lần
ước lượng σ cập nhật theo phương trình GARCH.

Đánh giá: tỷ lệ vượt VaR phải gần 1 − q (VaR 99% → ~1%). Kiểm định Kupiec (POF):
  LR = −2 ln[(1−p)^(n−x) p^x] + 2 ln[(1−x/n)^(n−x) (x/n)^x] ~ χ²(1), p-value > 5%: tần suất đúng.
Giữa các mô hình đạt, mô hình có VaR bình quân THẤP hơn tốt hơn (không giữ vốn dự phòng thừa).
Hạn chế: rổ mã lớn hiện tại (thiên lệch sống sót), giá điều chỉnh.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from vnsa.risk.garch import fit_garch
from vnsa.risk.jump import fit_jumps, jump_var

METHODS = ("lịch sử", "GARCH-N", "GARCH-TN", "bước nhảy")


def kupiec(n: int, x: int, p: float) -> float:
    """p-value kiểm định Kupiec cho x lần vượt trong n phiên, tần suất kỳ vọng p."""
    if n == 0:
        return float("nan")
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    phat = x / n
    ll1 = 0.0 if x in (0, n) else (n - x) * np.log(1 - phat) + x * np.log(phat)
    return float(1 - chi2.cdf(max(-2 * (ll0 - ll1), 0.0), 1))


def backtest_symbol(close: pd.Series, dates: pd.Series, qs=(0.95, 0.99), window: int = 750,
                    refit: int = 21, k: float = 3.0, band: float = 0.07) -> pd.DataFrame:
    """Một dòng mỗi (phiên dự báo, q): date, q, ret (lợi suất đơn phiên sau) và VaR từng cách."""
    c = close.astype(float).to_numpy()
    r = np.diff(np.log(c)) * 100                 # r[i] = lợi suất phiên i+1 (log %)
    simple = np.diff(c) / c[:-1]
    d = pd.to_datetime(dates).to_numpy()[1:]
    n = len(r)
    if n <= window + 2:
        return pd.DataFrame()
    hist = {q: pd.Series(simple).rolling(window).quantile(1 - q).to_numpy() for q in qs}
    rows = []
    fit = jf = None
    s2 = None
    for t in range(window - 1, n - 1):           # dự báo phiên t+1 bằng dữ liệu r[: t+1]
        if fit is None or (t - (window - 1)) % refit == 0:
            fit = fit_garch(r[t + 1 - window: t + 1])
            s2 = fit.next_sigma ** 2
            zq = {q: np.percentile(fit.resid[np.isfinite(fit.resid)], (1 - q) * 100) for q in qs}
            jf = fit_jumps(r[t + 1 - window: t + 1], fit.sigma, k, fit.mu)
        else:
            e = r[t] - fit.mu
            s2 = fit.omega + fit.alpha * e ** 2 + fit.beta * s2
        sig = float(np.sqrt(s2))
        for q in qs:
            to_loss = lambda x: float(min(-(np.exp((fit.mu + sig * x) / 100) - 1), band))  # noqa: E731
            rows.append({"date": d[t + 1], "q": q, "ret": simple[t + 1],
                         "lịch sử": float(-hist[q][t]),
                         "GARCH-N": to_loss(norm.ppf(1 - q)),
                         "GARCH-TN": to_loss(zq[q]),
                         "bước nhảy": jump_var(jf, sig, q, fit.mu, band)})
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Theo (q, cách): số phiên, tỷ lệ vượt VaR, p-value Kupiec, VaR bình quân."""
    out = []
    for q, g in rows.groupby("q"):
        for m in METHODS:
            v = g[["ret", m]].dropna()
            x = int((v["ret"] < -v[m]).sum())
            out.append({"q": q, "method": m, "n": len(v), "exceed": x / len(v) if len(v) else np.nan,
                        "kupiec_p": kupiec(len(v), x, 1 - q), "avg_var": v[m].mean()})
    return pd.DataFrame(out)


@dataclass
class VarStudy:
    symbols: list[str]
    summary: dict[str, pd.DataFrame]
    rows: pd.DataFrame
    skipped: list[str] = field(default_factory=list)


def study(data: dict[str, pd.DataFrame], split: str = "2022-01-01", start: str = "2016-01-01",
          window: int = 750, refit: int = 21, k: float = 3.0, bands: dict | None = None) -> VarStudy:
    frames = []
    for sym, px in data.items():
        df = backtest_symbol(px["close"], px["date"], window=window, refit=refit, k=k,
                             band=(bands or {}).get(sym, 0.07))
        if not df.empty:
            frames.append(df.assign(symbol=sym))
    if not frames:
        raise ValueError("không mã nào đủ dữ liệu (cần > 800 phiên)")
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[rows["date"] >= pd.Timestamp(start)]
    t1 = pd.Timestamp(split)
    summ = {f"{pd.Timestamp(start).year}–{t1.year - 1}": summarize(rows[rows["date"] < t1]),
            f"{t1.year}–nay": summarize(rows[rows["date"] >= t1]),
            "Toàn kỳ": summarize(rows)}
    return VarStudy(sorted(data), summ, rows)


def _p(x, d=1):
    return "—" if pd.isna(x) else f"{x:.{d}%}"


def format_var_study(res: VarStudy) -> str:
    lines = [f"Kiểm định VaR 1 ngày trên {len(res.symbols)} mã: {', '.join(res.symbols)}",
             "Mỗi phiên dự báo VaR chỉ từ dữ liệu quá khứ; 'vượt' = lỗ phiên sau lớn hơn VaR. Tần suất "
             "đúng: VaR 95% ~5%, VaR 99% ~1%; Kupiec p > 5% = đạt.", ""]
    lines.append(f"{'Giai đoạn':<12}{'VaR':>5}  {'Cách':<11}{'Số phiên':>9}{'Vượt':>8}{'Kupiec p':>10}"
                 f"{'VaR TB':>8}  Đạt")
    for period, tab in res.summary.items():
        for r in tab.itertuples():
            ok = "✓" if not pd.isna(r.kupiec_p) and r.kupiec_p > 0.05 else "✗"
            lines.append(f"{period:<12}{r.q:>5.0%}  {r.method:<11}{r.n:>9,}{_p(r.exceed, 2):>8}"
                         f"{_p(r.kupiec_p, 0):>10}{_p(r.avg_var):>8}  {ok}")
    lines += ["", "Đọc kết quả: chọn cách ĐẠT ở cả hai giai đoạn với VaR bình quân thấp nhất. "
              "Gộp nhiều mã nên tỷ lệ vượt chính xác hơn từng mã; Kupiec trên dữ liệu gộp khắt khe "
              "(n lớn) — chênh 0,2–0,3 điểm % đã có thể trượt."]
    return "\n".join(lines)
