"""Kiểm định quy tắc thời điểm trên nhiều mã, nhiều năm (đợt 4) — trả lời câu hỏi: trục "Thời điểm"
của tool có tốt hơn mua và nắm giữ không?

Phương pháp (walk-forward, chống "khớp quá mức" với quá khứ):
1. Giai đoạn chọn tham số (in-sample, mặc định 2016–2021): với mỗi quy tắc (strategies.rule_set) và
   mỗi hệ số cắt lỗ động k ∈ `atr_grid`, mô phỏng từng mã rồi gộp thành DANH MỤC chia đều vốn cho
   các mã (lợi nhuận ngày = trung bình lợi nhuận ngày của các mã). Chọn k có Sharpe danh mục cao nhất.
2. Giai đoạn kiểm tra độc lập (out-of-sample, 2022 → nay): chạy lại với k đã chọn, KHÔNG chỉnh gì.
3. So với danh mục mua & giữ cùng các mã. Một quy tắc "đạt" khi ở CẢ HAI giai đoạn có Sharpe cao hơn
   VÀ sụt giảm tối đa nhỏ hơn mua & giữ (cải thiện lợi nhuận đã điều chỉnh rủi ro, không chỉ may mắn
   ở một giai đoạn).

Hạn chế ghi rõ trong báo cáo: rổ mã là các mã lớn hiện tại (thiên lệch sống sót), giá điều chỉnh
(trần/sàn xấp xỉ), không tính trượt giá ngoài phí + thuế.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vnsa.recommend.backtest import TRADING_DAYS
from vnsa.recommend.strategies import NAMES, rule_set, simulate


@dataclass
class PeriodStats:
    cagr: float
    sharpe: float
    max_dd: float
    exposure: float = float("nan")
    trades: int = 0
    win_rate: float = float("nan")
    beat_hold: float = float("nan")     # tỷ lệ mã có Sharpe cao hơn mua & giữ của chính mã


@dataclass
class StudyResult:
    periods: dict[str, tuple[str, str]]                      # tên -> (từ, đến)
    table: pd.DataFrame                                      # dòng = (quy tắc, giai đoạn)
    chosen_k: dict[str, float] = field(default_factory=dict)
    passed: list[str] = field(default_factory=list)          # quy tắc đạt ở cả hai giai đoạn
    symbols: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def curve_stats(daily: pd.Series) -> tuple[float, float, float]:
    """(CAGR, Sharpe với r_f = 0, sụt giảm tối đa) từ chuỗi lợi nhuận ngày."""
    daily = daily.dropna()
    if len(daily) < 20:
        return float("nan"), float("nan"), float("nan")
    curve = (1 + daily).cumprod()
    years = len(daily) / TRADING_DAYS
    total = curve.iloc[-1] - 1
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else float("nan")
    sd = daily.std()
    sharpe = daily.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else float("nan")
    dd = (curve / curve.cummax() - 1).min()
    return float(cagr), float(sharpe), float(dd)


def _portfolio(curves: list[pd.Series]) -> pd.Series:
    """Danh mục chia đều: lợi nhuận ngày = trung bình các mã có giao dịch phiên đó."""
    rets = pd.concat([c.pct_change() for c in curves], axis=1)
    return rets.mean(axis=1, skipna=True).iloc[1:]


def run_period(data: dict[str, dict], name: str, k: float, start: pd.Timestamp,
               end: pd.Timestamp, cfg: dict) -> tuple[PeriodStats, PeriodStats]:
    """(chỉ số của quy tắc `name`, chỉ số mua & giữ) trên danh mục các mã trong `data`.
    data[sym] = {"ind", "score", "band"}; regime chung ở data["__regime__"] nếu có."""
    regime = data.get("__regime__", {}).get("series")
    curves, holds, trades, beats, exposures = [], [], [], [], []
    for sym, x in data.items():
        if sym.startswith("__"):
            continue
        rule = rule_set(x["ind"], x["score"], regime, k, cfg)[name]
        try:
            res = simulate(x["ind"], rule, x["band"], cfg["fee"], cfg["sell_tax"], start, end)
        except ValueError:
            continue
        curves.append(res.equity)
        holds.append(res.benchmark)
        trades += [t.ret for t in res.trades]
        exposures.append(res.exposure)
        s, b = res.stats["sharpe"], res.benchmark_stats["sharpe"]
        if not (pd.isna(s) or pd.isna(b)):
            beats.append(s > b)
    if not curves:
        nan = float("nan")
        return PeriodStats(nan, nan, nan), PeriodStats(nan, nan, nan)
    strat = PeriodStats(*curve_stats(_portfolio(curves)), exposure=float(np.mean(exposures)),
                        trades=len(trades),
                        win_rate=float(np.mean([r > 0 for r in trades])) if trades else float("nan"),
                        beat_hold=float(np.mean(beats)) if beats else float("nan"))
    hold = PeriodStats(*curve_stats(_portfolio(holds)), exposure=1.0)
    return strat, hold


def study(data: dict[str, dict], cfg: dict, split: str = "2022-01-01", start: str = "2016-01-01",
          atr_grid: tuple[float, ...] = (2.5, 3.0, 4.0)) -> StudyResult:
    """Chạy toàn bộ kiểm định. cfg = mục [backtest]."""
    t0, t1 = pd.Timestamp(start), pd.Timestamp(split)
    ends = [pd.Timestamp(x["ind"]["date"].max()) for s, x in data.items() if not s.startswith("__")]
    t2 = max(ends)
    periods = {"Chọn tham số": (t0, t1 - pd.Timedelta(days=1)), "Kiểm tra độc lập": (t1, t2)}
    names = [n for n in NAMES if n != "hold"]
    rows, chosen = [], {}
    hold_rows = {}
    for name in names:
        trail = name not in ("ta", "ta_vni")
        grid = atr_grid if trail else (cfg.get("atr_stop", 2.0),)
        best = None
        for k in grid:
            st, hold = run_period(data, name, k, *periods["Chọn tham số"], cfg)
            if best is None or (not pd.isna(st.sharpe) and (pd.isna(best[1].sharpe)
                                                           or st.sharpe > best[1].sharpe)):
                best = (k, st, hold)
        k, st_in, hold_in = best
        chosen[name] = k
        st_out, hold_out = run_period(data, name, k, *periods["Kiểm tra độc lập"], cfg)
        hold_rows = {"Chọn tham số": hold_in, "Kiểm tra độc lập": hold_out}
        for period, st in (("Chọn tham số", st_in), ("Kiểm tra độc lập", st_out)):
            rows.append({"rule": name, "period": period, "k": k, **st.__dict__})
    for period, hold in hold_rows.items():
        rows.append({"rule": "hold", "period": period, "k": np.nan, **hold.__dict__})
    table = pd.DataFrame(rows)
    passed = []
    for name in names:
        ok = True
        for period in periods:
            r = table[(table["rule"] == name) & (table["period"] == period)].iloc[0]
            h = table[(table["rule"] == "hold") & (table["period"] == period)].iloc[0]
            ok &= bool(r["sharpe"] > h["sharpe"] and r["max_dd"] > h["max_dd"])
        if ok:
            passed.append(name)
    syms = [s for s in data if not s.startswith("__")]
    return StudyResult({p: (f"{a:%d/%m/%Y}", f"{b:%d/%m/%Y}") for p, (a, b) in periods.items()},
                       table, chosen, passed, syms)


def format_study(r: StudyResult) -> str:
    lines = [f"Kiểm định quy tắc thời điểm trên {len(r.symbols)} mã (danh mục chia đều vốn): "
             + ", ".join(r.symbols)]
    for p, (a, b) in r.periods.items():
        lines.append(f"  {p}: {a} → {b}")
    lines.append("")
    head = (f"{'Quy tắc':<24}{'Giai đoạn':<18}{'k ATR':>6}{'CAGR':>8}{'Sharpe':>8}{'Sụt giảm':>10}"
            f"{'Cầm cp':>8}{'Lệnh':>6}{'Thắng':>7}{'Mã thắng M&G':>14}")
    lines.append(head)
    for row in r.table.itertuples():
        lines.append(
            f"{NAMES.get(row.rule, row.rule):<24}{row.period:<18}"
            f"{'' if pd.isna(row.k) else f'{row.k:g}':>6}{row.cagr:>8.1%}{row.sharpe:>8.2f}"
            f"{row.max_dd:>10.1%}{row.exposure:>8.0%}{row.trades:>6}"
            f"{'' if pd.isna(row.win_rate) else f'{row.win_rate:.0%}':>7}"
            f"{'' if pd.isna(row.beat_hold) else f'{row.beat_hold:.0%}':>14}")
    lines.append("")
    if r.passed:
        lines.append("Đạt ở CẢ HAI giai đoạn (Sharpe cao hơn VÀ sụt giảm nhỏ hơn mua & giữ): "
                     + ", ".join(NAMES[n] for n in r.passed))
    else:
        lines.append("KHÔNG quy tắc nào đạt ở cả hai giai đoạn: trục thời điểm chỉ nên dùng để quản "
                     "trị rủi ro (vùng mua, cắt lỗ), không dùng để đoán điểm vào.")
    if r.skipped:
        lines.append("Bỏ qua (thiếu dữ liệu): " + ", ".join(r.skipped))
    lines.append("Hạn chế: rổ mã lớn hiện tại (thiên lệch sống sót), giá điều chỉnh, chưa tính trượt "
                 "giá ngoài phí + thuế. Kết quả quá khứ không đảm bảo tương lai.")
    return "\n".join(lines)
