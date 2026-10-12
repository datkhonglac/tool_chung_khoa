"""Chấm lại các lượt sàng lọc đã lưu (bảng screen) sau N phiên — bộ lọc có chọn đúng không?

Với mỗi lượt (ngày D) và mỗi kỳ h phiên (`[screen] review_horizons`), cho từng mã đã qua lọc thanh
khoản lúc đó: lãi/lỗ = giá đóng cửa (điều chỉnh cổ tức, theo kho hiện tại) phiên D+h ÷ phiên D − 1;
vượt trội = lãi/lỗ − VNINDEX cùng kỳ. So sánh các tầng:
  - "Được chọn"        (tầng 4) — mã bộ lọc đưa ra;
  - "Nhóm dẫn đầu"     (tầng ≥ 3) — mọi mã trong các nhóm ngành điểm cao nhất;
  - "Qua thanh khoản"  (tầng ≥ 1) — rổ so sánh: mọi mã đủ thanh khoản lúc đó.
Bộ lọc có ích khi "Được chọn" và "Nhóm dẫn đầu" vượt trội hơn "Qua thanh khoản" một cách ổn định qua
nhiều lượt. Ít lượt (dưới ~10) thì chỉ mang tính tham khảo. Lượt chưa đủ h phiên thì chưa chấm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIERS = [("Được chọn", 4), ("Nhóm dẫn đầu", 3), ("Qua thanh khoản", 1)]


def evaluate(runs: pd.DataFrame, closes: pd.DataFrame, index: pd.Series,
             horizons: list[int]) -> pd.DataFrame:
    """Mỗi dòng = một mã (tầng ≥ 1) của một lượt × một kỳ đã đủ phiên.
    closes: bảng rộng giá đóng cửa (index = ngày, cột = mã); index: VNINDEX theo ngày."""
    if runs.empty or index.empty:
        return pd.DataFrame()
    days = index.sort_index().index
    rows = []
    for as_of, run in runs[runs["stage"] >= 1].groupby("as_of"):
        pos = days.searchsorted(pd.Timestamp(as_of), side="right") - 1
        if pos < 0:
            continue
        d0 = days[pos]
        for h in horizons:
            if pos + h >= len(days):
                continue
            d1 = days[pos + h]
            idx_ret = float(index[d1] / index[d0] - 1)
            for r in run.itertuples(index=False):
                if r.symbol not in closes:
                    continue
                c = closes[r.symbol]
                p0, p1 = c.asof(d0), c.asof(d1)
                if pd.isna(p0) or pd.isna(p1) or p0 <= 0:
                    continue
                ret = float(p1 / p0 - 1)
                rows.append({"as_of": pd.Timestamp(as_of), "horizon": h, "symbol": r.symbol,
                             "stage": int(r.stage), "group_name": r.group_name,
                             "action": r.action, "ret": ret, "index_ret": idx_ret,
                             "excess": ret - idx_ret})
    return pd.DataFrame(rows)


def summarize(ev: pd.DataFrame) -> pd.DataFrame:
    """Theo kỳ × tầng: số lượt, số mẫu, lãi/lỗ và vượt trội bình quân (mỗi lượt một phiếu: bình
    quân các mã trong lượt rồi bình quân các lượt), tỷ lệ lượt vượt rổ thanh khoản."""
    if ev.empty:
        return pd.DataFrame()
    out = []
    for h, g in ev.groupby("horizon"):
        base = g[g["stage"] >= 1].groupby("as_of")["ret"].mean()
        for name, stage in TIERS:
            part = g[g["stage"] >= stage]
            if part.empty:
                continue
            per_run = part.groupby("as_of").agg(ret=("ret", "mean"), excess=("excess", "mean"))
            beat = (per_run["ret"] > base.reindex(per_run.index)).mean() if stage > 1 else np.nan
            out.append({"horizon": h, "tier": name, "runs": len(per_run), "n": len(part),
                        "ret": per_run["ret"].mean(), "excess": per_run["excess"].mean(),
                        "beat_liquid": beat})
    return pd.DataFrame(out)


def format_review(summary: pd.DataFrame, n_runs: int) -> str:
    from vnsa.report.card import vn_num, vn_pct

    if summary.empty:
        return (f"Chấm lại sàng lọc: {n_runs} lượt đã lưu, chưa lượt nào đủ phiên để chấm "
                "(cần ≥ 20 phiên sau ngày sàng lọc).")
    lines = [f"CHẤM LẠI SÀNG LỌC — {n_runs} lượt đã lưu (lãi/lỗ đã gồm cổ tức, bình quân theo lượt)",
             (f"  {'Kỳ':>5}  {'Tầng':<17}{'Lượt':>5}{'Mẫu':>6}{'Lãi/lỗ':>9}{'Vượt VNINDEX':>14}"
              f"{'Lượt thắng rổ':>15}")]
    for r in summary.itertuples(index=False):
        beat = "—" if pd.isna(r.beat_liquid) else vn_pct(r.beat_liquid, 0)
        lines.append(f"  {r.horizon:>5}  {r.tier:<17}{r.runs:>5}{vn_num(r.n):>6}"
                     f"{vn_pct(r.ret, 1, sign=True):>9}{vn_pct(r.excess, 1, sign=True):>14}"
                     f"{beat:>15}")
    lines.append("  Lượt thắng rổ = tỷ lệ lượt mà nhóm này lãi hơn bình quân mọi mã qua thanh khoản.")
    if summary["runs"].max() < 10:
        lines.append("  Dưới 10 lượt: chỉ để tham khảo, chưa đủ để kết luận bộ lọc có ích.")
    return "\n".join(lines)
