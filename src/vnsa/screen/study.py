"""Kiểm định quá khứ phần DỰA TRÊN GIÁ của bộ sàng lọc (lệnh study-screen).

Chỉ kiểm định được phần giá: BCTC miễn phí chỉ có 4 năm và không lưu theo ngày công bố, điểm vĩ mô
không có lịch sử từng tháng — các phần đó theo dõi tiến bằng review-screen. Cách làm (mỗi cuối
tháng, chỉ dùng số liệu đến ngày đó):
  1. Mã đủ điều kiện: có ≥ 200 phiên giá trước đó, GTGD bình quân 20 phiên ≥ `min_value_bn` tỷ.
  2. Điểm nhóm ngành = thứ hạng giữa các nhóm của (trung vị RS 3–6 tháng so VNINDEX) và (tỷ lệ mã trên
     MA50/MA200), trộn theo tỷ lệ trọng số động lượng : độ rộng trong [screen.industry_weights];
     nhóm cần ≥ `min_members` mã.
  3. Danh mục: `top_industries` nhóm dẫn đầu, mỗi nhóm `per_industry` mã RS 6 tháng cao nhất,
     chia đều; giữ đến cuối tháng sau. Phí mua/bán + thuế bán theo [backtest] trên phần thay đổi.
  So với: rổ chia đều mọi mã đủ điều kiện (cùng tháng, không phí) và VNINDEX.
Hạn chế cần nói rõ: chỉ có các mã đang niêm yết (mã đã huỷ niêm yết không có giá — thiên lệch sống
sót), nhóm ngành theo phân loại hiện tại, ngưỡng GTGD danh nghĩa không đổi theo thời gian.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd


@dataclass
class StudyResult:
    monthly: pd.DataFrame        # mỗi tháng: date, strategy, top_groups_all, universe, index, n...
    periods: pd.DataFrame        # chỉ số theo giai đoạn × danh mục
    picks: pd.DataFrame          # mã đã chọn mỗi tháng


def _rank100(s: pd.Series) -> pd.Series:
    ok = s.dropna()
    if len(ok) <= 1:
        return s.where(s.isna(), 50.0)
    return ((ok.rank(method="average") - 1) / (len(ok) - 1) * 100).reindex(s.index)


def month_ends(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(days, index=days)
    return pd.DatetimeIndex(s.groupby(days.to_period("M")).max().to_numpy())


def snapshot(close: pd.DataFrame, value: pd.DataFrame, index: pd.Series, pos: int,
             min_value: float) -> pd.DataFrame:
    """Số liệu từng mã tại phiên thứ `pos` (chỉ dùng dữ liệu đến phiên đó)."""
    c = close.iloc[: pos + 1]
    have = c.notna().sum()
    last = c.iloc[-1]
    ok = (have >= 200) & last.notna()
    v20 = value.iloc[max(0, pos - 19): pos + 1].fillna(0).mean()
    ok &= v20 >= min_value
    cols = ok[ok].index
    out = pd.DataFrame(index=cols)
    i_now = index.iloc[pos]
    for key, n in (("rs_3m", 63), ("rs_6m", 126)):
        if pos - n < 0:
            out[key] = np.nan
            continue
        r = last[cols] / c.iloc[pos - n][cols] - 1
        out[key] = (1 + r) / (i_now / index.iloc[pos - n]) - 1
    out["ma50"] = (last[cols] > c.iloc[-50:][cols].mean()).astype(float)
    out["ma200"] = (last[cols] > c.iloc[-200:][cols].mean()).astype(float)
    return out


def choose(snap: pd.DataFrame, groups: pd.Series, cfg: dict) -> tuple[list[str], list[str]]:
    """(mã chọn, nhóm dẫn đầu) theo quy tắc phần giá của bộ lọc."""
    w = cfg.get("industry_weights", {})
    w_mom, w_br = w.get("momentum", 0.30), w.get("breadth", 0.10)
    df = snap.assign(group=groups.reindex(snap.index).fillna("Chưa rõ ngành"))
    df["rs"] = df[["rs_3m", "rs_6m"]].mean(axis=1)
    g = df.groupby("group").agg(n=("rs", "size"), rs=("rs", "median"),
                                br50=("ma50", "mean"), br200=("ma200", "mean"))
    g = g[g["n"] >= cfg.get("min_members", 3)]
    if g.empty:
        return [], []
    g["breadth"] = g[["br50", "br200"]].mean(axis=1)
    g["score"] = (w_mom * _rank100(g["rs"]) + w_br * _rank100(g["breadth"])) / (w_mom + w_br)
    top = g.sort_values(["score", "n"], ascending=False).head(cfg.get("top_industries", 5)).index
    picks = []
    for name in top:
        m = df[df["group"] == name].sort_values("rs_6m", ascending=False)
        picks += m.head(cfg.get("per_industry", 5)).index.tolist()
    return picks, list(top)


def _stats(r: pd.Series) -> dict:
    r = r.dropna()
    if r.empty:
        return {"months": 0, "cagr": np.nan, "vol": np.nan, "max_dd": np.nan}
    wealth = (1 + r).cumprod()
    years = len(r) / 12
    return {"months": len(r), "cagr": float(wealth.iloc[-1] ** (1 / years) - 1),
            "vol": float(r.std() * np.sqrt(12)),
            "max_dd": float((wealth / wealth.cummax() - 1).min())}


def run_study(close: pd.DataFrame, value: pd.DataFrame, index: pd.Series, groups: pd.Series,
              cfg: dict, costs: dict, split: str | None = None) -> StudyResult:
    """close, value: bảng rộng (ngày × mã) giá đóng cửa điều chỉnh và GTGD (đồng); index: VNINDEX."""
    days = index.dropna().index.intersection(close.index).sort_values()
    close, value, index = close.reindex(days), value.reindex(days), index.reindex(days)
    ends = [d for d in month_ends(days) if days.get_loc(d) >= 200]
    min_value = cfg.get("min_value_bn", 3.0) * 1e9
    fee, tax = costs.get("fee", 0.0015), costs.get("sell_tax", 0.001)
    rows, picks_log, held = [], [], set()
    for d0, d1 in pairwise(ends):
        pos = days.get_loc(d0)
        snap = snapshot(close, value, index, pos, min_value)
        if snap.empty:
            continue
        picks, top = choose(snap, groups, cfg)
        fwd = close.loc[d1] / close.loc[d0] - 1
        universe = float(fwd[snap.index].dropna().mean())
        top_all = snap.index[groups.reindex(snap.index).isin(top)]
        strat = float(fwd[picks].fillna(0).mean()) if picks else 0.0
        new = set(picks)
        bought = len(new - held) / len(new) if new else 0.0       # tỷ trọng mua mới
        sold = len(held - new) / len(held) if held else 0.0       # tỷ trọng bán ra
        cost = bought * fee + sold * (fee + tax)
        turnover = (bought + sold) / 2
        held = new
        rows.append({"date": d0, "strategy": strat - cost, "top_groups_all":
                     float(fwd[top_all].dropna().mean()) if len(top_all) else np.nan,
                     "universe": universe, "index": float(index[d1] / index[d0] - 1),
                     "n_universe": len(snap), "n_picks": len(picks), "turnover": turnover})
        picks_log += [{"date": d0, "symbol": s, "group": groups.get(s, "")} for s in picks]
    monthly = pd.DataFrame(rows)
    split_ts = pd.Timestamp(split) if split else None
    periods = []
    spans = [("Toàn bộ", None, None)]
    if split_ts is not None and len(monthly):
        spans += [(f"trước {split_ts:%m/%Y}", None, split_ts), (f"từ {split_ts:%m/%Y}", split_ts, None)]
    for label, a, b in spans:
        m = monthly
        if a is not None:
            m = m[m["date"] >= a]
        if b is not None:
            m = m[m["date"] < b]
        for col, name in (("strategy", "Bộ lọc (phần giá)"), ("top_groups_all", "Mọi mã nhóm dẫn đầu"),
                          ("universe", "Rổ chia đều mọi mã"), ("index", "VNINDEX")):
            st = _stats(m[col]) if len(m) else _stats(pd.Series(dtype=float))
            beat = (float((m[col] > m["universe"]).mean()) if len(m) and col != "universe"
                    else np.nan)
            periods.append({"period": label, "portfolio": name, **st, "beat_universe": beat})
    return StudyResult(monthly, pd.DataFrame(periods), pd.DataFrame(picks_log))


def format_study(res: StudyResult) -> str:
    from vnsa.report.card import vn_pct

    if res.monthly.empty:
        return "Kiểm định sàng lọc: không đủ dữ liệu giá (cần ≥ 200 phiên trước tháng đầu tiên)."
    m = res.monthly
    lines = [(f"KIỂM ĐỊNH SÀNG LỌC (phần giá) — {len(m)} tháng, {m['date'].min():%m/%Y} → "
              f"{m['date'].max():%m/%Y}, bình quân {m['n_universe'].mean():.0f} mã đủ điều kiện, "
              f"{m['n_picks'].mean():.0f} mã chọn, thay {m['turnover'].mean():.0%} danh mục/tháng"),
             (f"  {'Giai đoạn':<16}{'Danh mục':<22}{'Tháng':>6}{'Lợi nhuận/năm':>15}"
              f"{'Biến động':>11}{'Sụt giảm max':>14}{'Tháng thắng rổ':>16}")]
    for r in res.periods.itertuples(index=False):
        beat = "—" if pd.isna(r.beat_universe) else vn_pct(r.beat_universe, 0)
        lines.append(f"  {r.period:<16}{r.portfolio:<22}{r.months:>6}"
                     f"{vn_pct(r.cagr, 1, sign=True):>15}{vn_pct(r.vol, 0):>11}"
                     f"{vn_pct(r.max_dd, 0):>14}{beat:>16}")
    lines += ["  Bộ lọc đã trừ phí mua/bán và thuế bán; rổ chia đều và VNINDEX chưa trừ phí.",
              "  Hạn chế: chỉ mã đang niêm yết (thiên lệch sống sót), nhóm ngành theo phân loại hiện",
              "  tại; phần cơ bản (BCTC) và vĩ mô không kiểm định được — theo dõi bằng review-screen."]
    return "\n".join(lines)
