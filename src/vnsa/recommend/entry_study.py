"""Kiểm định ĐIỂM MUA (nâng cấp v2.1): vùng mua cũ (sát giá) so với thang mua 3 lớp.

Câu hỏi: đặt lệnh mua sâu hơn có tốt hơn không? Mua sâu thì giá vốn rẻ hơn nhưng hay lỡ mua những
mã tăng thẳng. Cách kiểm định (lệnh study-entry, không nhìn trước):

1. Mỗi `step` phiên, với từng mã trong rổ, tại phiên t lập vùng mua CHỈ từ dữ liệu đến phiên t
   (chỉ báo, hỗ trợ/kháng cự 250 phiên) theo ba cách:
     - "Vùng cũ": 2 lệnh LO, ½ ở đỉnh vùng, ½ ở đáy vùng (ta/entry.py, ladder tắt);
     - "Thang 3 lớp": 3 lệnh LO 30% / 40% / 30% (ladder bật);
     - "Vùng VA" (v4): như vùng cũ nhưng điểm neo là cận trên vùng giá trị / POC của hồ sơ khối
       lượng 120 phiên (ta/volume_profile.py, `[entry] anchor_mode = "va"`).
   Không có trần giá trị hợp lý (BCTC miễn phí không lưu theo ngày công bố) — phần đó giống nhau ở
   hai cách nên không làm lệch so sánh.
2. Lệnh LO khớp nếu trong `fill_sessions` phiên sau (t+1..t+20) có giá thấp nhất ≤ giá đặt; giá khớp
   = giá đặt, hoặc giá mở cửa nếu mở cửa đã thấp hơn.
3. Đánh giá ở phiên t + `hold_sessions` (60): lãi/lỗ của phần đã khớp; phần không khớp giữ tiền
   (lãi 0). So với "Mua ngay": toàn bộ vốn mua ở giá mở cửa phiên t+1.

Chỉ số báo cáo (bình quân trên mọi lần lập vùng):
  Khớp vốn       phần vốn khớp được (0–1)
  Giảm giá vốn   giá khớp bình quân thấp hơn giá đóng cửa phiên t bao nhiêu
  Lãi/vốn khớp   lãi 60 phiên trên phần vốn đã khớp
  Lãi/tổng vốn   lãi 60 phiên trên TOÀN BỘ vốn dự định (phần không khớp = 0) — thước đo chính
Chia hai giai đoạn như study-timing. Hạn chế: rổ mã lớn hiện tại (thiên lệch sống sót), giá điều
chỉnh, chưa trừ phí/thuế (như nhau ở mọi cách), khớp theo giá thấp nhất trong ngày là lạc quan.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vnsa.ta.entry import entry_zone
from vnsa.ta.levels import support_resistance

VARIANTS = ("Mua ngay", "Vùng cũ", "Thang 3 lớp", "Vùng VA")


def _orders(ind: pd.DataFrame, t: int, cfg_old: dict, cfg_ladder: dict, ta: dict, ticks
            ) -> dict[str, list[tuple[float, float, str]]]:
    """{cách: [(giá đặt, phần vốn, tên lệnh)]} tại phiên t (chỉ dùng dữ liệu đến t)."""
    hist = ind.iloc[: t + 1]
    sup, _ = support_resistance(hist, ta["swing_window"], ta["lookback"], ta["level_tolerance"])
    old = entry_zone(hist, sup, cfg_old, ticks)
    lad = entry_zone(hist, sup, cfg_ladder, ticks)
    va = entry_zone(hist, sup, {**cfg_old, "anchor_mode": "va"}, ticks)
    return {"Vùng cũ": [(old.high, 0.5, "đỉnh vùng"), (old.low, 0.5, "đáy vùng")],
            "Thang 3 lớp": [(x.price, x.weight, x.name) for x in lad.tiers],
            "Vùng VA": [(va.high, 0.5, "đỉnh vùng VA"), (va.low, 0.5, "đáy vùng VA")]}


def evaluate_symbol(ind: pd.DataFrame, cfg_entry: dict, ta: dict, ticks=None, step: int = 5,
                    fill_sessions: int = 20, hold_sessions: int = 60, warmup: int = 260
                    ) -> pd.DataFrame:
    """Một dòng cho mỗi (phiên lập vùng, cách, lệnh)."""
    d = ind.reset_index(drop=True)
    o, lo, c = (d[k].to_numpy(float) for k in ("open", "low", "close"))
    cfg_old = {**cfg_entry, "ladder": False, "anchor_mode": "classic"}
    cfg_lad = {**cfg_entry, "ladder": True, "anchor_mode": "classic"}
    rows = []
    for t in range(warmup, len(d) - hold_sessions, step):
        if pd.isna(d.at[t, "atr"]) or d.at[t, "atr"] <= 0:
            continue
        end = c[t + hold_sessions]
        base = {"date": d.at[t, "date"], "close": c[t]}
        rows.append({**base, "variant": "Mua ngay", "order": "mở cửa phiên sau", "price": o[t + 1],
                     "weight": 1.0, "filled": True, "fill_px": o[t + 1], "ret": end / o[t + 1] - 1})
        try:
            orders = _orders(d, t, cfg_old, cfg_lad, ta, ticks)
        except ValueError:
            continue
        win_lo, win_o = lo[t + 1: t + 1 + fill_sessions], o[t + 1: t + 1 + fill_sessions]
        for variant, legs in orders.items():
            for price, weight, name in legs:
                hit = np.flatnonzero(win_lo <= price)
                filled = len(hit) > 0
                px = min(price, win_o[hit[0]]) if filled else np.nan
                rows.append({**base, "variant": variant, "order": name, "price": price,
                             "weight": weight, "filled": filled, "fill_px": px,
                             "ret": end / px - 1 if filled else np.nan})
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Bình quân theo cách: khớp vốn, giảm giá vốn, lãi/vốn khớp, lãi/tổng vốn, số lần lập vùng."""
    r = rows.assign(w_fill=rows["weight"] * rows["filled"],
                    w_ret=(rows["weight"] * rows["ret"]).where(rows["filled"], 0.0),
                    w_disc=(rows["weight"] * (rows["fill_px"] / rows["close"] - 1)).where(
                        rows["filled"], 0.0))
    per = r.groupby(["variant", "symbol", "date"]).agg(
        filled=("w_fill", "sum"), gain=("w_ret", "sum"), disc=("w_disc", "sum"))
    per["ret_deployed"] = per["gain"] / per["filled"].where(per["filled"] > 0)
    per["disc"] = per["disc"] / per["filled"].where(per["filled"] > 0)
    out = per.groupby("variant").agg(
        fill=("filled", "mean"), discount=("disc", "mean"), ret_deployed=("ret_deployed", "mean"),
        ret_total=("gain", "mean"), n=("gain", "size"))
    order = [v for v in VARIANTS if v in out.index] + [v for v in out.index if v not in VARIANTS]
    return out.reindex(order)


def tier_summary(rows: pd.DataFrame) -> pd.DataFrame:
    """Từng lệnh: độ sâu so với giá lúc lập vùng, tỷ lệ khớp, lãi 60 phiên khi khớp."""
    r = rows[rows["variant"] != "Mua ngay"].assign(depth=lambda x: x["price"] / x["close"] - 1)
    return r.groupby(["variant", "order"], sort=False).agg(
        depth=("depth", "median"), fill=("filled", "mean"), ret=("ret", "mean"))


@dataclass
class EntryStudy:
    symbols: list[str]
    periods: dict[str, tuple[pd.Timestamp, pd.Timestamp]]
    summary: dict[str, pd.DataFrame]          # giai đoạn -> summarize
    tiers: pd.DataFrame                       # toàn kỳ
    rows: pd.DataFrame
    skipped: list[str] = field(default_factory=list)


def study(data: dict[str, dict], cfg_entry: dict, ta: dict, split: str = "2022-01-01",
          start: str = "2016-01-01", step: int = 5) -> EntryStudy:
    frames = []
    for sym, x in data.items():
        df = evaluate_symbol(x["ind"], cfg_entry, ta, x.get("ticks"), step)
        if not df.empty:
            frames.append(df.assign(symbol=sym))
    if not frames:
        raise ValueError("không mã nào đủ dữ liệu (cần > 320 phiên)")
    rows = pd.concat(frames, ignore_index=True)
    t0, t1 = pd.Timestamp(start), pd.Timestamp(split)
    rows = rows[rows["date"] >= t0]
    periods = {f"{t0.year}–{t1.year - 1}": (t0, t1 - pd.Timedelta(days=1)),
               f"{t1.year}–nay": (t1, pd.Timestamp(rows["date"].max()))}
    summary = {name: summarize(rows[(rows["date"] >= a) & (rows["date"] <= b)])
               for name, (a, b) in periods.items()}
    summary["Toàn kỳ"] = summarize(rows)
    return EntryStudy(sorted(data), periods, summary, tier_summary(rows), rows)


def _p(x: float, sign: bool = True) -> str:
    return "—" if pd.isna(x) else (f"{x:+.1%}" if sign else f"{x:.0%}")


def format_entry_study(res: EntryStudy) -> str:
    lines = [f"Kiểm định điểm mua trên {len(res.symbols)} mã: {', '.join(res.symbols)}",
             "Lập vùng mỗi 5 phiên chỉ từ dữ liệu đến phiên đó; lệnh LO chờ khớp 20 phiên; đánh giá "
             "sau 60 phiên. Phần không khớp giữ tiền.", ""]
    head = f"{'Giai đoạn':<12}{'Cách':<14}{'Khớp vốn':>10}{'Giảm giá vốn':>14}" \
           f"{'Lãi/vốn khớp':>14}{'Lãi/tổng vốn':>14}{'Số lần':>8}"
    lines.append(head)
    for period, tab in res.summary.items():
        for variant, r in tab.iterrows():
            lines.append(f"{period:<12}{variant:<14}{_p(r['fill'], False):>10}"
                         f"{_p(r['discount']):>14}{_p(r['ret_deployed']):>14}"
                         f"{_p(r['ret_total']):>14}{int(r['n']):>8}")
    lines += ["", "Từng lệnh (toàn kỳ): độ sâu trung vị so với giá lúc lập vùng · tỷ lệ khớp 20 phiên "
              "· lãi 60 phiên khi khớp"]
    for (variant, order), r in res.tiers.iterrows():
        lines.append(f"  {variant:<12} {order:<14} {_p(r['depth']):>7} · khớp {_p(r['fill'], False)}"
                     f" · lãi {_p(r['ret'])}")
    lines += ["", "Đọc kết quả: 'Lãi/tổng vốn' là thước đo chính (gồm cả cái giá của việc lỡ mua). "
              "Thang 3 lớp / Vùng VA (neo theo hồ sơ khối lượng, v4) đáng dùng khi Lãi/tổng vốn không "
              "thấp hơn Vùng cũ ở CẢ HAI giai đoạn.",
              "Hạn chế: rổ mã lớn hiện tại (thiên lệch sống sót), không có trần giá trị hợp lý, chưa trừ "
              "phí/thuế, khớp theo giá thấp nhất trong ngày là lạc quan."]
    return "\n".join(lines)
