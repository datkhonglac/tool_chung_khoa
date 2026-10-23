"""Phân tích nhật ký giao dịch (nâng cấp v3, lệnh `portfolio report`, tab "Danh mục").

1. Lợi suất theo thời gian (TWR) của PHẦN VỐN ĐÃ GIẢI NGÂN — sổ lệnh không ghi tiền mặt nên không
   tính được lợi suất cả tài khoản. Mỗi phiên d:
       r_d = (H_d − H_{d−1} − Mua_d + Bán_d + CổTức_d) / (H_{d−1} + Mua_d)
   H = Σ số lượng × giá đóng cửa CHƯA điều chỉnh (khớp giá trong sổ lệnh), Mua = tiền mua gồm phí,
   Bán = tiền bán sau phí + thuế, CổTức = cổ tức tiền nhận (sau thuế) ngày GDKHQ. TWR = Π(1 + r) − 1,
   so với VNINDEX cùng giai đoạn (từ phiên mua đầu tiên).
2. Thống kê lệnh bán (mỗi lệnh bán là một lần chốt): tỷ lệ thắng, lãi/lỗ bình quân (% trên giá vốn
   phần bán), payoff = lãi TB ÷ |lỗ TB|, kỳ vọng mỗi lần chốt, profit factor = Σ lãi ÷ |Σ lỗ|.
3. Điểm kỷ luật 0–100 (bình quân có trọng số các phần đo được, mỗi phần kèm số lần):
   - Mua theo kế hoạch (40): lệnh mua có khuyến nghị lưu trong `plan_days` ngày trước và giá mua
     ≤ đỉnh vùng mua × (1 + `zone_tolerance`). Mua không có khuyến nghị = không theo kế hoạch.
   - Đặt cắt lỗ (20): lần mở vị thế mới có lệnh CẮT LỖ trong `stop_sessions` phiên.
   - Tôn trọng cắt lỗ (40): lần giá đóng cửa xuống dưới mức cắt lỗ (cổ phiếu đã bán được) mà có
     lệnh bán trong `stop_sessions` phiên; dời cắt lỗ XUỐNG khi đang giữ cũng tính là một lần vi phạm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vnsa import portfolio as pf

DEFAULTS = {"plan_days": 7, "zone_tolerance": 0.01, "stop_sessions": 2,
            "weights": {"plan": 40, "stop_set": 20, "stop_kept": 40}}


@dataclass
class Discipline:
    score: float                    # 0–100, NaN nếu chưa đo được phần nào
    parts: dict[str, tuple[float, int, int]]   # phần -> (tỷ lệ đạt, số đạt, số lần)
    notes: list[str] = field(default_factory=list)


@dataclass
class Journal:
    curve: pd.DataFrame             # date, value, flow_in, flow_out, dividends, ret, twr, vni
    twr: float
    vni: float
    sales: pd.DataFrame             # date, symbol, quantity, price, pnl, pnl_pct, days
    stats: dict
    discipline: Discipline


def _cfg() -> dict:
    try:
        from vnsa.config import settings

        return {**DEFAULTS, **settings().get("journal", {})}
    except Exception:  # noqa: BLE001
        return dict(DEFAULTS)


def _raw(p: pd.DataFrame) -> pd.Series:
    """Giá đóng cửa chưa điều chỉnh theo ngày (thiếu close_raw thì dùng close)."""
    if p is None or p.empty:
        return pd.Series(dtype=float)
    c = p["close_raw"] if "close_raw" in p else p["close"]
    c = pd.to_numeric(c, errors="coerce").fillna(p["close"])
    return pd.Series(c.to_numpy(float), index=pd.to_datetime(p["date"]).dt.normalize())


def equity_curve(trades: pd.DataFrame, prices: dict[str, pd.DataFrame], index: pd.DataFrame,
                 as_of=None, actions: dict | None = None) -> pd.DataFrame:
    """Đường lợi suất phần vốn đã giải ngân theo từng phiên (lịch = ngày có giá VNINDEX)."""
    if trades.empty or index is None or index.empty:
        return pd.DataFrame()
    cal = pd.DatetimeIndex(pd.to_datetime(index["date"])).normalize()
    start = trades["date"].min().normalize()
    as_of = pd.Timestamp(as_of or cal.max()).normalize()
    days = cal[(cal >= start) & (cal <= as_of)]
    if not len(days):
        return pd.DataFrame()
    close = {s: _raw(p).reindex(days).ffill() for s, p in prices.items()}
    vni = _raw(index).reindex(days).ffill()
    buys = trades[trades["side"] == pf.BUY].assign(v=lambda x: x["quantity"] * x["price"] + x["fee"])
    sells = trades[trades["side"] == pf.SELL].assign(
        v=lambda x: x["quantity"] * x["price"] - x["fee"] - x["tax"])
    b_day = buys.groupby(buys["date"].dt.normalize())["v"].sum()
    s_day = sells.groupby(sells["date"].dt.normalize())["v"].sum()
    rows, prev_val, prev_div, twr = [], 0.0, 0.0, 1.0
    for d in days:
        pos = pf.positions(trades, d, cal, actions)
        val = 0.0
        for sym, p in pos.items():
            if p.quantity > 0:
                px = close.get(sym, pd.Series(dtype=float)).get(d, np.nan)
                val += p.quantity * (px if not pd.isna(px) else p.avg_cost)
        div_cum = sum(p.dividends for p in pos.values())
        fin, fout, div = float(b_day.get(d, 0.0)), float(s_day.get(d, 0.0)), div_cum - prev_div
        base = prev_val + fin
        r = (val - prev_val - fin + fout + div) / base if base > 0 else 0.0
        twr *= 1 + r
        rows.append({"date": d, "value": val, "flow_in": fin, "flow_out": fout, "dividends": div,
                     "ret": r, "twr": twr - 1, "vni": vni.get(d) / vni.iloc[0] - 1})
        prev_val, prev_div = val, div_cum
    return pd.DataFrame(rows)


def sales_table(trades: pd.DataFrame, as_of=None, calendar=None, actions=None) -> pd.DataFrame:
    rows = []
    for sym, p in pf.positions(trades, as_of, calendar, actions).items():
        for d, q, px, pnl, basis, first in p.sales or []:
            rows.append({"date": d, "symbol": sym, "quantity": q, "price": px, "pnl": pnl,
                         "pnl_pct": pnl / basis if basis > 0 else np.nan,
                         "days": (d - first).days if first is not None else np.nan})
    cols = ["date", "symbol", "quantity", "price", "pnl", "pnl_pct", "days"]
    return pd.DataFrame(rows, columns=cols).sort_values("date").reset_index(drop=True)


def trade_stats(sales: pd.DataFrame) -> dict:
    n = len(sales)
    if not n:
        return {"n": 0}
    win, loss = sales[sales["pnl"] > 0], sales[sales["pnl"] <= 0]
    avg_w = win["pnl_pct"].mean() if len(win) else np.nan
    avg_l = loss["pnl_pct"].mean() if len(loss) else np.nan
    rate = len(win) / n
    return {"n": n, "win_rate": rate, "avg_win": avg_w, "avg_loss": avg_l,
            "payoff": avg_w / abs(avg_l) if len(win) and len(loss) and avg_l != 0 else np.nan,
            "expectancy": rate * (avg_w if len(win) else 0) + (1 - rate) * (avg_l if len(loss) else 0),
            "profit_factor": win["pnl"].sum() / abs(loss["pnl"].sum())
            if len(loss) and loss["pnl"].sum() != 0 else np.nan,
            "pnl": sales["pnl"].sum(), "avg_days": sales["days"].mean()}


def discipline(trades: pd.DataFrame, signals: pd.DataFrame, prices: dict[str, pd.DataFrame],
               calendar: pd.DatetimeIndex, as_of=None, actions=None, cfg: dict | None = None
               ) -> Discipline:
    c = {**_cfg(), **(cfg or {})}
    cal = pd.DatetimeIndex(calendar).normalize()
    as_of = pd.Timestamp(as_of or (cal.max() if len(cal) else pd.Timestamp.today())).normalize()
    notes: list[str] = []
    sig = signals.assign(as_of=pd.to_datetime(signals["as_of"])) if signals is not None and \
        len(signals) else pd.DataFrame(columns=["as_of", "symbol", "buy_high"])

    # 1. Mua theo kế hoạch
    plan_ok = plan_n = 0
    for r in trades[trades["side"] == pf.BUY].itertuples(index=False):
        plan_n += 1
        s = sig[(sig["symbol"] == r.symbol) & (sig["as_of"] <= r.date)
                & (sig["as_of"] >= r.date - pd.Timedelta(days=int(c["plan_days"])))]
        s = s.sort_values("as_of").tail(1)
        hi = float(s["buy_high"].iloc[0]) if len(s) and not pd.isna(s["buy_high"].iloc[0]) else np.nan
        if not pd.isna(hi) and r.price <= hi * (1 + float(c["zone_tolerance"])):
            plan_ok += 1
        elif pd.isna(hi):
            notes.append(f"{r.date:%d/%m/%Y} mua {r.symbol}: không có khuyến nghị lưu trong "
                         f"{c['plan_days']} ngày trước (chạy analyze {r.symbol} trước khi mua)")
        else:
            notes.append(f"{r.date:%d/%m/%Y} mua {r.symbol} giá {r.price:,.0f} > đỉnh vùng mua "
                         f"{hi:,.0f} (đuổi giá)")

    # 2 + 3. Duyệt từng phiên: mở vị thế, đặt cắt lỗ, giá thủng cắt lỗ, dời cắt lỗ xuống
    k = int(c["stop_sessions"])
    days = cal[(cal >= trades["date"].min().normalize()) & (cal <= as_of)] if len(trades) else cal[:0]
    close = {s: _raw(p) for s, p in prices.items()}
    set_ok = set_n = kept_ok = kept_n = 0
    prev: dict[str, pf.Position] = {}
    pending_open: dict[str, int] = {}       # mã -> chỉ số phiên mở vị thế, chờ đặt cắt lỗ
    breach: dict[str, int] = {}             # mã -> chỉ số phiên thủng cắt lỗ, chờ bán
    counted: set[str] = set()               # đã tính vi phạm, chờ giá về trên cắt lỗ
    for i, d in enumerate(days):
        pos = pf.positions(trades, d, cal, actions)
        for sym, p in pos.items():
            was = prev.get(sym)
            held_before = was is not None and was.quantity > 0
            if p.quantity > 0 and not held_before:
                pending_open[sym] = i
            if sym in pending_open and p.stop > 0:
                set_ok, set_n = set_ok + 1, set_n + 1
                pending_open.pop(sym)
            elif sym in pending_open and (i - pending_open[sym] >= k or p.quantity <= 0):
                set_n += 1
                pending_open.pop(sym)
                notes.append(f"{d:%d/%m/%Y} {sym}: mở vị thế mà không đặt cắt lỗ trong {k} phiên")
            if sym in breach:
                if held_before and p.quantity < was.quantity:
                    kept_ok, kept_n = kept_ok + 1, kept_n + 1
                    breach.pop(sym)
                elif i - breach[sym] >= k:
                    kept_n += 1
                    notes.append(f"{days[breach[sym]]:%d/%m/%Y} {sym}: đóng cửa dưới cắt lỗ mà không "
                                 f"bán trong {k} phiên")
                    breach.pop(sym)
                    counted.add(sym)
            px = close.get(sym, pd.Series(dtype=float)).get(d, np.nan)
            below = p.quantity > 0 and p.stop > 0 and not pd.isna(px) and px < p.stop
            if not below:
                counted.discard(sym)
            elif p.sellable > 0 and sym not in breach and sym not in counted:
                breach[sym] = i
        prev = pos
    # dời cắt lỗ XUỐNG khi đang giữ (so với mức đang đặt đã điều chỉnh theo sự kiện quyền)
    for i, r in enumerate(trades.itertuples(index=False)):
        if r.side != pf.STOP or r.date > as_of:
            continue
        before = pf.positions(trades.iloc[:i], r.date, cal, actions).get(r.symbol)
        if before is not None and before.quantity > 0 and before.stop > 0 \
                and r.price < before.stop * 0.995:
            kept_n += 1
            notes.append(f"{r.date:%d/%m/%Y} {r.symbol}: dời cắt lỗ xuống {before.stop:,.0f} → "
                         f"{r.price:,.0f}")
    parts = {"plan": (plan_ok / plan_n if plan_n else np.nan, plan_ok, plan_n),
             "stop_set": (set_ok / set_n if set_n else np.nan, set_ok, set_n),
             "stop_kept": (kept_ok / kept_n if kept_n else np.nan, kept_ok, kept_n)}
    w = c["weights"]
    got = [(w[name], v[0]) for name, v in parts.items() if not pd.isna(v[0])]
    score = 100 * sum(a * b for a, b in got) / sum(a for a, _ in got) if got else np.nan
    return Discipline(score, parts, notes)


def build(trades: pd.DataFrame, prices: dict[str, pd.DataFrame], index: pd.DataFrame,
          signals: pd.DataFrame, as_of=None, actions=None) -> Journal:
    cal = pd.DatetimeIndex(pd.to_datetime(index["date"])).normalize() if index is not None and \
        len(index) else pd.DatetimeIndex([])
    curve = equity_curve(trades, prices, index, as_of, actions)
    sales = sales_table(trades, as_of, cal, actions)
    return Journal(curve, float(curve["twr"].iloc[-1]) if len(curve) else np.nan,
                   float(curve["vni"].iloc[-1]) if len(curve) else np.nan, sales,
                   trade_stats(sales), discipline(trades, signals, prices, cal, as_of, actions))


def load_journal(as_of=None) -> Journal | None:
    """Đọc sổ lệnh, giá, VNINDEX, khuyến nghị đã lưu và sự kiện quyền từ kho."""
    from vnsa.config import path, settings
    from vnsa.data.store import Store

    trades = pf.load()
    if trades.empty:
        return None
    syms = sorted(trades["symbol"].unique())
    start = f"{trades['date'].min():%Y-%m-%d}"
    with Store(path("warehouse")) as store:
        prices = {s: store.prices(s, start) for s in syms}
        index = store.prices(settings()["valuation"]["index"], start)
        signals = store.signals()
        actions = {s: store.dividends(s) for s in syms}
    return build(trades, prices, index, signals, as_of, actions)


PART_NAMES = {"plan": "Mua theo kế hoạch", "stop_set": "Đặt cắt lỗ khi mở vị thế",
              "stop_kept": "Tôn trọng cắt lỗ"}


def _p(x, sign=True) -> str:
    return "—" if x is None or pd.isna(x) else (f"{x:+.1%}" if sign else f"{x:.0%}")


def _x(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:.2f}"


def format_journal(j: Journal) -> str:
    lines = ["NHẬT KÝ GIAO DỊCH"]
    if len(j.curve):
        c = j.curve
        lines.append(f"Từ {c['date'].iloc[0]:%d/%m/%Y} đến {c['date'].iloc[-1]:%d/%m/%Y}: lợi suất "
                     f"phần vốn đã giải ngân (TWR) {_p(j.twr)} · VNINDEX {_p(j.vni)} · chênh lệch "
                     f"{_p(j.twr - j.vni)}")
        peak = (1 + c["twr"]).cummax()
        lines.append(f"Sụt giảm lớn nhất {_p(((1 + c['twr']) / peak - 1).min())} · số phiên có cổ "
                     f"phiếu {int((c['value'] > 0).sum())}/{len(c)}")
    else:
        lines.append("Chưa đủ dữ liệu giá để tính lợi suất (cần giá VNINDEX và các mã trong kho).")
    s = j.stats
    lines.append("")
    if s.get("n"):
        lines.append(f"Lệnh bán đã chốt: {s['n']} · thắng {_p(s['win_rate'], False)} · lãi TB "
                     f"{_p(s['avg_win'])} · lỗ TB {_p(s['avg_loss'])} · payoff {_x(s['payoff'])}"
                     f" · kỳ vọng/lần {_p(s['expectancy'])}")
        lines.append(f"Profit factor {_x(s['profit_factor'])} · tổng lãi/lỗ đã chốt "
                     f"{s['pnl']:,.0f}đ · giữ bình quân {s['avg_days']:.0f} ngày")
    else:
        lines.append("Chưa có lệnh bán nào để thống kê thắng/thua.")
    d = j.discipline
    lines += ["", f"ĐIỂM KỶ LUẬT: {'—' if pd.isna(d.score) else f'{d.score:.0f}/100'}"]
    for name, (rate, ok, n) in d.parts.items():
        lines.append(f"  {PART_NAMES[name]:<26} {_p(rate, False):>5}  ({ok}/{n})")
    if d.notes:
        lines.append("  Cần xem lại:")
        lines += [f"   - {x}" for x in d.notes[-10:]]
    lines += ["", "Ghi chú: sổ lệnh không ghi tiền mặt nên TWR chỉ đo phần vốn đã giải ngân (không "
              "tính tiền nhàn rỗi). Mỗi lệnh bán tính là một lần chốt."]
    return "\n".join(lines)
