"""Biểu đồ xoay vòng ngành RRG (nâng cấp v3, lệnh `rrg`, tab "Sàng lọc thị trường") — CHỈ HIỂN THỊ,
chưa vào điểm/xếp hạng (chưa kiểm định trên thị trường Việt Nam).

Công thức (xấp xỉ, công thức JdK gốc không công bố):
1. Chỉ số nhóm theo tuần: bình quân đều lợi suất tuần của các mã trong nhóm (giá đã điều chỉnh, mỗi
   tuần chặn ±`clip`), nhân dồn thành chỉ số; tuần có < `min_members` mã có giá thì bỏ.
2. Sức mạnh tương đối RS = chỉ số nhóm / VNINDEX (giá đóng cửa cuối tuần).
3. RS-Ratio = 100 × RS / EMA(RS, `ratio_weeks` tuần)  — trên 100: nhóm mạnh hơn xu hướng tương đối
   của chính nó (đang thắng thị trường).
4. RS-Momentum = 100 × RS-Ratio / RS-Ratio `mom_weeks` tuần trước — trên 100: sức mạnh tương đối
   đang tăng tốc.
Góc phần tư: Dẫn dắt (Ratio ≥ 100, Momentum ≥ 100) → Suy yếu (≥ 100, < 100) → Tụt hậu (< 100, < 100)
→ Cải thiện (< 100, ≥ 100) → ... (thường xoay theo chiều kim đồng hồ). "Đuôi" = vị trí `tail` tuần gần
nhất.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LEADING, WEAKENING, LAGGING, IMPROVING = "Dẫn dắt", "Suy yếu", "Tụt hậu", "Cải thiện"
DEFAULTS = {"ratio_weeks": 10, "mom_weeks": 4, "tail": 8, "min_members": 3, "clip": 0.5,
            "history_weeks": 60}


def cfg() -> dict:
    try:
        from vnsa.config import settings

        return {**DEFAULTS, **settings().get("rrg", {})}
    except Exception:  # noqa: BLE001
        return dict(DEFAULTS)


def weekly_close(prices: pd.DataFrame) -> pd.Series:
    """Giá đóng cửa phiên cuối mỗi tuần (tuần kết thúc thứ Sáu), index = ngày thứ Sáu."""
    if prices is None or prices.empty:
        return pd.Series(dtype=float)
    s = pd.Series(prices["close"].to_numpy(float), index=pd.to_datetime(prices["date"]))
    return s.resample("W-FRI").last().dropna()


def group_index(members: dict[str, pd.Series], min_members: int = 3, clip: float = 0.5) -> pd.Series:
    """Chỉ số nhóm bình quân đều từ giá tuần các mã (bắt đầu = 100)."""
    if not members:
        return pd.Series(dtype=float)
    rets = pd.DataFrame({s: w.pct_change() for s, w in members.items()}).clip(-clip, clip)
    n = rets.notna().sum(axis=1)
    r = rets.mean(axis=1).where(n >= min_members).dropna()
    return 100 * (1 + r).cumprod()


def quadrant(ratio: float, mom: float) -> str:
    if pd.isna(ratio) or pd.isna(mom):
        return ""
    if ratio >= 100:
        return LEADING if mom >= 100 else WEAKENING
    return IMPROVING if mom >= 100 else LAGGING


def rrg_series(group: pd.Series, bench: pd.Series, ratio_weeks: int = 10, mom_weeks: int = 4
               ) -> pd.DataFrame:
    """date, rs, ratio, momentum, quadrant của một nhóm so với chỉ số chuẩn."""
    both = pd.concat([group.rename("g"), bench.rename("b")], axis=1, join="inner").dropna()
    if both.empty:
        return pd.DataFrame(columns=["date", "rs", "ratio", "momentum", "quadrant"])
    rs = both["g"] / both["b"]
    ratio = 100 * rs / rs.ewm(span=ratio_weeks, adjust=False).mean()
    mom = 100 * ratio / ratio.shift(mom_weeks)
    out = pd.DataFrame({"date": both.index, "rs": rs.to_numpy(), "ratio": ratio.to_numpy(),
                        "momentum": mom.to_numpy()})
    out["quadrant"] = [quadrant(a, b) for a, b in zip(out["ratio"], out["momentum"])]
    # cần đủ lịch sử để EMA ổn định
    return out.iloc[max(ratio_weeks, mom_weeks):].reset_index(drop=True)


def compute(prices: dict[str, pd.DataFrame], groups: dict[str, str], bench: pd.DataFrame,
            c: dict | None = None) -> pd.DataFrame:
    """Bảng dài: group, members, date, rs, ratio, momentum, quadrant — `tail` tuần gần nhất mỗi
    nhóm. prices: {mã: giá ngày}; groups: {mã: tên nhóm}; bench: giá ngày VNINDEX."""
    c = {**DEFAULTS, **(c or {})}
    weekly = {s: weekly_close(p) for s, p in prices.items()}
    b = weekly_close(bench)
    frames = []
    by_group: dict[str, dict[str, pd.Series]] = {}
    for s, g in groups.items():
        if s in weekly and len(weekly[s]):
            by_group.setdefault(g, {})[s] = weekly[s]
    for g, members in sorted(by_group.items()):
        if len(members) < c["min_members"]:
            continue
        idx = group_index(members, c["min_members"], c["clip"])
        df = rrg_series(idx, b, c["ratio_weeks"], c["mom_weeks"]).dropna(subset=["momentum"])
        if df.empty:
            continue
        frames.append(df.tail(c["tail"]).assign(group=g, members=len(members)))
    if not frames:
        return pd.DataFrame(columns=["group", "members", "date", "rs", "ratio", "momentum",
                                     "quadrant"])
    out = pd.concat(frames, ignore_index=True)
    return out[["group", "members", "date", "rs", "ratio", "momentum", "quadrant"]]


def latest(tails: pd.DataFrame) -> pd.DataFrame:
    """Điểm mới nhất mỗi nhóm + góc phần tư 4 tuần trước (để thấy hướng xoay)."""
    if tails.empty:
        return tails
    last = tails.groupby("group").tail(1).set_index("group")
    prev = {g: (x["quadrant"].iloc[-5] if len(x) >= 5 else "") for g, x in tails.groupby("group")}
    last["prev_quadrant"] = [prev.get(g, "") for g in last.index]
    order = {LEADING: 0, IMPROVING: 1, WEAKENING: 2, LAGGING: 3}
    return (last.assign(_o=last["quadrant"].map(order))
            .sort_values(["_o", "ratio"], ascending=[True, False]).drop(columns="_o").reset_index())


def load(as_of=None, c: dict | None = None) -> pd.DataFrame:
    """Đọc kho: mã của lượt sàng lọc mới nhất đã qua thanh khoản (stage ≥ 1), nhóm theo
    screen/groups (macro_map.toml); chưa sàng lọc thì mọi mã có giá trong kho."""
    from vnsa.analysis import macro_mapping
    from vnsa.config import path, settings
    from vnsa.data.store import Store
    from vnsa.screen.groups import UNKNOWN, industry_group

    c = {**cfg(), **(c or {})}
    start = (pd.Timestamp(as_of or pd.Timestamp.today()) - pd.Timedelta(weeks=int(c["history_weeks"])
                                                                       + 2)).strftime("%Y-%m-%d")
    end = None if as_of is None else pd.Timestamp(as_of).strftime("%Y-%m-%d")
    mapping = macro_mapping()
    index = settings()["valuation"]["index"]
    with Store(path("warehouse")) as store:
        sc = store.screen()
        if len(sc):
            syms = sorted(sc.loc[sc["stage"] >= 1, "symbol"])
        else:
            inv = store.inventory()
            syms = [s for s in inv["symbol"].tolist() if s != index] if len(inv) else []
        bench = store.prices(index, start, end)
        allp = store.recent_prices(start, syms)
        companies = {s: store.company(s) for s in syms}
    if end is not None:
        allp = allp[allp["date"] <= pd.Timestamp(end)]
    prices, groups = {}, {}
    for s, p in allp.groupby("symbol"):
        g = industry_group(companies.get(s), mapping)
        if len(p) >= 20 and g != UNKNOWN:
            prices[s], groups[s] = p, g
    return compute(prices, groups, bench, c)


def format_rrg(tails: pd.DataFrame) -> str:
    if tails.empty:
        return ("Chưa đủ dữ liệu cho RRG (cần giá tuần của ≥ 3 mã mỗi nhóm và VNINDEX — chạy "
                "screen --stage liquidity trước).")
    last = latest(tails)
    d = pd.Timestamp(tails["date"].max())
    lines = [f"XOAY VÒNG NGÀNH (RRG) — tuần đến {d:%d/%m/%Y}, so với VNINDEX (chỉ tham khảo)",
             f"{'Nhóm':<28}{'Số mã':>6}{'RS-Ratio':>10}{'RS-Mom':>9}  Góc phần tư (4 tuần trước)"]
    for r in last.itertuples():
        frm = f" (từ {r.prev_quadrant})" if r.prev_quadrant and r.prev_quadrant != r.quadrant else ""
        lines.append(f"{r.group[:27]:<28}{r.members:>6}{r.ratio:>10.1f}{r.momentum:>9.1f}  "
                     f"{r.quadrant}{frm}")
    lines += ["", "RS-Ratio > 100: nhóm mạnh hơn thị trường; RS-Momentum > 100: đang tăng tốc. "
              "Thường xoay Cải thiện → Dẫn dắt → Suy yếu → Tụt hậu. Chưa kiểm định — không dùng "
              "làm tín hiệu mua bán đơn lẻ."]
    return "\n".join(lines)
