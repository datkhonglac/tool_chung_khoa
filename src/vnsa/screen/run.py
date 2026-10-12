"""Chạy sàng lọc toàn thị trường theo tầng; mỗi lượt ghi bảng `screen` và CSV."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from vnsa.config import path, settings
from vnsa.data.store import SCREEN_COLUMNS, Store
from vnsa.screen.liquidity import liquidity_check, price_stats, reason_group


def screen_cfg() -> dict:
    return settings().get("screen", {})


@dataclass
class ScreenResult:
    as_of: pd.Timestamp
    table: pd.DataFrame                                  # một dòng mỗi mã trong vũ trụ
    fetch: Counter = field(default_factory=Counter)      # ok/skip/error khi tải giá
    fetch_errors: list[str] = field(default_factory=list)
    listing: str = ""                                    # trạng thái danh sách niêm yết
    csv: str = ""

    def passed(self, stage: int) -> pd.DataFrame:
        return self.table[self.table["stage"] >= stage]

    def rejected(self, stage: int) -> Counter:
        """Số mã bị loại ở tầng `stage`, theo nhóm lý do."""
        t = self.table[self.table["stage"] == stage - 1]
        return Counter(reason_group(r) for r in t["reason"] if r)


def universe(store: Store, exchanges: list[str] | None) -> pd.DataFrame:
    """Tầng 0: mọi mã niêm yết trên các sàn được chọn (bảng companies)."""
    return store.companies(exchanges)


def _progress(log: Callable[[str], None] | None, every: int = 100):
    t0, errors = time.monotonic(), [0]

    def tick(i: int, n: int, sym: str, step) -> None:
        if step.status == "error":
            errors[0] += 1
        if log and (i % every == 0 or i == n):
            log(f"  … {i}/{n} mã · {(time.monotonic() - t0) / 60:.0f} phút · {errors[0]} lỗi")
    return tick


def liquidity(update: bool = True, exchanges: list[str] | None = None, limit: int | None = None,
              updater=None, log: Callable[[str], None] | None = print) -> ScreenResult:
    """Tầng 0–1: vũ trụ → tải giá 1 năm (mã chưa có) hoặc phần mới → lọc thanh khoản."""
    cfg = screen_cfg()
    exchanges = [e.upper() for e in (exchanges or cfg.get("exchanges", ["HOSE", "HNX", "UPCOM"]))]
    index = settings()["valuation"]["index"]
    listing = ""
    if update:
        from vnsa.data.autofetch import Updater

        updater = updater or Updater()
        listing = str(updater.listing()).strip()
    with Store(path("warehouse")) as store:
        uni = universe(store, exchanges)
    if uni.empty:
        raise ValueError("kho chưa có danh sách niêm yết: chạy lệnh screen không kèm --no-update")
    if limit:
        uni = uni.head(limit)
    res_fetch, errors = Counter(), []
    if update:
        steps = updater.quick_prices(uni["symbol"].tolist(), cfg.get("liquidity_years", 1),
                                     progress=_progress(log))
        res_fetch = Counter(s.status for s in steps.values())
        errors = [f"{sym}: {s.detail}" for sym, s in steps.items() if s.status == "error"]
    with Store(path("warehouse")) as store:
        last_idx = store.last_price_date(index)
        recent = store.recent_prices(
            (last_idx or pd.Timestamp.today()) - pd.Timedelta(days=400),
            [*uni["symbol"], index])
    if recent.empty:
        raise ValueError("kho chưa có giá: chạy lệnh screen không kèm --no-update")
    as_of = last_idx if last_idx is not None else recent["date"].max()
    groups = {s: g for s, g in recent.groupby("symbol")}
    idx = groups.get(index)
    idx_close = (pd.Series(idx["close"].to_numpy(float), index=idx["date"])
                 if idx is not None else None)
    rows = []
    for c in uni.itertuples(index=False):
        st = price_stats(groups.get(c.symbol), idx_close, as_of)
        reason = liquidity_check(st, cfg, as_of)
        rows.append({"as_of": as_of, "symbol": c.symbol, "exchange": c.exchange,
                     "industry": c.industry, "stage": 0 if reason else 1, "reason": reason, **st})
    table = pd.DataFrame(rows).reindex(columns=SCREEN_COLUMNS)
    res = ScreenResult(pd.Timestamp(as_of), table, res_fetch, errors, listing)
    res.csv = save(res)
    return res


def save(res: ScreenResult) -> str:
    """Ghi lượt sàng lọc vào kho và CSV; trả đường dẫn CSV."""
    with Store(path("warehouse")) as store:
        store.write_screen(res.table)
    out = path("screen_dir")
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"screen_{res.as_of:%Y-%m-%d}.csv"
    res.table.to_csv(f, index=False)
    return str(f)
