"""Chạy sàng lọc toàn thị trường theo tầng; mỗi lượt ghi bảng `screen` và CSV."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from vnsa.config import path, settings
from vnsa.data.store import INDUSTRY_COLUMNS, SCREEN_COLUMNS, Store
from vnsa.screen.groups import assign_groups
from vnsa.screen.liquidity import liquidity_check, price_stats, reason_group

STAGES = ("liquidity", "industry", "pick")     # chạy đến tầng 1, 3 hoặc 4


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
    industry: pd.DataFrame | None = None                 # điểm nhóm ngành (tầng 3)
    outside: pd.DataFrame | None = None                  # mã MUA ngoài nhóm dẫn đầu (tầng 4)
    regime: str = ""                                     # xu hướng VNINDEX lúc sàng lọc
    deep_fetch_errors: dict[str, str] = field(default_factory=dict)
    fallbacks: dict[str, str] = field(default_factory=dict)   # mã phải lấy giá vnstock (CafeF lỗi)
    stage: str = "liquidity"

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
              updater=None, log: Callable[[str], None] | None = print,
              save_result: bool = True) -> ScreenResult:
    """Tầng 0–1: vũ trụ → tải giá 1 năm (mã chưa có) hoặc phần mới → lọc thanh khoản."""
    from vnsa.analysis import macro_mapping

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
    uni = uni.assign(group_name=assign_groups(uni, macro_mapping()))
    res_fetch, errors, fallbacks = Counter(), [], {}
    if update:
        if log:
            log(f"Tầng 1: tải giá {len(uni)} mã ({cfg.get('workers', 4)} luồng)")
        steps = updater.quick_prices(uni["symbol"].tolist(), cfg.get("liquidity_years", 1),
                                     progress=_progress(log), workers=cfg.get("workers", 4))
        res_fetch = Counter(s.status for s in steps.values())
        errors = [f"{sym}: {s.detail}" for sym, s in steps.items() if s.status == "error"]
        fallbacks = dict(updater.fallbacks)
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
                     "industry": c.industry, "group_name": c.group_name, "model": c.model,
                     "stage": 0 if reason else 1, "reason": reason, **st})
    table = pd.DataFrame(rows).reindex(columns=SCREEN_COLUMNS)
    res = ScreenResult(pd.Timestamp(as_of), table, res_fetch, errors, listing,
                       fallbacks=fallbacks)
    if save_result:
        res.csv = save(res)
    return res


def run(stage: str = "pick", update: bool = True, exchanges: list[str] | None = None,
        limit: int | None = None, updater=None, log: Callable[[str], None] | None = print,
        deep_minutes: float | None = None) -> ScreenResult:
    """Chạy sàng lọc đến tầng `stage`: "liquidity" (tầng 1) | "industry" (tầng 2–3: phân tích từng
    mã qua lọc rồi chấm nhóm ngành) | "pick" (tầng 4: chọn mã trong các nhóm dẫn đầu)."""
    from vnsa import analysis
    from vnsa.screen import deep
    from vnsa.screen.groups import group_macro
    from vnsa.screen.industry import industry_scores

    if stage not in STAGES:
        raise ValueError(f"tầng không hợp lệ: {stage} (chọn {', '.join(STAGES)})")
    res = liquidity(update, exchanges, limit, updater, log, save_result=stage == "liquidity")
    res.stage = stage
    if stage == "liquidity":
        return res
    t = res.table.set_index("symbol")
    passed = (t[t["stage"] >= 1].sort_values("value_20d", ascending=False).index.tolist())
    cfg = screen_cfg()
    if update:
        minutes = deep_minutes if deep_minutes is not None else cfg.get("deep_minutes")
        if log:
            log(f"Tầng 2: tải dữ liệu đủ {settings()['auto']['years']} năm giá + BCTC cho "
                f"{len(passed)} mã (thanh khoản cao trước"
                + (f", tối đa {minutes:.0f} phút)" if minutes else ")"))
        res.deep_fetch_errors = deep.fetch_all(passed, updater, log, minutes,
                                               workers=cfg.get("workers", 4))
    rows, errors = deep.assess_all(passed, log)
    try:
        res.regime = str(analysis.market_regime())
    except Exception:  # noqa: BLE001 - thiếu chỉ số chỉ mất dòng bối cảnh
        res.regime = ""
    if rows:
        upd = pd.DataFrame.from_dict(rows, orient="index")
        for c in upd.columns:
            if not pd.api.types.is_numeric_dtype(upd[c]):     # chữ: cột NaN (float) -> object
                t[c] = t[c].astype(object)
            t.loc[upd.index, c] = upd[c]
        t.loc[upd.index, "stage"] = 2
    for sym, err in errors.items():
        t.loc[sym, "reason"] = f"phân tích lỗi: {err}"
    for sym, err in res.deep_fetch_errors.items():           # phân tích được nhưng thiếu dữ liệu
        if sym in rows and err.startswith("chưa tải"):
            t.loc[sym, "reason"] = err
    ind = industry_scores(t[t["stage"] >= 1].reset_index(),
                          group_macro(analysis.macro_mapping(), analysis.macro_readings()), cfg)
    if not ind.empty:
        ind.insert(0, "as_of", res.as_of)
        ranks = ind.set_index("group_name")
        t["group_rank"] = t["group_name"].map(ranks["rank"])
        t["group_score"] = t["group_name"].map(ranks["score"])
        top = cfg.get("top_industries", 5)
        t.loc[(t["stage"] >= 2) & (t["group_rank"] <= top), "stage"] = 3
    if stage == "pick":
        from vnsa.screen.pick import pick

        t, outside = pick(t, cfg)
        res.outside = outside.reset_index()          # pick() giữ mã ở chỉ mục
    res.table = t.reset_index().reindex(columns=SCREEN_COLUMNS)
    res.industry = ind.reindex(columns=INDUSTRY_COLUMNS) if not ind.empty else ind
    res.csv = save(res)
    return res


def save(res: ScreenResult) -> str:
    """Ghi lượt sàng lọc vào kho và CSV; trả đường dẫn CSV."""
    with Store(path("warehouse")) as store:
        store.write_screen(res.table, res.industry)
    out = path("screen_dir")
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"screen_{res.as_of:%Y-%m-%d}.csv"
    res.table.to_csv(f, index=False)
    if res.industry is not None and not res.industry.empty:
        res.industry.to_csv(out / f"industry_{res.as_of:%Y-%m-%d}.csv", index=False)
    return str(f)


def latest(as_of=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(bảng screen, điểm nhóm ngành) của lượt sàng lọc mới nhất trong kho (rỗng nếu chưa chạy)."""
    with Store(path("warehouse")) as store:
        return store.screen(as_of), store.screen_industry(as_of)


def outside_picks(table: pd.DataFrame) -> pd.DataFrame:
    """Mã MUA đạt mọi điều kiện tầng 4 nhưng nằm ngoài các nhóm dẫn đầu (đọc lại từ bảng screen)."""
    from vnsa.screen.pick import order

    m = (table["stage"] == 2) & (table["action"] == "MUA") & (
        table["reason"] == "đạt nhưng ngoài nhóm ngành dẫn đầu")
    return order(table[m]).head(screen_cfg().get("outside_max", 10))


def review(horizons: list[int] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """(bảng chấm từng mã × kỳ, tổng hợp theo kỳ × tầng, số lượt đã lưu) — screen/review.py."""
    from vnsa.screen.review import evaluate, summarize

    horizons = horizons or screen_cfg().get("review_horizons", [20, 60])
    index = settings()["valuation"]["index"]
    with Store(path("warehouse")) as store:
        runs = store.screen_history(min_stage=1)
        if runs.empty:
            return pd.DataFrame(), pd.DataFrame(), 0
        px = store.recent_prices(runs["as_of"].min() - pd.Timedelta(days=10),
                                 [*runs["symbol"].unique(), index])
    closes = px.pivot_table(index="date", columns="symbol", values="close").sort_index()
    idx = closes.pop(index) if index in closes else pd.Series(dtype=float)
    ev = evaluate(runs, closes, idx.dropna(), horizons)
    return ev, summarize(ev), runs["as_of"].nunique()


def study(years: int | None = None, update: bool = True, log: Callable[[str], None] | None = print):
    """Kiểm định quá khứ phần giá của bộ lọc (screen/study.py) trên các mã đủ thanh khoản ở lượt
    sàng lọc gần nhất; tải đủ `years` năm giá (mặc định [study] years). Lưu CSV ở data/study/."""
    from vnsa.screen.study import run_study

    cfg, scfg = screen_cfg(), settings().get("study", {})
    years = years or scfg.get("years", 10)
    index = settings()["valuation"]["index"]
    table, _ = latest()
    if table.empty:
        raise ValueError("chưa có lượt sàng lọc nào: chạy lệnh screen trước (lấy danh sách mã)")
    liquid = table[table["stage"] >= 1]
    groups = liquid.set_index("symbol")["group_name"]
    if update:
        from vnsa.data.autofetch import Updater

        if log:
            log(f"Tải {years} năm giá cho {len(liquid)} mã + {index} "
                f"({cfg.get('workers', 4)} luồng) ...")
        steps = Updater().long_prices(liquid["symbol"].tolist(), years,
                                      workers=cfg.get("workers", 4))
        bad = [s for s in steps if s.status == "error"]
        if log and bad:
            log(f"  ✗ {len(bad)} mã lỗi tải giá (bỏ qua)")
    with Store(path("warehouse")) as store:
        px = store.recent_prices(pd.Timestamp.today() - pd.Timedelta(days=round(365.25 * years)),
                                 [*liquid["symbol"], index])
    raw = px["close_raw"].where(px["close_raw"] > 0, px["close"])
    px = px.assign(value=raw * px["volume"].fillna(0))
    close = px.pivot_table(index="date", columns="symbol", values="close").sort_index()
    value = px.pivot_table(index="date", columns="symbol", values="value").sort_index()
    idx = close.pop(index) if index in close else pd.Series(dtype=float)
    value = value.drop(columns=[index], errors="ignore")
    res = run_study(close, value, idx, groups, cfg, settings().get("backtest", {}),
                    scfg.get("split"))
    out = path("warehouse").parent / "study"
    out.mkdir(parents=True, exist_ok=True)
    res.monthly.to_csv(out / "screen_study_monthly.csv", index=False)
    res.periods.to_csv(out / "screen_study.csv", index=False)
    res.picks.to_csv(out / "screen_study_picks.csv", index=False)
    return res
