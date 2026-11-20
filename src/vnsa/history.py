"""Lịch sử bền vững ngoài kho DuckDB: khuyến nghị đã lưu và BCTC năm/quý tích luỹ.

Kho `data/warehouse.duckdb` không đưa lên Git và bị xoá sau mỗi lần chạy GitHub Actions, nên:
- khuyến nghị (bảng signals) mất đi thì `review-signals` không bao giờ chấm được;
- bản miễn phí VCI chỉ trả 4 năm / 4 quý gần nhất — muốn có dài hơn phải tự tích luỹ qua thời gian.

`export_history(dir)` ghi các file CSV (signals, fundamentals, fundamentals_q, và các lượt sàng lọc
toàn thị trường: screen — chỉ mã qua lọc thanh khoản —, screen_industry); `import_history`
nạp lại (từ thư mục hoặc từ một nhánh Git, ví dụ `origin/signals` do workflow hằng ngày
`.github/workflows/daily.yml` cập nhật). Nạp chỉ THÊM/ghi đè theo khoá (mã, ngày | năm | quý), không
xoá dữ liệu đang có trong kho.

Danh sách niêm yết (companies: sàn, ngành, loại DN, số cp lưu hành) cũng được lưu từ 20/11/2026 vì
vnstock tắt nguồn VCI miễn phí (11/11/2026) nên kho mới (máy Windows, GitHub Actions không còn cache)
không tự tải được nữa. Nạp chỉ thêm mã kho chưa có; chưa có companies.csv thì lấy sàn/ngành từ lượt
sàng lọc gần nhất của mỗi mã trong screen.csv (các mã đã qua lọc thanh khoản — đủ cho lệnh `screen`).
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pandas as pd

from vnsa.config import PROJECT_ROOT, path

FILES = {"signals": "signals.csv", "fundamentals": "fundamentals.csv",
         "fundamentals_q": "fundamentals_q.csv", "screen": "screen.csv",
         "screen_industry": "screen_industry.csv", "companies": "companies.csv"}


def export_history(out_dir: str | Path) -> dict[str, int]:
    """Ghi toàn bộ lịch sử trong kho ra out_dir; trả về số dòng mỗi bảng."""
    from vnsa.data.store import Store

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    with Store(path("warehouse")) as store:
        tables = {"signals": store.signals().drop(columns="recorded_at", errors="ignore"),
                  "fundamentals": store.all_fundamentals(),
                  "fundamentals_q": store.all_quarterly(),
                  "screen": store.screen_history(min_stage=1),
                  "screen_industry": store.screen_industry_history(),
                  "companies": _companies(store)}
    for name, df in tables.items():
        for c in ("as_of", "last_date"):
            if c in df:
                df = df.assign(**{c: pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")})
        df.to_csv(out / FILES[name], index=False)
        counts[name] = len(df)
    return counts


def _companies(store) -> pd.DataFrame:
    """Danh sách niêm yết có sàn (kèm số cp lưu hành) để lưu ra lịch sử."""
    from vnsa.data.store import COMPANY_COLUMNS

    df = store.con.execute(f"SELECT {', '.join(COMPANY_COLUMNS)}, shares_out FROM companies "
                           "WHERE exchange IS NOT NULL ORDER BY symbol").df()
    return df


def companies_from_screen(scr: pd.DataFrame | None) -> pd.DataFrame:
    """Sàn/ngành/loại DN của mỗi mã theo lượt sàng lọc gần nhất có mã đó."""
    from vnsa.data.store import COMPANY_COLUMNS

    if scr is None or scr.empty or "exchange" not in scr:
        return pd.DataFrame(columns=COMPANY_COLUMNS)
    last = (scr.assign(as_of=pd.to_datetime(scr["as_of"])).sort_values("as_of")
            .dropna(subset=["exchange"]).groupby("symbol").tail(1))
    return last.reindex(columns=COMPANY_COLUMNS).reset_index(drop=True)


def _read_git(ref: str, name: str) -> pd.DataFrame | None:
    """Đọc history/<name> từ nhánh Git `ref` (ví dụ origin/signals) mà không cần checkout."""
    try:
        raw = subprocess.run(["git", "show", f"{ref}:history/{name}"], cwd=PROJECT_ROOT,
                             capture_output=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return pd.read_csv(io.BytesIO(raw)) if raw.strip() else None


def _missing(df: pd.DataFrame, have: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Các dòng của df chưa có trong kho theo khoá (số trong kho có thể mới hơn — không ghi đè)."""
    if have.empty:
        return df
    k_have = set(map(tuple, have[keys].astype(str).to_numpy()))
    mask = [tuple(r) not in k_have for r in df[keys].astype(str).to_numpy()]
    return df[mask]


def import_history(src: str | Path | None = None, git_ref: str | None = None) -> dict[str, int]:
    """Nạp lịch sử vào kho từ thư mục `src` hoặc nhánh Git `git_ref`; chỉ thêm dòng kho chưa có.
    BCTC nạp vào được đánh dấu "tải từ lâu" để lần phân tích sau vẫn tải số mới từ nguồn.
    Trả về số dòng đã thêm mỗi bảng."""
    from vnsa.data.store import Store

    frames: dict[str, pd.DataFrame | None] = {}
    for name, file in FILES.items():
        if git_ref:
            frames[name] = _read_git(git_ref, file)
        else:
            f = Path(src) / file
            frames[name] = pd.read_csv(f) if f.exists() and f.stat().st_size else None
    counts = dict.fromkeys(FILES, 0)
    with Store(path("warehouse")) as store:
        if frames["signals"] is not None and len(frames["signals"]):
            sig = frames["signals"].assign(as_of=lambda x: pd.to_datetime(x["as_of"]))
            have = store.signals().assign(as_of=lambda x: pd.to_datetime(x["as_of"]))
            counts["signals"] = store.write_signals(_missing(sig, have, ["as_of", "symbol"]))
        for name, keys, have, write in (
                ("fundamentals", ["symbol", "year"], store.all_fundamentals(),
                 store.write_fundamentals),
                ("fundamentals_q", ["symbol", "year", "quarter"], store.all_quarterly(),
                 store.write_quarterly)):
            df = frames[name]
            if df is None or df.empty:
                continue
            new = _missing(df, have, keys)
            if "source" not in new:
                new = new.assign(source="lịch sử")
            for src_name, part in new.groupby(new["source"].fillna("lịch sử")):
                write(part, str(src_name))
            store.mark_stale(name, new[keys])
            counts[name] = len(new)
        # danh sách niêm yết: companies.csv trước, bổ sung từ screen.csv; chỉ thêm mã kho chưa có sàn
        parts = [x for x in (frames["companies"], companies_from_screen(frames["screen"]))
                 if x is not None and len(x) and "exchange" in x]
        comp = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if len(comp):
            comp = comp.dropna(subset=["exchange"]).drop_duplicates("symbol")
            have = store.companies()
            have = set(have.loc[have["exchange"].notna(), "symbol"]) if len(have) else set()
            new = comp[~comp["symbol"].str.upper().isin(have)]
            if len(new):
                counts["companies"] = store.write_companies(new, "lịch sử")
                if "shares_out" in new:
                    for sym, n in new.dropna(subset=["shares_out"])[["symbol", "shares_out"]] \
                            .itertuples(index=False):
                        store.write_profile(sym, {"shares_out": float(n)}, stamp=False)
        # lượt sàng lọc: nạp nguyên lượt (ngày) kho chưa có, hoặc thay lượt cùng ngày nhưng chạy
        # nông hơn (vd. máy chỉ chạy --stage liquidity, GitHub chạy đủ 4 tầng cùng phiên)
        depth = store.screen_depth()
        take: set[pd.Timestamp] = set()
        scr = frames["screen"]
        if scr is not None and len(scr):
            scr = scr.assign(as_of=pd.to_datetime(scr["as_of"]))
            for day, stage in scr.groupby("as_of")["stage"].max().items():
                if day not in depth or stage > depth[day]:
                    take.add(day)
        if not take:
            return counts
        for name, write in (("screen", store.write_screen),
                            ("screen_industry", store.write_screen_industry)):
            df = frames[name]
            if df is None or df.empty:
                continue
            df = df.assign(as_of=pd.to_datetime(df["as_of"]))
            new = df[df["as_of"].isin(take)]
            if len(new):
                write(new)
                counts[name] = len(new)
    return counts
