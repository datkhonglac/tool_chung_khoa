"""Lịch sử bền vững ngoài kho DuckDB: khuyến nghị đã lưu và BCTC năm/quý tích luỹ.

Kho `data/warehouse.duckdb` không đưa lên Git và bị xoá sau mỗi lần chạy GitHub Actions, nên:
- khuyến nghị (bảng signals) mất đi thì `review-signals` không bao giờ chấm được;
- bản miễn phí VCI chỉ trả 4 năm / 4 quý gần nhất — muốn có dài hơn phải tự tích luỹ qua thời gian.

`export_history(dir)` ghi ba file CSV (signals, fundamentals, fundamentals_q); `import_history`
nạp lại (từ thư mục hoặc từ một nhánh Git, ví dụ `origin/signals` do workflow hằng ngày
`.github/workflows/daily.yml` cập nhật). Nạp chỉ THÊM/ghi đè theo khoá (mã, ngày | năm | quý), không
xoá dữ liệu đang có trong kho.
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pandas as pd

from vnsa.config import PROJECT_ROOT, path

FILES = {"signals": "signals.csv", "fundamentals": "fundamentals.csv",
         "fundamentals_q": "fundamentals_q.csv"}


def export_history(out_dir: str | Path) -> dict[str, int]:
    """Ghi toàn bộ lịch sử trong kho ra out_dir; trả về số dòng mỗi bảng."""
    from vnsa.data.store import Store

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    with Store(path("warehouse")) as store:
        tables = {"signals": store.signals().drop(columns="recorded_at", errors="ignore"),
                  "fundamentals": store.all_fundamentals(),
                  "fundamentals_q": store.all_quarterly()}
    for name, df in tables.items():
        if "as_of" in df:
            df = df.assign(as_of=pd.to_datetime(df["as_of"]).dt.strftime("%Y-%m-%d"))
        df.to_csv(out / FILES[name], index=False)
        counts[name] = len(df)
    return counts


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
    return counts
