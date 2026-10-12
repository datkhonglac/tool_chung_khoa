"""Cập nhật số liệu vĩ mô trong kho: tải chuỗi tự động, tính chuỗi dẫn xuất, nạp số nhập tay.

Dùng chung cho lệnh `fetch-macro` (tải hết) và autofetch khi phân tích (chỉ tải khi số đã cũ).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from vnsa.data.store import Store
from vnsa.macro.catalog import Series, catalog, manual_frame


@dataclass
class Result:
    series: str
    rows: int = 0
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    value: float = float("nan")
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def fetch_auto(store: Store, cat: dict[str, Series], provider, sids: list[str] | None = None,
               years: int = 10) -> list[Result]:
    """Tải các chuỗi tự động (`sids`, mặc định tất cả) rồi ghi vào kho; một chuỗi lỗi không chặn
    các chuỗi khác. Sau đó tính lại các chuỗi dẫn xuất (derived)."""
    from vnsa.data.providers.macro_provider import derive

    out = []
    for sid in sids or [s for s, spec in cat.items() if spec.auto]:
        try:
            df = provider.series(sid, years=years)
        except Exception as e:  # noqa: BLE001 - nguồn ngoài lỗi đủ kiểu (mạng, định dạng)
            out.append(Result(sid, error=f"{type(e).__name__}: {str(e)[:120]}"))
            continue
        if df.empty:
            out.append(Result(sid, error=f"không có số trong {years} năm"))
            continue
        store.write_macro(df)
        out.append(Result(sid, len(df), df["date"].min(), df["date"].max(),
                          float(df["value"].iloc[-1])))
    data = store.macro()
    for sid, spec in cat.items():
        if spec.source != "derived":
            continue
        df = derive(spec, data)
        if df.empty:
            out.append(Result(sid, error=f"thiếu chuỗi gốc {spec.code}"))
            continue
        store.write_macro(df)
        out.append(Result(sid, len(df), df["date"].min(), df["date"].max(),
                          float(df["value"].iloc[-1])))
    return out


def load_manual(store: Store, cat: dict[str, Series], file=None) -> tuple[int, list[str]]:
    """Nạp lại TOÀN BỘ số nhập tay (file là nguồn gốc). Trả (số dòng, các chuỗi chưa có số)."""
    man = manual_frame(file, cat)
    ids = [sid for sid, s in cat.items() if s.manual]
    n = store.replace_macro(man, ids, "nhập tay")
    have = set(man["series"])
    return n, [sid for sid in ids if sid not in have]


def refresh(max_age_hours: float = 20, years: int = 10, provider=None,
            log: Callable[[str], None] | None = None) -> tuple[str, str]:
    """Cho autofetch: nạp số nhập tay (nhanh) và tải lại chuỗi tự động nếu lần tải gần nhất cũ hơn
    `max_age_hours`. Trả (trạng thái ok|skip|error, mô tả)."""
    from vnsa.config import path
    from vnsa.data.providers.macro_provider import MacroProvider

    cat = catalog()
    with Store(path("warehouse")) as store:
        try:
            n_manual, _ = load_manual(store, cat)
        except ValueError as e:
            return "error", f"số nhập tay: {e}"
        auto = [sid for sid, s in cat.items() if s.auto]
        loaded = store.macro_loaded_at()
        # tuổi = lần tải cũ nhất trong các chuỗi ĐÃ có; chuỗi lỗi (nguồn hỏng) được thử lại ở lần
        # làm mới kế tiếp, không làm mọi lượt phân tích phải chờ nguồn đó
        times = [loaded[s] for s in auto if s in loaded]
        oldest = min(times) if times else None
        age = (pd.Timestamp.now() - oldest).total_seconds() / 3600 if oldest is not None else None
        if age is not None and age < max_age_hours:
            return "skip", f"số vĩ mô tải {age:.0f} giờ trước; {n_manual} số nhập tay"
        prov = provider or MacroProvider(cat, log=log)
        res = fetch_auto(store, cat, prov, years=years)
    bad = [r.series for r in res if not r.ok]
    detail = f"{len(res) - len(bad)}/{len(res)} chuỗi tự động, {n_manual} số nhập tay"
    if bad:
        detail += f"; lỗi: {', '.join(bad)}"
    return ("error" if len(bad) == len(res) else "ok"), detail
