"""Danh mục chuỗi vĩ mô và số liệu nhập tay.

config/macro_map.toml, mục [series]: mỗi chuỗi một dòng
    us_10y = { name = "...", source = "fred", code = "DGS10", freq = "D", kind = "diff", unit = "%" }
- source: fred | worldbank | yahoo | manual (nhập tay ở config/macro_manual.toml)
          | derived (tính từ hai chuỗi khác: code = "a/b", ví dụ JPY/VND = usd_vnd/usd_jpy)
- freq  : D (ngày) | M (tháng) | Q (quý). Chuỗi tháng/quý lưu ngày đầu kỳ (2026-09-01 = tháng 9).
- kind  : cách đọc biến động (dùng ở bước chấm điểm)
    change = % thay đổi sau `months` tháng (giá, tỷ giá, chỉ số giá)
    diff   = chênh lệch điểm % sau `months` tháng (lãi suất, lợi suất)
    level  = so mức hiện tại với `neutral`, đơn vị `scale` (tăng trưởng so cùng kỳ, PMI)
    ytd    = so với cùng kỳ năm trước của số luỹ kế từ đầu năm (tăng trưởng tín dụng)

config/macro_manual.toml: mỗi chuỗi nhập tay một bảng, khoá là kỳ "YYYY-MM" hoặc "YYYY-Qn":
    [vn_cpi_yoy]
    "2026-08" = 3.2
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from vnsa.config import path

SOURCES = ("fred", "worldbank", "yahoo", "manual", "derived")
KINDS = ("change", "diff", "level", "ytd")


@dataclass(frozen=True)
class Series:
    id: str
    name: str
    source: str
    code: str = ""
    freq: str = "M"
    kind: str = "change"
    unit: str = ""
    months: int = 3
    neutral: float | None = None
    scale: float | None = None
    note: str = ""

    @property
    def manual(self) -> bool:
        return self.source == "manual"

    @property
    def auto(self) -> bool:
        """Tải từ nguồn ngoài (không phải nhập tay, không phải tính từ chuỗi khác)."""
        return self.source not in ("manual", "derived")


def _load(file: Path | str | None, key: str) -> dict:
    f = Path(file) if file is not None else path(key)
    if not f.exists():
        return {}
    with f.open("rb") as fh:
        return tomllib.load(fh)


def catalog(file: Path | str | None = None) -> dict[str, Series]:
    """Mọi chuỗi khai báo ở mục [series] của macro_map.toml (kiểm tra nguồn và cách đọc)."""
    out: dict[str, Series] = {}
    for sid, spec in _load(file, "macro_map").get("series", {}).items():
        s = Series(id=sid, **spec)
        if s.source not in SOURCES:
            raise ValueError(f"chuỗi {sid}: nguồn '{s.source}' không hợp lệ ({', '.join(SOURCES)})")
        if s.kind not in KINDS:
            raise ValueError(f"chuỗi {sid}: kind '{s.kind}' không hợp lệ ({', '.join(KINDS)})")
        if s.kind == "level" and (s.neutral is None or not s.scale):
            raise ValueError(f"chuỗi {sid}: kind = 'level' cần neutral và scale")
        if not s.manual and not s.code:
            raise ValueError(f"chuỗi {sid}: thiếu code (mã ở nguồn {s.source})")
        out[sid] = s
    for s in out.values():
        if s.source == "derived":
            parts = s.code.split("/")
            if len(parts) != 2 or any(x.strip() not in out for x in parts):
                raise ValueError(f"chuỗi {s.id}: code '{s.code}' phải là 'a/b' với a, b trong danh mục")
    return out


def parse_period(key: str) -> pd.Timestamp:
    """"2026-09" -> 01/09/2026; "2026-Q3" -> 01/07/2026 (ngày đầu kỳ)."""
    k = str(key).strip().upper()
    m = re.fullmatch(r"(\d{4})-Q([1-4])", k)
    if m:
        return pd.Timestamp(int(m[1]), 3 * int(m[2]) - 2, 1)
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", k)
    if m and 1 <= int(m[2]) <= 12:
        return pd.Timestamp(int(m[1]), int(m[2]), 1)
    raise ValueError(f"kỳ '{key}' không hợp lệ (dùng YYYY-MM hoặc YYYY-Qn)")


def manual_frame(file: Path | str | None = None,
                 cat: dict[str, Series] | None = None) -> pd.DataFrame:
    """Số nhập tay -> bảng series, date, value. Chỉ nhận chuỗi có trong danh mục với nguồn manual
    (gõ sai tên chuỗi thì báo lỗi, không lặng lẽ bỏ qua)."""
    cat = catalog() if cat is None else cat
    rows = []
    for sid, periods in _load(file, "macro_manual").items():
        if not isinstance(periods, dict):
            continue
        if sid not in cat or not cat[sid].manual:
            raise ValueError(f"macro_manual.toml: chuỗi '{sid}' không có trong danh mục nhập tay "
                             "(config/macro_map.toml)")
        for k, v in periods.items():
            rows.append({"series": sid, "date": parse_period(k), "value": float(v)})
    return pd.DataFrame(rows, columns=["series", "date", "value"])
