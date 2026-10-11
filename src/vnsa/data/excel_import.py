"""Ghép nhiều file Excel lịch sử giá (ví dụ mỗi năm một file từ CafeF) thành một CSV chuẩn.

    data/raw/excel/FPT/*.xlsx  ->  data/raw/prices/FPT.csv

Giá đầu ra đã điều chỉnh (nếu file có cột "Giá điều chỉnh") và tính bằng đồng.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from vnsa.data.providers.csv_provider import _norm, normalize_prices


def _find_folder(base: Path, symbol: str) -> Path:
    if base.exists():
        for p in base.iterdir():
            if p.is_dir() and p.name.upper() == symbol.upper():
                return p
    raise FileNotFoundError(
        f"Không thấy thư mục {base / symbol.upper()}. Tạo thư mục đó và chép các file Excel vào."
    )


def _read_one(file: Path) -> pd.DataFrame:
    """Đọc một file Excel, tự tìm dòng tiêu đề (dòng có ô 'Ngày'/'Date')."""
    raw = pd.read_excel(file, header=None, dtype=object)
    header_row = None
    for i in range(min(15, len(raw))):
        cells = {_norm(v) for v in raw.iloc[i].tolist() if pd.notna(v)}
        if cells & {"ngay", "date", "ngay giao dich"}:
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"{file.name}: không tìm thấy dòng tiêu đề có cột 'Ngày'")
    df = raw.iloc[header_row + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[header_row].tolist()]
    return df.dropna(how="all")


def import_excel_folder(
    symbol: str, excel_base: Path, out_dir: Path, price_unit: str = "auto"
) -> tuple[Path, pd.DataFrame, list[str]]:
    """Trả về (đường dẫn CSV, bảng đã ghép, danh sách file đã đọc)."""
    folder = _find_folder(excel_base, symbol)
    files = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in {".xlsx", ".xls"}
         and not p.name.startswith("~$")],  # bỏ file tạm khi Excel đang mở
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        raise FileNotFoundError(f"Thư mục {folder} chưa có file .xlsx nào")

    # Chuẩn hoá từng file riêng: hệ số điều chỉnh tính theo từng dòng nên ghép sau vẫn đúng.
    # Ngày trùng giữa các file: giữ bản của file tải sau cùng (giá điều chỉnh mới nhất).
    parts = [normalize_prices(_read_one(f), symbol, price_unit) for f in files]
    df = (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{symbol.upper()}.csv"
    df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).drop(columns="symbol").to_csv(
        out, index=False
    )
    return out, df, [f.name for f in files]
