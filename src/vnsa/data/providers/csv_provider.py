"""Đọc giá từ file CSV tải thủ công (mỗi mã một file, tên file là mã: FPT.csv).

Chấp nhận cả tiêu đề tiếng Anh lẫn tiếng Việt thường gặp khi tải từ các trang
dữ liệu. Nếu file của bạn dùng tên cột khác, thêm vào COLUMN_ALIASES.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd

from vnsa.data.providers.base import PRICE_COLUMNS, DataProvider


def _norm(text: str) -> str:
    """'Giá Đóng cửa (nghìn VNĐ)' -> 'gia dong cua'. Bỏ dấu, chữ thường, bỏ ngoặc."""
    text = re.sub(r"\(.*?\)", "", str(text))
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return text


# tên chuẩn -> các tên có thể gặp (đã chuẩn hoá bằng _norm)
COLUMN_ALIASES: dict[str, list[str]] = {
    "date": ["date", "time", "trading date", "ngay", "ngay giao dich"],
    "open": ["open", "gia mo cua", "mo cua"],
    "high": ["high", "gia cao nhat", "cao nhat"],
    "low": ["low", "gia thap nhat", "thap nhat"],
    "close": ["close", "gia dong cua", "dong cua"],
    "adj_close": ["adj close", "adjusted close", "gia dieu chinh", "dieu chinh"],
    "close_raw": ["close raw", "gia dong cua chua dieu chinh"],   # file do tool tự ghi ra
    "volume": [
        "volume", "vol", "khoi luong", "khoi luong khop lenh", "kl khop lenh",
        "tong khoi luong", "kl",
    ],
}


def _map_columns(columns: list[str]) -> dict[str, str]:
    lookup = {alias: std for std, aliases in COLUMN_ALIASES.items() for alias in aliases}
    mapping: dict[str, str] = {}
    for col in columns:
        std = lookup.get(_norm(col))
        if std and std not in mapping.values():
            mapping[col] = std
    return mapping


def _to_number(s: pd.Series) -> pd.Series:
    """'1,234,500' hoặc '1.234.500' hoặc 98.5 -> số."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    s = s.astype(str).str.strip().str.replace(" ", "", regex=False)
    # Có cả '.' và ',' -> coi ',' là phân cách nghìn
    both = s.str.contains(",") & s.str.contains(r"\.")
    s = s.where(~both, s.str.replace(",", "", regex=False))
    # Chỉ có ',' -> nếu dạng 1,234,567 là phân cách nghìn, nếu dạng 98,5 là thập phân
    thousands_comma = s.str.fullmatch(r"-?\d{1,3}(,\d{3})+")
    s = s.where(~thousands_comma, s.str.replace(",", "", regex=False))
    s = s.str.replace(",", ".", regex=False)
    # Dạng 1.234.567 (nhiều dấu chấm) -> phân cách nghìn
    thousands_dot = s.str.fullmatch(r"-?\d{1,3}(\.\d{3}){2,}")
    s = s.where(~thousands_dot, s.str.replace(".", "", regex=False))
    return pd.to_numeric(s, errors="coerce")


def normalize_prices(
    raw: pd.DataFrame, symbol: str, price_unit: str = "auto", adjust: bool = True
) -> pd.DataFrame:
    """Đưa một bảng giá bất kỳ về PRICE_COLUMNS, giá tính bằng đồng.

    adjust=True và file có cột giá điều chỉnh: nhân open/high/low/close với hệ số
    (giá điều chỉnh / giá đóng cửa) để cả nến đều là giá điều chỉnh, tránh biểu đồ
    "gãy" ở ngày GDKHQ. Khối lượng giữ nguyên.
    """
    mapping = _map_columns(list(raw.columns))
    df = raw.rename(columns=mapping)
    missing = [c for c in ["date", "open", "high", "low", "close", "volume"] if c not in df]
    if missing:
        raise ValueError(
            f"{symbol}: thiếu cột {missing}. Cột trong file: {list(raw.columns)}. "
            "Thêm tên cột của bạn vào COLUMN_ALIASES trong csv_provider.py."
        )

    out = pd.DataFrame(index=df.index)
    dates = df["date"]
    if not pd.api.types.is_datetime64_any_dtype(dates):
        as_str = dates.astype(str).str.strip()
        iso = as_str.str.fullmatch(r"\d{4}-\d{2}-\d{2}.*").all()
        dates = pd.to_datetime(as_str, dayfirst=not iso, errors="coerce")
    out["date"] = dates.dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = _to_number(df[col])
    # Giá đóng cửa CHƯA điều chỉnh (giá khớp thật hôm đó): cần để tính P/E, P/B lịch sử đúng (giá đã
    # điều chỉnh cổ tức tiền mặt kéo giá quá khứ xuống, làm P/E lịch sử thấp hơn thực tế).
    out["close_raw"] = (_to_number(df["close_raw"]) if "close_raw" in df else
                        out["close"] if "adj_close" in df else float("nan"))
    if adjust and "adj_close" in df:
        adj = _to_number(df["adj_close"])
        factor = (adj / out["close"]).where((adj > 0) & (out["close"] > 0), 1.0)
        out[["open", "high", "low", "close"]] = out[["open", "high", "low", "close"]].mul(
            factor, axis=0
        )
    out["symbol"] = symbol.upper()

    prices = ["open", "high", "low", "close", "close_raw"]
    if price_unit == "nghin" or (price_unit == "auto" and out["close"].median() < 1000):
        out[prices] *= 1000
    out[prices] = out[prices].round(0)

    out = out.dropna(subset=["date", "close"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    out["volume"] = out["volume"].fillna(0).round().astype("int64")
    return out[[*PRICE_COLUMNS, "close_raw"]]


class CSVProvider(DataProvider):
    name = "csv"

    def __init__(self, folder: str | Path, price_unit: str = "auto"):
        self.folder = Path(folder)
        self.price_unit = price_unit

    def _file(self, symbol: str) -> Path:
        for ext in (".csv", ".CSV"):
            p = self.folder / f"{symbol.upper()}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"Không thấy file {symbol.upper()}.csv trong {self.folder}")

    def symbols(self) -> list[str]:
        if not self.folder.exists():
            return []
        return sorted(p.stem.upper() for p in self.folder.glob("*.csv"))

    def ohlcv(self, symbol: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        raw = pd.read_csv(self._file(symbol), encoding="utf-8-sig")
        df = normalize_prices(raw, symbol, self.price_unit)
        if start is not None:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
        return df.reset_index(drop=True)
