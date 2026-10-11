"""Lớp trừu tượng cho mọi nguồn dữ liệu.

Mọi module phía trên (FA, TA, Quant) chỉ làm việc với định dạng chuẩn ở đây.
Muốn đổi nguồn (CSV -> vnstock -> DNSE -> SSI) chỉ cần viết thêm một lớp con.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

# Định dạng chuẩn của bảng giá: giá tính bằng đồng, khối lượng bằng cổ phiếu.
PRICE_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]


class DataProvider(ABC):
    name: str = "base"

    @abstractmethod
    def symbols(self) -> list[str]:
        """Danh sách mã nguồn này có dữ liệu."""

    @abstractmethod
    def ohlcv(
        self, symbol: str, start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        """Giá ngày theo PRICE_COLUMNS, sắp xếp tăng dần theo ngày."""

    # Sẽ cài đặt ở ngày 2 (BCTC) — để sẵn chữ ký cho thống nhất.
    def financials(self, symbol: str, statement: str, period: str = "year") -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} chưa hỗ trợ BCTC")
