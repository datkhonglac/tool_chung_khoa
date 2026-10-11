"""Kho dữ liệu cục bộ bằng DuckDB (một file duy nhất: data/warehouse.duckdb).

Mọi module phân tích đọc từ đây, không gọi thẳng nguồn bên ngoài.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from vnsa.data.providers.base import PRICE_COLUMNS
from vnsa.fa.ratios import FIELDS

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    symbol  VARCHAR NOT NULL,
    date    DATE    NOT NULL,
    open    DOUBLE,
    high    DOUBLE,
    low     DOUBLE,
    close   DOUBLE,
    volume  BIGINT,
    source  VARCHAR,
    loaded_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (symbol, date)
);
"""

# Mỗi khoản mục BCTC chuẩn (FIELDS) là một cột, đơn vị đồng.
SCHEMA += f"""
CREATE TABLE IF NOT EXISTS fundamentals (
    symbol  VARCHAR NOT NULL,
    year    INTEGER NOT NULL,
{"".join(f"    {f} DOUBLE,{chr(10)}" for f in FIELDS)}    source  VARCHAR,
    loaded_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (symbol, year)
);
"""


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.db_path))
        self.con.execute(SCHEMA)
        for f in FIELDS:  # kho tạo từ bản cũ: thêm cột khoản mục mới nếu còn thiếu
            self.con.execute(f"ALTER TABLE fundamentals ADD COLUMN IF NOT EXISTS {f} DOUBLE")

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def write_prices(self, df: pd.DataFrame, source: str) -> int:
        """Ghi đè toàn bộ giá của các mã có trong df (nạp lại sạch, không trùng)."""
        if df.empty:
            return 0
        frame = df[PRICE_COLUMNS].copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        frame["source"] = source
        symbols = frame["symbol"].unique().tolist()
        self.con.register("incoming", frame)
        try:
            self.con.execute("BEGIN")
            self.con.execute(
                "DELETE FROM prices WHERE symbol IN (SELECT DISTINCT symbol FROM incoming)"
            )
            self.con.execute(
                "INSERT INTO prices (symbol, date, open, high, low, close, volume, source) "
                "SELECT symbol, date, open, high, low, close, volume, source FROM incoming"
            )
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        finally:
            self.con.unregister("incoming")
        return len(frame) if symbols else 0

    def prices(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        sql = "SELECT symbol, date, open, high, low, close, volume FROM prices WHERE symbol = ?"
        params: list = [symbol.upper()]
        if start:
            sql += " AND date >= ?"
            params.append(start)
        if end:
            sql += " AND date <= ?"
            params.append(end)
        df = self.con.execute(sql + " ORDER BY date", params).df()
        df["date"] = pd.to_datetime(df["date"])
        return df

    def write_fundamentals(self, df: pd.DataFrame, source: str) -> int:
        """Ghi đè BCTC năm của các mã có trong df. Cột: symbol, year + FIELDS (đồng)."""
        if df.empty:
            return 0
        frame = df[["symbol", "year", *FIELDS]].copy()
        frame["source"] = source
        cols = ", ".join(["symbol", "year", *FIELDS, "source"])
        self.con.register("incoming", frame)
        try:
            self.con.execute("BEGIN")
            self.con.execute(
                "DELETE FROM fundamentals WHERE symbol IN (SELECT DISTINCT symbol FROM incoming)"
            )
            self.con.execute(f"INSERT INTO fundamentals ({cols}) SELECT {cols} FROM incoming")
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        finally:
            self.con.unregister("incoming")
        return len(frame)

    def fundamentals(self, symbol: str) -> pd.DataFrame:
        cols = ", ".join(["symbol", "year", *FIELDS])
        return self.con.execute(
            f"SELECT {cols} FROM fundamentals WHERE symbol = ? ORDER BY year", [symbol.upper()]
        ).df()

    def inventory(self) -> pd.DataFrame:
        """Mỗi mã có bao nhiêu phiên, từ ngày nào đến ngày nào."""
        return self.con.execute(
            "SELECT symbol, COUNT(*) AS so_phien, MIN(date) AS tu_ngay, MAX(date) AS den_ngay, "
            "ANY_VALUE(source) AS nguon FROM prices GROUP BY symbol ORDER BY symbol"
        ).df()
