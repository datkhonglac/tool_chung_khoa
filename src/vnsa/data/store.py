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

-- Danh sách niêm yết: sàn (biên độ, bước giá), ngành và loại doanh nghiệp (mô hình FA)
CREATE TABLE IF NOT EXISTS companies (
    symbol   VARCHAR PRIMARY KEY,
    name     VARCHAR,
    exchange VARCHAR,     -- HOSE | HNX | UPCOM
    industry VARCHAR,
    model    VARCHAR,     -- general | bank | securities | insurance
    source   VARCHAR,
    loaded_at TIMESTAMP DEFAULT current_timestamp
);
"""

SCHEMA += """
-- Lịch sử khuyến nghị: mỗi mã một dòng cho mỗi ngày giá (chạy lại trong ngày thì ghi đè)
CREATE TABLE IF NOT EXISTS signals (
    as_of       DATE    NOT NULL,   -- phiên giá dùng để khuyến nghị
    symbol      VARCHAR NOT NULL,
    action      VARCHAR,            -- MUA | THEO DÕI | TRÁNH
    score       DOUBLE,
    fa_score    DOUBLE,
    ta_score    DOUBLE,
    price       DOUBLE,             -- giá đóng cửa phiên as_of
    fair_value  DOUBLE,
    zone_status VARCHAR,            -- MUA TRONG VÙNG | CHỜ GIÁ
    buy_low     DOUBLE,
    buy_high    DOUBLE,
    stop        DOUBLE,
    target      DOUBLE,             -- mục tiêu 1
    params      VARCHAR,            -- mã băm cấu hình chấm điểm (đổi quy tắc -> đổi mã)
    recorded_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (as_of, symbol)
);
"""

COMPANY_COLUMNS = ["symbol", "name", "exchange", "industry", "model"]
SIGNAL_COLUMNS = ["as_of", "symbol", "action", "score", "fa_score", "ta_score", "price",
                  "fair_value", "zone_status", "buy_low", "buy_high", "stop", "target", "params"]


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.db_path))
        self.con.execute(SCHEMA)
        for f in FIELDS:  # kho tạo từ bản cũ: thêm cột khoản mục mới nếu còn thiếu
            self.con.execute(f"ALTER TABLE fundamentals ADD COLUMN IF NOT EXISTS {f} DOUBLE")
        # giá đóng cửa chưa điều chỉnh (NULL nếu nguồn không có)
        self.con.execute("ALTER TABLE prices ADD COLUMN IF NOT EXISTS close_raw DOUBLE")

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
        frame = df.reindex(columns=[*PRICE_COLUMNS, "close_raw"]).copy()
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
                "INSERT INTO prices (symbol, date, open, high, low, close, volume, close_raw, source) "
                "SELECT symbol, date, open, high, low, close, volume, close_raw, source FROM incoming"
            )
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        finally:
            self.con.unregister("incoming")
        return len(frame) if symbols else 0

    def upsert_prices(self, df: pd.DataFrame, source: str) -> int:
        """Chỉ thay các phiên trong khoảng ngày của df (tải tăng dần), giữ nguyên phiên cũ hơn."""
        if df.empty:
            return 0
        frame = df.reindex(columns=[*PRICE_COLUMNS, "close_raw"]).copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        frame["source"] = source
        self.con.register("incoming", frame)
        try:
            self.con.execute("BEGIN")
            self.con.execute(
                "DELETE FROM prices p USING (SELECT symbol, MIN(date) AS d0, MAX(date) AS d1 "
                "FROM incoming GROUP BY symbol) r "
                "WHERE p.symbol = r.symbol AND p.date BETWEEN r.d0 AND r.d1"
            )
            self.con.execute(
                "INSERT INTO prices (symbol, date, open, high, low, close, volume, close_raw, source) "
                "SELECT symbol, date, open, high, low, close, volume, close_raw, source FROM incoming"
            )
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        finally:
            self.con.unregister("incoming")
        return len(frame)

    def price_source(self, symbol: str) -> str | None:
        """Nguồn của phiên mới nhất (để lần cập nhật sau dùng lại đúng nguồn đó)."""
        row = self.con.execute("SELECT source FROM prices WHERE symbol = ? ORDER BY date DESC "
                               "LIMIT 1", [symbol.upper()]).fetchone()
        return row[0] if row else None

    def last_price_date(self, symbol: str) -> pd.Timestamp | None:
        d = self.con.execute("SELECT MAX(date) FROM prices WHERE symbol = ?",
                             [symbol.upper()]).fetchone()[0]
        return None if d is None else pd.Timestamp(d)

    def prices(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        sql = ("SELECT symbol, date, open, high, low, close, volume, close_raw FROM prices "
               "WHERE symbol = ?")
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
        frame = df.reindex(columns=["symbol", "year", *FIELDS]).copy()  # khoản thiếu -> NULL
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

    def fundamentals_loaded_at(self, symbol: str) -> pd.Timestamp | None:
        t = self.con.execute("SELECT MAX(loaded_at) FROM fundamentals WHERE symbol = ?",
                             [symbol.upper()]).fetchone()[0]
        return None if t is None else pd.Timestamp(t)

    def write_companies(self, df: pd.DataFrame, source: str) -> int:
        """Ghi đè thông tin các mã có trong df (cột COMPANY_COLUMNS, thiếu cột nào để trống)."""
        if df.empty:
            return 0
        frame = df.reindex(columns=COMPANY_COLUMNS).drop_duplicates("symbol").copy()
        frame["symbol"] = frame["symbol"].str.upper()
        frame["source"] = source
        cols = ", ".join([*COMPANY_COLUMNS, "source"])
        self.con.register("incoming", frame)
        try:
            self.con.execute("BEGIN")
            self.con.execute("DELETE FROM companies WHERE symbol IN (SELECT symbol FROM incoming)")
            self.con.execute(f"INSERT INTO companies ({cols}) SELECT {cols} FROM incoming")
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        finally:
            self.con.unregister("incoming")
        return len(frame)

    def company(self, symbol: str) -> dict | None:
        """Sàn, ngành, loại doanh nghiệp của một mã; None nếu chưa có."""
        cols = [*COMPANY_COLUMNS, "loaded_at"]
        row = self.con.execute(f"SELECT {', '.join(cols)} FROM companies WHERE symbol = ?",
                               [symbol.upper()]).fetchone()
        return dict(zip(cols, row)) if row else None

    def peers(self, industry: str) -> list[str]:
        """Các mã cùng ngành (bảng companies) đã có BCTC trong kho."""
        rows = self.con.execute(
            "SELECT symbol FROM companies WHERE industry = ? "
            "AND symbol IN (SELECT DISTINCT symbol FROM fundamentals) ORDER BY symbol",
            [industry]).fetchall()
        return [r[0] for r in rows]

    def companies_loaded_at(self) -> pd.Timestamp | None:
        t = self.con.execute("SELECT MAX(loaded_at) FROM companies").fetchone()[0]
        return None if t is None else pd.Timestamp(t)

    def write_signals(self, df: pd.DataFrame) -> int:
        """Ghi đè khuyến nghị cùng (ngày, mã); cột SIGNAL_COLUMNS."""
        if df.empty:
            return 0
        frame = df.reindex(columns=SIGNAL_COLUMNS).copy()
        frame["as_of"] = pd.to_datetime(frame["as_of"]).dt.date
        cols = ", ".join(SIGNAL_COLUMNS)
        self.con.register("incoming", frame)
        try:
            self.con.execute("BEGIN")
            self.con.execute("DELETE FROM signals s USING incoming i "
                             "WHERE s.as_of = i.as_of AND s.symbol = i.symbol")
            self.con.execute(f"INSERT INTO signals ({cols}) SELECT {cols} FROM incoming")
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        finally:
            self.con.unregister("incoming")
        return len(frame)

    def signals(self, symbol: str | None = None) -> pd.DataFrame:
        sql = f"SELECT {', '.join(SIGNAL_COLUMNS)}, recorded_at FROM signals"
        params: list = []
        if symbol:
            sql += " WHERE symbol = ?"
            params.append(symbol.upper())
        df = self.con.execute(sql + " ORDER BY as_of, symbol", params).df()
        df["as_of"] = pd.to_datetime(df["as_of"])
        return df

    def inventory(self) -> pd.DataFrame:
        """Mỗi mã có bao nhiêu phiên, từ ngày nào đến ngày nào."""
        return self.con.execute(
            "SELECT symbol, COUNT(*) AS so_phien, MIN(date) AS tu_ngay, MAX(date) AS den_ngay, "
            "ANY_VALUE(source) AS nguon FROM prices GROUP BY symbol ORDER BY symbol"
        ).df()
