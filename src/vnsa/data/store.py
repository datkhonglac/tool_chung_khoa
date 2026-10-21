"""Kho dữ liệu cục bộ bằng DuckDB (một file duy nhất: data/warehouse.duckdb).

Mọi module phân tích đọc từ đây, không gọi thẳng nguồn bên ngoài.

Nhiều giao diện cùng mở kho (dòng lệnh, dashboard, MCP server cho Claude Desktop): DuckDB chỉ cho
MỘT tiến trình ghi tại một thời điểm và chặn cả tiến trình khác mở chỉ-đọc khi đang có tiến trình
ghi. Vì vậy:
- mở kết nối có thử lại khi file đang bận (`connect`, tối đa khoảng `LOCK_WAIT_SECONDS` giây);
- mọi chỗ mở kho đều mở ngắn (`with Store(...)`), và phần tải dữ liệu qua mạng dùng `LazyStore`:
  mỗi lời gọi mở kết nối rồi đóng ngay, nên không giữ khoá kho trong lúc chờ CafeF/vnstock;
- nâng cấp cấu trúc bảng (ALTER) chỉ chạy khi phiên bản cấu trúc trong bảng `meta` khác bản code.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from vnsa.data.providers.base import PRICE_COLUMNS
from vnsa.fa.ratios import STORED_FIELDS as FIELDS

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
    action      VARCHAR,            -- MUA | KHẢ QUAN | NẮM GIỮ | KÉM KHẢ QUAN | BÁN (bản cũ: THEO DÕI, TRÁNH)
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

# BCTC quý (bản miễn phí VCI: 4 quý gần nhất) — dùng tính số liệu 12 tháng gần nhất (TTM)
SCHEMA += f"""
CREATE TABLE IF NOT EXISTS fundamentals_q (
    symbol  VARCHAR NOT NULL,
    year    INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
{"".join(f"    {f} DOUBLE,{chr(10)}" for f in FIELDS)}    source  VARCHAR,
    loaded_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (symbol, year, quarter)
);

-- Cổ tức (sự kiện quyền): tiền mặt (đồng/cp) hoặc cổ phiếu/thưởng (tỷ lệ)
CREATE TABLE IF NOT EXISTS dividends (
    symbol         VARCHAR NOT NULL,
    exright_date   DATE    NOT NULL,
    cash_per_share DOUBLE,
    stock_ratio    DOUBLE,
    title          VARCHAR
);
"""

# Giao dịch khối ngoại và tự doanh theo phiên (CafeF), đồng
SCHEMA += """
CREATE TABLE IF NOT EXISTS flows (
    symbol          VARCHAR NOT NULL,
    date            DATE    NOT NULL,
    foreign_buy     DOUBLE,
    foreign_sell    DOUBLE,
    foreign_net     DOUBLE,
    foreign_net_vol DOUBLE,
    foreign_room    DOUBLE,
    foreign_pct     DOUBLE,
    prop_buy        DOUBLE,
    prop_sell       DOUBLE,
    prop_net        DOUBLE,
    loaded_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (symbol, date)
);
"""
# Chuỗi số liệu vĩ mô (lớp vĩ mô): mỗi dòng một kỳ của một chuỗi. Chuỗi tháng/quý ghi ngày đầu kỳ.
# Danh mục chuỗi (tên, đơn vị, nguồn) ở config/macro_map.toml.
SCHEMA += """
CREATE TABLE IF NOT EXISTS macro_series (
    series    VARCHAR NOT NULL,
    date      DATE    NOT NULL,
    value     DOUBLE,
    source    VARCHAR,
    loaded_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (series, date)
);
"""
# Tiêu đề tin theo mã (vnstock/VCI) — chỉ để tham khảo, không vào tính toán
SCHEMA += """
CREATE TABLE IF NOT EXISTS news (
    symbol    VARCHAR NOT NULL,
    date      TIMESTAMP NOT NULL,
    title     VARCHAR NOT NULL,
    url       VARCHAR,
    source    VARCHAR,
    loaded_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (symbol, date, title)
);
"""
# Độ sâu lịch sử giá đã yêu cầu ở lần tải toàn bộ gần nhất: phân biệt "kho mới tải ngắn" (sàng lọc
# chỉ tải 1 năm) với "mã mới niêm yết, nguồn không có hơn" (không tải lại vô ích)
SCHEMA += """
CREATE TABLE IF NOT EXISTS price_depth (
    symbol    VARCHAR PRIMARY KEY,
    full_from DATE,
    loaded_at TIMESTAMP DEFAULT current_timestamp
);
-- Kết quả sàng lọc toàn thị trường (screen/): mỗi lượt (as_of) một dòng mỗi mã trong vũ trụ
CREATE TABLE IF NOT EXISTS screen (
    as_of  DATE    NOT NULL,     -- phiên giá của lượt sàng lọc
    symbol VARCHAR NOT NULL,
    PRIMARY KEY (as_of, symbol)
);
CREATE TABLE IF NOT EXISTS screen_industry (
    as_of      DATE    NOT NULL,
    group_name VARCHAR NOT NULL,
    PRIMARY KEY (as_of, group_name)
);
"""
FLOW_COLUMNS = ["foreign_buy", "foreign_sell", "foreign_net", "foreign_net_vol", "foreign_room",
                "foreign_pct", "prop_buy", "prop_sell", "prop_net"]

COMPANY_COLUMNS = ["symbol", "name", "exchange", "industry", "model", "industry3", "industry4"]
# industry = ngành ICB cấp 2 (so sánh cùng ngành); industry3/4 = cấp 3/4 (ánh xạ vĩ mô)
# Hồ sơ doanh nghiệp (Company.overview), cập nhật theo từng mã
PROFILE_COLUMNS = ["shares_out", "foreign_pct", "foreign_max_pct", "state_pct", "free_float_pct",
                   "avg_value_1m", "analyst_target", "analyst_rating"]
PROFILE_TYPES = {c: "VARCHAR" if c == "analyst_rating" else "DOUBLE" for c in PROFILE_COLUMNS}
SIGNAL_COLUMNS = ["as_of", "symbol", "action", "score", "fa_score", "ta_score", "price",
                  "fair_value", "zone_status", "buy_low", "buy_high", "stop", "target", "params",
                  "macro_industry", "macro_market"]
# Cột bảng screen (ngoài as_of, symbol). stage = tầng cao nhất mã đã qua (0 = chỉ trong vũ trụ);
# reason = lý do bị loại ở tầng kế tiếp. Giá thật (đồng), GTGD đồng; tỷ lệ là số thập phân.
SCREEN_TYPES = {
    "exchange": "VARCHAR", "industry": "VARCHAR", "stage": "INTEGER", "reason": "VARCHAR",
    "last_date": "DATE", "price": "DOUBLE", "value_20d": "DOUBLE", "sessions": "INTEGER",
    "ret_1m": "DOUBLE", "ret_3m": "DOUBLE", "ret_6m": "DOUBLE", "rs_3m": "DOUBLE",
    "rs_6m": "DOUBLE", "above_ma50": "DOUBLE", "above_ma200": "DOUBLE",
    # tầng 2 (assess + thẻ báo cáo), tầng 3 (nhóm ngành)
    "group_name": "VARCHAR", "model": "VARCHAR", "action": "VARCHAR", "expected_return": "DOUBLE",
    "score": "DOUBLE", "fair_value": "DOUBLE", "upside": "DOUBLE", "dividend_yield": "DOUBLE",
    "fa_score": "DOUBLE", "ta_score": "DOUBLE", "quant": "DOUBLE", "confidence": "VARCHAR",
    "eps_growth": "DOUBLE", "growth_basis": "VARCHAR", "roe": "DOUBLE", "pe": "DOUBLE",
    "pb": "DOUBLE", "z_em": "DOUBLE", "macro_industry": "DOUBLE", "zone_status": "VARCHAR",
    "buy_low": "DOUBLE", "buy_high": "DOUBLE", "stop": "DOUBLE", "red_flags": "VARCHAR",
    "fin_basis": "VARCHAR", "group_rank": "DOUBLE", "group_score": "DOUBLE",
    "pick_rank": "DOUBLE",                     # tầng 4: thứ tự trong nhóm của mã được chọn
}
SCREEN_COLUMNS = ["as_of", "symbol", *SCREEN_TYPES]
# Điểm nhóm ngành mỗi lượt sàng lọc (screen/industry.py): số thô + điểm thành phần (pt_*) 0–100
INDUSTRY_TYPES = {
    "rank": "DOUBLE", "score": "DOUBLE", "members": "INTEGER", "rs": "DOUBLE", "breadth": "DOUBLE",
    "growth": "DOUBLE", "growing": "DOUBLE", "expected": "DOUBLE", "fa": "DOUBLE",
    "macro": "DOUBLE", "macro_coverage": "DOUBLE", "value_20d": "DOUBLE",
    "pt_momentum": "DOUBLE", "pt_breadth": "DOUBLE", "pt_earnings": "DOUBLE",
    "pt_valuation": "DOUBLE", "pt_quality": "DOUBLE", "pt_macro": "DOUBLE",
    "weight_used": "DOUBLE", "note": "VARCHAR",
}
INDUSTRY_COLUMNS = ["as_of", "group_name", *INDUSTRY_TYPES]


# Phiên bản cấu trúc: đổi khi SCHEMA hoặc danh sách cột thay đổi -> kho cũ được nâng cấp một lần.
# Thêm một câu ALTER viết tay trong Store._migrate thì tăng SCHEMA_REVISION.
SCHEMA_REVISION = 1
SCHEMA_VERSION = hashlib.sha1(repr((SCHEMA_REVISION, SCHEMA, list(FIELDS), PROFILE_TYPES,
                                    SCREEN_TYPES, INDUSTRY_TYPES)).encode()).hexdigest()[:16]
LOCK_WAIT_SECONDS = 60.0     # chờ tối đa khi tiến trình khác đang ghi kho


class WarehouseBusy(RuntimeError):
    """Kho đang bị tiến trình khác giữ quá lâu."""


def connect(db_path: str | Path, wait: float | None = None) -> duckdb.DuckDBPyConnection:
    """Mở kho; file đang bị tiến trình khác khoá thì chờ rồi thử lại (giãn dần 0,1 → 2 giây)."""
    wait = LOCK_WAIT_SECONDS if wait is None else wait
    deadline, delay = time.monotonic() + wait, 0.1
    while True:
        try:
            return duckdb.connect(str(db_path))
        except duckdb.IOException as e:
            if "lock" not in str(e).lower():
                raise
            if time.monotonic() + delay > deadline:
                raise WarehouseBusy(
                    f"Kho {db_path} đang bị tiến trình khác dùng quá {wait:.0f} giây (một lệnh "
                    "screen/fetch đang chạy?). Chờ lệnh đó xong rồi chạy lại.") from e
            time.sleep(delay)
            delay = min(delay * 1.5, 2.0)


class Store:
    def __init__(self, db_path: str | Path, wait: float | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = connect(self.db_path, wait)
        try:
            self._migrate()
        except Exception:
            self.con.close()
            raise

    @classmethod
    def lazy(cls, db_path: str | Path) -> "LazyStore":
        """Kho mở theo từng lời gọi (xem LazyStore) — dùng khi xen kẽ gọi mạng lâu."""
        return LazyStore(db_path)

    def _migrate(self) -> None:
        """Tạo bảng, thêm cột mới cho kho cũ. Bỏ qua nếu kho đã ở đúng phiên bản cấu trúc."""
        try:
            row = self.con.execute("SELECT value FROM meta WHERE key = 'schema'").fetchone()
            if row is not None and row[0] == SCHEMA_VERSION:
                return
        except duckdb.CatalogException:
            pass
        self.con.execute(SCHEMA)
        for f in FIELDS:  # kho tạo từ bản cũ: thêm cột khoản mục mới nếu còn thiếu
            for table in ("fundamentals", "fundamentals_q"):
                self.con.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {f} DOUBLE")
        # giá đóng cửa chưa điều chỉnh (NULL nếu nguồn không có)
        self.con.execute("ALTER TABLE prices ADD COLUMN IF NOT EXISTS close_raw DOUBLE")
        # điểm vĩ mô lúc khuyến nghị (lớp vĩ mô, để review-signals chấm xem vĩ mô có ích không)
        for c in ("macro_industry", "macro_market"):
            self.con.execute(f"ALTER TABLE signals ADD COLUMN IF NOT EXISTS {c} DOUBLE")
        for c, t in [("industry3", "VARCHAR"), ("industry4", "VARCHAR"), *PROFILE_TYPES.items(),
                     ("profile_at", "TIMESTAMP")]:
            self.con.execute(f"ALTER TABLE companies ADD COLUMN IF NOT EXISTS {c} {t}")
        for c, t in SCREEN_TYPES.items():
            self.con.execute(f"ALTER TABLE screen ADD COLUMN IF NOT EXISTS {c} {t}")
        for c, t in INDUSTRY_TYPES.items():
            self.con.execute(f"ALTER TABLE screen_industry ADD COLUMN IF NOT EXISTS {c} {t}")
        self.con.execute("CREATE TABLE IF NOT EXISTS meta (key VARCHAR PRIMARY KEY, value VARCHAR)")
        self.con.execute("INSERT OR REPLACE INTO meta VALUES ('schema', ?)", [SCHEMA_VERSION])

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

    def first_price_date(self, symbol: str) -> pd.Timestamp | None:
        d = self.con.execute("SELECT MIN(date) FROM prices WHERE symbol = ?",
                             [symbol.upper()]).fetchone()[0]
        return None if d is None else pd.Timestamp(d)

    def set_price_depth(self, symbol: str, full_from) -> None:
        """Ghi ngày bắt đầu đã yêu cầu ở lần tải toàn bộ giá gần nhất của mã."""
        self.con.execute("INSERT OR REPLACE INTO price_depth (symbol, full_from, loaded_at) "
                         "VALUES (?, ?, current_timestamp)",
                         [symbol.upper(), pd.Timestamp(full_from).date()])

    def price_depth(self, symbol: str) -> pd.Timestamp | None:
        row = self.con.execute("SELECT full_from FROM price_depth WHERE symbol = ?",
                               [symbol.upper()]).fetchone()
        return None if row is None or row[0] is None else pd.Timestamp(row[0])

    def recent_prices(self, start, symbols: list[str] | None = None) -> pd.DataFrame:
        """Giá mọi mã (hoặc `symbols`) từ ngày `start` — một truy vấn cho cả thị trường."""
        sql = ("SELECT symbol, date, close, close_raw, volume FROM prices WHERE date >= ?")
        params: list = [pd.Timestamp(start).date()]
        if symbols is not None:
            sql += f" AND symbol IN ({', '.join('?' * len(symbols))})" if symbols else " AND FALSE"
            params += [s.upper() for s in symbols]
        df = self.con.execute(sql + " ORDER BY symbol, date", params).df()
        df["date"] = pd.to_datetime(df["date"])
        return df

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
        """Ghi đè BCTC năm theo (mã, năm) có trong df; năm cũ hơn không có trong df được GIỮ (nguồn
        miễn phí chỉ trả 4 năm gần nhất — kho tự tích luỹ dài dần). Cột: symbol, year + FIELDS."""
        if df.empty:
            return 0
        frame = df.reindex(columns=["symbol", "year", *FIELDS]).copy()  # khoản thiếu -> NULL
        frame["source"] = source
        cols = ", ".join(["symbol", "year", *FIELDS, "source"])
        self.con.register("incoming", frame)
        try:
            self.con.execute("BEGIN")
            self.con.execute("DELETE FROM fundamentals f USING incoming i WHERE f.symbol = i.symbol "
                             "AND f.year = i.year")
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

    def mark_stale(self, table: str, keys: pd.DataFrame) -> None:
        """Đặt loaded_at rất cũ cho các dòng (BCTC nạp từ lịch sử) để autofetch vẫn tải số mới."""
        if keys.empty or table not in ("fundamentals", "fundamentals_q"):
            return
        cond = " AND ".join(f"t.{k} = k.{k}" for k in keys.columns)
        self.con.register("stale_keys", keys)
        try:
            self.con.execute(f"UPDATE {table} t SET loaded_at = TIMESTAMP '2000-01-01' "
                             f"FROM stale_keys k WHERE {cond}")
        finally:
            self.con.unregister("stale_keys")

    def all_fundamentals(self) -> pd.DataFrame:
        """Mọi BCTC năm trong kho (kèm nguồn) — để lưu lịch sử (history.py)."""
        cols = ", ".join(["symbol", "year", *FIELDS, "source"])
        return self.con.execute(f"SELECT {cols} FROM fundamentals ORDER BY symbol, year").df()

    def all_quarterly(self) -> pd.DataFrame:
        cols = ", ".join(["symbol", "year", "quarter", *FIELDS, "source"])
        return self.con.execute(f"SELECT {cols} FROM fundamentals_q ORDER BY symbol, year, "
                                "quarter").df()

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
        updates = ", ".join(f"{c} = excluded.{c}" for c in [*COMPANY_COLUMNS[1:], "source"])
        try:
            self.con.execute("BEGIN")
            # cập nhật đè sàn/ngành, giữ nguyên hồ sơ (số cp lưu hành...) đã ghi riêng
            self.con.execute(f"INSERT INTO companies ({cols}) SELECT {cols} FROM incoming "
                             f"ON CONFLICT (symbol) DO UPDATE SET {updates}, "
                             "loaded_at = now()")
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        finally:
            self.con.unregister("incoming")
        return len(frame)

    def write_profile(self, symbol: str, profile: dict, stamp: bool = True) -> None:
        """Ghi hồ sơ doanh nghiệp vào dòng companies của mã (thêm dòng nếu chưa có). `stamp` = False:
        không ghi thời điểm tải (profile_at) để lần sau tải lại (vd. chưa lấy được cổ tức)."""
        sym = symbol.upper()
        if self.company(sym) is None:
            self.con.execute("INSERT INTO companies (symbol) VALUES (?)", [sym])
        cols = [c for c in PROFILE_COLUMNS if c in profile]
        sets = ", ".join(f"{c} = ?" for c in cols)
        when = ", profile_at = current_timestamp" if stamp else ""
        self.con.execute(f"UPDATE companies SET {sets}{when} "
                         "WHERE symbol = ?", [*(None if isinstance(profile[c], float)
                                                 and np.isnan(profile[c]) else profile[c]
                                                 for c in cols), sym])

    def write_quarterly(self, df: pd.DataFrame, source: str) -> int:
        """Ghi đè BCTC quý của các mã có trong df. Cột: symbol, year, quarter + FIELDS (đồng)."""
        if df.empty:
            return 0
        frame = df.reindex(columns=["symbol", "year", "quarter", *FIELDS]).copy()
        frame["source"] = source
        cols = ", ".join(["symbol", "year", "quarter", *FIELDS, "source"])
        self.con.register("incoming", frame)
        try:
            self.con.execute("BEGIN")
            self.con.execute("DELETE FROM fundamentals_q f USING incoming i WHERE f.symbol = "
                             "i.symbol AND f.year = i.year AND f.quarter = i.quarter")
            self.con.execute(f"INSERT INTO fundamentals_q ({cols}) SELECT {cols} FROM incoming")
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        finally:
            self.con.unregister("incoming")
        return len(frame)

    def write_flows(self, df: pd.DataFrame) -> int:
        """Ghi đè giao dịch khối ngoại/tự doanh theo (mã, ngày)."""
        if df.empty:
            return 0
        frame = df.reindex(columns=["symbol", "date", *FLOW_COLUMNS]).copy()
        frame["symbol"] = frame["symbol"].str.upper()
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        cols = ", ".join(["symbol", "date", *FLOW_COLUMNS])
        self.con.register("incoming", frame)
        try:
            self.con.execute("BEGIN")
            self.con.execute("DELETE FROM flows f USING incoming i WHERE f.symbol = i.symbol "
                             "AND f.date = i.date")
            self.con.execute(f"INSERT INTO flows ({cols}) SELECT {cols} FROM incoming")
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        finally:
            self.con.unregister("incoming")
        return len(frame)

    def flows(self, symbol: str) -> pd.DataFrame:
        cols = ", ".join(["symbol", "date", *FLOW_COLUMNS])
        df = self.con.execute(f"SELECT {cols} FROM flows WHERE symbol = ? ORDER BY date",
                              [symbol.upper()]).df()
        df["date"] = pd.to_datetime(df["date"])
        return df

    def quarterly(self, symbol: str) -> pd.DataFrame:
        cols = ", ".join(["symbol", "year", "quarter", *FIELDS])
        return self.con.execute(f"SELECT {cols} FROM fundamentals_q WHERE symbol = ? "
                                "ORDER BY year, quarter", [symbol.upper()]).df()

    def quarterly_loaded_at(self, symbol: str) -> pd.Timestamp | None:
        t = self.con.execute("SELECT MAX(loaded_at) FROM fundamentals_q WHERE symbol = ?",
                             [symbol.upper()]).fetchone()[0]
        return None if t is None else pd.Timestamp(t)

    def write_dividends(self, df: pd.DataFrame, symbol: str) -> int:
        """Ghi đè toàn bộ lịch sử cổ tức của một mã."""
        self.con.execute("DELETE FROM dividends WHERE symbol = ?", [symbol.upper()])
        if df.empty:
            return 0
        frame = df.reindex(columns=["symbol", "exright_date", "cash_per_share", "stock_ratio",
                                    "title"]).copy()
        frame["exright_date"] = pd.to_datetime(frame["exright_date"]).dt.date
        self.con.register("incoming", frame)
        try:
            self.con.execute("INSERT INTO dividends SELECT * FROM incoming")
        finally:
            self.con.unregister("incoming")
        return len(frame)

    def dividends(self, symbol: str) -> pd.DataFrame:
        df = self.con.execute("SELECT exright_date, cash_per_share, stock_ratio, title FROM "
                              "dividends WHERE symbol = ? ORDER BY exright_date",
                              [symbol.upper()]).df()
        df["exright_date"] = pd.to_datetime(df["exright_date"])
        return df

    def company(self, symbol: str) -> dict | None:
        """Sàn, ngành, loại doanh nghiệp và hồ sơ (số cp lưu hành...) của một mã; None nếu chưa có."""
        cols = [*COMPANY_COLUMNS, "loaded_at", *PROFILE_COLUMNS, "profile_at"]
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

    def companies(self, exchanges: list[str] | None = None) -> pd.DataFrame:
        """Danh sách niêm yết (cột COMPANY_COLUMNS), lọc theo sàn nếu có."""
        sql = f"SELECT {', '.join(COMPANY_COLUMNS)} FROM companies"
        params: list = []
        if exchanges:
            sql += f" WHERE exchange IN ({', '.join('?' * len(exchanges))})"
            params = [e.upper() for e in exchanges]
        return self.con.execute(sql + " ORDER BY symbol", params).df()

    def has_icb_detail(self) -> bool:
        """Danh sách niêm yết đã có ngành ICB cấp 4 (kho tạo trước 10/2026 thì chưa: tải lại)."""
        return bool(self.con.execute("SELECT COUNT(*) FROM companies WHERE industry4 IS NOT NULL")
                    .fetchone()[0])

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

    def write_macro(self, df: pd.DataFrame, source: str | None = None) -> int:
        """Ghi chuỗi vĩ mô (cột series, date, value[, source]). Chỉ thay các kỳ trong khoảng ngày của
        mỗi chuỗi có trong df; kỳ cũ hơn được GIỮ (nguồn chỉ trả số gần đây thì kho vẫn tích luỹ)."""
        if df.empty:
            return 0
        frame = df.reindex(columns=["series", "date", "value", "source"]).copy()
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        if source is not None:
            frame["source"] = source
        frame = frame.dropna(subset=["value"]).drop_duplicates(["series", "date"], keep="last")
        self.con.register("incoming", frame)
        try:
            self.con.execute("BEGIN")
            self.con.execute(
                "DELETE FROM macro_series m USING (SELECT series, MIN(date) AS d0, MAX(date) AS d1 "
                "FROM incoming GROUP BY series) r "
                "WHERE m.series = r.series AND m.date BETWEEN r.d0 AND r.d1")
            self.con.execute("INSERT INTO macro_series (series, date, value, source) "
                             "SELECT series, date, value, source FROM incoming")
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        finally:
            self.con.unregister("incoming")
        return len(frame)

    def replace_macro(self, df: pd.DataFrame, series: list[str], source: str) -> int:
        """Thay TOÀN BỘ các chuỗi `series` bằng df (số nhập tay: file là nguồn gốc, xoá kỳ đã bỏ)."""
        if series:
            self.con.execute("DELETE FROM macro_series WHERE series IN (SELECT UNNEST(?))",
                             [list(series)])
        return self.write_macro(df, source) if not df.empty else 0

    def macro(self, series: list[str] | str | None = None) -> pd.DataFrame:
        """Chuỗi vĩ mô dạng dài: series, date, value, source (sắp theo chuỗi, ngày)."""
        sql = "SELECT series, date, value, source FROM macro_series"
        params: list = []
        if series:
            sql += " WHERE series IN (SELECT UNNEST(?))"
            params.append([series] if isinstance(series, str) else list(series))
        df = self.con.execute(sql + " ORDER BY series, date", params).df()
        df["date"] = pd.to_datetime(df["date"])
        return df

    def macro_loaded_at(self) -> dict[str, pd.Timestamp]:
        """Lần tải gần nhất của từng chuỗi vĩ mô."""
        rows = self.con.execute("SELECT series, MAX(loaded_at) FROM macro_series "
                                "GROUP BY series").fetchall()
        return {r[0]: pd.Timestamp(r[1]) for r in rows}

    def write_news(self, df: pd.DataFrame, symbol: str) -> int:
        """Thay toàn bộ tiêu đề tin của một mã (nguồn trả danh sách mới nhất)."""
        sym = symbol.upper()
        frame = df.reindex(columns=["symbol", "date", "title", "url", "source"]).copy()
        frame["symbol"] = sym
        frame = frame.drop_duplicates(["date", "title"])
        self.con.register("incoming", frame)
        try:
            self.con.execute("BEGIN")
            self.con.execute("DELETE FROM news WHERE symbol = ?", [sym])
            self.con.execute("INSERT INTO news (symbol, date, title, url, source) "
                             "SELECT symbol, date, title, url, source FROM incoming")
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        finally:
            self.con.unregister("incoming")
        return len(frame)

    def news(self, symbol: str, limit: int = 10, days: int | None = None) -> pd.DataFrame:
        sql = "SELECT date, title, url, source FROM news WHERE symbol = ?"
        params: list = [symbol.upper()]
        if days is not None:
            sql += " AND date >= ?"
            params.append(pd.Timestamp.now().normalize() - pd.Timedelta(days=days))
        df = self.con.execute(sql + " ORDER BY date DESC LIMIT ?", [*params, limit]).df()
        df["date"] = pd.to_datetime(df["date"])
        return df

    def news_loaded_at(self, symbol: str) -> pd.Timestamp | None:
        t = self.con.execute("SELECT MAX(loaded_at) FROM news WHERE symbol = ?",
                             [symbol.upper()]).fetchone()[0]
        return None if t is None else pd.Timestamp(t)

    def _replace_run(self, table: str, columns: list[str], df: pd.DataFrame,
                     dates: tuple[str, ...]) -> int:
        """Thay toàn bộ dòng của các ngày as_of có trong df (một lượt sàng lọc)."""
        if df.empty:
            return 0
        frame = df.reindex(columns=columns).copy()
        for c in dates:
            frame[c] = pd.to_datetime(frame[c]).dt.date
        self.con.register("incoming", frame)
        try:
            self.con.execute("BEGIN")
            self.con.execute(f"DELETE FROM {table} WHERE as_of IN "
                             "(SELECT DISTINCT as_of FROM incoming)")
            self.con.execute(f"INSERT INTO {table} ({', '.join(columns)}) "
                             f"SELECT {', '.join(columns)} FROM incoming")
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise
        finally:
            self.con.unregister("incoming")
        return len(frame)

    def write_screen(self, df: pd.DataFrame, industry: pd.DataFrame | None = None) -> int:
        """Ghi một lượt sàng lọc (bảng screen, và screen_industry nếu có điểm nhóm ngành): thay
        toàn bộ dòng của các ngày as_of có trong df (mỗi bảng thay theo ngày của chính nó)."""
        n = self._replace_run("screen", SCREEN_COLUMNS, df, ("as_of", "last_date"))
        if industry is not None and not industry.empty:
            self._replace_run("screen_industry", INDUSTRY_COLUMNS, industry, ("as_of",))
        return n

    def write_screen_industry(self, industry: pd.DataFrame) -> int:
        return self._replace_run("screen_industry", INDUSTRY_COLUMNS, industry, ("as_of",))

    def _latest_run(self, table: str, as_of):
        if as_of is None:
            as_of = self.con.execute(f"SELECT MAX(as_of) FROM {table}").fetchone()[0]
        return None if as_of is None else pd.Timestamp(as_of).date()

    def screen(self, as_of=None) -> pd.DataFrame:
        """Một lượt sàng lọc (mặc định lượt mới nhất); rỗng nếu chưa chạy."""
        day = self._latest_run("screen", as_of)
        if day is None:
            return pd.DataFrame(columns=SCREEN_COLUMNS)
        df = self.con.execute(f"SELECT {', '.join(SCREEN_COLUMNS)} FROM screen WHERE as_of = ? "
                              "ORDER BY symbol", [day]).df()
        for c in ("as_of", "last_date"):
            df[c] = pd.to_datetime(df[c])
        return df

    def screen_dates(self) -> list[pd.Timestamp]:
        return [pd.Timestamp(r[0]) for r in
                self.con.execute("SELECT DISTINCT as_of FROM screen ORDER BY as_of").fetchall()]

    def screen_depth(self) -> dict[pd.Timestamp, int]:
        """{ngày as_of: tầng cao nhất đã chạy} — lượt chỉ lọc thanh khoản có tầng 1."""
        return {pd.Timestamp(d): int(s) for d, s in self.con.execute(
            "SELECT as_of, MAX(stage) FROM screen GROUP BY as_of").fetchall()}

    def screen_history(self, min_stage: int = 1) -> pd.DataFrame:
        """Mọi lượt sàng lọc đã lưu, chỉ các mã đạt từ tầng `min_stage` (để chấm lại, lưu lịch sử)."""
        df = self.con.execute(f"SELECT {', '.join(SCREEN_COLUMNS)} FROM screen WHERE stage >= ? "
                              "ORDER BY as_of, symbol", [min_stage]).df()
        for c in ("as_of", "last_date"):
            df[c] = pd.to_datetime(df[c])
        return df

    def screen_industry_history(self) -> pd.DataFrame:
        df = self.con.execute(f"SELECT {', '.join(INDUSTRY_COLUMNS)} FROM screen_industry "
                              "ORDER BY as_of, rank NULLS LAST").df()
        df["as_of"] = pd.to_datetime(df["as_of"])
        return df

    def screen_industry(self, as_of=None) -> pd.DataFrame:
        """Điểm nhóm ngành của một lượt sàng lọc (mặc định mới nhất), xếp theo thứ hạng."""
        day = self._latest_run("screen", as_of)
        if day is None:
            return pd.DataFrame(columns=INDUSTRY_COLUMNS)
        df = self.con.execute(f"SELECT {', '.join(INDUSTRY_COLUMNS)} FROM screen_industry "
                              "WHERE as_of = ? ORDER BY rank NULLS LAST, members DESC",
                              [day]).df()
        df["as_of"] = pd.to_datetime(df["as_of"])
        return df

    def inventory(self) -> pd.DataFrame:
        """Mỗi mã có bao nhiêu phiên, từ ngày nào đến ngày nào."""
        return self.con.execute(
            "SELECT symbol, COUNT(*) AS so_phien, MIN(date) AS tu_ngay, MAX(date) AS den_ngay, "
            "ANY_VALUE(source) AS nguon FROM prices GROUP BY symbol ORDER BY symbol"
        ).df()


class LazyStore:
    """Như Store nhưng MỖI lời gọi phương thức mở kết nối riêng rồi đóng ngay.

    Dùng cho phần tải dữ liệu (autofetch, vĩ mô): giữa hai lần ghi là các lần gọi CafeF/vnstock
    10–30 giây; giữ một kết nối suốt thời gian đó sẽ khoá kho, làm dashboard/MCP/lệnh khác báo lỗi
    "Could not set lock". Mỗi phương thức của Store trả DataFrame/giá trị thường nên đóng kết nối
    sau lời gọi là an toàn."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def __enter__(self) -> "LazyStore":
        return self

    def __exit__(self, *exc) -> None:
        pass

    def close(self) -> None:
        pass

    def __getattr__(self, name: str):
        if name.startswith("_") or not callable(getattr(Store, name, None)):
            raise AttributeError(name)

        def call(*args, **kwargs):
            with Store(self.db_path) as store:
                return getattr(store, name)(*args, **kwargs)

        call.__name__ = name
        return call
