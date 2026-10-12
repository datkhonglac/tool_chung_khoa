"""Nguồn số liệu vĩ mô miễn phí (lớp vĩ mô). Đã dò thật bằng scripts/probe_macro.py trên Actions.

- Yahoo Finance (chart API, 10 năm ngày): USD/VND, USD/JPY, lợi suất Mỹ 10 năm (^TNX), dầu Brent,
  chỉ số USD, quặng sắt, thép HRC, bông, đường (hợp đồng kỳ hạn gần nhất).
- World Bank Pink Sheet (CMO-Historical-Data-Monthly.xlsx): giá hàng hoá tháng (urê, DAP, đá phốt
  phát, than, cao su). Tên cột so khớp bỏ dấu * chú thích và khoảng trắng thừa ("Urea " = "Urea").
- FRED (fredgraph.csv, không cần khoá): lãi suất Fed, bán lẻ quần áo Mỹ. Phải gửi User-Agent kiểu
  curl — với User-Agent trình duyệt, FRED treo kết nối từ máy chủ GitHub (đã dò 12/10/2026).
Số vĩ mô Việt Nam (CPI, tín dụng, PMI, TPCP 10 năm...) không có nguồn miễn phí tự động: nhập tay ở
config/macro_manual.toml (vnsa.macro.catalog.manual_frame).

Mọi hàm trả bảng chuẩn: series, date, value, source.
"""

from __future__ import annotations

import io
import json
import re
import time
import urllib.request
from collections.abc import Callable
from datetime import date

import pandas as pd

from vnsa.data.providers.base import DataProvider
from vnsa.macro.catalog import Series

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
FRED_HEADERS = {"User-Agent": "curl/8.5.0", "Accept": "*/*"}
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{code}?range={years}y&interval=1d"
PINK_PAGE = "https://www.worldbank.org/en/research/commodity-markets"
COLUMNS = ["series", "date", "value", "source"]


def _http_get(url: str, timeout: float = 30) -> bytes:
    req = urllib.request.Request(url, headers=FRED_HEADERS if "stlouisfed" in url else HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parse_fred_csv(raw: bytes) -> pd.Series:
    """CSV của FRED (2 cột: ngày, giá trị; "." = thiếu) -> Series theo ngày."""
    df = pd.read_csv(io.BytesIO(raw))
    if df.shape[1] < 2:
        raise ValueError("FRED trả về không đúng định dạng")
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df.iloc[:, 0], errors="coerce"))
    return s[s.index.notna()].dropna().sort_index()


def parse_yahoo(raw: bytes) -> pd.Series:
    """JSON chart của Yahoo -> giá đóng cửa theo ngày (giờ Việt Nam, bỏ giờ)."""
    res = json.loads(raw)["chart"]["result"][0]
    close = res["indicators"]["quote"][0]["close"]
    idx = (pd.to_datetime(res["timestamp"], unit="s", utc=True)
           .tz_convert("Asia/Ho_Chi_Minh").tz_localize(None).normalize())
    s = pd.Series(close, index=idx, dtype=float).dropna()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    # Yahoo đôi khi có phiên giá lỗi (lệch hẳn khỏi các phiên quanh nó): bỏ điểm lệch > 20% so với
    # trung vị 5 phiên xung quanh.
    med = s.rolling(5, center=True, min_periods=3).median()
    return s[(s / med - 1).abs().fillna(0) <= 0.2]


def pink_sheet_url(page_html: str) -> str:
    """Link file CMO-Historical-Data-Monthly.xlsx trên trang Commodity Markets của World Bank."""
    urls = re.findall(r'https?://[^"\'\s>]+CMO-Historical-Data-Monthly\.xlsx', page_html)
    if not urls:
        raise ValueError("không thấy link Pink Sheet trên trang World Bank (trang đổi cấu trúc?)")
    return urls[0]


def norm_name(name: str) -> str:
    """Tên cột Pink Sheet để so khớp: bỏ dấu * chú thích, khoảng trắng thừa, chữ hoa."""
    return " ".join(str(name).replace("*", " ").split()).lower()


def parse_pink_sheet(raw: bytes) -> pd.DataFrame:
    """Sheet "Monthly Prices" -> bảng theo tháng (ngày đầu tháng), mỗi cột một mặt hàng (tên cột
    đúng như hàng tiêu đề của World Bank, ví dụ "Urea", "Iron ore, cfr spot")."""
    xl = pd.ExcelFile(io.BytesIO(raw))
    sheet = next((s for s in xl.sheet_names if "monthly" in s.lower() and "price" in s.lower()),
                 None)
    if sheet is None:
        raise ValueError(f"Pink Sheet không có sheet giá tháng: {xl.sheet_names}")
    df = xl.parse(sheet, header=None)
    hdr = next((i for i in range(min(15, len(df)))
                if df.iloc[i].astype(str).str.contains("Crude oil", case=False).any()), None)
    if hdr is None:
        raise ValueError("Pink Sheet: không thấy hàng tiêu đề (Crude oil...)")
    names = [str(x).strip() for x in df.iloc[hdr].tolist()]
    body = df.iloc[hdr + 1:].copy()
    body.columns = names
    period = body.iloc[:, 0].astype(str).str.strip()
    m = period.str.extract(r"^(\d{4})M(\d{2})$")
    ok = m[0].notna()
    out = body[ok].iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    out.index = pd.to_datetime(m[ok][0] + "-" + m[ok][1] + "-01")
    return out.loc[:, [c for c in out.columns if c and c != "nan"]]


class MacroProvider(DataProvider):
    """Tải các chuỗi vĩ mô tự động trong danh mục (nguồn khác manual)."""

    name = "macro"

    def __init__(self, cat: dict[str, Series], get: Callable[[str], bytes] = _http_get,
                 pause: float = 0.3, log: Callable[[str], None] | None = print):
        self.cat = cat
        self.get = get          # thay được trong test, không cần mạng
        self.pause = pause
        self.log = log
        self._pink: pd.DataFrame | None = None

    def symbols(self) -> list[str]:
        return [sid for sid, s in self.cat.items() if s.auto]

    def ohlcv(self, symbol: str, start: date | None = None, end: date | None = None):
        raise NotImplementedError("nguồn vĩ mô không có giá OHLCV — dùng series()")

    def _pink_sheet(self) -> pd.DataFrame:
        if self._pink is None:
            url = pink_sheet_url(self.get(PINK_PAGE).decode("utf-8", "ignore"))
            self._pink = parse_pink_sheet(self.get(url))
        return self._pink

    def _raw(self, s: Series, years: int) -> pd.Series:
        if s.source == "fred":
            return parse_fred_csv(self.get(FRED_URL.format(code=s.code)))
        if s.source == "yahoo":
            return parse_yahoo(self.get(YAHOO_URL.format(code=s.code, years=years)))
        if s.source == "worldbank":
            pink = self._pink_sheet()
            cols = {norm_name(c): c for c in pink.columns}
            if norm_name(s.code) not in cols:
                raise ValueError(f"Pink Sheet không có cột '{s.code}'")
            return pink[cols[norm_name(s.code)]].dropna()
        raise ValueError(f"chuỗi {s.id} ({s.source}) không tải từ nguồn ngoài")

    def series(self, sid: str, years: int = 10) -> pd.DataFrame:
        """Một chuỗi trong danh mục, `years` năm gần nhất -> bảng series, date, value, source."""
        s = self.cat[sid]
        raw = self._raw(s, years)
        start = pd.Timestamp.today().normalize() - pd.DateOffset(years=years)
        raw = raw[raw.index >= start]
        if s.freq in ("M", "Q"):
            raw.index = raw.index.to_period(s.freq).to_timestamp()
            raw = raw[~raw.index.duplicated(keep="last")]
        if self.pause:
            time.sleep(self.pause)
        return pd.DataFrame({"series": sid, "date": raw.index, "value": raw.to_numpy(),
                             "source": s.source}, columns=COLUMNS)


def derive(spec: Series, data: pd.DataFrame) -> pd.DataFrame:
    """Chuỗi tính từ hai chuỗi trong kho: code "a/b" -> a ÷ b trên các ngày chung."""
    a, b = (x.strip() for x in spec.code.split("/"))
    wide = data[data["series"].isin([a, b])].pivot_table(index="date", columns="series",
                                                         values="value")
    if a not in wide or b not in wide:
        return pd.DataFrame(columns=COLUMNS)
    ratio = (wide[a] / wide[b]).replace([float("inf")], float("nan")).dropna()
    return pd.DataFrame({"series": spec.id, "date": ratio.index, "value": ratio.to_numpy(),
                         "source": "derived"}, columns=COLUMNS)
