"""Dò các nguồn số liệu vĩ mô miễn phí trước khi viết code đọc (lớp vĩ mô, bước 1).

    python scripts/probe_macro.py

In ra nguồn nào gọi được, có bao nhiêu dòng, ngày đầu/cuối và vài giá trị; không ghi gì vào kho.
Chạy trên GitHub Actions (workflow "Vĩ mô") vì nhiều mạng chặn các địa chỉ này.
"""

from __future__ import annotations

import io
import json
import os
import re
import signal
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

os.environ.setdefault("VNSTOCK_TELEMETRY", "off")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
TODAY = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).date()
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}

# Mã FRED cần thử (đã biết + đoán): lãi suất Mỹ, USD, dầu, yên; giá hàng hoá IMF hằng tháng;
# chỉ số giá sản xuất (PPI) Mỹ của phân bón, thép làm chuỗi đại diện.
FRED = [
    "DGS10", "FEDFUNDS", "DTWEXBGS", "DCOILBRENTEU", "DEXJPUS", "RSCCAS",
    "PIORECRUSDM", "PCOALAUUSDM", "PCOTTINDUSDM", "PSUGAISAUSDM", "PRUBBUSDM", "PNGASJPUSDM",
    "PRICENPQUSDM", "PCOPPUSDM", "PALUMUSDM", "POILBREUSDM", "PSHRIUSDM",
    "PUREAUSDM", "PDAPUSDM", "PPHOSUSDM", "PFERTINDEXM",
    "PCU325311325311", "PCU325312325312", "WPU0652", "WPU1017", "PCU331110331110",
    "CCUSMA02VNM618N", "VNMCPIALLMINMEI", "FPCPITOTLZGVNM",
]


def get(url: str, timeout: float = 30, headers: dict | None = None) -> bytes:
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def section(title: str) -> None:
    print(f"\n===== {title}", flush=True)


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout("quá 45 giây")


def try_(label: str, fn, limit: int = 45) -> None:
    """Gọi thử một nguồn, tối đa `limit` giây (thư viện không đặt timeout cũng không treo cả job)."""
    t0 = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(limit)
    try:
        out = fn()
    except BaseException as e:  # noqa: BLE001
        print(f"  {label}: LỖI {type(e).__name__}: {str(e)[:200]}", flush=True)
        return
    finally:
        signal.alarm(0)
    print(f"  {label}: {out}  ({time.time() - t0:.1f}s)", flush=True)


def fred(code: str) -> str:
    raw = get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}", timeout=20)
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", "value"]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()
    return (f"{len(df)} dòng {df['date'].iloc[0]} → {df['date'].iloc[-1]}, "
            f"cuối = {df['value'].iloc[-1]:g}")


def pink_sheet() -> None:
    section("World Bank Pink Sheet (giá hàng hoá hằng tháng)")
    urls: list[str] = []
    try:
        page = get("https://www.worldbank.org/en/research/commodity-markets").decode("utf-8", "ignore")
        urls = sorted(set(re.findall(r'https?://[^"\']+CMO-Historical-Data-Monthly\.xlsx', page)))
        print(f"  link tìm thấy trên trang: {urls}")
    except Exception as e:  # noqa: BLE001
        print(f"  trang commodity-markets: LỖI {type(e).__name__}: {e}")
    for url in urls[:2]:
        try:
            raw = get(url, timeout=60)
        except Exception as e:  # noqa: BLE001
            print(f"  tải {url}: LỖI {type(e).__name__}: {e}")
            continue
        xl = pd.ExcelFile(io.BytesIO(raw))
        print(f"  {len(raw) / 1e6:.1f} MB, sheet: {xl.sheet_names}")
        sheet = next((s for s in xl.sheet_names if "monthly" in s.lower() and "price" in s.lower()),
                     xl.sheet_names[0])
        df = xl.parse(sheet, header=None)
        for i in range(min(10, len(df))):
            print(f"  dòng {i}: {[str(x)[:22] for x in df.iloc[i].tolist()[:12]]}")
        hdr = next((i for i in range(10) if df.iloc[i].astype(str).str.contains("Crude").any()), 4)
        names = [str(x) for x in df.iloc[hdr].tolist()]
        keys = ("Urea", "DAP", "Phosphate", "Iron", "Coal", "Cotton", "Sugar", "Rubber", "Brent",
                "TSP", "Potassium", "Natural gas")
        print(f"  hàng tiêu đề {hdr}; cột liên quan: "
              f"{[(j, n) for j, n in enumerate(names) if any(k.lower() in n.lower() for k in keys)]}")
        print(f"  3 dòng cuối: {df.tail(3).iloc[:, :6].to_string()}")
        break


def yahoo(sym: str) -> str:
    raw = get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=10y&interval=1d")
    res = json.loads(raw)["chart"]["result"][0]
    ts, close = res["timestamp"], res["indicators"]["quote"][0]["close"]
    s = pd.Series(close, index=pd.to_datetime(ts, unit="s")).dropna()
    return f"{len(s)} dòng {s.index[0]:%Y-%m-%d} → {s.index[-1]:%Y-%m-%d}, cuối = {s.iloc[-1]:g}"


def vnstock_probe() -> None:
    section("vnstock: tỷ giá, ngành ICB, tin tức")
    try:
        import vnstock
    except ImportError:
        print("  chưa cài vnstock")
        return
    print(f"  vnstock {getattr(vnstock, '__version__', '?')}")
    end = TODAY.isoformat()

    def fx():
        from vnstock import Vnstock
        df = Vnstock().fx(symbol="USDVND", source="MSN").quote.history(
            start="2016-01-01", end=end, interval="1D")
        return f"{df.shape} cột {list(df.columns)}\n{df.tail(3).to_string()}"
    try_("Vnstock().fx USDVND (MSN)", fx)

    def fx_jpy():
        from vnstock import Vnstock
        df = Vnstock().fx(symbol="JPYVND", source="MSN").quote.history(
            start="2024-01-01", end=end, interval="1D")
        return f"{df.shape}\n{df.tail(2).to_string()}"
    try_("Vnstock().fx JPYVND (MSN)", fx_jpy)

    def vcb():
        from vnstock.explorer.misc import vcb_exchange_rate
        d = (TODAY - timedelta(days=3)).isoformat()
        df = vcb_exchange_rate(date=d)
        return f"{df.shape} cột {list(df.columns)}\n{df.head(3).to_string()}"
    try_("vcb_exchange_rate", vcb)

    def gold():
        from vnstock.explorer.misc import sjc_gold_price
        df = sjc_gold_price()
        return f"{df.shape}\n{df.head(3).to_string()}"
    try_("sjc_gold_price", gold)

    def icb():
        from vnstock import Listing
        df = Listing(source="VCI").symbols_by_industries()
        want = ["HPG", "FPT", "DGC", "VCB", "ACV", "GMD", "TNG", "MWG", "VNM", "SSI", "PVS"]
        cols = [c for c in df.columns if "icb" in c.lower() or c in ("symbol", "com_type_code")]
        return (f"{df.shape} cột {list(df.columns)}\n"
                f"{df[df['symbol'].isin(want)][cols].to_string()}")
    try_("Listing.symbols_by_industries (ICB)", icb)

    def news():
        from vnstock import Company
        df = Company(symbol="FPT", source="VCI").news()
        return f"{df.shape} cột {list(df.columns)}\n{df.head(3).to_string()[:1500]}"
    try_("Company.news (VCI)", news)

    def news_tcbs():
        from vnstock import Company
        df = Company(symbol="FPT", source="TCBS").news()
        return f"{df.shape} cột {list(df.columns)}\n{df.head(3).to_string()[:1500]}"
    try_("Company.news (TCBS)", news_tcbs)


YAHOO = ["VND=X", "JPYVND=X", "JPY=X", "^TNX", "^IRX", "BZ=F", "DX-Y.NYB", "CT=F", "SB=F",
         "HG=F", "TIO=F", "HRC=F", "RB=F"]


def fred_variants() -> None:
    """FRED không trả lời máy chủ GitHub qua fredgraph.csv (lần dò 1): thử địa chỉ khác."""
    for label, url, hdr in [
        ("fredgraph curl UA", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10",
         {"User-Agent": "curl/8.5.0", "Accept": "*/*"}),
        ("data/DGS10.txt", "https://fred.stlouisfed.org/data/DGS10.txt", {}),
        ("api (không khoá)",
         "https://api.stlouisfed.org/fred/series/observations?series_id=DGS10&file_type=json", {}),
    ]:
        try_(label, lambda u=url, h=hdr: f"{len(get(u, timeout=15, headers=h))} byte", limit=20)


def main() -> int:
    section("Yahoo Finance (tỷ giá, lợi suất, hàng hoá kỳ hạn)")
    for sym in YAHOO:
        try_(sym, lambda s=sym: yahoo(s))
    section("FRED (thử địa chỉ khác)")
    fred_variants()
    if os.environ.get("PROBE_FRED_ALL"):
        for code in FRED:
            try_(code, lambda c=code: fred(c), limit=25)
    try_("Pink Sheet", lambda: pink_sheet() or "xong", limit=150)
    vnstock_probe()
    section("Trang số liệu Việt Nam (chỉ xem có truy cập được không)")
    for url in ("https://www.gso.gov.vn/", "https://sbv.gov.vn/",
                "https://www.worldgovernmentbonds.com/bond-historical-data/vietnam/10-years/",
                "https://cafef.vn/thi-truong-chung-khoan.rss"):
        try_(url, lambda u=url: f"{len(get(u, timeout=20))} byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
