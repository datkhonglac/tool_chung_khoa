"""Dò cấu trúc dữ liệu vnstock (nguồn VCI) trước khi viết code đọc: BCTC quý, chỉ số tài chính,
thông tin doanh nghiệp (số cổ phiếu lưu hành), cổ tức, giao dịch khối ngoại.

    python scripts/probe_vnstock.py FPT VCB

Chỉ in ra tên cột và vài dòng đầu; không ghi gì vào kho. Chạy trên GitHub Actions (input `probe`).
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

os.environ.setdefault("VNSTOCK_TELEMETRY", "off")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 40)


def show(title: str, fn) -> None:
    print(f"\n===== {title}")
    try:
        df = fn()
    except Exception as e:  # noqa: BLE001
        print(f"  LỖI: {type(e).__name__}: {e}")
        return
    if df is None:
        print("  (None)")
        return
    if isinstance(df, pd.DataFrame):
        print(f"  shape={df.shape}")
        cols = list(df.columns)
        print(f"  columns ({len(cols)}): {cols[:60]}")
        if isinstance(df.index, pd.MultiIndex) or df.index.name:
            print(f"  index: {df.index.names}")
        print(df.head(6).to_string()[:3000])
    else:
        print(f"  type={type(df).__name__}: {str(df)[:1500]}")
    time.sleep(2)


def items(symbols: list[str]) -> int:
    """--items: in toàn bộ item_id / tên khoản mục BCTC quý (để ánh xạ chỉ số ngân hàng, CK)."""
    from vnstock import Finance

    for sym in symbols:
        fin = Finance(source="VCI", symbol=sym)
        for name in ("income_statement", "balance_sheet", "cash_flow"):
            try:
                df = getattr(fin, name)(period="quarter")
            except Exception as e:  # noqa: BLE001
                print(f"{sym} {name}: LỖI {e}")
                continue
            last = [c for c in df.columns if str(c)[:4].isdigit()][:1]
            print(f"\n===== {sym} {name} ({len(df)} dòng)")
            for _, r in df.iterrows():
                v = r[last[0]] if last else ""
                print(f"  {r['item_id']:<70} | {str(r['item'])[:60]:<60} | {v}")
            time.sleep(2)
    return 0


def main() -> int:
    from vnstock import Company, Finance, Quote

    if sys.argv[1:2] == ["--items"]:
        return items([s.upper() for s in sys.argv[2:]])
    for sym in [s.upper() for s in sys.argv[1:]] or ["FPT", "VCB"]:
        fin = Finance(source="VCI", symbol=sym)
        for name in ("income_statement", "balance_sheet", "cash_flow"):
            show(f"{sym} {name} quarter", lambda n=name, f=fin: getattr(f, n)(period="quarter"))
        show(f"{sym} ratio quarter", lambda f=fin: f.ratio(period="quarter"))
        show(f"{sym} ratio year", lambda f=fin: f.ratio(period="year"))
        comp = Company(source="VCI", symbol=sym)
        for name in ("overview", "dividends", "events", "shareholders"):
            if hasattr(comp, name):
                show(f"{sym} company.{name}", getattr(comp, name))
        q = Quote(source="VCI", symbol=sym)
        show(f"{sym} quote.history 1D (5 dòng)",
             lambda q=q: q.history(start="2026-09-01", end="2026-10-10", interval="1D"))
        for name in ("intraday", "price_depth"):
            if hasattr(q, name):
                print(f"  (Quote có {name})")
        print(f"\n  Company methods: {[m for m in dir(comp) if not m.startswith('_')]}")
        print(f"  Finance methods: {[m for m in dir(fin) if not m.startswith('_')]}")
    try:
        from vnstock import Trading

        t = Trading(source="VCI")
        print(f"\n  Trading methods: {[m for m in dir(t) if not m.startswith('_')]}")
        show("Trading.price_board FPT VCB", lambda: t.price_board(["FPT", "VCB"]))
    except Exception as e:  # noqa: BLE001
        print(f"Trading: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
