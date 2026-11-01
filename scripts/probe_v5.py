"""Dò cấu trúc vnstock thế hệ 5 (Market / Reference / Fundamental) để chuyển provider sang cách viết mới
trước khi thế hệ 4 (Company/Finance/Quote/Trading/Listing theo source) ngừng hoạt động.

    python scripts/probe_v5.py FPT

Chỉ in tên thuộc tính, chữ ký hàm, tên cột và vài dòng đầu; không ghi gì vào kho.
"""

from __future__ import annotations

import inspect
import os
import sys
import time

import pandas as pd

os.environ.setdefault("VNSTOCK_TELEMETRY", "off")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 30)


def members(obj, name, depth=0):
    pub = [m for m in dir(obj) if not m.startswith("_")]
    print(f"{'  ' * depth}{name}: {pub[:60]}")
    for m in pub:
        try:
            attr = getattr(obj, m)
        except Exception as e:  # noqa: BLE001
            print(f"{'  ' * depth}  .{m}: LỖI {type(e).__name__}")
            continue
        if callable(attr):
            try:
                sig = str(inspect.signature(attr))
            except (TypeError, ValueError):
                sig = "(?)"
            print(f"{'  ' * depth}  .{m}{sig[:160]}")
        elif depth < 1 and not isinstance(attr, (str, int, float, dict, list, tuple)):
            members(attr, f"{name}.{m}", depth + 1)


def show(title, fn):
    print(f"\n===== {title}")
    try:
        df = fn()
    except BaseException as e:  # noqa: BLE001
        print(f"  LỖI: {type(e).__name__}: {str(e)[:300]}")
        return
    if isinstance(df, pd.DataFrame):
        print(f"  shape={df.shape}")
        print(f"  columns: {list(df.columns)[:40]}")
        print(df.head(4).to_string()[:2500])
    else:
        print(f"  {type(df).__name__}: {str(df)[:1500]}")
    time.sleep(3)


def main() -> int:
    sym = (sys.argv[1:] or ["FPT"])[0].upper()
    import vnstock

    print(f"vnstock {getattr(vnstock, '__version__', '?')}")
    try:
        import vnai
        print(f"vnai {getattr(vnai, '__version__', '?')}")
    except Exception:  # noqa: BLE001
        pass
    print("Có lớp cũ:", {n: hasattr(vnstock, n) for n in
                         ("Company", "Finance", "Quote", "Trading", "Listing", "Vnstock")})
    print("Có lớp mới:", {n: hasattr(vnstock, n) for n in ("Market", "Reference", "Fundamental")})
    from vnstock import Fundamental, Market, Reference

    m, r, f = Market(), Reference(), Fundamental()
    for obj, name in ((m, "Market()"), (r, "Reference()"), (f, "Fundamental()")):
        members(obj, name)
    objs = {"Market().equity(sym)": m.equity(sym), "Market().index('VNINDEX')": m.index("VNINDEX"),
            "Reference().company(sym)": r.company(sym), "Reference().equity()": r.equity(),
            "Reference().events(sym)": r.events(sym), "Reference().industry()": r.industry(),
            "Reference().market()": r.market(), "Fundamental().equity(sym)": f.equity(sym)}
    for name, obj in objs.items():
        members(obj, name)
    eq, co, ev, fe = objs["Market().equity(sym)"], objs["Reference().company(sym)"], \
        objs["Reference().events(sym)"], objs["Fundamental().equity(sym)"]

    def call(obj, meth, *a, **k):
        return lambda: getattr(obj, meth)(*a, **k)

    tries = [("equity.ohlcv", eq, "ohlcv", (), {"start": "2026-10-01", "end": "2026-10-31"}),
             ("equity.ohlcv default", eq, "ohlcv", (), {}),
             ("equity.trades/intraday", eq, "intraday", (), {}),
             ("index.ohlcv", objs["Market().index('VNINDEX')"], "ohlcv", (),
              {"start": "2026-10-01", "end": "2026-10-31"})]
    for meth in ("info", "overview", "profile", "shareholders", "news", "events", "dividends",
                 "officers", "ratio", "ratios"):
        tries.append((f"company.{meth}", co, meth, (), {}))
    for meth in ("dividends", "list", "events", "all"):
        tries.append((f"events.{meth}", ev, meth, (), {}))
    for meth in ("list", "symbols", "listing", "all_symbols", "by_exchange", "by_industries"):
        tries.append((f"Reference.equity.{meth}", objs["Reference().equity()"], meth, (), {}))
    for meth in ("balance_sheet", "income_statement", "cash_flow", "ratio"):
        tries.append((f"fundamental.{meth} year", fe, meth, (), {"period": "year"}))
    tries.append(("fundamental.income_statement quarter", fe, "income_statement", (),
                  {"period": "quarter"}))
    for meth in ("price_board", "board", "quote", "depth"):
        tries.append((f"Market.equity.{meth}", eq, meth, (), {}))
    for title, obj, meth, a, k in tries:
        if hasattr(obj, meth):
            show(title, call(obj, meth, *a, **k))
    return 0


if __name__ == "__main__":
    sys.exit(main())
