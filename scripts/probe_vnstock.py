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


def foreign(symbols: list[str]) -> int:
    """--foreign: dò nguồn giao dịch khối ngoại miễn phí (CafeF GDKhoiNgoai, vnstock)."""
    import requests

    for sym in symbols:
        for page in ("GDKhoiNgoai", "GDTuDoanh"):
            url = (f"https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/{page}.ashx?Symbol={sym}"
                   "&StartDate=&EndDate=&PageIndex=1&PageSize=3")
            print(f"\n===== CafeF {page} {sym}")
            try:
                r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
                print(f"  HTTP {r.status_code}: {r.text[:2500]}")
            except Exception as e:  # noqa: BLE001
                print(f"  LỖI: {e}")
            time.sleep(1)
    try:
        import vnstock

        for name in ("Trading", "Quote", "Company"):
            cls = getattr(vnstock, name, None)
            if cls is not None:
                obj = cls(source="VCI", symbol=symbols[0]) if name != "Trading" else cls(source="VCI")
                print(f"\n  {name}: {[m for m in dir(obj) if not m.startswith('_')]}")
    except Exception as e:  # noqa: BLE001
        print(f"vnstock: {e}")
    return 0


def board(symbols: list[str]) -> int:
    """--board: bảng giá trong phiên (Trading.price_board) và Quote.intraday — cho lệnh monitor."""
    from vnstock import Quote, Trading

    t = Trading(source="VCI")
    try:
        df = _call_plain(t.price_board, symbols)
        cols = list(df.columns)
        print(f"price_board shape={df.shape}")
        print(f"columns ({len(cols)}): {cols}")
        with pd.option_context("display.max_rows", 300):
            print(df.head(len(symbols)).T.to_string()[:12000])
    except Exception as e:  # noqa: BLE001
        print(f"price_board LỖI: {type(e).__name__}: {e}")
    time.sleep(2)
    try:
        q = Quote(source="VCI", symbol=symbols[0])
        df = q.intraday(page_size=5)
        print(f"\nintraday {symbols[0]}: columns={list(df.columns)}")
        print(df.tail(5).to_string())
    except Exception as e:  # noqa: BLE001
        print(f"intraday LỖI: {type(e).__name__}: {e}")
    return 0


def _call_plain(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def items_grep(symbols: list[str], pattern: str) -> int:
    """--grep MẪU MÃ...: chỉ in item_id/tên khoản mục BCTC năm khớp MẪU (regex, không phân biệt hoa
    thường) kèm số năm gần nhất — để ánh xạ khoản mục mới (phải thu, tồn kho, khấu hao...)."""
    import re

    from vnstock import Finance

    rx = re.compile(pattern, re.IGNORECASE)
    for sym in symbols:
        fin = Finance(source="VCI", symbol=sym)
        for name in ("income_statement", "balance_sheet", "cash_flow"):
            try:
                df = getattr(fin, name)(period="year")
            except Exception as e:  # noqa: BLE001
                print(f"{sym} {name}: LỖI {e}")
                continue
            last = [c for c in df.columns if str(c)[:4].isdigit()][:1]
            print(f"== {sym} {name} ({len(df)} dòng, cột {last})")
            for _, r in df.iterrows():
                if rx.search(str(r["item_id"])) or rx.search(str(r["item"])):
                    v = r[last[0]] if last else ""
                    print(f"  {r['item_id']} | {str(r['item'])[:70]} | {v}")
            time.sleep(2)
    return 0


def news(symbols: list[str]) -> int:
    """--news: tiêu đề tin (Company.news) để soạn bộ từ khoá phân loại tin."""
    from vnstock import Company

    for sym in symbols:
        try:
            df = Company(source="VCI", symbol=sym).news()
            print(f"== {sym} news columns={list(df.columns)}")
            col = next((c for c in ("news_title", "title") if c in df.columns), df.columns[0])
            for x in df[col].head(25):
                print(f"  - {str(x)[:140]}")
        except Exception as e:  # noqa: BLE001
            print(f"{sym} news LỖI: {e}")
        time.sleep(2)
    return 0


def limits(n: int = 45) -> int:
    """Giới hạn số lần gọi của vnstock (vnai): in cấu hình trong mã nguồn vnai rồi gọi liên tục
    `n` lần một hàm nhẹ, ghi thời gian và kiểu lỗi (kể cả SystemExit) để biết khi vượt giới hạn thì
    thư viện chờ, báo lỗi hay thoát hẳn chương trình."""
    import pathlib
    import re

    import vnai

    root = pathlib.Path(vnai.__file__).parent
    pat = re.compile(r"(per_?minute|per_?hour|rate.?limit|quota|requests?_per|max_requests|"
                     r"sys\.exit|SystemExit|time\.sleep|tier|guest|community)", re.IGNORECASE)
    print(f"vnai {getattr(vnai, '__version__', '?')} tại {root}")
    shown = 0
    for f in sorted(root.rglob("*.py")):
        for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if pat.search(line) and shown < 120:
                print(f"  {f.relative_to(root)}:{i}: {line.strip()[:150]}")
                shown += 1
    from vnstock import Company

    t0 = time.monotonic()
    for k in range(1, n + 1):
        t = time.monotonic()
        try:
            Company(source="VCI", symbol="FPT").overview()
            status = "ok"
        except BaseException as e:  # noqa: BLE001 - muốn thấy cả SystemExit
            status = f"{type(e).__name__}: {str(e)[:160]}"
        print(f"  lần {k:>2}: {time.monotonic() - t:5.1f}s (tổng {time.monotonic() - t0:6.1f}s) {status}")
        if status != "ok" and "SystemExit" in status:
            break
    return 0


def main() -> int:
    from vnstock import Company, Finance, Quote

    if sys.argv[1:2] == ["--limits"]:
        return limits(int(sys.argv[2]) if len(sys.argv) > 2 else 45)

    if sys.argv[1:2] == ["--items"]:
        return items([s.upper() for s in sys.argv[2:]])
    if sys.argv[1:2] == ["--board"]:
        return board([s.upper() for s in sys.argv[2:]] or ["FPT", "VCB"])
    if sys.argv[1:2] == ["--grep"]:
        return items_grep([s.upper() for s in sys.argv[3:]] or ["FPT"], sys.argv[2])
    if sys.argv[1:2] == ["--news"]:
        return news([s.upper() for s in sys.argv[2:]] or ["FPT"])
    if sys.argv[1:2] == ["--foreign"]:
        return foreign([s.upper() for s in sys.argv[2:]] or ["FPT"])
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
