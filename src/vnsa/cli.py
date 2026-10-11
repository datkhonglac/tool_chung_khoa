"""Dòng lệnh của tool.

  python -m vnsa.cli fetch            # tải 5 năm giá của danh sách theo dõi từ CafeF rồi nạp
  python -m vnsa.cli fetch-bctc       # tải BCTC năm (vnstock/VCI) của danh sách theo dõi vào kho
  python -m vnsa.cli ratios FPT       # bộ chỉ số tài chính năm: ROE, ROA, biên LN, D/E, CFO/LNST, EPS
  python -m vnsa.cli import-excel FPT  # ghép data/raw/excel/FPT/*.xlsx -> data/raw/prices/FPT.csv rồi nạp
  python -m vnsa.cli load            # nạp mọi CSV trong data/raw/prices vào DuckDB
  python -m vnsa.cli load --sample   # nạp dữ liệu MẪU (giả lập) để chạy thử
  python -m vnsa.cli check FPT       # kiểm tra chất lượng dữ liệu của một mã
  python -m vnsa.cli list            # xem trong kho đang có gì
  python -m vnsa.cli chart FPT       # xuất biểu đồ nến ra file HTML
"""

from __future__ import annotations

import argparse
import sys

from vnsa.config import PROJECT_ROOT, path, settings
from vnsa.data.cleaning import summarize, validate_prices
from vnsa.data.providers import CSVProvider


def _provider(sample: bool) -> CSVProvider:
    folder = path("sample_prices_dir" if sample else "raw_prices_dir")
    return CSVProvider(folder, price_unit=settings()["data"]["price_unit"])


def cmd_load(args) -> int:
    from vnsa.data.store import Store

    prov = _provider(args.sample)
    symbols = [s.upper() for s in args.symbols] or prov.symbols()
    if not symbols:
        print(f"Chưa có file CSV nào trong {prov.folder}. Xem README, mục 'Lấy dữ liệu giá'.")
        return 1
    band = settings()["market"]["band_hose"]
    total, failed = 0, []
    with Store(path("warehouse")) as store:
        for sym in symbols:
            try:
                df = prov.ohlcv(sym)
            except Exception as e:  # noqa: BLE001 - báo lỗi từng mã, chạy tiếp mã khác
                print(f"  ✗ {sym}: {e}")
                failed.append(sym)
                continue
            issues = validate_prices(df, band=band)
            errors = [i for i in issues if i.level == "LOI"]
            if errors and not args.force:
                print(f"  ✗ {sym}: {len(errors)} lỗi, không nạp (dùng --force để nạp vẫn).")
                for i in errors:
                    print(f"      {i}")
                failed.append(sym)
                continue
            n = store.write_prices(df, source=f"csv{'-sample' if args.sample else ''}")
            total += n
            warn = sum(i.level == "CANH_BAO" for i in issues)
            s = summarize(df)
            print(f"  ✓ {sym}: {n} phiên ({s['tu_ngay']} → {s['den_ngay']})"
                  + (f", {warn} cảnh báo — chạy 'check {sym}'" if warn else ""))
    print(f"Đã nạp {total} dòng vào {path('warehouse').name}. Lỗi: {failed or 'không'}")
    return 1 if failed else 0


def cmd_import_excel(args) -> int:
    from vnsa.data.excel_import import import_excel_folder

    code = 0
    for sym in args.symbols:
        try:
            out, df, files = import_excel_folder(
                sym, path("raw_excel_dir"), path("raw_prices_dir"),
                price_unit=settings()["data"]["price_unit"],
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {sym.upper()}: {e}")
            code = 1
            continue
        first, last = df["date"].min().date(), df["date"].max().date()
        print(f"  ✓ {sym.upper()}: ghép {len(files)} file → {len(df)} phiên ({first} → {last})")
        print(f"      đã ghi {out.relative_to(PROJECT_ROOT)}")
    if args.no_load or code:
        return code
    args.sample, args.force = False, False
    return cmd_load(args)


def cmd_fetch(args) -> int:
    from datetime import date, timedelta

    from vnsa.data.providers.cafef_provider import CafeFProvider

    symbols = [s.upper() for s in args.symbols] or settings()["watchlist"]["symbols"]
    end = date.today()
    start = end - timedelta(days=round(365.25 * args.years))
    out_dir = path("raw_prices_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = CafeFProvider(pause=args.pause)
    done, code = [], 0
    for sym in symbols:
        print(f"  … {sym}: đang tải {start} → {end} từ CafeF")
        try:
            df = prov.ohlcv(sym, start, end)
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {sym}: {e}")
            code = 1
            continue
        out = out_dir / f"{sym}.csv"
        df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).drop(columns="symbol").to_csv(
            out, index=False
        )
        first, last = df["date"].min().date(), df["date"].max().date()
        print(f"  ✓ {sym}: {len(df)} phiên ({first} → {last}) → {out.relative_to(PROJECT_ROOT)}")
        if prov.repaired:
            print(f"      sửa cao/thấp cho {len(prov.repaired)} phiên CafeF ghi sai: "
                  + ", ".join(prov.repaired))
        done.append(sym)
    if args.no_load or not done:
        return code
    args.symbols, args.sample, args.force = done, False, False
    return cmd_load(args) or code


def cmd_fetch_bctc(args) -> int:
    import time

    from vnsa.data.providers.vnstock_provider import VnstockProvider
    from vnsa.data.store import Store

    symbols = [s.upper() for s in args.symbols] or settings()["watchlist"]["symbols"]
    out_dir = path("raw_bctc_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = VnstockProvider()
    code = 0
    with Store(path("warehouse")) as store:
        for i, sym in enumerate(symbols):
            if i:
                time.sleep(args.pause)   # bản miễn phí giới hạn số lượt gọi mỗi phút
            try:
                df = prov.fundamentals(sym)
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ {sym}: {e}")
                code = 1
                continue
            df.to_csv(out_dir / f"{sym}.csv", index=False)
            store.write_fundamentals(df, source=prov.name)
            years = df["year"].tolist()
            print(f"  ✓ {sym}: BCTC năm {years[0]}–{years[-1]} ({len(years)} năm)"
                  + (f", thiếu khoản mục: {', '.join(prov.missing)}" if prov.missing else ""))
    return code


def cmd_ratios(args) -> int:
    import pandas as pd

    from vnsa.data.store import Store
    from vnsa.fa.ratios import compute_ratios, format_ratios

    with Store(path("warehouse")) as store:
        fin = store.fundamentals(args.symbol)
    if fin.empty:
        print(f"{args.symbol.upper()} chưa có BCTC trong kho. Chạy 'fetch-bctc {args.symbol}' trước.")
        return 1
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"{args.symbol.upper()} — chỉ số tài chính năm (nguồn: BCTC hợp nhất, VCI)")
        print(format_ratios(compute_ratios(fin)).to_string())
    return 0


def cmd_check(args) -> int:
    from vnsa.data.store import Store

    band = settings()["market"]["band_hose"]
    with Store(path("warehouse")) as store:
        df = store.prices(args.symbol)
    if df.empty:
        print(f"{args.symbol.upper()} chưa có trong kho. Chạy 'load' trước.")
        return 1
    print(summarize(df))
    issues = validate_prices(df, band=band)
    if not issues:
        print("Không phát hiện vấn đề.")
    for i in issues:
        print(i)
    return 0


def cmd_list(args) -> int:
    from vnsa.data.store import Store

    with Store(path("warehouse")) as store:
        inv = store.inventory()
    print(inv.to_string(index=False) if len(inv) else "Kho đang trống.")
    return 0


def cmd_chart(args) -> int:
    from vnsa.data.store import Store
    from vnsa.report.charts import candlestick

    with Store(path("warehouse")) as store:
        df = store.prices(args.symbol, start=args.start)
    if df.empty:
        print(f"{args.symbol.upper()} chưa có trong kho.")
        return 1
    out = path("charts_dir")
    out.mkdir(parents=True, exist_ok=True)
    file = out / f"{args.symbol.upper()}.html"
    candlestick(df).write_html(file, include_plotlyjs="cdn")
    print(f"Đã lưu biểu đồ: {file}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows: tránh lỗi in tiếng Việt khi chuyển hướng output
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="vnsa", description="VN Stock Analyzer")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("load", help="Nạp CSV vào kho DuckDB")
    pl.add_argument("symbols", nargs="*", help="Bỏ trống = nạp tất cả")
    pl.add_argument("--sample", action="store_true", help="Dùng dữ liệu mẫu giả lập")
    pl.add_argument("--force", action="store_true", help="Nạp kể cả khi có lỗi")
    pl.set_defaults(func=cmd_load)

    pi = sub.add_parser("import-excel", help="Ghép các file Excel của một mã thành CSV rồi nạp")
    pi.add_argument("symbols", nargs="+")
    pi.add_argument("--no-load", action="store_true", help="Chỉ tạo CSV, không nạp vào kho")
    pi.set_defaults(func=cmd_import_excel)

    pf = sub.add_parser("fetch", help="Tải giá từ CafeF (tự chia từng năm) rồi nạp")
    pf.add_argument("symbols", nargs="*", help="Bỏ trống = danh sách theo dõi trong settings.toml")
    pf.add_argument("--years", type=int, default=5)
    pf.add_argument("--pause", type=float, default=0.3, help="Giây nghỉ giữa các lần gọi")
    pf.add_argument("--no-load", action="store_true", help="Chỉ tạo CSV, không nạp vào kho")
    pf.set_defaults(func=cmd_fetch)

    pb = sub.add_parser("fetch-bctc", help="Tải BCTC năm qua vnstock (VCI) rồi nạp vào kho")
    pb.add_argument("symbols", nargs="*", help="Bỏ trống = danh sách theo dõi trong settings.toml")
    pb.add_argument("--pause", type=float, default=3.0, help="Giây nghỉ giữa các mã")
    pb.set_defaults(func=cmd_fetch_bctc)

    pr = sub.add_parser("ratios", help="Bộ chỉ số tài chính năm từ BCTC trong kho")
    pr.add_argument("symbol")
    pr.set_defaults(func=cmd_ratios)

    pc = sub.add_parser("check", help="Kiểm tra chất lượng dữ liệu")
    pc.add_argument("symbol")
    pc.set_defaults(func=cmd_check)

    sub.add_parser("list", help="Xem kho có gì").set_defaults(func=cmd_list)

    pg = sub.add_parser("chart", help="Xuất biểu đồ nến HTML")
    pg.add_argument("symbol")
    pg.add_argument("--start", help="YYYY-MM-DD")
    pg.set_defaults(func=cmd_chart)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
