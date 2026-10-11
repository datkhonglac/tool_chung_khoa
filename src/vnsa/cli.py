"""Dòng lệnh của tool.

  python -m vnsa.cli fetch            # tải 5 năm giá của danh sách theo dõi từ CafeF rồi nạp
  python -m vnsa.cli fetch-bctc       # tải BCTC năm (vnstock/VCI) của danh sách theo dõi vào kho
  python -m vnsa.cli ratios FPT       # bộ chỉ số tài chính năm: ROE, ROA, biên LN, D/E, CFO/LNST, EPS
  python -m vnsa.cli value FPT        # định giá: beta, WACC, DCF-FCFF, P/E, P/B, ma trận độ nhạy
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
    index = settings()["valuation"]["index"]
    if index not in symbols:
        symbols = [*symbols, index]   # cần chỉ số thị trường để tính beta
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


def cmd_value(args) -> int:
    import pandas as pd

    from vnsa.data.store import Store
    from vnsa.fa import valuation as val

    sym = args.symbol.upper()
    a = val.Assumptions.from_settings(settings()["valuation"])
    with Store(path("warehouse")) as store:
        fin = store.fundamentals(sym)
        prices = store.prices(sym)
        index = store.prices(settings()["valuation"]["index"])
    if fin.empty or prices.empty:
        print(f"{sym}: cần cả giá và BCTC trong kho. Chạy 'fetch {sym}' và 'fetch-bctc {sym}' trước.")
        return 1
    b, weeks = (val.beta(prices, index, a.beta_years) if not index.empty else (float("nan"), 0))
    price = float(prices["close"].iloc[-1])
    try:
        v = val.value(sym, fin, price, a, beta_value=b, beta_weeks=weeks,
                      g1=args.growth, wacc_override=args.wacc)
    except ValueError as e:
        print(f"  ✗ {e}")
        return 1

    def bn(x: float) -> str:
        return "—" if pd.isna(x) else f"{x / 1e9:,.0f} tỷ"

    def pct(x: float) -> str:
        return "—" if pd.isna(x) else f"{x:.1%}"

    day = prices["date"].iloc[-1].date()
    print(f"{sym} — giá {price:,.0f} đ ({day}), {v.shares / 1e6:,.1f} triệu cp, "
          f"vốn hoá {bn(price * v.shares)}")
    print(f"  P/E {v.pe:.1f} lần · P/B {v.pb:.2f} lần (BCTC năm {int(fin['year'].iloc[-1])})")
    beta_src = "mặc định"
    if not v.beta_is_default:
        beta_src = f"{v.beta_weeks} tuần so với VN-Index" + (
            f", thô {b:.2f}, điều chỉnh Blume" if a.adjust_beta else "")
    print(f"\nWACC {pct(v.wacc)}")
    print(f"  beta {v.beta:.2f} ({beta_src}) → k_e = {pct(a.risk_free)} + {v.beta:.2f} × "
          f"{pct(a.equity_risk_premium)} = {pct(v.cost_of_equity)}")
    print(f"  k_d {pct(v.cost_of_debt)}, thuế {pct(v.tax_rate)}, tỷ trọng vốn chủ {pct(v.weight_equity)}")
    print("\nFCFF lịch sử (CFO + lãi vay × (1−t) − capex):")
    print("  " + " · ".join(f"{y}: {bn(x)}" for y, x in v.fcff_history.items()))
    path_ = " → ".join(pct(x) for x in val.growth_path(v.g1, v.g, a.forecast_years, a.fade_growth))
    print(f"  gốc (TB 3 năm) {bn(v.fcff_base)}; tăng {path_} trong {a.forecast_years} năm, "
          f"sau đó {pct(v.g)}/năm")
    d, br = v.dcf, v.net_debt_bridge
    print(f"\nDCF: PV dòng tiền {bn(d['pv_flows'])} + PV giá trị cuối {bn(d['pv_terminal'])} "
          f"= EV {bn(d['ev'])}")
    print(f"  − vay {bn(br['debt'])} + tiền & ĐTNH {bn(br['cash'])} − CĐ thiểu số {bn(br['minority'])}"
          f" = vốn chủ {bn(v.equity_value)}")
    if not pd.isna(v.value_per_share):
        print(f"  → {v.value_per_share:,.0f} đ/cp, so với giá hiện tại {v.upside:+.1%}")
    for n in v.notes:
        print(f"  ! {n}")
    table = val.sensitivity(v, a)
    if table.notna().any().any():
        print("\nĐộ nhạy giá trị/cp (nghìn đồng): hàng = WACC, cột = tăng trưởng dài hạn g")
        t = (table / 1000).round(1)
        t.index = [pct(w) for w in t.index]
        t.columns = [pct(g) for g in t.columns]
        print(t.to_string())
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

    pv = sub.add_parser("value", help="Định giá: WACC, DCF-FCFF, P/E, P/B, độ nhạy")
    pv.add_argument("symbol")
    pv.add_argument("--growth", type=float, help="Tăng trưởng FCFF giai đoạn 1, vd 0.12 (mặc định: "
                    "CAGR LNST cty mẹ, tối đa max_stage1_growth)")
    pv.add_argument("--wacc", type=float, help="Dùng WACC này thay vì tự tính, vd 0.13")
    pv.set_defaults(func=cmd_value)

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
