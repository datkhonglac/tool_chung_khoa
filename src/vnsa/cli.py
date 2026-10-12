"""Dòng lệnh của tool.

  python -m vnsa.cli analyze SHS VCB  # MỌI mã HOSE/HNX/UPCoM: tự tải giá, BCTC, sàn rồi phân tích
  python -m vnsa.cli fetch            # tải 5 năm giá của danh sách theo dõi từ CafeF rồi nạp
  python -m vnsa.cli fetch-bctc       # tải BCTC năm (vnstock/VCI) của danh sách theo dõi vào kho
  python -m vnsa.cli ratios FPT       # bộ chỉ số tài chính năm: ROE, ROA, biên LN, D/E, CFO/LNST, EPS
  python -m vnsa.cli value FPT        # định giá: beta, WACC, DCF-FCFF, P/E, P/B, ma trận độ nhạy
  python -m vnsa.cli fa-score FPT     # Piotroski F-Score, Altman Z, điểm FA 0–100
  python -m vnsa.cli ta FPT           # chỉ báo kỹ thuật, hỗ trợ/kháng cự, điểm TA 0–100
  python -m vnsa.cli plan FPT         # kế hoạch mua: giá vào, cắt lỗ theo ATR, mục tiêu, khối lượng
  python -m vnsa.cli risk FPT         # GARCH, Monte Carlo P10–P90, VaR/CVaR
  python -m vnsa.cli recommend        # xếp hạng MUA…BÁN + thời điểm cho danh sách theo dõi (tự lưu)
  python -m vnsa.cli review-signals   # chấm lại khuyến nghị đã lưu sau 5/20/60 phiên
  python -m vnsa.cli backtest FPT     # backtest chiến lược theo điểm TA
  python -m vnsa.cli import-excel FPT  # ghép data/raw/excel/FPT/*.xlsx -> data/raw/prices/FPT.csv rồi nạp
  python -m vnsa.cli load            # nạp mọi CSV trong data/raw/prices vào DuckDB
  python -m vnsa.cli load --sample   # nạp dữ liệu MẪU (giả lập) để chạy thử
  python -m vnsa.cli check FPT       # kiểm tra chất lượng dữ liệu của một mã
  python -m vnsa.cli list            # xem trong kho đang có gì
  python -m vnsa.cli chart FPT       # xuất biểu đồ nến ra file HTML
"""

from __future__ import annotations

import argparse
import math
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
    from vnsa import market

    total, failed = 0, []
    with Store(path("warehouse")) as store:
        for sym in symbols:
            try:
                df = prov.ohlcv(sym)
            except Exception as e:  # noqa: BLE001 - báo lỗi từng mã, chạy tiếp mã khác
                print(f"  ✗ {sym}: {e}")
                failed.append(sym)
                continue
            issues = validate_prices(df, band=market.band(sym))
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


def cmd_fetch_macro(args) -> int:
    """Tải các chuỗi vĩ mô tự động (Yahoo, World Bank, FRED) và nạp số nhập tay vào kho."""
    from vnsa.data.providers.macro_provider import MacroProvider
    from vnsa.data.store import Store
    from vnsa.macro.catalog import catalog
    from vnsa.macro.update import fetch_auto, load_manual

    cat = catalog()
    unknown = [x for x in args.series if x not in cat]
    if unknown:
        print(f"Không có trong config/macro_map.toml: {', '.join(unknown)}")
        return 1
    auto = [sid for sid in (args.series or list(cat)) if cat[sid].auto]
    with Store(path("warehouse")) as store:
        res = fetch_auto(store, cat, MacroProvider(cat, pause=args.pause), auto,
                         args.years) if auto or not args.series else []
        for r in res:
            src = cat[r.series].source
            if r.ok:
                print(f"  ✓ {r.series:<18} {r.rows:>5} kỳ  {r.first:%m/%Y} → {r.last:%d/%m/%Y}  "
                      f"cuối = {r.value:,.4g}  ({src})")
            else:
                print(f"  ✗ {r.series:<18} {src}: {r.error}")
        try:
            n, missing = load_manual(store, cat)
        except ValueError as e:
            print(f"  ✗ số nhập tay: {e}")
            return 1
        n_manual = sum(1 for s in cat.values() if s.manual)
        print(f"  Nhập tay (config/macro_manual.toml): {n} số, {n_manual - len(missing)}/{n_manual} "
              "chuỗi có số")
        if missing:
            print(f"    chưa có số: {', '.join(missing)}")
    bad = [r.series for r in res if not r.ok]
    if bad:
        print(f"Lỗi {len(bad)}/{len(res)} chuỗi: {', '.join(bad)}")
    return 1 if res and len(bad) == len(res) else 0


def cmd_macro(args) -> int:
    """Bảng số liệu vĩ mô mới nhất trong kho: giá trị, biến động, mức bất thường (z)."""
    from vnsa.data.store import Store
    from vnsa.macro.catalog import catalog
    from vnsa.macro.signals import fmt_change, fmt_value, readings

    cat = catalog()
    with Store(path("warehouse")) as store:
        data = store.macro()
    if data.empty:
        print("Kho chưa có số liệu vĩ mô. Chạy: python -m vnsa.cli fetch-macro")
        return 1
    cfg = settings().get("macro", {})
    rs = readings(data, cat, z_years=cfg.get("z_years", 5), z_cap=cfg.get("z_cap", 2.0))
    print(f"{'Chuỗi':<17}{'Tên':<34}{'Kỳ cuối':>11}{'Giá trị':>13}{'Biến động':>16}{'z':>6}  Ghi chú")
    for sid, r in rs.items():
        if r.date is None:
            continue
        when = f"{r.date:%d/%m/%Y}" if cat[sid].freq == "D" else (
            f"Q{r.date.quarter}/{r.date.year}" if cat[sid].freq == "Q" else f"{r.date:%m/%Y}")
        z = "" if not r.ok else f"{r.z:+.1f}".replace(".", ",")
        note = "CŨ" if r.stale else ""
        note += ("" if r.basis.startswith("z") else (" " if note else "") + r.basis)
        print(f"{sid:<17}{r.name[:33]:<34}{when:>11}{fmt_value(r):>13}{fmt_change(r):>16}{z:>6}  "
              f"{note}")
    empty = [sid for sid, r in rs.items() if r.date is None]
    if empty:
        print(f"\nChưa có số ({len(empty)}): {', '.join(empty)}")
        print("  Số Việt Nam nhập tay ở config/macro_manual.toml rồi chạy lại fetch-macro.")
    print("\nBiến động: % sau 3 tháng (giá, tỷ giá) · điểm % (lãi suất) · so mốc (tăng trưởng, PMI). "
          "z: so với 5 năm của chính chuỗi đó, chặn ±2.")
    return 0


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


nan = float("nan")


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _bn(x: float) -> str:
    return "—" if _nan(x) else f"{x / 1e9:,.0f} tỷ"


def _pct(x: float, digits: int = 1) -> str:
    return "—" if _nan(x) else f"{x:.{digits}%}"


def _num(x: float, digits: int = 0) -> str:
    return "—" if _nan(x) else f"{x:,.{digits}f}"


def _load_data(sym: str, need_fin: bool = True, need_prices: bool = True):
    """analysis.Data đầy đủ (BCTC năm + quý/TTM, hồ sơ, cổ tức, khối ngoại); in lệnh cần chạy và
    trả về None nếu thiếu."""
    from vnsa import analysis

    try:
        return analysis.load(sym, need_fin, need_prices)
    except analysis.DataMissing as e:
        print(e)
        return None


def _load(sym: str, need_fin: bool = True, need_prices: bool = True):
    """(BCTC, giá, giá chỉ số) từ kho; in lệnh cần chạy và trả về None nếu thiếu."""
    d = _load_data(sym, need_fin, need_prices)
    return None if d is None else (d.fin, d.prices, d.index)


def cmd_value(args) -> int:
    import pandas as pd

    from vnsa.fa import valuation as val

    sym = args.symbol.upper()
    d = _load_data(sym)
    if d is None:
        return 1
    from vnsa import analysis
    from vnsa.market import financial_note

    if financial_note(sym):
        return _fin_value(d)
    try:
        a, v, b = analysis.valuation(d, args.growth, args.wacc)
    except ValueError as e:
        print(f"  ✗ {e}")
        return 1
    price = v.price
    bn, pct = _bn, _pct

    day = d.prices["date"].iloc[-1].date()
    print(f"{sym} — giá {price:,.0f} đ ({day}), {v.shares / 1e6:,.1f} triệu cp, "
          f"vốn hoá {bn(price * v.shares)}")
    basis = d.current.basis if d.current is not None else f"năm {int(d.fin['year'].iloc[-1])}"
    print(f"  P/E {v.pe:.1f} lần · P/B {v.pb:.2f} lần ({basis})")
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
    dcf, br = v.dcf, v.net_debt_bridge
    print(f"\nDCF: PV dòng tiền {bn(dcf['pv_flows'])} + PV giá trị cuối {bn(dcf['pv_terminal'])} "
          f"= EV {bn(dcf['ev'])}")
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
    _print_fair(d, v)
    return 0


def _print_fair(d, v) -> None:
    """Giá trị hợp lý gộp DCF + P/E, P/B lịch sử (fa/fair_value.py)."""
    from vnsa import analysis
    from vnsa.fa import fair_value as fv

    fair = analysis.general_fair_value(d, v)
    hist = fv.historical_multiples(d.fin, d.prices, settings()["valuation"]["par_value"])
    c = d.current
    print(f"\nBội số lịch sử (vốn hoá bình quân năm / LNST, vốn chủ; {hist['basis'].iloc[-1]}):")
    print("  " + " · ".join(f"{int(r.year)}: P/E {_num(r.pe, 1)}, P/B {_num(r.pb, 2)}"
                            for r in hist.itertuples()))
    print(f"  Trung vị lịch sử: P/E {_num(fair.pe_median, 1)}, P/B {_num(fair.pb_median, 2)}; mục tiêu "
          f"thu hẹp {settings()['valuation'].get('multiple_reversion', 1):.0%} khoảng cách từ mức hiện tại")
    print(f"  P/E mục tiêu {_num(fair.pe_target, 1)} × EPS {_num(c.eps if c else nan)} = "
          f"{_num(fair.methods['pe_hist'])} · P/B mục tiêu {_num(fair.pb_target, 2)} × BVPS "
          f"{_num(c.bvps if c else nan)} = {_num(fair.methods['pb_hist'])}"
          + (f" ({c.basis}, {c.shares / 1e6:,.1f} triệu cp)" if c else ""))
    from vnsa.fa.fair_value import METHOD_NAMES

    print("⇒ Giá trị hợp lý (bình quân có trọng số: "
          + ", ".join(f"{METHOD_NAMES[k]} {w:.0%}" for k, w in fair.weights.items())
          + f"): {_num(fair.value)} đ/cp, {_pct(fair.upside)} so với giá")
    for n in fair.notes:
        print(f"  ! {n}")


def _fin_score(d) -> int:
    """Điểm FA của định chế tài chính (scores.financial_score)."""
    import pandas as pd

    from vnsa import analysis
    from vnsa.fa.scores import FIN_GROUPS

    try:
        s, _ = analysis.financial(d)
    except ValueError as e:
        print(f"  ✗ {e}")
        return 1
    print(f"{d.symbol} — chất lượng tài chính, bộ chỉ tiêu cho định chế tài chính "
          f"(BCTC năm {int(d.fin['year'].iloc[0])}–{int(d.fin['year'].iloc[-1])})")
    names = {"sinh_loi": "Sinh lời", "tang_truong": "Tăng trưởng", "an_toan": "Đệm vốn",
             "hieu_qua": "Hiệu quả", "rui_ro": "Rủi ro", "chat_luong_tai_san": "Chất lượng tài sản",
             "dinh_gia": "Định giá"}
    labels = {"roe": ("ROE", "pct"), "ni_cagr": ("Tăng trưởng kép LNST cty mẹ", "pct"),
              "equity_to_assets": ("Vốn chủ / tổng tài sản", "pct"),
              "nim": ("NIM (xấp xỉ)", "pct"), "cir": ("CIR (chi phí/thu nhập)", "pct"),
              "credit_cost": ("Chi phí tín dụng", "pct"),
              "margin_to_equity": ("Cho vay ký quỹ / vốn chủ (lần)", "x"),
              "npl": ("Nợ xấu (nhập tay)", "pct"), "npl_coverage": ("Bao phủ nợ xấu (nhập tay)", "pct"),
              "casa": ("CASA (nhập tay)", "pct"), "car": ("CAR (nhập tay)", "pct"),
              "pb_rel": ("P/B so với trung vị lịch sử (1 = bằng)", "x"),
              "fair_upside": ("Chênh lệch so với giá trị hợp lý", "pct")}
    print(f"\nĐiểm FA: {_num(s.total)}/100 — {s.rating}")
    for g, keys in FIN_GROUPS.items():
        shown = [k for k in keys if not pd.isna(s.items[k][0])]
        if not shown:
            continue
        w = settings()["fa_score"]["financials"]["weights"][g]
        print(f"  {names[g]} ({w}%): {_num(s.groups[g])}")
        for k in shown:
            val_, pts = s.items[k]
            label, kind = labels[k]
            print(f"      {label}: {_pct(val_) if kind == 'pct' else _num(val_, 2)} → {_num(pts)} điểm")
    sm = analysis.sector(d)
    if sm is not None:
        from vnsa.fa.sector import LABELS

        print(f"\n  Chỉ số ngành ({sm.basis}): " + " · ".join(
            f"{LABELS[k]} {_num(v, 2) if k == 'margin_to_equity' else _pct(v)}"
            for k, v in sm.values.items() if not pd.isna(v)))
    for n in s.notes:
        print(f"  ! {n}")
    return 0


def _fin_value(d) -> int:
    """Định giá định chế tài chính: thu nhập thặng dư, P/B theo ROE, P/B lịch sử."""
    from vnsa import analysis
    from vnsa.fa.fair_value import METHOD_NAMES

    try:
        _, fair = analysis.financial(d)
    except ValueError as e:
        print(f"  ✗ {e}")
        return 1
    ri, a = fair.ri, settings()["valuation"]
    ke, b, raw, weeks = analysis.cost_of_equity(d)
    print(f"{d.symbol} — định giá định chế tài chính, giá {fair.price:,.0f} đ "
          f"({d.prices['date'].iloc[-1]:%d/%m/%Y})")
    print(f"  k_e = {_pct(a['risk_free'])} + {b:.2f} × {_pct(a['equity_risk_premium'])} = {_pct(ke)}"
          + (f" (beta {weeks} tuần, thô {raw:.2f})" if not _nan(raw) else " (beta mặc định)"))
    print(f"  Giá trị sổ sách/cp B0 = {ri.book0:,.0f} · P/B hiện tại {fair.price / ri.book0:.2f}")
    print(f"  ROE bình quân {_pct(ri.roe0)} → dài hạn {_pct(ri.roe_terminal)}: "
          + " → ".join(_pct(x) for x in ri.roe_path))
    print(f"\nThu nhập thặng dư: B0 {ri.book0:,.0f} + PV(RI 5 năm) {ri.pv_ri:,.0f} "
          f"+ PV(giá trị cuối) {ri.pv_terminal:,.0f} = {ri.value_per_share:,.0f} đ/cp")
    print(f"P/B hợp lý theo ROE = (ROE − g) / (k_e − g) = {_num(ri.justified_pb, 2)} "
          f"→ {_num(ri.justified_pb * ri.book0)} đ/cp")
    print(f"P/B lịch sử: trung vị {_num(fair.pb_median, 2)}, mục tiêu {_num(fair.pb_target, 2)} "
          f"→ {_num(fair.pb_target * ri.book0)} đ/cp")
    print("\n⇒ Giá trị hợp lý (bình quân có trọng số: "
          + ", ".join(f"{METHOD_NAMES[k]} {w:.0%}" for k, w in fair.weights.items())
          + f"): {_num(fair.value)} đ/cp, {_pct(fair.upside)} so với giá")
    print("  Phương pháp dùng: " + ", ".join(f"{METHOD_NAMES[k]} {v:,.0f}" for k, v in fair.used.items()))
    for n in fair.notes:
        print(f"  ! {n}")
    return 0


def cmd_fa_score(args) -> int:
    import pandas as pd

    from vnsa.fa import scores as sc

    sym = args.symbol.upper()
    d = _load_data(sym, need_prices=False)
    if d is None:
        return 1
    from vnsa import analysis

    fin = d.fin
    from vnsa.market import financial_note

    if financial_note(sym):
        return _fin_score(d)
    try:
        f, z, score, v = analysis.fa(d)
    except ValueError as e:
        print(f"  ✗ {e}")
        return 1
    if v is None and not d.prices.empty:
        print("  ! không định giá được; bỏ nhóm định giá")

    print(f"{sym} — chất lượng tài chính (BCTC năm {int(fin['year'].iloc[0])}–"
          f"{int(fin['year'].iloc[-1])})")
    print("\nPiotroski F-Score (1 = đạt, 0 = không đạt, — = thiếu số liệu)")
    t = f.set_index("year")[list(sc.FSCORE_CRITERIA) + ["f_score"]].T
    t.index = [*sc.FSCORE_CRITERIA.values(), "F-Score (/9)"]
    t = t.map(lambda x: "—" if pd.isna(x) else f"{x:.0f}")
    t.loc["(tiền thu phát hành cp, tỷ)"] = [_num(x / 1e9) for x in fin["equity_issued"]]
    print(t.to_string())

    print("\nAltman Z")
    zt = z.set_index("year")[["z_em", "z_orig"]]
    for y, row in zt.iterrows():
        print(f"  {y}: Z'' = {_num(row['z_em'], 2)} ({sc.z_zone(row['z_em'])})"
              f" · Z gốc = {_num(row['z_orig'], 2)} ({sc.z_zone(row['z_orig'], 'orig')})")
    print("  Z'' dùng cho mọi ngành (chuẩn thị trường mới nổi); Z gốc chỉ hợp với DN sản xuất.")

    s = score
    names = {"sinh_loi": "Sinh lời", "tang_truong": "Tăng trưởng", "suc_khoe": "Sức khoẻ TC",
             "chat_luong": "Chất lượng LN", "dinh_gia": "Định giá"}
    labels = {"roe": ("ROE", "pct"), "ni_cagr": ("Tăng trưởng kép LNST cty mẹ", "pct"),
              "z_em": ("Altman Z''", "x"), "debt_to_equity": ("Vay/VCSH", "x"),
              "f_score": ("F-Score", "x0"), "cfo_to_ni": ("CFO/LNST cộng dồn 3 năm", "x"),
              "pe_rel": ("P/E so với trung vị lịch sử (1 = bằng)", "x"),
              "pb_rel": ("P/B so với trung vị lịch sử (1 = bằng)", "x"),
              "dcf_upside": ("Chênh lệch DCF", "pct")}
    print(f"\nĐiểm FA: {_num(s.total)}/100 — {s.rating}")
    for g, keys in sc.GROUPS.items():
        w = settings()["fa_score"]["weights"][g]
        print(f"  {names[g]} ({w}%): {_num(s.groups[g])}")
        for k in keys:
            val_, pts = s.items[k]
            label, kind = labels[k]
            shown = (_pct(val_) if kind == "pct" else _num(val_, 0 if kind == "x0" else
                                                            1 if kind == "x1" else 2))
            print(f"      {label}: {shown} → {_num(pts)} điểm")
    for n in s.notes:
        print(f"  ! {n}")
    return 0


def _indicators(prices):
    from vnsa import analysis

    return analysis.indicators(prices)


def cmd_ta(args) -> int:
    from vnsa.ta.levels import support_resistance
    from vnsa.ta.score import ta_score

    sym = args.symbol.upper()
    loaded = _load(sym, need_fin=False)
    if loaded is None:
        return 1
    ind = _indicators(loaded[1])
    if len(ind) < 220:
        print(f"{sym}: cần ít nhất 220 phiên để tính SMA200, mới có {len(ind)}.")
        return 1
    cfg = settings()["ta"]
    last = ind.iloc[-1]
    s = ta_score(ind, settings()["ta_score"]["weights"])
    sup, res = support_resistance(ind, cfg["swing_window"], cfg["lookback"], cfg["level_tolerance"])
    close = last["close"]

    def vs(x):
        return "—" if _nan(x) else f"{_num(x)} ({close / x - 1:+.1%})"

    print(f"{sym} — kỹ thuật phiên {last['date']:%d/%m/%Y}, giá {close:,.0f} đ")
    print(f"  SMA20 {vs(last['sma20'])} · SMA50 {vs(last['sma50'])} · SMA200 {vs(last['sma200'])}")
    print(f"  RSI14 {_num(last['rsi'], 1)} · MACD {_num(last['macd'])} / tín hiệu "
          f"{_num(last['macd_signal'])} (histogram {_num(last['macd_hist'])})")
    print(f"  Bollinger {_num(last['bb_lower'])}–{_num(last['bb_upper'])} (%B {_num(last['bb_pctb'], 2)})"
          f" · ATR14 {_num(last['atr'])} ({last['atr'] / close:.1%} giá)")
    print(f"  ADX {_num(last['adx'], 1)} (+DI {_num(last['plus_di'], 1)}, −DI {_num(last['minus_di'], 1)})"
          f" · KL TB20 {_num(last['vol_ma20'])} / TB50 {_num(last['vol_ma50'])}")
    print(f"\nHỗ trợ ({cfg['lookback']} phiên gần nhất): "
          + ("; ".join(str(lv) for lv in sup[:3]) or "không có"))
    print("Kháng cự: " + ("; ".join(str(lv) for lv in res[:3]) or "không có (giá ở vùng đỉnh)"))
    names = {"xu_huong": "Xu hướng", "dong_luong": "Động lượng", "suc_manh": "Sức mạnh (ADX)",
             "dong_tien": "Dòng tiền (KL)"}
    print(f"\nĐiểm TA: {_num(s.total)}/100 — {s.rating}")
    print("  " + " · ".join(f"{names[g]} {_num(x)}" for g, x in s.groups.items()))
    return 0


def _plan(ind):
    from vnsa import analysis

    return analysis.plan(ind)


def _print_zone(z) -> None:
    if z is None:
        return
    print(f"  Giá đóng cửa {z.close:,.0f} · VWAP20 {z.vwap20:,.0f} · neo {z.anchor_source} "
          f"{z.anchor:,.0f}")
    if not _nan(z.fair_value):
        print(f"  Giá trị hợp lý {z.fair_value:,.0f} − biên an toàn {z.margin_of_safety:.0%} "
              f"= trần giá mua {z.value_cap:,.0f}")
    if z.status == "CHỜ GIÁ":
        print(f"  ⇒ CHỜ GIÁ: chỉ mua khi giá về ≤ {z.high:,.0f} ({z.discount:+.1%} so với giá hiện tại)")
    else:
        print(f"  ⇒ VÙNG MUA {z.low:,.0f} – {z.high:,.0f} ({z.low / z.close - 1:+.1%} đến "
              f"{z.discount:+.1%} so với giá đóng cửa)")
        print(f"    Gợi ý: 2 lệnh LO, ½ khối lượng ở {z.high:,.0f} và ½ ở {z.low:,.0f} "
              f"→ giá bình quân dự kiến {z.mid:,.0f}")
    for n in z.notes:
        print(f"  ! {n}")


def cmd_plan(args) -> int:
    from vnsa.ta.score import ta_score

    sym = args.symbol.upper()
    loaded = _load(sym, need_fin=False)
    if loaded is None:
        return 1
    ind = _indicators(loaded[1])
    try:
        p = _plan(ind)
    except ValueError as e:
        print(f"  ✗ {sym}: {e}")
        return 1
    s = ta_score(ind, settings()["ta_score"]["weights"]) if len(ind) >= 220 else None
    print(f"{sym} — kế hoạch mua tham khảo (dữ liệu đến phiên {ind['date'].iloc[-1]:%d/%m/%Y})")
    _print_zone(p.zone)
    print(f"  Giá vào   {p.entry:,.0f} (giá bình quân dự kiến của vùng mua, không phải giá đóng cửa)")
    print(f"  Cắt lỗ    {p.stop:,.0f} ({p.stop_pct:+.1%}; {p.stop_source}; ATR14 = {p.atr:,.0f})")
    for i, t in enumerate(p.targets, 1):
        print(f"  Mục tiêu {i} {t.price:,.0f} ({t.price / p.entry - 1:+.1%}; {t.reward_risk:.1f}R; "
              f"{t.source})")
    print(f"  Khối lượng {p.shares:,} cp ≈ {p.position_value / 1e6:,.1f} triệu "
          f"({p.position_value / p.capital:.0%} vốn {p.capital / 1e6:,.0f} triệu); "
          f"lỗ tối đa {p.max_loss / 1e6:,.2f} triệu (giới hạn {p.risk_budget / 1e6:,.2f} triệu)")
    if s is not None:
        print(f"  Điểm TA {_num(s.total)}/100 ({s.rating})"
              + ("; xu hướng chưa ủng hộ, cân nhắc chờ" if s.total < 45 else ""))
    for n in p.notes:
        print(f"  ! {n}")
    print("  Đây là kế hoạch tính theo quy tắc, không phải khuyến nghị đầu tư.")
    return 0


def cmd_risk(args) -> int:
    from vnsa import analysis

    sym = args.symbol.upper()
    loaded = _load(sym, need_fin=False)
    if loaded is None:
        return 1
    prices = loaded[1]
    try:
        r = analysis.risk(prices, [args.days] if args.days else None, args.paths)
    except ValueError as e:
        print(f"  ✗ {sym}: {e}")
        return 1
    from vnsa import market

    fit, sim, band = r.fit, r.sim, market.band(sym)
    print(f"{sym} — rủi ro, giá {r.price:,.0f} đ ({prices['date'].iloc[-1]:%d/%m/%Y})")
    print(f"\nGARCH(1,1) trên {len(fit.resid)} phiên: alpha {fit.alpha:.3f}, beta {fit.beta:.3f}, "
          f"bền vững {fit.persistence:.3f} (cú sốc giảm một nửa sau ~{fit.half_life:.1f} phiên)"
          + ("" if fit.converged else " — ! tối ưu chưa hội tụ"))
    print(f"  Biến động năm: hiện tại {fit.current_vol_annual:.1%} · dài hạn "
          f"{fit.long_run_vol_annual:.1%} · lịch sử 1 năm {r.hist_vol:.1%}")
    print("\nVaR 1 ngày (lỗ tối đa với xác suất tin cậy):")
    for q, v in r.var.items():
        print(f"  {q:.0%}: lịch sử {v['hist']:.2%} (CVaR {v['cvar']:.2%}) · GARCH {v['garch']:.2%}"
              f" (phân phối chuẩn {v['garch_normal']:.2%})")
    print(f"\nMonte Carlo {sim.paths.shape[0]:,} đường (FHS, chặn biên độ ±{band:.0%}/phiên):")
    for h in r.horizons:
        pc = sim.percentiles(h)
        print(f"  Sau {h} phiên: " + " · ".join(f"P{k} {v:,.0f}" for k, v in pc.items()))
        print(f"      xác suất cao hơn giá hiện tại {sim.prob_above(day=h):.0%} · VaR "
              + " · ".join(f"{q:.0%} {sim.var(q, h):.1%}" for q in r.var))
    p, ft = r.plan, r.first_touch
    if p and ft:
        print(f"\nTheo kế hoạch (cắt lỗ {p.stop:,.0f}, mục tiêu 1 {p.targets[0].price:,.0f}) "
              f"trong {max(r.horizons)} phiên: chạm mục tiêu trước {ft['target']:.0%} · chạm cắt "
              f"lỗ trước {ft['stop']:.0%} · chưa chạm {ft['neither']:.0%}")
        if p.shares:
            q = next(iter(r.var))
            loss = r.var[q]["hist"] * p.position_value
            print(f"  Vị thế {p.shares:,} cp ({p.position_value / 1e6:,.1f} triệu): "
                  f"VaR {q:.0%} 1 ngày ≈ {loss / 1e6:,.2f} triệu")
    print("  Mô phỏng dựa trên biến động quá khứ, không dự báo được sự kiện bất thường.")
    return 0


def cmd_recommend(args) -> int:
    from vnsa import analysis

    symbols = [s.upper() for s in args.symbols] or settings()["watchlist"]["symbols"]
    rows, plans, code = [], [], 0
    for sym in symbols if getattr(args, "update", False) else []:
        rep = analysis.update(sym)
        print(f"  … {sym}: cập nhật dữ liệu" + "".join(f"\n    ✗ {e}" for e in rep.errors))
    for sym in symbols:
        try:
            rec, p = analysis.recommendation(sym)
        except (analysis.DataMissing, ValueError) as e:
            print(f"  ✗ {e}")
            code = 1
            continue
        rows.append(rec)
        plans.append(p)
    if not rows:
        return code
    analysis.record_signals(list(zip(rows, plans, strict=True)))
    x0 = rows[0].inputs
    print(f"Khuyến nghị theo quy tắc — giá đóng cửa {x0.date:%d/%m/%Y} · VNINDEX {x0.regime}")
    zones = {r.symbol: p.zone for r, p in zip(rows, plans, strict=True) if p is not None}
    print(f"{'Mã':<6}{'Giá':>9}{'FA':>5}{'TA':>5}{'LN kỳ vọng':>12}  {'Xếp hạng':<14}"
          f"{'Vùng mua':<17}Thời điểm")
    from vnsa.recommend.engine import RATINGS

    def rank(r):
        er = r.expected_return
        return (RATINGS.index(r.action) if r.action in RATINGS else 9, 9 if _nan(er) else -er)

    for r in sorted(rows, key=rank):
        x, z = r.inputs, zones.get(r.symbol)
        zone = ("—" if z is None else f"≤ {z.high:,.0f} (chờ)" if z.status == "CHỜ GIÁ"
                else f"{z.low:,.0f}–{z.high:,.0f}")
        print(f"{r.symbol:<6}{x.price:>9,.0f}{_num(x.fa_score):>5}{_num(x.ta_score):>5}"
              f"{_pct(r.expected_return, 0):>12}  {r.action:<14}{zone:<17}{r.timing}")
    print()
    for r in rows:
        print(f"{r.symbol} — {r.action} · {r.timing}: " + "; ".join(r.reasons))
    print("\nĐây là kết quả sàng lọc theo quy tắc trong settings.toml, không phải lời khuyên đầu tư.")
    return code


def cmd_backtest(args) -> int:
    """So sánh các quy tắc thời điểm (điểm TA, + VNINDEX, quy tắc đang dùng) với mua & giữ."""
    from vnsa import analysis

    sym = args.symbol.upper()
    if _load(sym, need_fin=False) is None:
        return 1
    d = analysis.load(sym, need_fin=False)
    try:
        results = analysis.backtests(d, args.entry, args.exit)
    except ValueError as e:
        print(f"  ✗ {sym}: {e}")
        return 1
    cfg = analysis.backtest_config(sym)
    entry = cfg["entry_score"] if args.entry is None else args.entry
    exit_ = cfg["exit_score"] if args.exit is None else args.exit
    first = next(iter(results.values()))
    d0, d1 = first.equity.index[0], first.equity.index[-1]
    print(f"{sym} — backtest {d0:%d/%m/%Y} → {d1:%d/%m/%Y} (khớp mở cửa phiên sau, không mua ở giá "
          f"trần / không bán khi nằm sàn, T+2, phí {cfg['fee']:.2%}/chiều + thuế bán "
          f"{cfg['sell_tax']:.1%}; điểm TA: mua ≥ {entry:g}, bán < {exit_:g}, cắt lỗ "
          f"{cfg['atr_stop']:g} ATR)")
    cols = [(name, r.stats) for name, r in results.items()]
    cols.append(("Mua & giữ", first.benchmark_stats))
    width = max(14, *(len(n) + 2 for n, _ in cols))
    print(f"{'':<22}" + "".join(f"{c:>{width}}" for c, _ in cols))
    for key, label in (("total", "Lợi nhuận cả kỳ"), ("cagr", "Lợi nhuận/năm (CAGR)"),
                       ("max_drawdown", "Sụt giảm tối đa")):
        print(f"{label:<22}" + "".join(f"{st[key]:>{width}.1%}" for _, st in cols))
    print(f"{'Sharpe (r_f = 0)':<22}" + "".join(f"{st['sharpe']:>{width}.2f}" for _, st in cols))
    for name, st in cols[:-1]:
        print(f"{name}: {st['trades']} lệnh, thắng {_pct(st['win_rate'], 0)}, lãi/lỗ TB mỗi lệnh "
              f"{_pct(st['avg_trade'])}, thời gian cầm cổ phiếu {st['exposure']:.0%}")
    rule = settings()["recommend"].get("timing_rule", "ta")
    last_name = list(results)[-1]
    print(f"Lệnh gần nhất ({last_name}; trục thời điểm đang dùng: {rule}):")
    for t in results[last_name].trades[-args.last:]:
        print(f"  {t.entry_date:%d/%m/%Y} mua {t.entry:,.0f} → {t.exit_date:%d/%m/%Y} bán "
              f"{t.exit:,.0f} ({t.ret:+.1%}, {t.reason})")
    print("Kết quả quá khứ không đảm bảo tương lai; chưa tính trượt giá. Trục xếp hạng (giá trị hợp "
          "lý) không backtest được trung thực với 4 năm BCTC miễn phí: theo dõi bằng review-signals.")
    return 0


def cmd_analyze(args) -> int:
    """Một mã bất kỳ: tự tải dữ liệu còn thiếu/cũ rồi in thẻ báo cáo (report/card.py);
    --detail in thêm từng phần chi tiết."""
    from vnsa import analysis, market
    from vnsa.report.card import build_card, format_card

    code = 0
    for i, raw in enumerate(args.symbols):
        sym = raw.upper().strip()
        if i:
            print("\n" + "═" * 78 + "\n")
        if not args.offline:
            print(f"{sym} — cập nhật dữ liệu (giá: CafeF; BCTC, sàn, ngành: vnstock/VCI)")
            rep = analysis.update(sym, args.refresh, args.full, log=print if args.verbose else None)
            print(rep)
            if not rep.ok:
                print(f"  ✗ {sym}: chưa có giá nên không phân tích được. Kiểm tra lại mã.")
                code = 1
                continue
        prof = market.profile(sym)
        where = (f"sàn {prof['exchange']} (biên độ ±{market.band(sym):.0%})" if prof["known"] else
                 f"chưa rõ sàn, tạm tính như {prof['exchange']} (±{market.band(sym):.0%})")
        print("\n" + " · ".join(x for x in (f"{sym}", prof["name"], where, prof["industry"]) if x))
        try:
            a = analysis.assess(sym)
        except (analysis.DataMissing, ValueError) as e:
            print(f"  ✗ {e}")
            code = 1
            continue
        analysis.record_signals([(a.rec, a.plan)])
        print("\n" + format_card(build_card(a)))
        if not args.detail:
            continue
        fin_note = market.financial_note(sym)
        one = argparse.Namespace(symbol=sym, symbols=[sym], days=None, paths=None, growth=None,
                                 wacc=None, update=False)
        sections = [("KỸ THUẬT", cmd_ta), ("KẾ HOẠCH GIAO DỊCH", cmd_plan), ("RỦI RO", cmd_risk)]
        if analysis.load(sym, need_fin=False).fin.empty:
            fin_note = None
            sections.append(("TÀI CHÍNH", lambda a: print(
                "  Chưa có BCTC (nguồn vnstock lỗi hoặc chưa cài): bỏ qua chỉ số, điểm FA và "
                f"định giá. Thử lại: analyze {a.symbol} --refresh") or 0))
        else:
            sections += ([("CHỈ SỐ TÀI CHÍNH", cmd_ratios)] if fin_note else [])
            sections += [("CHẤT LƯỢNG TÀI CHÍNH", cmd_fa_score), ("ĐỊNH GIÁ", cmd_value)]
        for title, fn in sections:
            print(f"\n── {title} " + "─" * (74 - len(title)))
            try:
                if fn(one):
                    code = 1
            except Exception as e:  # noqa: BLE001 - một phần lỗi không làm dừng cả báo cáo
                print(f"  ✗ không tính được: {e}")
                code = 1
        if fin_note:
            print(f"  ! {fin_note}")
    return code


def cmd_review_signals(args) -> int:
    """Chấm lại khuyến nghị đã lưu sau N phiên (recommend/review.py)."""
    from vnsa import analysis

    for sym in args.update or []:
        analysis.update(sym)
    ev, summary, n = analysis.review(args.horizons)
    print(f"Đã lưu {n} khuyến nghị (bảng signals, bản sao {path('signals_csv').name}).")
    if summary.empty:
        print("Chưa có khuyến nghị nào đủ số phiên để chấm. Chạy recommend/analyze đều đặn; sau "
              f"{min(args.horizons or settings()['signals']['horizons'])} phiên sẽ có kết quả đầu tiên.")
        return 0
    print(f"\n{'Kỳ':>4} {'KN':<9}{'Mẫu':>5}{'Lãi/lỗ TB':>11}{'Vượt VNI':>10}{'Thắng':>7}"
          f"{'Chạm MT trước':>15}{'Khớp vùng':>11}{'Lãi từ vùng':>13}")
    for r in summary.itertuples():
        print(f"{r.horizon:>4} {r.action:<9}{r.n:>5}{_pct(r.avg_ret):>11}{_pct(r.avg_excess):>10}"
              f"{_pct(r.win_rate, 0):>7}{_pct(r.target_first, 0):>15}{_pct(r.zone_fill, 0):>11}"
              f"{_pct(r.avg_ret_from_zone):>13}")
    print("\nMUA/KHẢ QUAN tốt khi 'Vượt VNI' dương, KÉM KHẢ QUAN/BÁN tốt khi âm. Dưới ~30 mẫu chỉ "
          "để tham khảo.")
    if args.detail:
        print("\n" + ev.to_string(index=False))
    return 0


def cmd_study_timing(args) -> int:
    """Kiểm định các quy tắc thời điểm trên nhiều mã, nhiều năm (recommend/study.py)."""
    from vnsa import analysis
    from vnsa.recommend.study import format_study

    cfg = settings()["study"]
    syms = [s.upper() for s in args.symbols] or cfg["universe"]
    if not args.no_fetch:
        from vnsa.data.autofetch import Updater

        print(f"Tải giá {args.years or cfg['years']} năm cho {len(syms)} mã + VNINDEX ...")
        for step in Updater().long_prices(syms, args.years or cfg["years"]):
            print(step)
    try:
        res = analysis.timing_study(syms, cfg["split"], cfg["start"])
    except ValueError as e:
        print(f"  ✗ {e}")
        return 1
    text = format_study(res)
    print("\n" + text)
    out = path("warehouse").parent / "study"
    out.mkdir(parents=True, exist_ok=True)
    (out / "timing_study.txt").write_text(text, encoding="utf-8")
    res.table.to_csv(out / "timing_study.csv", index=False)
    print(f"\nĐã lưu {out / 'timing_study.txt'} và timing_study.csv")
    return 0


def cmd_history(args) -> int:
    """Lưu/nạp lịch sử khuyến nghị và BCTC năm/quý tích luỹ (history.py)."""
    from vnsa import history

    if args.action == "export":
        counts = history.export_history(args.dir)
        print(f"Đã ghi lịch sử vào {args.dir}: " + ", ".join(f"{k} {v} dòng" for k, v in counts.items()))
        return 0
    if args.git:
        import subprocess

        branch = args.git.split("/", 1)[-1]
        subprocess.run(["git", "fetch", "-q", "origin", f"{branch}:refs/remotes/origin/{branch}"],
                       cwd=PROJECT_ROOT, check=False)
    counts = history.import_history(None if args.git else args.dir, git_ref=args.git)
    where = args.git or args.dir
    print(f"Đã nạp từ {where} (chỉ thêm dòng kho chưa có): "
          + ", ".join(f"{k} {v} dòng" for k, v in counts.items()))
    return 0


def cmd_check(args) -> int:
    from vnsa import market
    from vnsa.data.store import Store

    band = market.band(args.symbol)
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

    pfm = sub.add_parser("fetch-macro", help="Tải số liệu vĩ mô (FRED, World Bank, tỷ giá) "
                         "và nạp số nhập tay")
    pfm.add_argument("series", nargs="*", help="Bỏ trống = mọi chuỗi trong config/macro_map.toml")
    pfm.add_argument("--years", type=int, default=10)
    pfm.add_argument("--pause", type=float, default=0.3, help="Giây nghỉ giữa các lần gọi")
    pfm.set_defaults(func=cmd_fetch_macro)

    pmc = sub.add_parser("macro", help="Số liệu vĩ mô mới nhất trong kho và biến động")
    pmc.set_defaults(func=cmd_macro)

    pr = sub.add_parser("ratios", help="Bộ chỉ số tài chính năm từ BCTC trong kho")
    pr.add_argument("symbol")
    pr.set_defaults(func=cmd_ratios)

    pv = sub.add_parser("value", help="Định giá: WACC, DCF-FCFF, P/E, P/B, độ nhạy")
    pv.add_argument("symbol")
    pv.add_argument("--growth", type=float, help="Tăng trưởng FCFF giai đoạn 1, vd 0.12 (mặc định: "
                    "CAGR LNST cty mẹ, tối đa max_stage1_growth)")
    pv.add_argument("--wacc", type=float, help="Dùng WACC này thay vì tự tính, vd 0.13")
    pv.set_defaults(func=cmd_value)

    pfa = sub.add_parser("fa-score", help="Piotroski F-Score, Altman Z, điểm FA 0–100")
    pfa.add_argument("symbol")
    pfa.set_defaults(func=cmd_fa_score)

    pta = sub.add_parser("ta", help="Chỉ báo kỹ thuật, hỗ trợ/kháng cự, điểm TA 0–100")
    pta.add_argument("symbol")
    pta.set_defaults(func=cmd_ta)

    ppl = sub.add_parser("plan", help="Kế hoạch mua: giá vào, cắt lỗ theo ATR, mục tiêu, khối lượng")
    ppl.add_argument("symbol")
    ppl.set_defaults(func=cmd_plan)

    prk = sub.add_parser("risk", help="GARCH, Monte Carlo P10–P90, VaR/CVaR")
    prk.add_argument("symbol")
    prk.add_argument("--days", type=int, help="Số phiên mô phỏng (mặc định: horizons trong settings)")
    prk.add_argument("--paths", type=int, help="Số đường mô phỏng (mặc định: n_paths trong settings)")
    prk.set_defaults(func=cmd_risk)

    prc = sub.add_parser("recommend", help="Xếp hạng MUA…BÁN và thời điểm mua cho danh sách mã")
    prc.add_argument("symbols", nargs="*", help="Bỏ trống = danh sách theo dõi trong settings.toml")
    prc.add_argument("--update", action="store_true",
                     help="Tự tải/cập nhật dữ liệu từng mã trước (như analyze)")
    prc.set_defaults(func=cmd_recommend)

    pan = sub.add_parser("analyze", help="Mã bất kỳ: tự tải dữ liệu còn thiếu rồi in thẻ báo cáo")
    pan.add_argument("symbols", nargs="+", help="Một hoặc nhiều mã HOSE/HNX/UPCoM, vd: SHS VCB ACV")
    pan.add_argument("--detail", action="store_true",
                     help="In thêm chi tiết: chỉ báo, kế hoạch, rủi ro, BCTC, định giá từng bước")
    pan.add_argument("--brief", action="store_true", help=argparse.SUPPRESS)  # giữ tương thích
    pan.add_argument("--refresh", action="store_true",
                     help="Tải lại BCTC và danh sách sàn/ngành dù chưa cũ")
    pan.add_argument("--full", action="store_true", help="Tải lại toàn bộ giá thay vì phần mới")
    pan.add_argument("--offline", action="store_true", help="Không tải gì, chỉ dùng dữ liệu trong kho")
    pan.add_argument("--verbose", action="store_true", help="In tiến trình tải từng đoạn của CafeF")
    pan.set_defaults(func=cmd_analyze)

    pbt = sub.add_parser("backtest", help="Backtest chiến lược theo điểm TA")
    pbt.add_argument("symbol")
    pbt.add_argument("--entry", type=float, help="Ngưỡng điểm TA để mua (mặc định trong settings)")
    pbt.add_argument("--exit", type=float, help="Ngưỡng điểm TA để bán (mặc định trong settings)")
    pbt.add_argument("--last", type=int, default=10, help="Số lệnh gần nhất in ra")
    pbt.set_defaults(func=cmd_backtest)

    prv = sub.add_parser("review-signals", help="Chấm lại các khuyến nghị đã lưu sau N phiên")
    prv.add_argument("--horizons", type=int, nargs="+", help="Số phiên, vd: 5 20 60")
    prv.add_argument("--update", nargs="*", metavar="MÃ",
                     help="Cập nhật giá các mã này trước khi chấm (để có phiên mới)")
    prv.add_argument("--detail", action="store_true", help="In từng khuyến nghị")
    prv.set_defaults(func=cmd_review_signals)

    pst = sub.add_parser("study-timing", help="Kiểm định quy tắc thời điểm trên nhiều mã, 10 năm")
    pst.add_argument("symbols", nargs="*", help="Bỏ trống = [study] universe trong settings")
    pst.add_argument("--years", type=int, help="Số năm giá cần có (mặc định [study] years)")
    pst.add_argument("--no-fetch", action="store_true", help="Không tải, chỉ dùng giá trong kho")
    pst.set_defaults(func=cmd_study_timing)

    ph = sub.add_parser("history", help="Lưu/nạp lịch sử khuyến nghị và BCTC tích luỹ")
    ph.add_argument("action", choices=["export", "import"])
    ph.add_argument("dir", nargs="?", default="history", help="Thư mục CSV (mặc định history/)")
    ph.add_argument("--git", help="Nạp từ nhánh Git, ví dụ origin/signals (workflow hằng ngày)")
    ph.set_defaults(func=cmd_history)

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
