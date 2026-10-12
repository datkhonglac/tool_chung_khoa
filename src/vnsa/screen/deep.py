"""Tầng 2 của sàng lọc: phân tích đầy đủ từng mã đã qua lọc thanh khoản.

Hai lượt để so sánh ngành công bằng:
  1. Tải dữ liệu cho MỌI mã (autofetch: đủ `[auto] years` năm giá, BCTC năm/quý, hồ sơ, cổ tức;
     bỏ khối ngoại và tin tức vì không vào điểm) — để P/E, P/B ngành của từng mã thấy đủ mã cùng ngành.
  2. analysis.assess() + thẻ báo cáo cho từng mã: đúng các số lệnh `analyze` in ra (xếp hạng hai
     trục, giá trị hợp lý, FA, TA, độ tin cậy, vùng mua, cắt lỗ, điểm vĩ mô ngành).
Mã lỗi được ghi lý do, không dừng cả lượt.

Tăng trưởng lợi nhuận (dùng cho điểm nhóm ngành), LNST công ty mẹ:
  - đủ 8 quý liền nhau: 4 quý gần nhất so với 4 quý trước đó ("TTM so cùng kỳ");
  - không thì TTM so với năm gần nhất nếu TTM mới hơn; không nữa thì năm gần nhất so với năm trước.
  Kỳ gốc ≤ 0 thì để trống (không tính được % tăng).
"""

from __future__ import annotations

import time
from collections.abc import Callable

import pandas as pd

from vnsa.config import settings

NAN = float("nan")


def _consecutive(q: pd.DataFrame) -> pd.DataFrame:
    """Các quý liền nhau tính từ quý gần nhất trở về trước."""
    q = q.sort_values(["year", "quarter"]).reset_index(drop=True)
    idx = (q["year"] * 4 + q["quarter"]).to_numpy()
    start = len(q) - 1
    while start > 0 and idx[start] - idx[start - 1] == 1:
        start -= 1
    return q.iloc[start:]


def earnings_growth(fin: pd.DataFrame, quarterly: pd.DataFrame | None, current=None
                    ) -> tuple[float, str]:
    """(tăng trưởng LNST công ty mẹ, cơ sở so sánh) — xem đầu file."""
    if quarterly is not None and len(quarterly) >= 8:
        q = _consecutive(quarterly.dropna(subset=["net_income_parent"]))
        if len(q) >= 8:
            now, before = q["net_income_parent"].iloc[-4:].sum(), q["net_income_parent"].iloc[-8:-4].sum()
            last = q.iloc[-1]
            basis = f"TTM Q{int(last['quarter'])}/{int(last['year'])} so cùng kỳ"
            return (float(now / before - 1) if before > 0 else NAN), basis
    annual = (fin.sort_values("year").dropna(subset=["net_income_parent"])
              if fin is not None and not fin.empty else pd.DataFrame())
    if (current is not None and str(current.basis).startswith("TTM") and len(annual)):
        base = annual["net_income_parent"].iloc[-1]
        basis = f"{current.basis} so năm {int(annual['year'].iloc[-1])}"
        return (float(current.net_income_parent / base - 1) if base > 0 else NAN), basis
    if len(annual) >= 2:
        now, base = annual["net_income_parent"].iloc[-1], annual["net_income_parent"].iloc[-2]
        basis = f"năm {int(annual['year'].iloc[-1])} so {int(annual['year'].iloc[-2])}"
        return (float(now / base - 1) if base > 0 else NAN), basis
    return NAN, ""


def assessment_row(a, card) -> dict:
    """Các số của một mã (Assessment + thẻ báo cáo) ghi vào bảng screen."""
    d, rec, p = a.data, a.rec, a.plan
    zone = p.zone if p is not None else None
    growth, basis = earnings_growth(d.fin, d.quarterly, d.current)
    cur = d.current
    return {
        "model": (d.profile or {}).get("model") or "general",
        "action": rec.action, "expected_return": rec.expected_return, "score": rec.score,
        "fair_value": a.fair.value if a.fair is not None else NAN,
        "upside": a.fair.upside if a.fair is not None else NAN,
        "dividend_yield": card.dividend_yield, "fa_score": a.fa_score, "ta_score": a.ta.total,
        "quant": a.quant, "confidence": card.confidence,
        "eps_growth": growth, "growth_basis": basis,
        "roe": cur.roe if cur is not None else NAN, "pe": a.pe, "pb": a.pb, "z_em": a.z_em,
        "macro_industry": rec.inputs.macro_industry,
        "zone_status": zone.status if zone is not None else "",
        "buy_low": zone.low if zone is not None else NAN,
        "buy_high": zone.high if zone is not None else NAN,
        "stop": p.stop if p is not None else NAN,
        "red_flags": "; ".join(card.red_flags),
        "fin_basis": card.basis,
    }


def _ticker(log: Callable[[str], None] | None, what: str, n: int, every: int = 25):
    t0 = time.monotonic()

    def tick(i: int, errors: int) -> None:
        if log and (i % every == 0 or i == n):
            log(f"  … {what} {i}/{n} mã · {(time.monotonic() - t0) / 60:.0f} phút · {errors} lỗi")
    return tick


def fetch_all(symbols: list[str], updater=None, log: Callable[[str], None] | None = print,
              minutes: float | None = None, workers: int = 1) -> dict[str, str]:
    """Lượt 1: tải dữ liệu cho mọi mã (theo thứ tự truyền vào — mã thanh khoản cao trước). Trả
    {mã: lỗi khi tải} để ghi chú; mã vẫn còn giá trong kho thì lượt 2 vẫn phân tích (thiếu BCTC thì
    thành KHÔNG XẾP HẠNG, như lệnh analyze). Quá `minutes` phút thì dừng tải (hạn mức vnstock chế độ
    Khách ~140 mã/giờ); mã chưa tải ghi "chưa tải lượt này" — kho giữ lại thì lượt sau tải tiếp.
    `workers` > 1: trước đó tải song song đủ `[auto] years` năm GIÁ cho mọi mã (CafeF chỉ trả 30
    phiên mỗi trang nên 5 năm ≈ 45 lần gọi mỗi mã); thời gian `minutes` chỉ tính phần BCTC."""
    from vnsa.data.autofetch import Updater

    if updater is None:
        cfg = {**settings()["auto"], "news": False, "flows": False}
        updater = Updater(cfg=cfg)
    errors: dict[str, str] = {}
    if workers > 1 and symbols:
        years = updater.cfg.get("years", 5)
        t_px = time.monotonic()
        steps = updater.long_prices(symbols, years, workers=workers)
        if log:
            bad = sum(st.status == "error" for st in steps)
            log(f"  … giá {years} năm cho {len(symbols)} mã ({workers} luồng): "
                f"{(time.monotonic() - t_px) / 60:.0f} phút, {bad} lỗi")
    tick = _ticker(log, "tải dữ liệu", len(symbols))
    t0 = time.monotonic()
    for i, sym in enumerate(symbols, 1):
        if minutes is not None and (time.monotonic() - t0) / 60 > minutes:
            rest = symbols[i - 1:]
            for s in rest:
                errors[s] = "chưa tải BCTC lượt này (hết thời gian, lượt sau tải tiếp)"
            if log:
                log(f"  … dừng tải sau {minutes:.0f} phút: còn {len(rest)} mã để lượt sau")
            break
        try:
            rep = updater.ensure(sym)
            if rep.errors or not rep.ok:
                errors[sym] = "; ".join(rep.errors) or "không có giá"
        except Exception as e:  # noqa: BLE001 - một mã lỗi không dừng cả lượt
            errors[sym] = f"{type(e).__name__}: {e}"
        tick(i, len(errors))
    return errors


def assess_all(symbols: list[str], log: Callable[[str], None] | None = print
               ) -> tuple[dict[str, dict], dict[str, str]]:
    """Lượt 2: assess() + thẻ báo cáo từng mã. Trả ({mã: số liệu}, {mã: lỗi})."""
    from vnsa import analysis
    from vnsa.report.card import build_card

    rows, errors = {}, {}
    tick = _ticker(log, "phân tích", len(symbols))
    for i, sym in enumerate(symbols, 1):
        try:
            a = analysis.assess(sym)
            rows[sym] = assessment_row(a, build_card(a))
        except Exception as e:  # noqa: BLE001 - một mã lỗi không dừng cả lượt
            errors[sym] = f"{type(e).__name__}: {str(e)[:200]}"
        tick(i, len(errors))
    return rows, errors
