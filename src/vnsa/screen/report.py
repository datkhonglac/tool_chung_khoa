"""In kết quả sàng lọc (lệnh screen)."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from vnsa.report.card import vn_num, vn_pct
from vnsa.screen.run import ScreenResult, screen_cfg

EXCHANGES = ["HOSE", "HNX", "UPCOM"]


def _by_exchange(df: pd.DataFrame) -> str:
    c = df["exchange"].value_counts()
    return " · ".join(f"{ex} {vn_num(c.get(ex, 0))}" for ex in EXCHANGES if c.get(ex, 0))


def _counter(c: Counter) -> str:
    return " · ".join(f"{k} {vn_num(v)}" for k, v in c.most_common())


def format_liquidity(res: ScreenResult, top: int = 20) -> str:
    cfg = screen_cfg()
    t = res.table
    ok = res.passed(1)
    lines = [
        f"SÀNG LỌC TOÀN THỊ TRƯỜNG — tầng 1 (thanh khoản) · phiên {res.as_of:%d/%m/%Y}",
        (f"Ngưỡng: GTGD bình quân 20 phiên ≥ {vn_num(cfg.get('min_value_bn', 3.0), 1)} tỷ · "
         f"giá ≥ {vn_num(cfg.get('min_price', 5000))}đ · ≥ {cfg.get('min_sessions', 200)} phiên "
         f"có giao dịch/năm · còn giao dịch trong {cfg.get('max_stale_days', 10)} ngày"),
    ]
    if res.listing:
        lines.append(f"Danh sách niêm yết: {res.listing.lstrip('✓·✗ ').split(': ', 1)[-1]}")
    if res.fetch:
        names = {"ok": "tải mới", "skip": "đã mới", "error": "lỗi"}
        lines.append("Tải giá: " + " · ".join(f"{names[k]} {vn_num(res.fetch.get(k, 0))}"
                                             for k in ("ok", "skip", "error") if res.fetch.get(k)))
    lines += [
        f"Vũ trụ {vn_num(len(t))} mã ({_by_exchange(t)}) → qua tầng 1: {vn_num(len(ok))} mã"
        + (f" ({_by_exchange(ok)})" if len(ok) else ""),
        f"Bị loại: {_counter(res.rejected(1))}" if len(ok) < len(t) else "Không mã nào bị loại",
    ]
    if len(ok):
        ind = (t.assign(qua=t["stage"] >= 1).groupby(t["industry"].fillna("(chưa rõ ngành)"))
               ["qua"].agg(["sum", "count"]).sort_values(["sum", "count"], ascending=False))
        lines += ["", "Theo ngành ICB cấp 2 — số mã qua / tổng:"]
        cells = [f"{name} {int(r['sum'])}/{int(r['count'])}" for name, r in ind.iterrows()
                 if r["sum"] > 0]
        for i in range(0, len(cells), 4):
            lines.append("  " + " · ".join(cells[i:i + 4]))
        head = ok.sort_values("value_20d", ascending=False).head(top)
        lines += ["", f"{min(top, len(ok))} mã GTGD bình quân 20 phiên lớn nhất:",
                  (f"  {'Mã':<8}{'Sàn':<7}{'Ngành':<26}{'Giá':>9}{'GTGD (tỷ)':>11}"
                   f"{'3 tháng':>9}{'6 tháng':>9}{'RS 6T':>8}")]
        for r in head.itertuples(index=False):
            ind_name = (r.industry or "")[:24]
            lines.append(f"  {r.symbol:<8}{r.exchange or '':<7}{ind_name:<26}"
                         f"{vn_num(r.price):>9}{vn_num(r.value_20d / 1e9, 1):>11}"
                         f"{vn_pct(r.ret_3m, 0, sign=True):>9}{vn_pct(r.ret_6m, 0, sign=True):>9}"
                         f"{vn_pct(r.rs_6m, 0, sign=True):>8}")
        lines.append("  RS 6T = sức mạnh tương đối 6 tháng so với VNINDEX")
    if res.fetch_errors:
        lines += ["", f"Lỗi tải giá ({len(res.fetch_errors)} mã, xem CSV cột reason):"]
        lines += [f"  ✗ {e[:110]}" for e in res.fetch_errors[:10]]
        if len(res.fetch_errors) > 10:
            lines.append(f"  … và {len(res.fetch_errors) - 10} mã khác")
    if res.csv:
        lines += ["", f"Đã lưu bảng screen trong kho và {res.csv}"]
    lines.append("Sàng lọc theo quy tắc trong config/settings.toml, không phải lời khuyên đầu tư.")
    return "\n".join(lines)
