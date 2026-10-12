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


PT = [("pt_momentum", "ĐL"), ("pt_breadth", "ĐR"), ("pt_earnings", "LN"), ("pt_valuation", "ĐG"),
      ("pt_quality", "CL"), ("pt_macro", "VM")]
LEGEND = ("ĐL động lượng (RS so VNINDEX) · ĐR độ rộng (giá trên MA50/MA200) · LN tăng trưởng lợi "
          "nhuận · ĐG định giá (lợi nhuận kỳ vọng) · CL chất lượng (điểm FA) · VM vĩ mô")
RATING_ORDER = ["MUA", "KHẢ QUAN", "NẮM GIỮ", "KÉM KHẢ QUAN", "BÁN", "KHÔNG XẾP HẠNG"]


def _pt(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:.0f}"


def _weights_line(cfg: dict) -> str:
    from vnsa.screen.industry import DEFAULT_WEIGHTS

    w = {**DEFAULT_WEIGHTS, **cfg.get("industry_weights", {})}
    names = {"momentum": "động lượng", "breadth": "độ rộng", "earnings": "lợi nhuận",
             "valuation": "định giá", "quality": "chất lượng", "macro": "vĩ mô"}
    return " · ".join(f"{names[k]} {vn_num(v * 100)}%" for k, v in w.items() if v)


def format_industry(res: ScreenResult, top_groups: int | None = None) -> str:
    cfg = screen_cfg()
    t, ind = res.table, res.industry
    top_groups = top_groups or cfg.get("top_industries", 5)
    analysed = t[t["stage"] >= 2]
    rated = ind[ind["score"].notna()] if ind is not None and not ind.empty else pd.DataFrame()
    lines = [
        f"SÀNG LỌC TOÀN THỊ TRƯỜNG — tầng 1–3 · phiên {res.as_of:%d/%m/%Y}",
        (f"Vũ trụ {vn_num(len(t))} mã → qua thanh khoản {vn_num(len(res.passed(1)))} → phân tích "
         f"được {vn_num(len(analysed))} → {vn_num(len(rated))} nhóm ngành được chấm"),
    ]
    if len(res.passed(1)) < len(t):
        lines.append(f"Loại ở tầng 1: {_counter(res.rejected(1))}")
    bad = t[(t["stage"] == 1) & t["reason"].fillna("").str.startswith("phân tích lỗi")]
    if len(bad):
        lines.append(f"Lỗi phân tích ({len(bad)} mã): "
                     + " · ".join(f"{r.symbol} ({r.reason[15:70]})" for r in bad.head(5).itertuples())
                     + (" …" if len(bad) > 5 else ""))
    if rated.empty:
        lines += ["", "Chưa nhóm ngành nào đủ số liệu để chấm."]
        return "\n".join(lines)
    lines += ["", "NHÓM NGÀNH — điểm 0–100 (thành phần là thứ hạng giữa các nhóm)",
              f"Trọng số: {_weights_line(cfg)}",
              (f"  {'#':>2}  {'Nhóm ngành':<28}{'Mã':>4}{'Điểm':>6}  "
               + "".join(f"{lab:>4}" for _, lab in PT)
               + f"{'RS':>7}{'LN tăng':>9}{'LN kỳ vọng':>12}")]
    for r in ind.itertuples(index=False):
        if pd.isna(r.score):
            continue
        lines.append(
            f"  {int(r.rank):>2}  {r.group_name[:27]:<28}{int(r.members):>4}{_pt(r.score):>6}  "
            + "".join(f"{_pt(getattr(r, k)):>4}" for k, _ in PT)
            + f"{vn_pct(r.rs, 0, sign=True):>7}{vn_pct(r.growth, 0, sign=True):>9}"
            + f"{vn_pct(r.expected, 0, sign=True):>12}")
    lines.append(f"  {LEGEND}")
    lines.append("  RS, LN tăng, LN kỳ vọng: trung vị các mã trong nhóm")
    skipped = ind[ind["score"].isna()] if ind is not None else pd.DataFrame()
    if len(skipped):
        lines.append("Không xếp hạng: " + " · ".join(
            f"{r.group_name} ({int(r.members)} mã)" for r in skipped.itertuples(index=False)))
    lines += ["", f"{top_groups} NHÓM DẪN ĐẦU — xếp hạng các mã (chưa qua bộ lọc tầng 4)"]
    for r in rated.head(top_groups).itertuples(index=False):
        g = analysed[analysed["group_name"] == r.group_name]
        counts = g["action"].value_counts()
        lines.append(f"  {int(r.rank)}. {r.group_name}: " + " · ".join(
            f"{a} {counts[a]}" for a in RATING_ORDER if counts.get(a, 0)))
        good = g[g["action"].isin(["MUA", "KHẢ QUAN"])]
        good = good.assign(_o=good["action"].map(RATING_ORDER.index)).sort_values(
            ["_o", "score"], ascending=[True, False])
        if len(good):
            lines.append("     " + ", ".join(
                f"{x.symbol} {x.action} ({vn_pct(x.expected_return, 0, sign=True)})"
                for x in good.head(8).itertuples(index=False)))
    if res.csv:
        lines += ["", f"Đã lưu bảng screen, screen_industry trong kho và {res.csv}"]
    lines.append("Sàng lọc theo quy tắc trong config/settings.toml, không phải lời khuyên đầu tư.")
    return "\n".join(lines)


def format_taxonomy(tax: pd.DataFrame) -> str:
    """Ngành ICB cấp 2/3/4 → nhóm (kiểm tra quy tắc `match` trong config/macro_map.toml)."""
    lines = ["NHÓM NGÀNH ← ngành ICB (cấp 2 › cấp 3 › cấp 4: số mã)"]
    for name, g in tax.groupby("group", sort=True):
        lines.append(f"{name} — {int(g['n'].sum())} mã")
        for r in g.itertuples(index=False):
            lines.append(f"    {r.industry} › {r.industry3} › {r.industry4}: {r.n}")
    return "\n".join(lines)


def _industry_rows(ind: pd.DataFrame, n: int) -> list[str]:
    lines = [(f"  {'#':>2}  {'Nhóm ngành':<28}{'Mã':>4}{'Điểm':>6}  "
              + "".join(f"{lab:>4}" for _, lab in PT)
              + f"{'RS':>7}{'LN tăng':>9}{'LN kỳ vọng':>12}")]
    for r in ind[ind["score"].notna()].head(n).itertuples(index=False):
        lines.append(
            f"  {int(r.rank):>2}  {r.group_name[:27]:<28}{int(r.members):>4}{_pt(r.score):>6}  "
            + "".join(f"{_pt(getattr(r, k)):>4}" for k, _ in PT)
            + f"{vn_pct(r.rs, 0, sign=True):>7}{vn_pct(r.growth, 0, sign=True):>9}"
            + f"{vn_pct(r.expected, 0, sign=True):>12}")
    return lines


def _pick_rows(df: pd.DataFrame, group: bool = False) -> list[str]:
    from vnsa.screen.pick import why_picked

    head = (f"     {'Mã':<7}{'Xếp hạng':<10}{'Giá':>8}{'GT hợp lý':>11}{'LN kỳ vọng':>12}  "
            f"{'Vùng mua':<16}{'Cắt lỗ':>8}{'FA':>4}{'TA':>4}  Tin cậy")
    lines = [head]
    for r in df.itertuples(index=False):
        zone = (f"{vn_num(r.buy_low)}–{vn_num(r.buy_high)}" if not pd.isna(r.buy_low)
                else (r.zone_status or "—"))
        lines.append(f"     {r.symbol:<7}{r.action:<10}{vn_num(r.price):>8}{vn_num(r.fair_value):>11}"
                     f"{vn_pct(r.expected_return, 0, sign=True):>12}  {zone[:15]:<16}"
                     f"{vn_num(r.stop):>8}{_pt(r.fa_score):>4}{_pt(r.ta_score):>4}  "
                     f"{r.confidence or '—'}")
        extra = f"{r.group_name} · " if group else ""
        if isinstance(r.zone_status, str) and r.zone_status and r.zone_status != "MUA TRONG VÙNG":
            extra += f"{r.zone_status.lower()} · "
        lines.append(f"     {'':<7}{extra}{why_picked(r)}")
    return lines


def format_picks(res: ScreenResult) -> str:
    """Báo cáo cuối: nhóm ngành tiềm năng và mã tiềm năng trong từng nhóm (tầng 1–4)."""
    cfg = screen_cfg()
    t, ind = res.table, res.industry
    n_top = cfg.get("top_industries", 5)
    analysed = t[t["stage"] >= 2]
    picks = t[t["stage"] >= 4]
    rated = ind[ind["score"].notna()] if ind is not None and not ind.empty else pd.DataFrame()
    lines = [f"SÀNG LỌC TOÀN THỊ TRƯỜNG · phiên {res.as_of:%d/%m/%Y}"
             + (f" · {res.regime}" if res.regime else ""),
             (f"Vũ trụ {vn_num(len(t))} mã → thanh khoản {vn_num(len(res.passed(1)))} → phân tích "
              f"{vn_num(len(analysed))} → {vn_num(len(rated))} nhóm ngành → {min(n_top, len(rated))}"
              f" nhóm dẫn đầu → {vn_num(len(picks))} mã")]
    if len(res.passed(1)) < len(t):
        lines.append(f"Loại ở tầng 1 (thanh khoản): {_counter(res.rejected(1))}")
    pending = t["reason"].fillna("").str.startswith("chưa tải")
    if pending.any():
        lines.append(f"Chưa đủ dữ liệu: {int(pending.sum())} mã chưa tải BCTC lượt này (hạn mức "
                     "vnstock) — lượt sau tải tiếp; các mã này đang là KHÔNG XẾP HẠNG")
    bad = t["reason"].fillna("").str.startswith("phân tích lỗi")
    if bad.any():
        lines.append(f"Lỗi phân tích: {int(bad.sum())} mã (xem cột reason trong CSV)")
    if rated.empty:
        lines += ["", "Chưa nhóm ngành nào đủ số liệu để chấm."]
        return "\n".join(lines)
    lines += ["", "NHÓM NGÀNH TIỀM NĂNG — điểm 0–100 (thành phần là thứ hạng giữa các nhóm)",
              f"Trọng số: {_weights_line(cfg)}", *_industry_rows(ind, max(10, n_top)),
              f"  {LEGEND}", "  RS, LN tăng, LN kỳ vọng: trung vị các mã trong nhóm"]
    lines += ["", (f"MÃ TIỀM NĂNG TRONG {min(n_top, len(rated))} NHÓM DẪN ĐẦU "
                   f"(MUA/KHẢ QUAN, FA ≥ {cfg.get('min_fa', 50)}, không cờ đỏ)")]
    for r in rated.head(n_top).itertuples(index=False):
        macro = "" if pd.isna(r.macro) else f" · vĩ mô {r.macro:+.2f}".replace(".", ",").replace(
            "-", "−")
        lines.append(f"{int(r.rank)}. {r.group_name} — điểm {_pt(r.score)} · {int(r.members)} mã · "
                     f"RS {vn_pct(r.rs, 0, sign=True)} · LN tăng {vn_pct(r.growth, 0, sign=True)}"
                     f" · LN kỳ vọng {vn_pct(r.expected, 0, sign=True)}{macro}")
        g = picks[picks["group_name"] == r.group_name].sort_values("pick_rank")
        if len(g):
            lines += _pick_rows(g)
        else:
            members = t[(t["group_name"] == r.group_name) & (t["stage"] == 3)]
            why = Counter(str(x).split(":")[0] for x in members["reason"].dropna() if x)
            lines.append("     Không mã nào đạt" + (f" — {_counter(why)}" if why else ""))
    if res.outside is not None and len(res.outside):
        lines += ["", "MÃ MUA NGOÀI CÁC NHÓM DẪN ĐẦU (đạt mọi điều kiện, ngành chưa mạnh)"]
        lines += _pick_rows(res.outside, group=True)
    in_top = t[t["stage"] == 3]
    if len(in_top):
        why = Counter(str(x).split(":")[0] for x in in_top["reason"].dropna() if x)
        lines += ["", f"Không chọn trong nhóm dẫn đầu: {_counter(why)}"]
    if res.csv:
        lines += ["", f"Đã lưu bảng screen, screen_industry trong kho và {res.csv}"]
    lines += [("Thứ hạng nhóm ngành và mã là sàng lọc theo quy tắc trong config/settings.toml, chưa "
               "được kiểm định trên dữ liệu quá khứ; không phải lời khuyên đầu tư."),
              "Xem thẻ đầy đủ từng mã: python -m vnsa.cli analyze MÃ"]
    return "\n".join(lines)
