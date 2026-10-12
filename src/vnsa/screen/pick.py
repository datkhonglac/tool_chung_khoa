"""Tầng 4 của sàng lọc: chọn mã tiềm năng trong các nhóm ngành dẫn đầu.

Loại (theo thứ tự, ghi lý do đầu tiên gặp):
  - xếp hạng không phải MUA/KHẢ QUAN (gồm KHÔNG XẾP HẠNG: thiếu giá trị hợp lý);
  - điểm FA < `min_fa`; Altman Z'' < `min_z` (doanh nghiệp thường); có cờ đỏ BCTC;
  - độ tin cậy thấp hơn `min_confidence`.
Xếp: MUA trước KHẢ QUAN → điểm sắp xếp (FA 60% + TA 40%, như lệnh recommend) → sức mạnh giá 6 tháng.
Mỗi nhóm dẫn đầu lấy `per_industry` mã (stage 4). Mã MUA đạt mọi điều kiện nhưng nằm ngoài các nhóm
dẫn đầu được liệt kê riêng (`outside`) để không bỏ sót cổ phiếu tốt ở ngành đang yếu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GOOD = ["MUA", "KHẢ QUAN"]
CONFIDENCE = ["Thấp", "Trung bình", "Cao"]


def _fmt(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def exclusion(row, cfg: dict) -> str:
    """Lý do mã bị loại ở tầng 4 ("" = đạt)."""
    if row.get("action") not in GOOD:
        return f"xếp hạng {row.get('action') or 'chưa có'}"
    fa = row.get("fa_score")
    if fa is not None and not pd.isna(fa) and fa < cfg.get("min_fa", 50):
        return f"FA {fa:.0f} < {cfg.get('min_fa', 50)}"
    z = row.get("z_em")
    if (row.get("model") in (None, "general") and z is not None and not pd.isna(z)
            and z < cfg.get("min_z", 1.1)):
        return f"Altman Z'' {_fmt(z, 2)} < {_fmt(cfg.get('min_z', 1.1))}"
    flags = row.get("red_flags")
    if isinstance(flags, str) and flags.strip():
        return f"cờ đỏ BCTC: {flags.split(';')[0]}"
    need = cfg.get("min_confidence", "Thấp")
    conf = row.get("confidence")
    if need in CONFIDENCE and conf in CONFIDENCE and CONFIDENCE.index(conf) < CONFIDENCE.index(need):
        return f"độ tin cậy {conf} < {need}"
    return ""


def order(df: pd.DataFrame) -> pd.DataFrame:
    """MUA trước KHẢ QUAN, rồi điểm sắp xếp, rồi sức mạnh giá 6 tháng."""
    key = df["action"].map(lambda a: GOOD.index(a) if a in GOOD else 9)
    return (df.assign(_k=key, _s=df["score"].fillna(-1), _r=df["rs_6m"].fillna(-9))
            .sort_values(["_k", "_s", "_r"], ascending=[True, False, False])
            .drop(columns=["_k", "_s", "_r"]))


def pick(table: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(bảng screen đã đánh dấu tầng 4 + lý do + pick_rank, danh sách mã MUA ngoài nhóm dẫn đầu).
    table: bảng screen sau tầng 3 (stage 3 = thuộc nhóm dẫn đầu)."""
    t = table.copy()
    t["pick_rank"] = np.nan
    if "reason" in t:
        t["reason"] = t["reason"].astype(object)
    analysed = t["stage"] >= 2
    why = pd.Series("", index=t.index, dtype=object)
    why[analysed] = [exclusion(r, cfg) for r in t[analysed].to_dict("records")]
    per = cfg.get("per_industry", 5)
    top = t[(t["stage"] >= 3) & (why == "")]
    for _, g in top.groupby("group_name"):
        chosen = order(g).head(per)
        t.loc[chosen.index, "stage"] = 4
        t.loc[chosen.index, "pick_rank"] = np.arange(1, len(chosen) + 1)
    in_top = t["stage"] == 3
    t.loc[in_top & (why != ""), "reason"] = why[in_top & (why != "")]
    t.loc[in_top & (why == ""), "reason"] = f"ngoài {per} mã đầu nhóm"
    outside = t[(t["stage"] == 2) & (why == "") & (t["action"] == "MUA")]
    t.loc[(t["stage"] == 2) & (why != ""), "reason"] = why[(t["stage"] == 2) & (why != "")]
    t.loc[(t["stage"] == 2) & (why == ""), "reason"] = "đạt nhưng ngoài nhóm ngành dẫn đầu"
    return t, order(outside).head(cfg.get("outside_max", 10))


def why_picked(r) -> str:
    """Lý do chọn trong một dòng, từ các số đã tính."""
    def pct(x):
        return "—" if pd.isna(x) else f"{x * 100:+.0f}%".replace("-", "−")

    parts = [f"LN kỳ vọng {pct(r.expected_return)}", f"FA {r.fa_score:.0f}" if not pd.isna(
        r.fa_score) else "", f"RS 6T {pct(r.rs_6m)}"]
    if not pd.isna(r.eps_growth):
        parts.append(f"LNST {pct(r.eps_growth)} ({r.growth_basis})")
    return " · ".join(p for p in parts if p)
