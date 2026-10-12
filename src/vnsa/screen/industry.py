"""Tầng 3 của sàng lọc: chấm điểm nhóm ngành 0–100 từ các mã đã qua lọc thanh khoản.

Thành phần (trọng số ở [screen.industry_weights]); mỗi thành phần là trung vị/tỷ lệ của các mã trong
nhóm, rồi đổi ra THỨ HẠNG giữa các nhóm (nhóm tốt nhất 100, kém nhất 0) nên đơn vị khác nhau vẫn cộng
được:
  momentum  : trung vị sức mạnh tương đối so với VNINDEX (bình quân 3 và 6 tháng) — chỉ cần giá
  breadth   : tỷ lệ mã có giá trên MA50 và trên MA200 (bình quân hai tỷ lệ)
  earnings  : bình quân thứ hạng của (trung vị tăng trưởng LNST, tỷ lệ doanh nghiệp lợi nhuận tăng)
  valuation : trung vị lợi nhuận kỳ vọng (upside giá trị hợp lý + cổ tức) của các mã có định giá
  quality   : trung vị điểm FA
  macro     : điểm vĩ mô của nhóm (−1..+1) nếu độ phủ ≥ `macro_coverage`; không thì bỏ
Thành phần thiếu số thì bỏ và chia lại trọng số; nhóm còn dưới `min_weight` tổng trọng số hoặc
dưới `min_members` mã thì không xếp hạng.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENTS = ("momentum", "breadth", "earnings", "valuation", "quality", "macro")
DEFAULT_WEIGHTS = {"momentum": 0.30, "breadth": 0.10, "earnings": 0.25, "valuation": 0.15,
                   "quality": 0.10, "macro": 0.10}


def _median(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.median()) if len(s) else float("nan")


def _share(s: pd.Series, cond) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(cond(s).mean()) if len(s) else float("nan")


def group_raw(members: pd.DataFrame) -> dict:
    """Số liệu thô của một nhóm (các mã đã qua tầng 1, có cột tầng 2 nếu đã phân tích)."""
    rs = members[["rs_3m", "rs_6m"]].mean(axis=1, skipna=True)
    ma = members[["above_ma50", "above_ma200"]]
    out = {
        "members": len(members),
        "rs": _median(rs),
        "breadth": float(np.nanmean([_share(ma["above_ma50"], lambda x: x > 0.5),
                                     _share(ma["above_ma200"], lambda x: x > 0.5)]))
        if ma.notna().any().any() else float("nan"),
        "growth": _median(members.get("eps_growth", pd.Series(dtype=float))),
        "growing": _share(members.get("eps_growth", pd.Series(dtype=float)), lambda x: x > 0),
        "expected": _median(members.get("expected_return", pd.Series(dtype=float))),
        "fa": _median(members.get("fa_score", pd.Series(dtype=float))),
        "value_20d": float(pd.to_numeric(members["value_20d"], errors="coerce").sum()),
    }
    return out


def _rank100(s: pd.Series) -> pd.Series:
    """Thứ hạng giữa các nhóm: tốt nhất 100, kém nhất 0, giữ NaN; chỉ một nhóm có số thì 50."""
    ok = s.dropna()
    if len(ok) == 0:
        return s * np.nan
    if len(ok) == 1:
        return s.where(s.isna(), 50.0)
    r = ok.rank(method="average")
    return ((r - 1) / (len(ok) - 1) * 100).reindex(s.index)


def industry_scores(table: pd.DataFrame, macro: dict[str, tuple[float, float]] | None = None,
                    cfg: dict | None = None) -> pd.DataFrame:
    """Bảng điểm nhóm ngành. table: bảng screen có cột `group_name`, chỉ các mã đã qua tầng 1.
    macro: {nhóm: (điểm, độ phủ)}. Trả một dòng mỗi nhóm, xếp theo điểm giảm dần (nhóm không xếp
    hạng ở cuối, `rank` = NaN, `note` ghi lý do)."""
    cfg = cfg or {}
    weights = {**DEFAULT_WEIGHTS, **cfg.get("industry_weights", {})}
    min_members = cfg.get("min_members", 3)
    min_weight = cfg.get("min_weight", 0.5)
    macro_cov = cfg.get("macro_coverage", 0.6)
    macro = macro or {}
    rows = []
    for name, g in table.groupby("group_name"):
        raw = group_raw(g)
        score, cov = macro.get(name, (float("nan"), 0.0))
        raw["macro"] = score if cov >= macro_cov else float("nan")
        raw["macro_coverage"] = cov
        rows.append({"group_name": name, **raw})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    big = df["members"] >= min_members
    comp = pd.DataFrame(index=df.index)
    comp["momentum"] = _rank100(df["rs"].where(big))
    comp["breadth"] = _rank100(df["breadth"].where(big))
    earn = pd.concat([_rank100(df["growth"].where(big)), _rank100(df["growing"].where(big))],
                     axis=1)
    comp["earnings"] = earn.mean(axis=1, skipna=True)
    comp["valuation"] = _rank100(df["expected"].where(big))
    comp["quality"] = _rank100(df["fa"].where(big))
    comp["macro"] = ((df["macro"].clip(-1, 1) + 1) * 50).where(big)   # −1..+1 → 0..100
    w = pd.Series({c: float(weights.get(c, 0.0)) for c in COMPONENTS})
    have = comp[list(COMPONENTS)].notna()
    w_used = have.mul(w, axis=1).sum(axis=1)
    total = (comp[list(COMPONENTS)].fillna(0).mul(w, axis=1).sum(axis=1)
             / w_used.where(w_used > 0))
    enough = big & (w_used >= min_weight * w.sum())
    df = pd.concat([df, comp.add_prefix("pt_")], axis=1)
    df["score"] = total.where(enough)
    df["weight_used"] = w_used / w.sum()
    df["note"] = np.where(~big, f"dưới {min_members} mã qua lọc thanh khoản",
                          np.where(~enough, "thiếu số liệu cho phần lớn thành phần", ""))
    df = df.sort_values(["score", "members"], ascending=[False, False], na_position="last")
    df["rank"] = np.where(df["score"].notna(), np.arange(1, len(df) + 1), np.nan)
    return df.reset_index(drop=True)
