"""Nhóm ngành của bộ sàng lọc: nhóm kiểu Việt Nam trong config/macro_map.toml ([[industry]]: Ngân
hàng, Chứng khoán, Thép, Bất động sản...) để điểm vĩ mô khớp đúng nhóm; mã không khớp nhóm nào thì
xếp theo ngành ICB cấp 2. Cùng quy tắc khớp với lớp vĩ mô (macro.score.match_industry)."""

from __future__ import annotations

import pandas as pd

from vnsa.macro.score import match_industry, score_drivers

UNKNOWN = "Chưa rõ ngành"


def industry_group(company: dict | None, mapping: dict) -> str:
    found = match_industry(company, mapping)
    if found is not None:
        return found[0].get("name", found[2])
    return (company or {}).get("industry") or UNKNOWN


def assign_groups(companies: pd.DataFrame, mapping: dict) -> pd.Series:
    """Tên nhóm cho từng dòng của bảng companies (index giữ nguyên)."""
    return pd.Series([industry_group(r, mapping) for r in companies.to_dict("records")],
                     index=companies.index, dtype=object)


def group_macro(mapping: dict, readings: tuple[dict, dict] | None) -> dict[str, tuple[float, float]]:
    """{tên nhóm: (điểm vĩ mô −1..+1, độ phủ)} cho các nhóm có yếu tố vĩ mô; {} nếu chưa có số."""
    if readings is None:
        return {}
    rs, cfg = readings
    out = {}
    for ind in mapping.get("industry", []):
        if ind.get("drivers"):
            m = score_drivers(ind["name"], ind["drivers"], rs, cfg)
            out[ind["name"]] = (m.score, m.coverage)
    return out


def taxonomy(companies: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """Bảng ngành ICB cấp 2/3/4 → nhóm (để kiểm tra quy tắc khớp trong macro_map.toml)."""
    df = companies.assign(group=assign_groups(companies, mapping))
    keys = ["group", "industry", "industry3", "industry4"]
    return (df.fillna({k: "—" for k in keys[1:]}).groupby(keys).size().rename("n").reset_index()
            .sort_values(["group", "n"], ascending=[True, False]))
