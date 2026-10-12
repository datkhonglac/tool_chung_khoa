"""Điểm vĩ mô: bối cảnh Việt Nam (theo nhóm) và mức thuận lợi của ngành đối với từng mã.

Mỗi yếu tố (driver) = một chuỗi vĩ mô + dấu (+1: chuỗi tăng bất thường là tốt, −1: là xấu) + trọng
số + lý do, khai báo ở config/macro_map.toml (mục [[industry]], [symbol.MÃ], [market.NHÓM]).

    điểm = Σ (dấu × trọng số × z) ÷ (z_cap × Σ trọng số của yếu tố có số)      ∈ [−1; +1]
    độ phủ = Σ trọng số yếu tố có số ÷ Σ trọng số mọi yếu tố

z lấy từ macro/signals.py (so với 5 năm của chính chuỗi, chặn ±z_cap). Yếu tố thiếu số hoặc số CŨ
không vào điểm và được liệt kê riêng. Độ phủ < `min_coverage` -> "Chưa đủ số liệu" (không xếp loại).
Ngưỡng: điểm ≥ `favorable` -> Thuận lợi, ≤ `unfavorable` -> Bất lợi, còn lại Trung tính.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from vnsa.macro.signals import Reading

DEFAULTS = {"favorable": 0.3, "unfavorable": -0.3, "min_coverage": 0.5, "z_cap": 2.0}
LABELS = ("Thuận lợi", "Trung tính", "Bất lợi", "Chưa đủ số liệu")


@dataclass
class Driver:
    series: str
    sign: int
    weight: float
    why: str
    reading: Reading | None
    contribution: float = float("nan")      # phần đóng góp vào điểm (tổng các phần = điểm)

    @property
    def used(self) -> bool:
        return not np.isnan(self.contribution)


@dataclass
class MacroScore:
    scope: str                      # "ngành Thép (ICB cấp 4: ...)" | "mã FPT" | "Tiền tệ"
    score: float
    label: str
    coverage: float
    drivers: list[Driver] = field(default_factory=list)

    @property
    def missing(self) -> list[Driver]:
        return [d for d in self.drivers if d.reading is None or d.reading.date is None]

    @property
    def stale(self) -> list[Driver]:
        return [d for d in self.drivers if d.reading is not None and d.reading.date is not None
                and d.reading.stale]


def _cfg(cfg: dict | None) -> dict:
    return {**DEFAULTS, **(cfg or {})}


def label(score: float, coverage: float, cfg: dict | None = None) -> str:
    c = _cfg(cfg)
    if np.isnan(score) or coverage < c["min_coverage"]:
        return "Chưa đủ số liệu"
    if score >= c["favorable"]:
        return "Thuận lợi"
    if score <= c["unfavorable"]:
        return "Bất lợi"
    return "Trung tính"


def score_drivers(scope: str, specs: list[dict], rs: dict[str, Reading],
                  cfg: dict | None = None) -> MacroScore:
    """Chấm một nhóm yếu tố (danh sách {series, sign, weight, why}) từ các chuỗi đã đọc."""
    c = _cfg(cfg)
    drivers = [Driver(s["series"], int(np.sign(s.get("sign", 1))) or 1,
                      float(s.get("weight", 1.0)), s.get("why", ""), rs.get(s["series"]))
               for s in specs]
    total = sum(d.weight for d in drivers)
    usable = [d for d in drivers if d.reading is not None and d.reading.ok and not d.reading.stale]
    w_used = sum(d.weight for d in usable)
    if not usable or total <= 0:
        return MacroScore(scope, float("nan"), "Chưa đủ số liệu", 0.0, drivers)
    for d in usable:
        d.contribution = d.sign * d.weight * d.reading.z / (c["z_cap"] * w_used)
    score = float(sum(d.contribution for d in usable))
    coverage = w_used / total
    return MacroScore(scope, score, label(score, coverage, c), coverage, drivers)


def _norm(text: str | None) -> str:
    return (text or "").lower()


def drivers_for(symbol: str, company: dict | None, mapping: dict) -> tuple[str, list[dict]] | None:
    """(phạm vi, yếu tố) của một mã: khai báo riêng [symbol.MÃ] nếu có, ngược lại mục [[industry]]
    đầu tiên có `match` khớp tên ngành ICB cấp 4, rồi cấp 3, rồi cấp 2 (hoặc `model` khớp loại
    doanh nghiệp). None nếu không có ánh xạ."""
    own = mapping.get("symbol", {}).get(symbol.upper())
    if own:
        return own.get("name", f"mã {symbol.upper()}"), own.get("drivers", [])
    company = company or {}
    levels = [(4, company.get("industry4")), (3, company.get("industry3")),
              (2, company.get("industry"))]
    for level, name in levels:
        if not name:
            continue
        for ind in mapping.get("industry", []):
            pat = ind.get("match")
            if pat and re.search(pat, _norm(name), flags=re.IGNORECASE):
                return f"{ind.get('name', name)} (ICB cấp {level}: {name})", ind.get("drivers", [])
    model = company.get("model")
    for ind in mapping.get("industry", []):
        if model and ind.get("model") == model:
            return ind.get("name", model), ind.get("drivers", [])
    return None


def industry_score(symbol: str, company: dict | None, rs: dict[str, Reading], mapping: dict,
                   cfg: dict | None = None) -> MacroScore | None:
    found = drivers_for(symbol, company, mapping)
    if found is None:
        return None
    scope, specs = found
    return score_drivers(scope, specs, rs, cfg)


def market_context(rs: dict[str, Reading], mapping: dict,
                   cfg: dict | None = None) -> tuple[MacroScore, dict[str, MacroScore]]:
    """Bối cảnh vĩ mô Việt Nam: điểm từng nhóm ([market.NHÓM]) và điểm chung = trung bình các nhóm
    đã xếp loại được (mỗi nhóm một phiếu)."""
    groups = {key: score_drivers(g.get("name", key), g.get("drivers", []), rs, cfg)
              for key, g in mapping.get("market", {}).items()}
    rated = [g for g in groups.values() if g.label != "Chưa đủ số liệu"]
    if not rated:
        return MacroScore("Bối cảnh Việt Nam", float("nan"), "Chưa đủ số liệu", 0.0), groups
    score = float(np.mean([g.score for g in rated]))
    coverage = len(rated) / len(groups)
    return MacroScore("Bối cảnh Việt Nam", score, label(score, coverage, cfg), coverage), groups
