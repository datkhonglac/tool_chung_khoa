"""Đọc config/settings.toml. Mọi đường dẫn đều tính từ thư mục gốc dự án.

Khi đọc, cấu hình được kiểm tra (`validate`): sửa nhầm một tham số (trọng số âm, ngưỡng đảo chiều,
tăng trưởng dài hạn ≥ chi phí vốn...) thì tool dừng ngay và chỉ rõ dòng cần sửa, thay vì chạy tiếp
và cho ra số sai. Kiểm tra tay: `python -m vnsa.config`.

Lưu ý: KHÔNG bắt tổng trọng số bằng 100. Code tự chia lại trọng số khi thiếu chỉ tiêu/phương pháp
(ví dụ chưa đủ mã cùng ngành), nên [valuation.weights] cố ý có tổng > 1 và điểm FA định chế tài
chính có tổng 110; chỉ cần trọng số ≥ 0 và tổng > 0."""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_FILE = PROJECT_ROOT / "config" / "settings.toml"


class ConfigError(ValueError):
    """config/settings.toml có tham số sai."""


REQUIRED = ["data", "market", "auto", "watchlist", "valuation", "fa_score", "ta", "ta_score",
            "plan", "entry", "signals", "recommend", "report", "backtest", "risk"]
# Ngưỡng [mức xấu, mức tốt]: các chỉ tiêu "số nhỏ là tốt" phải có mức xấu > mức tốt, còn lại ngược.
LOWER_IS_BETTER = {"debt_to_equity", "pe_rel", "pb_rel", "cir", "credit_cost", "margin_to_equity",
                   "npl", "vol_annual"}
WEIGHT_TABLES = ["fa_score.weights", "fa_score.financials.weights", "ta_score.weights",
                 "valuation.weights.general", "valuation.weights.financial",
                 "valuation.weights.cyclical", "valuation.weights.real_estate",
                 "screen.industry_weights"]
THRESHOLD_TABLES = ["fa_score.thresholds", "fa_score.financials.thresholds",
                    "fa_score.financials.relative", "fa_score.financials.equity_to_assets",
                    "recommend.quant"]


def _get(cfg: dict, dotted: str):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _num(x) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool)


def validate(cfg: dict) -> list[str]:
    """Danh sách lỗi (rỗng = hợp lệ). Mỗi lỗi ghi [mục] khoá và cách sửa."""
    err: list[str] = []
    for sec in REQUIRED:
        if not isinstance(cfg.get(sec), dict):
            err.append(f"thiếu mục [{sec}]")
    for tbl in WEIGHT_TABLES:
        w = _get(cfg, tbl)
        if w is None:
            continue
        for k, v in w.items():
            if not _num(v) or v < 0:
                err.append(f"[{tbl}] {k} = {v!r}: trọng số phải là số ≥ 0")
        if all(_num(v) for v in w.values()) and sum(w.values()) <= 0:
            err.append(f"[{tbl}]: tổng trọng số phải > 0")
    for tbl in THRESHOLD_TABLES:
        t = _get(cfg, tbl)
        if t is None:
            continue
        for k, v in t.items():
            if not isinstance(v, list):
                continue                       # vd min_bank_peers: số đơn, không phải ngưỡng
            if len(v) != 2 or not all(_num(x) for x in v) or v[0] == v[1]:
                err.append(f"[{tbl}] {k} = {v}: cần 2 số khác nhau [mức xấu, mức tốt]")
            elif tbl.endswith("equity_to_assets"):
                if v[0] > v[1]:
                    err.append(f"[{tbl}] {k} = {v}: mức xấu phải nhỏ hơn mức tốt")
            elif (k in LOWER_IS_BETTER) != (v[0] > v[1]):
                hint = ("số nhỏ là tốt nên mức xấu phải LỚN hơn mức tốt" if k in LOWER_IS_BETTER
                        else "mức xấu phải NHỎ hơn mức tốt")
                err.append(f"[{tbl}] {k} = {v}: {hint} (có thể đã đảo ngược hai số)")
    v = cfg.get("valuation", {})
    rf, erp, g = v.get("risk_free"), v.get("equity_risk_premium"), v.get("terminal_growth")
    if not (_num(rf) and 0 < rf < 0.2):
        err.append(f"[valuation] risk_free = {rf!r}: lãi suất phi rủi ro dạng thập phân, 0 < rf < 0,2 "
                   "(0.035 = 3,5%, không phải 3.5)")
    if not (_num(erp) and 0 < erp < 0.2):
        err.append(f"[valuation] equity_risk_premium = {erp!r}: dạng thập phân, 0 < ERP < 0,2")
    if not _num(g) or (_num(rf) and _num(erp) and g >= rf + erp) or (_num(g) and g < -0.05):
        err.append(f"[valuation] terminal_growth = {g!r}: phải nhỏ hơn chi phí vốn (r_f + ERP); "
                   "nếu không, công thức Gordon cho giá trị vô hạn hoặc âm")
    for k in ("multiple_reversion",):
        x = v.get(k)
        if x is not None and not (_num(x) and 0 <= x <= 1):
            err.append(f"[valuation] {k} = {x!r}: phải trong [0; 1]")
    if _num(v.get("consensus_cap")) and v["consensus_cap"] < 0:
        err.append("[valuation] consensus_cap: phải ≥ 0 (0 = tắt)")
    th = _get(cfg, "recommend.rating_thresholds")
    if th is not None and (len(th) != 4 or not all(_num(x) for x in th)
                           or any(a <= b for a, b in zip(th, th[1:], strict=False))):
        err.append(f"[recommend] rating_thresholds = {th}: cần 4 số giảm dần "
                   "(MUA > KHẢ QUAN > NẮM GIỮ > KÉM KHẢ QUAN)")
    r = cfg.get("report", {})
    if _num(r.get("medium_confidence")) and _num(r.get("high_confidence")) and not (
            0 <= r["medium_confidence"] <= r["high_confidence"] <= 6):
        err.append("[report] cần 0 ≤ medium_confidence ≤ high_confidence ≤ 6 (6 điều kiện)")
    p = cfg.get("plan", {})
    if _num(p.get("risk_per_trade")) and not 0 < p["risk_per_trade"] <= 0.05:
        err.append(f"[plan] risk_per_trade = {p['risk_per_trade']}: rủi ro mỗi lệnh 0 < x ≤ 0,05 "
                   "(0.01 = 1% vốn)")
    if _num(p.get("max_position_pct")) and not 0 < p["max_position_pct"] <= 1:
        err.append("[plan] max_position_pct: phải trong (0; 1]")
    if (_num(p.get("min_stop_atr")) and _num(p.get("max_stop_atr"))
            and p["min_stop_atr"] > p["max_stop_atr"]):
        err.append("[plan] min_stop_atr phải ≤ max_stop_atr")
    e = cfg.get("entry", {})
    if _num(e.get("margin_of_safety")) and not 0 <= e["margin_of_safety"] < 1:
        err.append("[entry] margin_of_safety: phải trong [0; 1)")
    for k in ("band_hose", "band_hnx", "band_upcom"):
        b = _get(cfg, f"market.{k}")
        if b is not None and not (_num(b) and 0 < b < 0.5):
            err.append(f"[market] {k} = {b!r}: biên độ dạng thập phân (0.07 = 7%)")
    sma = _get(cfg, "ta.sma")
    if sma is not None and list(sma) != sorted(set(sma)):
        err.append(f"[ta] sma = {sma}: các độ dài phải tăng dần, không trùng")
    return err


def check(cfg: dict) -> dict:
    errors = validate(cfg)
    if errors:
        raise ConfigError("config/settings.toml có tham số sai:\n  - " + "\n  - ".join(errors))
    return cfg


@lru_cache(maxsize=1)
def settings() -> dict:
    with SETTINGS_FILE.open("rb") as f:
        return check(tomllib.load(f))


def path(key: str) -> Path:
    """Đường dẫn tuyệt đối cho một khoá trong mục [data], ví dụ path("warehouse").

    Biến môi trường VNSA_<KHOÁ> (ví dụ VNSA_WAREHOUSE) ghi đè đường dẫn — dùng cho test hoặc khi
    MCP server / dashboard chạy từ thư mục khác."""
    override = os.environ.get(f"VNSA_{key.upper()}")
    return Path(override) if override else PROJECT_ROOT / settings()["data"][key]


if __name__ == "__main__":
    try:
        settings()
    except ConfigError as e:
        raise SystemExit(str(e)) from None
    print(f"{SETTINGS_FILE}: hợp lệ")
