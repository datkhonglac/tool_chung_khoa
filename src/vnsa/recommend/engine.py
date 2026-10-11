"""Engine khuyến nghị: gộp điểm FA, điểm TA, định giá và rủi ro thành MUA / THEO DÕI / TRÁNH.

Quy tắc (ngưỡng ở mục [recommend] trong settings):
- Điểm tổng = w_fa × điểm FA + w_ta × điểm TA (thiếu BCTC thì chỉ dùng điểm TA).
- TRÁNH nếu điểm tổng < `sell_below`, hoặc Altman Z'' < 1,1 (vùng nguy cơ).
- MUA nếu điểm tổng ≥ `buy_above` VÀ điểm TA ≥ `min_ta_buy` (xu hướng không xấu) VÀ chênh lệch so
  với giá trị hợp lý (fa/fair_value.py; thiếu thì dùng DCF) ≥ `min_upside_buy` (nếu có định giá).
- Còn lại: THEO DÕI. Lý do đi kèm nói rõ điều kiện nào chưa đạt.
Đây là quy tắc tính toán để sàng lọc, không phải lời khuyên đầu tư.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from vnsa.ta.score import rating as ta_rating


@dataclass
class Inputs:
    symbol: str
    price: float
    date: pd.Timestamp
    fa_score: float
    ta_score: float
    z_em: float
    dcf_upside: float
    pe: float
    pb: float
    vol_annual: float        # biến động năm hiện tại (GARCH)
    prob_up: float           # xác suất giá cao hơn hiện tại sau `horizon` phiên (Monte Carlo)
    first_target_rr: float   # lãi/lỗ của mục tiêu 1 trong kế hoạch
    horizon: int = 60
    fair_value: float = float("nan")    # giá trị hợp lý/cp (fa/fair_value.py)
    fair_upside: float = float("nan")   # giá trị hợp lý so với giá hiện tại


@dataclass
class Recommendation:
    symbol: str
    action: str              # "MUA" | "THEO DÕI" | "TRÁNH"
    score: float
    inputs: Inputs
    reasons: list[str] = field(default_factory=list)


def _fa_rating(x: float) -> str:
    if pd.isna(x):
        return "—"
    return "tốt" if x >= 70 else ("trung bình" if x >= 50 else "yếu")


def recommend(x: Inputs, cfg: dict, notes: list[str] | None = None) -> Recommendation:
    w_fa, w_ta = float(cfg["weight_fa"]), float(cfg["weight_ta"])
    if pd.isna(x.fa_score):
        score = x.ta_score
    elif pd.isna(x.ta_score):
        score = x.fa_score
    else:
        score = (w_fa * x.fa_score + w_ta * x.ta_score) / (w_fa + w_ta)

    reasons = []
    if not pd.isna(x.fa_score):
        reasons.append(f"điểm FA {x.fa_score:.0f}/100 ({_fa_rating(x.fa_score)})")
    if not pd.isna(x.ta_score):
        reasons.append(f"điểm TA {x.ta_score:.0f}/100 ({ta_rating(x.ta_score).lower()})")
    upside = x.fair_upside if not pd.isna(x.fair_upside) else x.dcf_upside
    if not pd.isna(x.fair_upside):
        reasons.append(f"giá trị hợp lý {x.fair_value:,.0f} ({x.fair_upside:+.0%} so với giá)")
    elif not pd.isna(x.dcf_upside):
        reasons.append(f"giá trị DCF {x.dcf_upside:+.0%} so với giá")
    distress = not pd.isna(x.z_em) and x.z_em < 1.1

    if pd.isna(score):
        action = "THEO DÕI"
        reasons.append("thiếu dữ liệu để chấm điểm")
    elif score < cfg["sell_below"] or distress:
        action = "TRÁNH"
        if distress:
            reasons.append(f"Altman Z'' {x.z_em:.2f} ở vùng nguy cơ tài chính")
    elif score >= cfg["buy_above"]:
        blockers = []
        if pd.isna(x.ta_score) or x.ta_score < cfg["min_ta_buy"]:
            blockers.append("chờ xu hướng kỹ thuật cải thiện "
                            f"(điểm TA cần ≥ {cfg['min_ta_buy']:g})")
        if not pd.isna(upside) and upside < cfg["min_upside_buy"]:
            blockers.append(f"giá đã cao hơn giá trị hợp lý quá {-cfg['min_upside_buy']:.0%}")
        action = "THEO DÕI" if blockers else "MUA"
        reasons += blockers
    else:
        action = "THEO DÕI"

    if not pd.isna(x.first_target_rr) and x.first_target_rr < cfg.get("min_reward_risk", 2):
        reasons.append(f"mục tiêu gần nhất chỉ {x.first_target_rr:.1f}R (kháng cự sát)")
    if not pd.isna(x.vol_annual) and x.vol_annual > cfg.get("high_vol", 0.40):
        reasons.append(f"biến động cao ({x.vol_annual:.0%}/năm): giảm khối lượng")
    if not pd.isna(x.prob_up):
        reasons.append(f"mô phỏng: {x.prob_up:.0%} khả năng giá cao hơn sau {x.horizon} phiên")
    reasons += notes or []
    return Recommendation(x.symbol, action, float(score), x, reasons)
