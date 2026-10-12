"""Engine khuyến nghị hai trục — như cách các công ty chứng khoán VN làm, không gộp FA + TA thành
một điểm (gộp thì "doanh nghiệp tốt đang giảm giá" và "doanh nghiệp tầm thường đi ngang" ra cùng
một con số).

Trục 1 — XẾP HẠNG theo tổng lợi nhuận kỳ vọng 12 tháng = chênh lệch giá trị hợp lý so với giá
(fa/fair_value.py) + tỷ suất cổ tức tiền 12 tháng. Ngưỡng `rating_thresholds` ([recommend]):
    ≥ 15% MUA · ≥ 5% KHẢ QUAN · ≥ −5% NẮM GIỮ · ≥ −15% KÉM KHẢ QUAN · còn lại BÁN.
  Rào chắn (ghi rõ trong lý do):
  - Thị trường chung đang giảm (VNINDEX dưới MA200, MA50 đi xuống — ta/regime.py): mọi ngưỡng cao
    thêm `bear_market_extra` (cần biên lợi nhuận lớn hơn mới mua).
  - Điểm FA < `min_fa_quality`: hạ một bậc (không xếp MUA doanh nghiệp yếu chỉ vì rẻ).
  - Altman Z'' < 1,1 (nguy cơ tài chính): tối đa NẮM GIỮ (tránh bẫy giá trị).
  - Không có giá trị hợp lý (thiếu BCTC): KHÔNG XẾP HẠNG, chỉ có trục thời điểm.
Trục 2 — THỜI ĐIỂM, theo `timing_rule` (xem timing()). Mặc định "none" vì kiểm định 10 năm × 45 mã
(lệnh study-timing, docs/kiem_dinh_thoi_diem.md) cho thấy KHÔNG quy tắc kỹ thuật nào (kể cả điểm TA
cũ) vừa có Sharpe cao hơn vừa sụt giảm nhỏ hơn mua & giữ ở cả giai đoạn chọn tham số lẫn giai đoạn
kiểm tra độc lập — nên trục này chỉ quản trị rủi ro:
  - giá đang cao hơn trần giá trị (giá trị hợp lý trừ biên an toàn) → "Chờ giá về vùng mua";
  - còn lại → "Trong vùng mua (giữ kỷ luật cắt lỗ)".
  "ta" (cách cũ, theo điểm TA) hoặc tên quy tắc trong recommend/strategies.py vẫn chọn được.
Điểm tổng (`score`) = w_fa × FA + w_ta × TA chỉ để sắp xếp danh sách, KHÔNG quyết định xếp hạng;
điểm Quant chỉ để tham khảo (xác suất tăng của Monte Carlo chủ yếu kéo dài xu hướng quá khứ).
Đây là quy tắc sàng lọc, không phải lời khuyên đầu tư.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from vnsa.fa.scores import scale
from vnsa.ta.score import rating as ta_rating

RATINGS = ["MUA", "KHẢ QUAN", "NẮM GIỮ", "KÉM KHẢ QUAN", "BÁN"]
NOT_RATED = "KHÔNG XẾP HẠNG"


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
    quant_score: float = float("nan")   # điểm Quant 0–100 (tham khảo)
    dividend_yield: float = 0.0         # cổ tức tiền 12 tháng / giá
    regime: str = "trung tính"          # trạng thái VNINDEX: tăng | trung tính | giảm
    zone_status: str = ""               # MUA TRONG VÙNG | CHỜ GIÁ (ta/entry.py)
    rule_state: dict | None = None      # trạng thái quy tắc thời điểm đã kiểm định (strategies)
    macro_industry: float = float("nan")  # điểm vĩ mô ngành −1..+1 (chỉ lưu và chấm lại, chưa vào điểm)
    macro_market: float = float("nan")    # điểm bối cảnh Việt Nam −1..+1


@dataclass
class Recommendation:
    symbol: str
    action: str              # một trong RATINGS hoặc NOT_RATED
    score: float             # điểm sắp xếp (FA/TA), không quyết định xếp hạng
    inputs: Inputs
    reasons: list[str] = field(default_factory=list)
    timing: str = ""
    expected_return: float = float("nan")


def quant_score(prob_up: float, vol_annual: float, reward_risk: float, cfg: dict) -> float:
    """Điểm Quant 0–100 (cfg = mục [recommend.quant]); chỉ tiêu thiếu thì bỏ qua."""
    pts = [scale(prob_up, *cfg["prob_up"]), scale(vol_annual, *cfg["vol_annual"]),
           scale(reward_risk, *cfg["reward_risk"])]
    pts = [p for p in pts if not pd.isna(p)]
    return float(sum(pts) / len(pts)) if pts else float("nan")


def _fa_rating(x: float) -> str:
    if pd.isna(x):
        return "—"
    return "tốt" if x >= 70 else ("trung bình" if x >= 50 else "yếu")


def rate(expected: float, thresholds: list[float]) -> str:
    for label, t in zip(RATINGS, thresholds, strict=False):
        if expected >= t:
            return label
    return RATINGS[-1]


def timing(x: Inputs, cfg: dict) -> str:
    """Trục thời điểm theo `timing_rule` ([recommend]):
    - "ta": điểm TA (cách ban đầu);
    - tên quy tắc trong strategies (vd "trend"): đang trong lệnh của quy tắc thì "Vào được", chưa
      thì chờ tín hiệu — chỉ dùng quy tắc đã ĐẠT kiểm định (lệnh study-timing);
    - "none": chỉ quản trị rủi ro (vùng mua + cắt lỗ), không đoán điểm vào."""
    if x.zone_status == "CHỜ GIÁ":
        return "Chờ giá về vùng mua"
    rule = cfg.get("timing_rule", "ta")
    if rule == "none":
        return "Trong vùng mua (giữ kỷ luật cắt lỗ)"
    if rule != "ta":
        st = x.rule_state
        if st is None:
            return "Chưa đủ dữ liệu kỹ thuật"
        if st["holding"]:
            stop = f", cắt lỗ động {st['stop']:,.0f}" if not pd.isna(st["stop"]) else ""
            return f"Vào được — xu hướng tăng xác nhận{stop}"
        return "Chờ tín hiệu xu hướng"
    if pd.isna(x.ta_score):
        return "Chưa đủ dữ liệu kỹ thuật"
    if x.ta_score < cfg.get("timing_wait", 40):
        return "Chờ xu hướng cải thiện"
    if x.ta_score >= cfg.get("timing_go", 55):
        return "Vào được trong vùng mua"
    return "Mua dần trong vùng mua"


def recommend(x: Inputs, cfg: dict, notes: list[str] | None = None) -> Recommendation:
    w_fa, w_ta = float(cfg["weight_fa"]), float(cfg["weight_ta"])
    parts = [(w, v) for w, v in ((w_fa, x.fa_score), (w_ta, x.ta_score)) if not pd.isna(v)]
    score = sum(w * v for w, v in parts) / sum(w for w, _ in parts) if parts else float("nan")

    reasons = []
    expected = float("nan")
    if pd.isna(x.fair_upside):
        action = NOT_RATED
        reasons.append("chưa có giá trị hợp lý (thiếu BCTC): chỉ xét thời điểm kỹ thuật")
    else:
        dy = 0.0 if pd.isna(x.dividend_yield) else x.dividend_yield
        expected = x.fair_upside + dy
        fmt = "+.1%" if abs(expected) < 0.1 else "+.0%"
        reasons.append(f"lợi nhuận kỳ vọng {expected:{fmt}} (giá trị hợp lý {x.fair_value:,.0f} "
                       f"{x.fair_upside:+.0%}" + (f", cổ tức {dy:.1%}" if dy > 0 else "") + ")")
        thresholds = list(cfg["rating_thresholds"])
        if x.regime == "giảm":
            extra = float(cfg.get("bear_market_extra", 0.0))
            thresholds = [t + extra for t in thresholds]
            reasons.append(f"thị trường chung đang giảm: ngưỡng xếp hạng cao thêm {extra:.0%}")
        action = rate(expected, thresholds)
        level = RATINGS.index(action)
        if not pd.isna(x.fa_score) and x.fa_score < cfg.get("min_fa_quality", 50) and level < 4:
            level += 1
            reasons.append(f"điểm FA {x.fa_score:.0f} < {cfg.get('min_fa_quality', 50)}: hạ một "
                           "bậc (doanh nghiệp yếu dù rẻ)")
        if not pd.isna(x.z_em) and x.z_em < 1.1 and level < 2:
            level = 2
            reasons.append(f"Altman Z'' {x.z_em:.2f} vùng nguy cơ: tối đa NẮM GIỮ")
        action = RATINGS[level]
    if not pd.isna(x.fa_score):
        reasons.append(f"điểm FA {x.fa_score:.0f}/100 ({_fa_rating(x.fa_score)})")
    if not pd.isna(x.ta_score):
        reasons.append(f"điểm TA {x.ta_score:.0f}/100 ({ta_rating(x.ta_score).lower()})")
    if not pd.isna(x.first_target_rr) and x.first_target_rr < cfg.get("min_reward_risk", 2):
        reasons.append(f"mục tiêu gần nhất chỉ {x.first_target_rr:.1f}R (kháng cự sát)")
    if not pd.isna(x.vol_annual) and x.vol_annual > cfg.get("high_vol", 0.40):
        reasons.append(f"biến động cao ({x.vol_annual:.0%}/năm): giảm khối lượng")
    reasons += notes or []
    return Recommendation(x.symbol, action, float(score), x, reasons, timing(x, cfg), expected)
