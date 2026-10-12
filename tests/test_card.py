"""Thẻ báo cáo (report/card.py) và điểm Quant (recommend/engine.py)."""

import pandas as pd
import pytest
from conftest import _fundamentals

from vnsa.recommend.engine import quant_score
from vnsa.report.card import format_card, red_flags, vn_num, vn_pct

QCFG = {"prob_up": [0.40, 0.60], "vol_annual": [0.60, 0.20], "reward_risk": [1.0, 3.0]}


def test_vietnamese_number_format():
    assert vn_num(50000) == "50.000" and vn_num(0.6, 1) == "0,6" and vn_num(float("nan")) == "—"
    assert vn_pct(0.185) == "18,5%" and vn_pct(0.22, 0, sign=True) == "+22%"
    assert vn_pct(-0.031) == "−3,1%"


def test_quant_score_averages_available_items():
    assert quant_score(0.60, 0.20, 3.0, QCFG) == 100
    assert quant_score(0.50, 0.40, 2.0, QCFG) == pytest.approx(50)
    assert quant_score(0.60, float("nan"), float("nan"), QCFG) == 100     # thiếu thì bỏ qua
    assert pd.isna(quant_score(float("nan"), float("nan"), float("nan"), QCFG))


def test_red_flags_general_and_financial():
    fin = _fundamentals("X")
    assert red_flags(fin, False, z_em=3.0, f_score=7) == []
    bad = fin.assign(net_income_parent=[90e9, 80e9, 70e9, -5e9],
                     short_debt=fin["equity"] * 2.5, equity_issued=fin["equity"] * 0.2)
    flags = " | ".join(red_flags(bad, False, z_em=0.8, f_score=2))
    for text in ("âm", "giảm 2 năm", "đòn bẩy cao", "vùng nguy cơ", "F-Score thấp", "pha loãng"):
        assert text in flags, text
    bank = fin.assign(net_income_parent=fin["equity"] * 0.05)
    assert any("ROE thấp" in f for f in red_flags(bank, True, float("nan"), float("nan")))


def test_report_card_from_warehouse(warehouse):
    from vnsa import analysis

    card = analysis.report_card("DEMO_A")
    assert card.action in {"MUA", "KHẢ QUAN", "NẮM GIỮ", "KÉM KHẢ QUAN", "BÁN", "KHÔNG XẾP HẠNG"} and card.label
    assert card.confidence in {"Cao", "Trung bình", "Thấp"} and len(card.confidence_checks) == 6
    assert card.p10 <= card.p50 <= card.p90 and card.target == card.p50
    assert 0 <= card.quant <= 100 and not pd.isna(card.total)
    text = format_card(card)
    for title in ("ĐỊNH GIÁ", "ĐIỂM SỐ (0-100)", "CƠ BẢN", "KỸ THUẬT", "DỰ BÁO 20 PHIÊN",
                  "KẾ HOẠCH GIAO DỊCH", "Giá trị nội tại (P10 / P50 / P90)", "Tỷ trọng gợi ý"):
        assert title in text, title
    no_fin = format_card(analysis.report_card("DEMO_C"))               # chưa có BCTC
    assert "Chưa có BCTC" in no_fin and "DỰ BÁO 20 PHIÊN" in no_fin


def test_confidence_needs_independent_method_and_consensus(warehouse):
    from vnsa import analysis
    from vnsa.config import settings
    from vnsa.data.store import Store
    from vnsa.report.card import _confidence, build_card, format_card

    with Store(warehouse) as store:
        store.write_profile("DEMO_A", {"analyst_target": 10_000.0, "analyst_rating": "BUY"})
    a = analysis.assess("DEMO_A")
    level, _ = _confidence(a, settings()["report"])
    gap = a.fair.value / 10_000 - 1
    assert gap > 0.3 and level != "Cao"                           # lệch xa giá mục tiêu CTCK
    text = format_card(build_card(a))
    assert "Giá mục tiêu CTCK (VCI tổng hợp): 10.000đ · BUY" in text
    assert "so với giá mục tiêu CTCK" in text
