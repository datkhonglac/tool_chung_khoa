"""Kiểm tra chất lượng BCTC (Beneish M-Score...), phân loại tin theo từ khoá, chặn mua, cảnh báo thẻ."""

from types import SimpleNamespace

import pandas as pd
import pytest

from vnsa import news as nw
from vnsa.fa import forensic as fx
from vnsa.recommend.engine import Inputs, recommend


def _year(year, **kw):
    base = dict(year=year, revenue=1000.0, gross_profit=300.0, receivables=100.0,
                current_assets=500.0, ppe=300.0, total_assets=1000.0, depreciation=30.0,
                selling_expenses=50.0, admin_expenses=50.0, current_liabilities=300.0,
                long_debt=100.0, net_income=100.0, cfo=100.0, equity=500.0, cash=100.0,
                short_term_investments=50.0, short_debt=50.0, other_receivables=20.0,
                other_receivables_lt=10.0, customer_advances=0.0, customer_advances_lt=0.0)
    base.update(kw)
    return base


def test_beneish_neutral_company():
    # năm t giống hệt t−1, dồn tích 0: mọi chỉ số = 1, TATA = 0
    m, comp, neutral = fx.beneish(pd.Series(_year(2025)), pd.Series(_year(2024)))
    assert all(comp[k] == pytest.approx(1.0) for k in ("dsri", "gmi", "aqi", "sgi", "depi", "sgai",
                                                        "lvgi"))
    assert comp["tata"] == 0 and not neutral
    assert m == pytest.approx(-4.84 + 0.92 + 0.528 + 0.404 + 0.892 + 0.115 - 0.172 - 0.327)


def test_beneish_manipulator_flags_and_hand_calc():
    prev = _year(2024)
    cur = _year(2025, revenue=1500.0, gross_profit=375.0, receivables=300.0, net_income=200.0,
                cfo=20.0, total_assets=1200.0)
    m, c, _ = fx.beneish(pd.Series(cur), pd.Series(prev))
    assert c["dsri"] == pytest.approx((300 / 1500) / (100 / 1000))          # 2,0
    assert c["gmi"] == pytest.approx(0.3 / 0.25)
    assert c["sgi"] == pytest.approx(1.5)
    assert c["tata"] == pytest.approx(180 / 1200)
    assert c["aqi"] == pytest.approx((1 - 800 / 1200) / (1 - 800 / 1000))
    expected = (-4.84 + 0.92 * c["dsri"] + 0.528 * c["gmi"] + 0.404 * c["aqi"] + 0.892 * c["sgi"]
                + 0.115 * c["depi"] - 0.172 * c["sgai"] + 4.679 * c["tata"] - 0.327 * c["lvgi"])
    assert m == pytest.approx(expected) and m > -1.78
    res = fx.analyze(pd.DataFrame([prev, cur]))
    assert res.year == 2025 and any("M-Score" in w for w in res.warnings)
    assert "DSRI" in res.warnings[0]


def test_beneish_missing_data():
    prev, cur = _year(2024), _year(2025, receivables=float("nan"), depreciation=float("nan"))
    m, c, neutral = fx.beneish(pd.Series(cur), pd.Series(prev))
    assert pd.isna(m) and "DEPI" in neutral
    # năm liền trước bị thiếu (2023 -> 2025): không tính M-Score
    assert pd.isna(fx.analyze(pd.DataFrame([_year(2023), _year(2025)])).mscore)


def test_other_receivables_backlog_net_cash():
    df = pd.DataFrame([_year(2024, customer_advances=500.0),
                       _year(2025, other_receivables=150.0, other_receivables_lt=50.0,
                             customer_advances=800.0, customer_advances_lt=200.0)])
    res = fx.analyze(df, market_cap=1000.0, real_estate=True)
    assert res.other_receivables == pytest.approx(0.4)
    assert any("phải thu khác 40%" in w for w in res.warnings)
    assert res.backlog == pytest.approx(1.0) and res.backlog_prev == pytest.approx(0.5)
    assert res.net_cash == pytest.approx((150 - 150) / 1000)
    assert fx.analyze(df).backlog != fx.analyze(df).backlog        # không phải BĐS: NaN


# ---------- tin tức ----------

@pytest.mark.parametrize("title,label", [
    ("DIG: Ông Nguyễn Hùng Cường - Chủ tịch HĐQT đăng ký mua 5 triệu cổ phiếu", nw.POSITIVE),
    ("DIG: Vợ TGĐ đăng ký mua 1 triệu cổ phiếu", nw.POSITIVE),
    ("Gia đình Chủ tịch DIC Corp (DIG) Nguyễn Hùng Cường bị bán giải chấp thêm hơn 8,1 triệu cổ phiếu",
     nw.NEGATIVE),
    ("HPG: Ông Nguyễn Ngọc Quang - Thành viên HĐQT đăng ký bán 6,6 triệu cổ phiếu", nw.INSIDER_SELL),
    ("HPG: Kết quả kinh doanh quý 2/2026", nw.NEUTRAL),
    ("FPT: Thông báo về việc giao dịch chứng khoán thay đổi đăng ký niêm yết", nw.NEUTRAL),
    ("HOSE đưa cổ phiếu ABC vào diện kiểm soát", nw.SEVERE),
    ("ABC: Quyết định đình chỉ giao dịch cổ phiếu", nw.SEVERE),
    ("ABC: Hủy niêm yết trái phiếu ABC2025", nw.NEUTRAL),
    ("ABC được đưa ra khỏi diện kiểm soát", nw.POSITIVE),
    ("Cổ phiếu ABC bị đưa vào diện cảnh báo do BCTC kiểm toán có ý kiến ngoại trừ", nw.NEGATIVE),
    ("ABC báo lãi quý 3 kỷ lục", nw.POSITIVE),
    ("Cục Thuế xử phạt và truy thu ABC 12 tỷ đồng", nw.NEGATIVE),
])
def test_classify_headlines(title, label):
    assert nw.classify(title, nw.DEFAULTS) == label


def _news(rows):
    return pd.DataFrame(rows, columns=["date", "title"])


def test_block_until_exit_and_recent_negatives():
    asof = pd.Timestamp("2026-10-22")
    n = _news([("2026-06-01", "HOSE đưa cổ phiếu ABC vào diện kiểm soát"),
               ("2026-10-20", "ABC bị xử phạt vi phạm công bố thông tin"),
               ("2026-08-01", "ABC chậm nộp báo cáo tài chính")])
    res = nw.check(n, asof, nw.DEFAULTS)
    assert "diện kiểm soát" in res.blocked
    assert len(res.negatives) == 1 and res.negatives[0].startswith("20/10")
    n2 = pd.concat([n, _news([("2026-09-01", "ABC được đưa ra khỏi diện kiểm soát")])])
    assert nw.check(n2, asof, nw.DEFAULTS).blocked == ""
    old = _news([("2026-01-01", "HOSE đưa cổ phiếu ABC vào diện kiểm soát")])
    assert nw.check(old, asof, nw.DEFAULTS).blocked == ""          # quá 180 ngày


def _inputs(**kw):
    base = dict(symbol="ABC", price=10_000, date=pd.Timestamp("2026-10-22"), fa_score=70,
                ta_score=60, z_em=3.0, dcf_upside=0.4, pe=8, pb=1, vol_annual=0.3, prob_up=0.5,
                first_target_rr=2.5, fair_value=14_000, fair_upside=0.4)
    base.update(kw)
    return Inputs(**base)


def test_engine_block_caps_at_hold():
    cfg = {"weight_fa": 60, "weight_ta": 40, "rating_thresholds": [0.15, 0.05, -0.05, -0.15],
           "timing_rule": "none"}
    assert recommend(_inputs(), cfg).action == "MUA"
    rec = recommend(_inputs(blocked="diện kiểm soát"), cfg)
    assert rec.action == "NẮM GIỮ" and rec.timing.startswith("Không mua")
    assert any("chặn mua" in r and "tối đa NẮM GIỮ" in r for r in rec.reasons)


def test_card_quality_warnings_downgrade_per_source():
    from vnsa.report.card import quality_warnings

    a = SimpleNamespace(forensic=fx.Forensic(2025, warnings=["Beneish M-Score -1,2 > -1,78"]),
                        news=nw.NewsCheck(pd.DataFrame(), negatives=["20/10 ABC bị xử phạt"]))
    warn, steps = quality_warnings(a)
    assert steps == 2 and any(w.startswith("BCTC:") for w in warn)
    a2 = SimpleNamespace(forensic=fx.Forensic(2025), news=None)
    assert quality_warnings(a2) == ([], 0)
