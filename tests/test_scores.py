import numpy as np
import pandas as pd
import pytest

from vnsa.fa import scores as sc

B = 1_000_000_000


def _fin():
    """Hai năm: năm 2025 cải thiện ở mọi tiêu chí F-Score."""
    return pd.DataFrame({
        "year": [2024, 2025],
        "revenue": [800 * B, 1000 * B],
        "gross_profit": [200 * B, 300 * B],          # biên gộp 25% -> 30%
        "net_income": [40 * B, 60 * B],
        "net_income_parent": [40 * B, 60 * B],
        "total_assets": [1000 * B, 1000 * B],         # ROA 4% -> 6%, vòng quay 0,8 -> 1,0
        "total_liabilities": [500 * B, 400 * B],
        "equity": [500 * B, 600 * B],
        "long_debt": [200 * B, 100 * B],             # vay dài hạn/TS 20% -> 10%
        "short_debt": [100 * B, 100 * B],
        "cfo": [50 * B, 90 * B],                     # CFO > LNST
        "current_assets": [400 * B, 500 * B],
        "current_liabilities": [400 * B, 250 * B],   # thanh toán hiện hành 1,0 -> 2,0
        "retained_earnings": [100 * B, 150 * B],
        "pbt": [50 * B, 75 * B],
        "interest_expense": [20 * B, 15 * B],
        "paid_in_capital": [300 * B, 300 * B],
        "equity_issued": [0.0, 0.0],
    })


def test_piotroski_all_pass_and_first_year_incomplete():
    f = sc.piotroski(_fin())
    assert f.loc[1, "f_score"] == 9
    assert np.isnan(f.loc[0, "f_score"])        # năm đầu không có năm trước để so
    assert f.loc[0, "f_roa"] == 1 and f.loc[0, "f_cfo"] == 1


def test_piotroski_failures_and_share_issue_fallback():
    fin = _fin()
    fin.loc[1, "cfo"] = 50 * B                    # CFO < LNST -> trượt tiêu chí dồn tích
    fin.loc[1, "equity_issued"] = 80 * B          # có phát hành thu tiền
    f = sc.piotroski(fin).iloc[1]
    assert (f["f_accrual"], f["f_no_issue"], f["f_score"]) == (0, 0, 7)

    fin = _fin()
    fin.loc[1, "equity_issued"] = 5 * B           # ESOP nhỏ (< 1% vốn chủ 600 tỷ) -> bỏ qua
    assert sc.piotroski(fin).iloc[1]["f_no_issue"] == 1

    fin = _fin()
    fin["equity_issued"] = np.nan                 # nguồn không có: so vốn góp
    fin.loc[1, "paid_in_capital"] = 330 * B       # vốn góp tăng (kể cả cổ phiếu thưởng)
    assert sc.piotroski(fin).iloc[1]["f_no_issue"] == 0


def test_piotroski_no_long_debt_both_years_counts_as_pass():
    fin = _fin()
    fin["long_debt"] = 0.0
    assert sc.piotroski(fin).iloc[1]["f_dlever"] == 1


def test_altman_by_hand():
    z = sc.altman(_fin(), market_caps={2025: 1200 * B}).iloc[1]
    x1, x2, x3 = (500 - 250) / 1000, 150 / 1000, (75 + 15) / 1000
    x4b, x4m, x5 = 600 / 400, 1200 / 400, 1000 / 1000
    assert z["z_em"] == pytest.approx(6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4b)
    assert z["z_orig"] == pytest.approx(1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4m + x5)
    assert np.isnan(sc.altman(_fin()).iloc[1]["z_orig"])   # thiếu vốn hoá -> không đoán
    assert sc.z_zone(3.0) == "an toàn" and sc.z_zone(2.0) == "vùng xám"
    assert sc.z_zone(1.0) == "nguy cơ" and sc.z_zone(2.0, "orig") == "vùng xám"


def test_year_end_market_cap_uses_last_session_of_year():
    prices = pd.DataFrame({"date": pd.to_datetime(["2024-12-30", "2024-12-31", "2025-06-30"]),
                           "close": [10_000.0, 11_000.0, 12_000.0]})
    caps = sc.year_end_market_caps(_fin(), prices)
    assert caps == {2024: 11_000 * 30_000_000, 2025: 12_000 * 30_000_000}


def test_scale_handles_both_directions():
    assert sc.scale(0.15, 0.05, 0.25) == pytest.approx(50)
    assert sc.scale(0.50, 0.05, 0.25) == 100 and sc.scale(-1, 0.05, 0.25) == 0
    assert sc.scale(16.5, 25, 8) == pytest.approx(50)       # P/E: nhỏ là tốt
    assert np.isnan(sc.scale(np.nan, 0, 1))


def test_fa_score_reweights_when_valuation_missing():
    cfg = {"weights": {"sinh_loi": 25, "tang_truong": 20, "suc_khoe": 20, "chat_luong": 15,
                       "dinh_gia": 20},
           "thresholds": {"roe": [0.05, 0.25], "ni_cagr": [0.0, 0.20], "z_em": [1.1, 2.6],
                          "debt_to_equity": [1.5, 0.3], "f_score": [2, 8], "cfo_to_ni": [0.5, 1.0],
                          "pe_rel": [1.3, 0.7], "pb_rel": [1.3, 0.7], "dcf_upside": [-0.2, 0.3]}}
    s = sc.fa_score(_fin(), cfg)
    assert np.isnan(s.groups["dinh_gia"])
    expected = sum(s.groups[g] * w for g, w in cfg["weights"].items() if g != "dinh_gia") / 80
    assert s.total == pytest.approx(expected)
    assert s.items["f_score"] == (9.0, 100.0)
    assert s.items["ni_cagr"][0] == pytest.approx(0.5)          # 40 -> 60 tỷ trong 1 năm
    assert s.items["cfo_to_ni"][0] == pytest.approx(140 / 100)  # cộng dồn, không lấy TB tỷ lệ
    with_val = sc.fa_score(_fin(), cfg, pe=8, pb=1, dcf_upside=0.3, pe_hist=12, pb_hist=1.5)
    assert with_val.groups["dinh_gia"] == 100                  # rẻ hơn lịch sử 33% -> 100 điểm
    pricey = sc.fa_score(_fin(), cfg, pe=18, pb=2.25, pe_hist=12, pb_hist=1.5)
    assert pricey.items["pe_rel"] == (1.5, 0.0)                # đắt hơn lịch sử 50% -> 0 điểm
    assert np.isnan(sc.fa_score(_fin(), cfg, pe=8).items["pe_rel"][0])   # thiếu mốc lịch sử
