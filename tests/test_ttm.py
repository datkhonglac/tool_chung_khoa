"""Đợt 1 dữ liệu: BCTC quý + TTM (fa/ttm.py), hồ sơ và cổ tức (vnstock_provider), giá chưa điều
chỉnh và bội số lịch sử theo vốn hoá thật (fa/fair_value.py)."""

import numpy as np
import pandas as pd
import pytest
from conftest import _fundamentals

from vnsa.data.providers.csv_provider import normalize_prices
from vnsa.data.providers.vnstock_provider import (
    standardize,
    standardize_dividends,
    standardize_profile,
)
from vnsa.fa import fair_value as fv
from vnsa.fa.ttm import current, ttm

B = 1_000_000_000


def _quarters(n=4, start=(2025, 3)):
    y, q = start
    rows = []
    for i in range(n):
        rows.append({"symbol": "X", "year": y, "quarter": q, "net_income_parent": (i + 1) * B,
                     "net_income": (i + 1) * B, "revenue": 10 * B, "cfo": 2 * B,
                     "equity": (100 + i) * B, "minority_interest": 0.0, "total_assets": 300 * B,
                     "paid_in_capital": 50 * B, "eps": 999.0})
        q += 1
        if q == 5:
            y, q = y + 1, 1
    return pd.DataFrame(rows)


def test_ttm_sums_flows_and_takes_latest_balances():
    t = ttm(_quarters())
    assert t["net_income_parent"] == 10 * B and t["revenue"] == 40 * B     # 1+2+3+4 tỷ
    assert t["equity"] == 103 * B and t["equity_parent_start"] == 100 * B
    assert int(t["year"]) == 2026 and int(t["quarter"]) == 2 and np.isnan(t["eps"])
    assert ttm(_quarters(3)) is None                                       # chưa đủ 4 quý
    gap = _quarters().drop(index=1)
    assert ttm(pd.concat([gap, _quarters(1, (2026, 3))])) is None          # thiếu quý giữa


def test_current_prefers_newer_ttm_and_current_shares():
    fin = _fundamentals("X")                                               # năm 2022–2025
    c = current(fin, _quarters(), shares_out=4e6)
    assert c.basis == "TTM Q2/2026"
    assert c.roe == pytest.approx(10 * B / ((103 + 100) / 2 * B))
    assert c.eps == pytest.approx(10 * B / 4e6) and c.bvps == pytest.approx(103 * B / 4e6)
    old_q = _quarters(start=(2024, 3))                                     # TTM cũ hơn BCTC năm
    c2 = current(fin, old_q, shares_out=np.nan)
    assert c2.basis == "năm 2025" and c2.shares == pytest.approx(300 * B / 10_000)


def test_standardize_quarterly_vci_columns():
    def stmt(items):
        rows = [{"item": k, "item_en": k, "item_id": k, "2026-Q2": v, "2026-Q1": v / 2,
                 "2025-Q4": v / 4, "2025-Q3": v / 8} for k, v in items.items()]
        return pd.DataFrame(rows)

    statements = {"income_statement": stmt({"net_sales": 800.0,
                                            "attributable_to_parent_company": 80.0}),
                  "balance_sheet": stmt({"owners_equity": 1000.0}),
                  "cash_flow": stmt({"purchases_of_fixed_assets_and_other_long_term_assets": -40.0})}
    df, missing = standardize(statements, "fpt", quarterly=True)
    assert list(df[["year", "quarter"]].itertuples(index=False, name=None)) == [
        (2025, 3), (2025, 4), (2026, 1), (2026, 2)]
    assert df["revenue"].tolist() == [100, 200, 400, 800]
    assert df["capex"].iloc[-1] == 40                                      # chi phí lưu số dương
    assert "net_income_parent" not in missing and "cfo" in missing


def test_standardize_profile_handles_duplicate_issue_share_columns():
    # VCI trả 4 cột issue_share trùng tên (số trước và sau cổ phiếu thưởng) — đã gặp với FPT
    ov = pd.DataFrame([[57900.0, 1.091836e14, 1.885728e9, 1714326422, 1.885728e9, 1714326422,
                        0.28312, 0.49, 0.05634, 0.85, 4.06378e11, 90300.0, "BUY"]],
                      columns=["current_price", "market_cap", "issue_share", "issue_share",
                               "issue_share", "issue_share", "foreigner_percentage",
                               "maximum_foreign_percentage", "state_percentage",
                               "free_float_percentage", "average_match_value1_month",
                               "target_price", "rating"])
    p = standardize_profile(ov)
    assert p["shares_out"] == pytest.approx(1.885728e9, rel=1e-4)        # sau thưởng 10%
    assert p["foreign_pct"] == pytest.approx(0.28312) and p["analyst_rating"] == "BUY"
    assert standardize_profile(pd.DataFrame()) == {}


def test_standardize_dividends_splits_cash_and_stock():
    ev = pd.DataFrame({"category": ["DIVIDEND", "DIVIDEND", "OTHER"],
                       "exright_date": ["2026-06-10", "2026-09-21", "2026-08-26"],
                       "value_per_share": [1000.0, np.nan, np.nan],
                       "exercise_ratio": [np.nan, 0.1, np.nan],
                       "event_title_vi": ["Cổ tức tiền", "Cổ phiếu thưởng 10%", "Niêm yết thêm"]})
    dup = pd.concat([ev, ev.iloc[[0]]])                                    # nguồn trả trùng 1 đợt
    d = standardize_dividends(dup, "fpt")
    assert len(d) == 2 and d["symbol"].iloc[0] == "FPT"
    assert d["cash_per_share"].iloc[0] == 1000 and np.isnan(d["stock_ratio"].iloc[0])
    assert d["stock_ratio"].iloc[1] == 0.1 and np.isnan(d["cash_per_share"].iloc[1])


def test_unadjusted_close_kept_and_scaled():
    raw = pd.DataFrame({"Ngày": ["09/10/2026", "08/10/2026"], "Giá mở cửa": [57.5, 58.0],
                        "Giá cao nhất": [58.2, 58.5], "Giá thấp nhất": [57.1, 57.6],
                        "Giá đóng cửa": [57.9, 58.1], "Giá điều chỉnh": [52.11, 52.29],
                        "Khối lượng khớp lệnh": [1000, 2000]})
    df = normalize_prices(raw, "FPT", price_unit="nghin")
    assert df["close"].iloc[-1] == 52110 and df["close_raw"].iloc[-1] == 57900
    again = normalize_prices(df.drop(columns="symbol"), "FPT")             # đọc lại CSV tool ghi
    assert again["close_raw"].iloc[-1] == 57900 and again["close"].iloc[-1] == 52110


def test_historical_multiples_use_market_cap_from_unadjusted_price():
    fin = _fundamentals("X")             # vốn góp 300 tỷ = 30 triệu cp, LNST cty mẹ 75 tỷ năm 2022
    dates = pd.bdate_range("2022-01-01", "2025-12-31")
    px = pd.DataFrame({"date": dates, "close": 36_000.0, "close_raw": 40_000.0})
    hist = fv.historical_multiples(fin, px)
    assert (hist["basis"] == "giá thật").all()
    assert hist["pe"].iloc[0] == pytest.approx(40_000 * 30e6 / (75 * B))  # giá thật, không phải 36k
    no_raw = fv.historical_multiples(fin, px.drop(columns="close_raw"))
    assert (no_raw["basis"] == "giá điều chỉnh").all()
    assert no_raw["pe"].iloc[0] < hist["pe"].iloc[0]                       # giá điều chỉnh lệch thấp


def test_analysis_uses_ttm_and_profile(warehouse):
    from vnsa import analysis
    from vnsa.data.store import Store

    q = _quarters().assign(symbol="DEMO_A", net_income_parent=[30 * B, 30 * B, 30 * B, 30 * B],
                           equity=[1200 * B] * 4, minority_interest=[40 * B] * 4)
    with Store(warehouse) as store:
        store.write_quarterly(q, "test")
        store.write_profile("DEMO_A", {"shares_out": 33e6})
    d = analysis.load("DEMO_A")
    assert d.current.basis == "TTM Q2/2026" and d.current.shares == 33e6
    pe, pb = analysis.multiples(d)
    price = float(d.prices["close"].iloc[-1])
    assert pe == pytest.approx(price / (120 * B / 33e6))
    assert pb == pytest.approx(price / (1160 * B / 33e6))
    card = analysis.report_card("DEMO_A")
    assert card.basis == "TTM Q2/2026"
