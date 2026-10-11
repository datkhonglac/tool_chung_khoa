import math

import pandas as pd

from vnsa.fa.ratios import compute_ratios, format_ratios

B = 1_000_000_000  # 1 tỷ đồng


def _fin():
    # Số tròn để tính tay được.
    return pd.DataFrame({
        "year": [2023, 2024],
        "revenue": [1000 * B, 1200 * B],
        "gross_profit": [400 * B, 450 * B],
        "net_income": [100 * B, 150 * B],
        "net_income_parent": [90 * B, 132 * B],
        "eps": [2000, 2500],
        "total_assets": [2000 * B, 2200 * B],
        "total_liabilities": [1200 * B, 1300 * B],
        "equity": [800 * B, 900 * B],
        "minority_interest": [0, 0],
        "short_debt": [300 * B, 250 * B],
        "long_debt": [100 * B, 200 * B],
        "cfo": [120 * B, 90 * B],
    })


def test_ratios_match_hand_calculation():
    r = compute_ratios(_fin()).set_index("year")
    # Năm đầu: không có đầu kỳ -> dùng cuối kỳ
    assert math.isclose(r.loc[2023, "roe"], 90 / 800)
    assert math.isclose(r.loc[2023, "roa"], 100 / 2000)
    # Năm sau: bình quân đầu - cuối kỳ
    assert math.isclose(r.loc[2024, "roe"], 132 / ((800 + 900) / 2))     # 15,53%
    assert math.isclose(r.loc[2024, "roa"], 150 / ((2000 + 2200) / 2))   # 7,14%
    assert math.isclose(r.loc[2024, "gross_margin"], 450 / 1200)
    assert math.isclose(r.loc[2024, "net_margin"], 150 / 1200)
    assert math.isclose(r.loc[2024, "debt_to_equity"], (250 + 200) / 900)
    assert math.isclose(r.loc[2024, "liabilities_to_equity"], 1300 / 900)
    assert math.isclose(r.loc[2024, "cfo_to_net_income"], 90 / 150)
    assert math.isclose(r.loc[2024, "eps_growth"], 2500 / 2000 - 1)
    assert math.isclose(r.loc[2024, "net_income_parent_growth"], 132 / 90 - 1)
    assert math.isnan(r.loc[2023, "eps_growth"])


def test_roe_excludes_minority_interest():
    fin = _fin()
    fin["minority_interest"] = [100 * B, 150 * B]   # VCSH công ty mẹ: 700 rồi 750
    r = compute_ratios(fin).set_index("year")
    assert math.isclose(r.loc[2024, "roe"], 132 / ((700 + 750) / 2))
    assert math.isclose(r.loc[2024, "roa"], 150 / ((2000 + 2200) / 2))   # ROA không đổi


def test_missing_items_give_nan_not_guess():
    fin = _fin().drop(columns=["gross_profit", "short_debt", "long_debt"])
    fin.loc[0, "eps"] = -500        # EPS năm trước âm -> không tính tăng trưởng
    fin.loc[1, "revenue"] = 0       # chia cho 0 -> NaN
    r = compute_ratios(fin).set_index("year")
    assert r["gross_margin"].isna().all()
    assert r["debt_to_equity"].isna().all()
    assert math.isnan(r.loc[2024, "net_margin"])
    assert math.isnan(r.loc[2024, "eps_growth"])


def test_year_gap_does_not_use_wrong_opening_balance():
    fin = _fin()
    fin.loc[1, "year"] = 2025       # thiếu năm 2024
    r = compute_ratios(fin).set_index("year")
    assert math.isclose(r.loc[2025, "roe"], 132 / 900)
    assert math.isnan(r.loc[2025, "eps_growth"])


def test_format_ratios_shows_percent():
    t = format_ratios(compute_ratios(_fin()))
    assert t.loc["ROE (%)", 2024] == 15.5
    assert t.loc["Vay/VCSH (lần)", 2024] == 0.5


def _vci_frame(rows):
    """Giả lập bảng vnstock nguồn VCI: item, item_en, item_id, cột năm mới -> cũ."""
    return pd.DataFrame(
        [{"item": en, "item_en": en, "item_id": iid, "2025": a, "2024": b} for iid, en, a, b in rows]
    )


def test_vnstock_vci_tables_are_standardized():
    from vnsa.data.providers.vnstock_provider import VnstockProvider

    # Số thật của FPT (VCI, hợp nhất) lấy từ lần chạy thử, tỷ đồng
    statements = {
        "income_statement": _vci_frame([
            ("sales", "Sales", 70207.69 * B, 62962.65 * B),
            ("net_sales", "Net sales", 70112.83 * B, 62848.79 * B),
            ("gross_profit", "Gross Profit", 25888.53 * B, 23698.35 * B),
            ("net_profit_loss_after_tax", "Net profit/(loss) after tax", 11232.34 * B, 9427.42 * B),
            ("attributable_to_parent_company", "Attributable to parent company",
             9376.13 * B, 7856.77 * B),
            ("eps_basic_vnd", "EPS basic (VND)", 5216, 4292),
        ]),
        "balance_sheet": _vci_frame([
            ("short_term_investments", "Short-term investments", 29630.99 * B, 21785.21 * B),
            ("short_term_investments", "Short-term investments", 0, 0),   # trùng id, lấy dòng đầu
            ("total_assets", "Total Assets", 88141.99 * B, 72000.0 * B),
            ("liabilities", "Liabilities", 44393.95 * B, 36272.46 * B),
            ("short_term_borrowings", "Short-term borrowings", 19169.70 * B, 14446.24 * B),
            ("long_term_borrowings", "Long-term borrowings", 1903.79 * B, 501.12 * B),
            ("owners_equity", "OWNER'S EQUITY", 43748.04 * B, 35727.54 * B),
            ("minority_interests_before_2015", "Minority interests", 0, 0),   # dòng cũ toàn 0
            ("minority_interests", "Minority interests", 6900.0 * B, 5700.0 * B),
        ]),
        "cash_flow": _vci_frame([
            ("net_cash_inflows_outflows_from_operating_activities",
             "Net cash inflows/(outflows) from operating activities", 12000 * B, 10000 * B),
        ]),
    }
    prov = VnstockProvider(get_statements=lambda sym: statements)
    df = prov.fundamentals("fpt")
    core = ["revenue", "gross_profit", "net_income", "net_income_parent", "eps", "total_assets",
            "total_liabilities", "equity", "minority_interest", "short_debt", "long_debt", "cfo"]
    assert not set(core) & set(prov.missing)
    assert df["year"].tolist() == [2024, 2025]
    assert df["symbol"].tolist() == ["FPT", "FPT"]
    row = df.set_index("year").loc[2025]
    assert row["revenue"] == 70112.83 * B           # doanh thu THUẦN, không phải doanh thu gộp
    assert row["eps"] == 5216
    assert row["equity"] == 43748.04 * B
    r = compute_ratios(df).set_index("year")
    assert row["minority_interest"] == 6900.0 * B   # bỏ qua dòng toàn số 0
    assert math.isclose(r.loc[2025, "roe"], 9376.13 / ((43748.04 - 6900 + 35727.54 - 5700) / 2))


def test_missing_statement_items_are_reported():
    from vnsa.data.providers.vnstock_provider import standardize

    statements = {"income_statement": _vci_frame([("net_sales", "Net sales", 10.0, 8.0)])}
    df, missing = standardize(statements, "X")
    assert "equity" in missing and "revenue" not in missing
    assert df["equity"].isna().all()


def test_store_fundamentals_roundtrip(tmp_path):
    from vnsa.data.store import Store

    fin = _fin().assign(symbol="ABC")
    with Store(tmp_path / "w.duckdb") as store:
        assert store.write_fundamentals(fin, source="test") == 2
        store.write_fundamentals(fin, source="test")    # nạp lại không bị trùng
        out = store.fundamentals("abc")
    assert out["year"].tolist() == [2023, 2024]
    assert out.loc[1, "equity"] == 900 * B
