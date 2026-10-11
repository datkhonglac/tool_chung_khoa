"""Đợt 3: chỉ số ngân hàng / công ty chứng khoán từ khoản mục BCTC (fa/sector.py)."""

import numpy as np
import pandas as pd
import pytest

from vnsa.data.providers.vnstock_provider import standardize
from vnsa.fa import sector
from vnsa.fa.scores import financial_score

B = 1_000_000_000


def _bank_q(n=4, nii=10 * B):
    rows = []
    for i in range(n):
        rows.append({"symbol": "BNK", "year": 2025 + (i + 2) // 4, "quarter": (i + 2) % 4 + 1,
                     "nii": nii, "toi": 14 * B, "bank_opex": 5 * B, "credit_provision": 2 * B,
                     "loans_gross": (900 + 100 * i) * B, "loan_loss_reserve": 12 * B,
                     "customer_deposits": 1100 * B, "sbv_balances": 50 * B,
                     "interbank_placements": 150 * B, "trading_securities": 0.0,
                     "investment_securities": 100 * B, "equity": 120 * B,
                     "total_assets": 1500 * B})
    return pd.DataFrame(rows)


def _bank_y():
    return pd.DataFrame({"symbol": "BNK", "year": [2024, 2025], "nii": [30 * B, 36 * B],
                         "loans_gross": [800 * B, 1000 * B], "equity": [100 * B, 115 * B],
                         "total_assets": [1300 * B, 1450 * B]})


def test_bank_metrics_from_ttm():
    q = _bank_q()
    assert list(zip(q["year"], q["quarter"], strict=True)) == [(2025, 3), (2025, 4), (2026, 1),
                                                               (2026, 2)]
    m = sector.bank(_bank_y(), q)
    assert m.basis == "TTM Q2/2026"
    ea_first, ea_last = 50 + 150 + 100 + 900, 50 + 150 + 100 + 1200
    assert m.values["nim"] == pytest.approx(40 / ((ea_first + ea_last) / 2))
    assert m.values["cir"] == pytest.approx(20 / 56)
    assert m.values["credit_cost"] == pytest.approx(8 / 1050)
    assert m.values["llr"] == pytest.approx(12 / 1200)
    assert m.values["ldr"] == pytest.approx(1200 / 1100)
    assert m.values["loan_growth"] == pytest.approx(0.25)              # BCTC năm 800 -> 1000
    assert "nợ xấu (NPL)" in m.missing and "CASA" in m.missing


def test_bank_metrics_fall_back_to_annual_and_none_without_items():
    y = _bank_y().assign(toi=[50 * B, 60 * B], bank_opex=[20 * B, 21 * B])
    m = sector.bank(y, None)
    assert m.basis == "năm 2025" and m.values["cir"] == pytest.approx(21 / 60)
    assert m.values["nim"] == pytest.approx(36 / 900)                  # chỉ có dư nợ: bình quân 2 năm
    assert sector.bank(y.drop(columns="nii"), None) is None
    assert sector.metrics("general", y) is None


def test_securities_metrics():
    fin = pd.DataFrame({"symbol": "CK", "year": [2025], "equity": [10_000 * B],
                        "margin_loans": [15_000 * B], "fvtpl_assets": [5_000 * B],
                        "brokerage_revenue": [1_000 * B], "operating_revenue": [4_000 * B]})
    m = sector.securities(fin)
    assert m.values == pytest.approx({"margin_to_equity": 1.5, "fvtpl_to_equity": 0.5,
                                      "brokerage_share": 0.25})


def test_financial_score_uses_sector_groups():
    fin = pd.DataFrame({"year": [2023, 2024, 2025], "net_income_parent": [10 * B, 12 * B, 14 * B],
                        "net_income": [10 * B, 12 * B, 14 * B], "equity": [80 * B, 90 * B, 100 * B],
                        "total_assets": [1000 * B] * 3})
    cfg = {"weights": {"sinh_loi": 30, "tang_truong": 15, "an_toan": 10, "hieu_qua": 10,
                       "rui_ro": 10, "dinh_gia": 25},
           "thresholds": {"roe": [0.08, 0.20], "ni_cagr": [0.0, 0.20], "pb_rel": [1.3, 0.7],
                          "fair_upside": [-0.2, 0.3], "nim": [0.025, 0.045], "cir": [0.5, 0.3],
                          "credit_cost": [0.025, 0.005], "margin_to_equity": [1.8, 0.8]},
           "equity_to_assets": {"bank": [0.05, 0.11]}}
    sm = sector.SectorMetrics("bank", "năm 2025", {"nim": 0.045, "cir": 0.30, "credit_cost": 0.015},
                              ["CASA"])
    s = financial_score(fin, cfg, "bank", sector=sm)
    assert s.groups["hieu_qua"] == 100 and s.groups["rui_ro"] == pytest.approx(50)
    assert any("CASA" in n for n in s.notes)
    bare = financial_score(fin, cfg, "bank")
    assert np.isnan(bare.groups["hieu_qua"]) and any("fetch-bctc" in n for n in bare.notes)


def test_standardize_reads_bank_items_by_id():
    def stmt(items):
        return pd.DataFrame([{"item": k, "item_en": k, "item_id": k, "2025": v, "2024": v}
                             for k, v in items.items()])

    statements = {
        "income_statement": stmt({"net_interest_income": 100.0, "total_operating_income": 140.0,
                                  "general_and_admin_expenses": -50.0,
                                  "provision_for_credit_losses": -20.0}),
        "balance_sheet": stmt({"loans_and_advances_to_customers": 900.0,
                               "less_provision_for_losses_on_loans_and_advances_to_customers": -9.0,
                               "deposits_from_customers": 1000.0}),
        "cash_flow": stmt({"net_cash_flows_from_operating_activities": 1.0}),
    }
    df, missing = standardize(statements, "vcb")
    last = df.iloc[-1]
    assert last["nii"] == 100 and last["bank_opex"] == 50 and last["credit_provision"] == 20
    assert last["loan_loss_reserve"] == 9 and last["customer_deposits"] == 1000
    assert "nii" not in missing                                        # không báo thiếu khoản ngành


def test_manual_bank_file_fills_missing_items(tmp_path):
    f = tmp_path / "bank_manual.toml"
    f.write_text('[BNK."2026-Q1"]\nnpl = 0.02\n[BNK."2026-Q2"]\nnpl = 0.015\ncasa = 0.3\n'
                 '# ghi chú\n', encoding="utf-8")
    period, vals = sector.manual_bank("bnk", f)
    assert period == "2026-Q2" and vals == {"npl": 0.015, "casa": 0.3}       # kỳ mới nhất
    assert sector.manual_bank("XYZ", f) is None
    m = sector.bank(_bank_y(), _bank_q(), (period, vals))
    assert m.values["npl"] == 0.015 and m.manual == "2026-Q2"
    assert m.missing == ["bao phủ nợ xấu", "CAR"]                            # chỉ còn thiếu 2


def test_financial_score_manual_and_peer_relative():
    fin = pd.DataFrame({"year": [2023, 2024, 2025], "net_income_parent": [10 * B, 12 * B, 14 * B],
                        "net_income": [10 * B, 12 * B, 14 * B], "equity": [80 * B, 90 * B, 100 * B],
                        "total_assets": [1000 * B] * 3})
    cfg = {"weights": {"sinh_loi": 30, "tang_truong": 15, "an_toan": 10, "hieu_qua": 10,
                       "rui_ro": 10, "chat_luong_tai_san": 10, "dinh_gia": 25},
           "thresholds": {"roe": [0.08, 0.20], "ni_cagr": [0.0, 0.20], "pb_rel": [1.3, 0.7],
                          "fair_upside": [-0.2, 0.3], "nim": [0.025, 0.045], "cir": [0.5, 0.3],
                          "credit_cost": [0.025, 0.005], "margin_to_equity": [1.8, 0.8],
                          "npl": [0.03, 0.01], "npl_coverage": [0.5, 1.5], "casa": [0.1, 0.4],
                          "car": [0.09, 0.13]},
           "relative": {"nim": [0.8, 1.2], "cir": [1.2, 0.8], "credit_cost": [1.5, 0.5]},
           "equity_to_assets": {"bank": [0.05, 0.11]}}
    sm = sector.SectorMetrics("bank", "TTM", {"nim": 0.027, "cir": 0.34, "credit_cost": 0.003,
                                              "npl": 0.01, "npl_coverage": 1.5}, ["CASA", "CAR"],
                              "2026-Q2")
    absolute = financial_score(fin, cfg, "bank", sector=sm)
    assert absolute.items["nim"][1] == pytest.approx(10)                  # 2,7% gần mức xấu 2,5%
    assert absolute.groups["chat_luong_tai_san"] == 100
    peer = {"nim": 0.027, "cir": 0.40, "credit_cost": 0.01, "n": 5}
    rel = financial_score(fin, cfg, "bank", sector=sm, peer=peer)
    assert rel.items["nim"][1] == pytest.approx(50)                       # bằng trung vị ngành
    assert rel.items["credit_cost"][1] == 100                             # 0,3% so 1% ngành
    assert any("5 ngân hàng" in n for n in rel.notes)
    assert any("nhập tay kỳ 2026-Q2" in n for n in rel.notes)
