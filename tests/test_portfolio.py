"""Sổ lệnh: giá vốn bình quân, T+2 theo lịch giao dịch, lãi đã chốt, cắt lỗ, sự kiện quyền."""

import numpy as np
import pandas as pd
import pytest

from vnsa import portfolio as pf

CAL = pd.bdate_range("2026-10-01", "2026-10-16")      # lịch giao dịch giả (thứ Hai–Sáu)


def _add(file, side, sym, qty, price, date, **kw):
    return pf.add(side, sym, qty, price, date, file=file, calendar=CAL, **kw)


def test_sessions_after_uses_calendar_then_weekdays():
    cal = pd.DatetimeIndex(["2026-10-01", "2026-10-02", "2026-10-06"])   # 05/10 nghỉ
    assert pf.sessions_after(pd.Timestamp("2026-10-01"), pd.Timestamp("2026-10-06"), cal) == 2
    # sau ngày cuối của lịch: đếm ngày thường (07, 08/10)
    assert pf.sessions_after(pd.Timestamp("2026-10-06"), pd.Timestamp("2026-10-08"), cal) == 2
    assert pf.sessions_after(pd.Timestamp("2026-10-06"), pd.Timestamp("2026-10-06"), cal) == 0


def test_average_cost_and_realized(tmp_path):
    f = tmp_path / "trades.csv"
    _add(f, pf.BUY, "fpt", 100, 100_000, "2026-10-01", fee=0)
    _add(f, pf.BUY, "FPT", 100, 120_000, "2026-10-02", fee=0)
    _add(f, pf.SELL, "FPT", 50, 130_000, "2026-10-08", fee=0, tax=0)
    p = pf.positions(pf.load(f), "2026-10-08", CAL)["FPT"]
    assert p.quantity == 150 and p.avg_cost == pytest.approx(110_000)
    assert p.realized == pytest.approx(50 * (130_000 - 110_000))


def test_default_fee_and_tax(tmp_path):
    f = tmp_path / "trades.csv"
    row = _add(f, pf.BUY, "HPG", 1_000, 25_000, "2026-10-01")
    assert row["fee"] == pytest.approx(25_000_000 * 0.0015) and row["tax"] == 0
    row = _add(f, pf.SELL, "HPG", 1_000, 26_000, "2026-10-05")
    assert row["tax"] == pytest.approx(26_000_000 * 0.001)


def test_t2_blocks_early_sell(tmp_path):
    f = tmp_path / "trades.csv"
    _add(f, pf.BUY, "MWG", 200, 60_000, "2026-10-07")        # thứ Tư
    with pytest.raises(ValueError, match="T\\+2"):
        _add(f, pf.SELL, "MWG", 100, 61_000, "2026-10-08")   # T+1: chưa về
    _add(f, pf.SELL, "MWG", 100, 61_000, "2026-10-09")       # T+2: bán được
    # mua thêm lô mới: chỉ phần lô cũ bán được ngay
    _add(f, pf.BUY, "MWG", 300, 59_000, "2026-10-09")
    p = pf.positions(pf.load(f), "2026-10-12", CAL)["MWG"]
    assert p.quantity == 400 and p.sellable == 100           # 12/10 mới là T+1 của lô 09/10


def test_stop_and_status(tmp_path):
    f = tmp_path / "trades.csv"
    _add(f, pf.BUY, "VCB", 100, 90_000, "2026-10-01", fee=0)
    with pytest.raises(ValueError, match="chưa có lệnh mua"):
        _add(f, pf.STOP, "ACB", 0, 20_000, "2026-10-01")
    _add(f, pf.STOP, "VCB", 0, 85_000, "2026-10-02")
    tab = pf.status(pf.load(f), {"VCB": 95_000}, {"VCB": "Ngân hàng"}, 100_000_000,
                    "2026-10-08", CAL)
    r = tab.iloc[0]
    assert r["Lãi/lỗ"] == pytest.approx(500_000) and r["Tỷ trọng"] == pytest.approx(0.095)
    assert r["Cách cắt lỗ"] == pytest.approx(95_000 / 85_000 - 1)


def test_rejects_price_in_thousands_and_undo(tmp_path):
    f = tmp_path / "trades.csv"
    with pytest.raises(ValueError, match="đồng"):
        _add(f, pf.BUY, "FPT", 100, 98.5, "2026-10-01")
    _add(f, pf.BUY, "FPT", 100, 98_500, "2026-10-01")
    _add(f, pf.BUY, "FPT", 100, 99_000, "2026-10-02")
    assert "99000" in pf.undo(f)
    assert len(pf.load(f)) == 1


def test_correlation_factor():
    assert pf.correlation_factor(0.5) == 1.0
    assert pf.correlation_factor(0.8) == pytest.approx(0.75)
    assert pf.correlation_factor(1.0) == pytest.approx(0.5)


def _px(rets, start=100_000.0):
    import numpy as np

    close = start * np.cumprod(1 + np.asarray(rets))
    return pd.DataFrame({"date": pd.bdate_range("2026-03-02", periods=len(close)), "close": close})


def test_sizing_limits_correlation_and_sector(tmp_path):
    import numpy as np

    f = tmp_path / "trades.csv"
    _add(f, pf.BUY, "VCB", 200, 90_000, "2026-10-01", fee=0)
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.02, 150)
    hist = {"VCB": _px(base), "FPT": _px(rng.normal(0, 0.02, 150))}
    ctx = (CAL, {"VCB": 100_000}, {"VCB": "Ngân hàng", "TCB": "Ngân hàng", "FPT": "Công nghệ"},
           hist)
    # mã mới đi gần như cùng VCB (tương quan cao) và cùng nhóm ngân hàng
    adj, caps = pf.sizing_limits("TCB", _px(base + rng.normal(0, 0.003, 150)), 100_000_000,
                                 trades=pf.load(f), context=ctx)
    assert adj is not None and 0.5 <= adj[0] < 0.6 and "VCB" in adj[1]
    assert caps[0][0] == pytest.approx(0.30 * 100_000_000 - 200 * 100_000)
    # mã khác ngành, không tương quan: không giảm rủi ro, trần nhóm còn nguyên 30%
    adj, caps = pf.sizing_limits("FPT", hist["FPT"], 100_000_000, trades=pf.load(f), context=ctx)
    assert adj is None and caps[0][0] == pytest.approx(30_000_000)


def test_sizing_limits_empty_ledger():
    assert pf.sizing_limits("FPT", _px([0.01] * 10), 1e8, trades=pd.DataFrame(columns=pf.COLUMNS)) \
        == (None, [])


def test_trade_plan_applies_portfolio_limits():
    from vnsa.ta.plan import trade_plan

    cfg = {"atr_stop": 2.0, "support_buffer_atr": 0.5, "max_stop_atr": 3.0, "min_stop_atr": 1.0,
           "reward_multiples": [2.0, 3.0], "capital": 100_000_000, "risk_per_trade": 0.01,
           "lot_size": 100, "max_position_pct": 0.15}
    base = trade_plan(50_000, 1_000, [], [], cfg)
    cut = trade_plan(50_000, 1_000, [], [], cfg, risk_adjust=(0.5, "tương quan cao"),
                     extra_caps=[(5_000_000, "trần nhóm")])
    assert base.shares == 300 and cut.shares == 100
    assert "tương quan cao" in cut.notes and any("trần nhóm" in n for n in cut.notes)


def _acts(**rows):
    return {sym: pd.DataFrame([{"exright_date": pd.Timestamp(d), "cash_per_share": c,
                                "stock_ratio": r} for d, c, r in evs])
            for sym, evs in rows.items()}


def test_cash_dividend_adjusts_stop_and_counts_income(tmp_path):
    """v2.1: cổ tức tiền — nhận tiền sau thuế 5%, cắt lỗ hạ đúng bằng cổ tức, giá vốn giữ nguyên."""
    f = tmp_path / "trades.csv"
    _add(f, pf.BUY, "VNM", 1_000, 60_000, "2026-10-01", fee=0)
    _add(f, pf.STOP, "VNM", 0, 56_000, "2026-10-02")
    acts = _acts(VNM=[("2026-10-08", 2_000, np.nan)])
    p = pf.positions(pf.load(f), "2026-10-12", CAL, acts)["VNM"]
    assert p.stop == pytest.approx(54_000) and p.avg_cost == pytest.approx(60_000)
    assert p.dividends == pytest.approx(1_000 * 2_000 * 0.95)
    # giá sau GDKHQ 57.500 (đã trừ cổ tức): KHÔNG báo chạm cắt lỗ
    tab = pf.status(pf.load(f), {"VNM": 57_500}, {}, 1e8, "2026-10-12", CAL, acts)
    assert tab.iloc[0]["Cách cắt lỗ"] > 0 and tab.iloc[0]["Cổ tức nhận"] == pytest.approx(1.9e6)


def test_stock_dividend_adds_shares_and_lowers_cost(tmp_path):
    f = tmp_path / "trades.csv"
    _add(f, pf.BUY, "FPT", 1_050, 100_000, "2026-10-01", fee=0)
    _add(f, pf.STOP, "FPT", 0, 90_000, "2026-10-02")
    acts = _acts(FPT=[("2026-10-08", np.nan, 0.10)])
    p = pf.positions(pf.load(f), "2026-10-12", CAL, acts)["FPT"]
    assert p.quantity == 1_155                                  # floor(1.050 × 10%) = 105
    assert p.avg_cost == pytest.approx(105_000_000 / 1_155)
    assert p.stop == pytest.approx(90_000 / 1.1)
    assert p.sellable == 1_050                                  # cp thưởng chưa niêm yết


def test_dividend_only_for_shares_held_before_exdate(tmp_path):
    f = tmp_path / "trades.csv"
    _add(f, pf.BUY, "HPG", 1_000, 25_000, "2026-10-01", fee=0)
    _add(f, pf.BUY, "HPG", 500, 25_000, "2026-10-08", fee=0)     # mua đúng ngày GDKHQ: không hưởng
    acts = _acts(HPG=[("2026-10-08", 500, np.nan)])
    p = pf.positions(pf.load(f), "2026-10-12", CAL, acts)["HPG"]
    assert p.dividends == pytest.approx(1_000 * 500 * 0.95)
