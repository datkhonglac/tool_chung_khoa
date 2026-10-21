"""Sổ lệnh: giá vốn bình quân, T+2 theo lịch giao dịch, lãi đã chốt, cắt lỗ."""

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
