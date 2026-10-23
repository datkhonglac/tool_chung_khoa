"""Nhật ký giao dịch: TWR phần vốn giải ngân, thống kê lệnh bán, điểm kỷ luật."""

import numpy as np
import pandas as pd
import pytest

from vnsa import journal
from vnsa import portfolio as pf

DAYS = pd.bdate_range("2026-09-01", "2026-09-30")


K = 1000          # giá trong test viết theo nghìn đồng, sổ lệnh ghi bằng đồng


def _px(values, sym="X"):
    v = [x * K for x in values]
    v = v + [v[-1]] * (len(DAYS) - len(v))
    return pd.DataFrame({"symbol": sym, "date": DAYS, "close": v, "close_raw": v})


def _ledger(tmp_path, rows):
    f = tmp_path / "trades.csv"
    for side, sym, qty, price, date in rows:
        pf.add(side, sym, qty, price * K, date, fee=0, tax=0, file=f, calendar=DAYS)
    return pf.load(f)


def test_twr_ignores_flows_and_counts_dividends(tmp_path):
    # mua 100 @100 ngày 1, giá lên 110; mua thêm 100 @110 ngày 3 (dòng tiền vào không phải lãi);
    # giá về 121 -> TWR = 1,1 × 1,1 − 1 = 21%
    px = [100, 110, 110, 121]
    trades = _ledger(tmp_path, [(pf.BUY, "X", 100, 100, DAYS[0]), (pf.BUY, "X", 100, 110, DAYS[2])])
    idx = _px([1000, 1010, 1020, 1030], "VNINDEX")
    c = journal.equity_curve(trades, {"X": _px(px)}, idx, DAYS[3])
    assert c["twr"].iloc[-1] == pytest.approx(0.21)
    assert c["vni"].iloc[-1] == pytest.approx(0.03)
    # cổ tức tiền 5đ/cp (thuế 5%) ngày 4: giá điều chỉnh giảm 5 nhưng lợi suất không đổi
    acts = {"X": pd.DataFrame({"exright_date": [DAYS[3]], "cash_per_share": [5.0 * K],
                                "stock_ratio": [np.nan]})}
    c2 = journal.equity_curve(trades, {"X": _px([100, 110, 110, 116])}, idx, DAYS[3], acts)
    # ngày 4: (116×200 + 950 − 110×200) / 22000 = 9,77%
    assert c2["ret"].iloc[-1] == pytest.approx((116 * 200 + 200 * 5 * 0.95 - 22_000) / 22_000)


def test_sell_flow_and_trade_stats(tmp_path):
    trades = _ledger(tmp_path, [(pf.BUY, "X", 100, 100, DAYS[0]), (pf.SELL, "X", 50, 120, DAYS[3]),
                                (pf.SELL, "X", 50, 90, DAYS[5]), (pf.BUY, "Y", 10, 50, DAYS[0]),
                                (pf.SELL, "Y", 10, 60, DAYS[4])])
    sales = journal.sales_table(trades, DAYS[6], DAYS)
    assert list(sales["pnl"]) == [1000 * K, 100 * K, -500 * K]
    st = journal.trade_stats(sales)
    assert st["n"] == 3 and st["win_rate"] == pytest.approx(2 / 3)
    assert st["avg_win"] == pytest.approx((0.2 + 0.2) / 2) and st["avg_loss"] == pytest.approx(-0.1)
    assert st["payoff"] == pytest.approx(2.0) and st["profit_factor"] == pytest.approx(1100 / 500)
    c = journal.equity_curve(trades, {"X": _px([100] * 3 + [120] * 3), "Y": _px([50] * 4 + [60])},
                             _px([1000] * 10, "VNINDEX"), DAYS[6])
    assert c["value"].iloc[-1] == 0                    # đã bán hết
    assert c["twr"].iloc[3] == pytest.approx(5_500 / 5_500 * 0 + (12_000 + 500 - 10_500) / 10_500)


def test_discipline_scores(tmp_path):
    trades = _ledger(tmp_path, [
        (pf.BUY, "X", 100, 101, DAYS[1]),      # trong vùng (≤ 100 × 1,01)
        (pf.STOP, "X", 0, 95, DAYS[1]),
        (pf.BUY, "Y", 100, 120, DAYS[1]),      # đuổi giá
        (pf.BUY, "Z", 100, 50, DAYS[1]),       # không có khuyến nghị, không đặt cắt lỗ
        (pf.STOP, "Y", 0, 110, DAYS[2]),
        (pf.STOP, "Y", 0, 100, DAYS[6]),       # dời cắt lỗ xuống
        (pf.SELL, "X", 100, 93, DAYS[9]),      # thủng cắt lỗ ngày 8, bán ngày 9 (≤ 2 phiên)
    ])
    sig = pd.DataFrame({"as_of": [DAYS[0], DAYS[0]], "symbol": ["X", "Y"],
                        "buy_high": [100.0 * K, 110.0 * K]})
    prices = {"X": _px([100] * 8 + [94, 93]), "Y": _px([120]), "Z": _px([50])}
    d = journal.discipline(trades, sig, prices, DAYS, DAYS[15])
    assert d.parts["plan"][1:] == (1, 3)
    assert d.parts["stop_set"][1:] == (2, 3)
    assert d.parts["stop_kept"][1:] == (1, 2)           # 1 lần bán đúng, 1 lần dời xuống
    expected = 100 * (40 * 1 / 3 + 20 * 2 / 3 + 40 * 0.5) / 100
    assert d.score == pytest.approx(expected)
    assert any("đuổi giá" in n for n in d.notes) and any("dời cắt lỗ" in n for n in d.notes)


def test_breach_without_sale_is_violation(tmp_path):
    trades = _ledger(tmp_path, [(pf.BUY, "X", 100, 100, DAYS[0]), (pf.STOP, "X", 0, 95, DAYS[0])])
    d = journal.discipline(trades, pd.DataFrame(), {"X": _px([100] * 5 + [90])}, DAYS, DAYS[12])
    assert d.parts["stop_kept"][1:] == (0, 1)
    j = journal.build(trades, {"X": _px([100] * 5 + [90])}, _px([1000] * 22, "VNINDEX"),
                      pd.DataFrame(), DAYS[12])
    text = journal.format_journal(j)
    assert "ĐIỂM KỶ LUẬT" in text and "Chưa có lệnh bán" in text
