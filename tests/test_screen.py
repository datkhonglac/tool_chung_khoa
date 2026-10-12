"""Sàng lọc toàn thị trường (src/vnsa/screen/): tầng 0–1 lọc thanh khoản, tải giá nhanh nhiều mã."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from test_autofetch import AUTO, FakeCafeF, FakeVnstock

from vnsa.data.autofetch import VN_TZ, Updater
from vnsa.data.store import Store
from vnsa.screen.liquidity import liquidity_check, price_stats, reason_group

NOW = datetime(2026, 10, 9, 16, 0, tzinfo=VN_TZ)
LISTING = pd.DataFrame({
    "symbol": ["DEMO_A", "DEMO_B", "DEMO_C", "LOWVOL", "NEWBIE", "OLD", "GONE"],
    "name": ["A", "B", "C", "Ít thanh khoản", "Mới niêm yết", "Ngừng giao dịch", "Không có giá"],
    "exchange": ["HOSE", "HNX", "UPCOM", "HOSE", "HOSE", "UPCOM", "HNX"],
    "industry": ["Công nghệ", "Ngân hàng", "Bán lẻ", "Công nghệ", "Bán lẻ", "Bán lẻ", "Ngân hàng"],
    "model": ["general", "bank", "general", "general", "general", "general", "bank"]})
CFG = {"min_value_bn": 3.0, "min_price": 5000, "min_sessions": 200, "max_stale_days": 10}


class MarketCafeF(FakeCafeF):
    """Giá mẫu + các mã giả lập: ít thanh khoản, mới niêm yết, ngừng giao dịch."""

    def __init__(self, **kw):
        super().__init__(**kw)
        a, c = self.data["DEMO_A"], self.data["DEMO_C"]
        self.data["LOWVOL"] = a.assign(symbol="LOWVOL", volume=a["volume"] // 1000)
        self.data["NEWBIE"] = c.tail(100).assign(symbol="NEWBIE")
        self.data["OLD"] = c[c["date"] <= "2026-08-31"].assign(symbol="OLD")


class MarketVn(FakeVnstock):
    def listing(self):
        return LISTING


@pytest.fixture
def market_db(tmp_path, monkeypatch):
    db = tmp_path / "screen.duckdb"
    monkeypatch.setenv("VNSA_WAREHOUSE", str(db))
    monkeypatch.setenv("VNSA_RAW_PRICES_DIR", str(tmp_path / "prices"))
    monkeypatch.setenv("VNSA_RAW_BCTC_DIR", str(tmp_path / "bctc"))
    monkeypatch.setenv("VNSA_SCREEN_DIR", str(tmp_path / "screen"))
    return db


def _updater(cafef=None):
    return Updater(cafef or MarketCafeF(), MarketVn(), AUTO, now=lambda: NOW)


def test_price_stats_relative_strength_and_liquidity():
    dates = pd.bdate_range(end="2026-10-09", periods=260)
    close = np.linspace(10_000, 20_000, 260)                # cổ phiếu tăng đều
    px = pd.DataFrame({"date": dates, "close": close, "close_raw": close * 1.1,
                       "volume": 1_000_000})
    idx = pd.Series(1000.0, index=dates)                    # chỉ số đi ngang
    st = price_stats(px, idx, pd.Timestamp("2026-10-09"))
    assert st["price"] == pytest.approx(22_000)             # giá khớp thật (close_raw)
    assert st["value_20d"] == pytest.approx((close[-20:] * 1.1 * 1e6).mean())
    assert st["ret_6m"] == pytest.approx(close[-1] / close[-127] - 1)
    assert st["rs_6m"] == pytest.approx(st["ret_6m"])        # chỉ số đi ngang: RS = lợi nhuận
    assert st["above_ma50"] == 1.0 and st["above_ma200"] == 1.0
    assert 250 <= st["sessions"] <= 262
    short = price_stats(px.tail(30), idx, pd.Timestamp("2026-10-09"))
    assert np.isnan(short["ret_3m"]) and np.isnan(short["above_ma50"])
    assert np.isnan(price_stats(px.iloc[:0], idx, pd.Timestamp("2026-10-09"))["price"])


def test_liquidity_check_reasons_in_order():
    as_of = pd.Timestamp("2026-10-09")
    ok = {"last_date": as_of, "sessions": 240, "price": 25_000, "value_20d": 5e9}
    assert liquidity_check(ok, CFG, as_of) == ""
    assert reason_group(liquidity_check({**ok, "last_date": pd.NaT}, CFG, as_of)) == "không có giá"
    stale = liquidity_check({**ok, "last_date": pd.Timestamp("2026-08-31")}, CFG, as_of)
    assert stale == "ngừng giao dịch: phiên cuối 31/08/2026"
    assert reason_group(liquidity_check({**ok, "sessions": 90}, CFG, as_of)) == "ít phiên"
    low = liquidity_check({**ok, "price": 3_650}, CFG, as_of)
    assert low == "giá thấp: 3.650đ < 5.000đ"
    thin = liquidity_check({**ok, "value_20d": 1.25e9}, CFG, as_of)
    assert thin == "thanh khoản thấp: GTGD 20 phiên 1,2 tỷ < 3,0 tỷ"
    assert reason_group(liquidity_check({**ok, "value_20d": float("nan")}, CFG, as_of)) \
        == "thanh khoản thấp"


def test_liquidity_stage_end_to_end(market_db, tmp_path):
    from vnsa.screen.report import format_liquidity
    from vnsa.screen.run import liquidity

    cafef = MarketCafeF()
    res = liquidity(updater=_updater(cafef), log=None)
    t = res.table.set_index("symbol")
    assert res.as_of == pd.Timestamp("2026-10-09")
    assert set(res.passed(1)["symbol"]) == {"DEMO_A", "DEMO_C"}
    groups = {s: reason_group(r) for s, r in t["reason"].items() if r}
    assert groups == {"DEMO_B": "giá thấp", "LOWVOL": "thanh khoản thấp", "NEWBIE": "ít phiên",
                      "OLD": "ngừng giao dịch", "GONE": "không có giá"}
    assert res.fetch["error"] == 1 and res.fetch_errors[0].startswith("GONE")
    assert t.loc["DEMO_A", "exchange"] == "HOSE" and t.loc["DEMO_C", "industry"] == "Bán lẻ"
    # mã chưa có giá: chỉ 1 năm, mỗi mã 1 lần gọi (kể cả VNINDEX)
    first = {sym: start for sym, start, _ in cafef.calls}
    assert first["DEMO_A"] == pd.Timestamp("2025-10-09").date()
    with Store(market_db) as store:
        assert store.first_price_date("DEMO_A") >= pd.Timestamp("2025-10-09")
        assert store.price_depth("DEMO_A") == pd.Timestamp("2025-10-09")
        saved = store.screen()
    assert len(saved) == len(LISTING) and (tmp_path / "screen" / "screen_2026-10-09.csv").exists()
    text = format_liquidity(res)
    assert "qua tầng 1: 2 mã" in text and "Bị loại:" in text and "DEMO_A" in text
    # chạy lại cùng ngày: thay lượt cũ, không nhân đôi; giá đã mới thì không tải lại (mã ngừng
    # giao dịch / không có giá vẫn được hỏi lại, có thể đã giao dịch trở lại)
    again = MarketCafeF()
    liquidity(updater=_updater(again), log=None)
    assert {c[0] for c in again.calls} == {"OLD", "GONE"}
    with Store(market_db) as store:
        assert len(store.screen()) == len(LISTING)


def test_deep_update_tops_up_short_history_once(market_db):
    from vnsa.screen.run import liquidity

    liquidity(updater=_updater(), log=None)                  # kho chỉ có 1 năm giá
    cafef = MarketCafeF()
    rep = _updater(cafef).ensure("DEMO_A")
    assert rep.ok
    with Store(market_db) as store:
        assert store.first_price_date("DEMO_A") == pd.Timestamp("2021-10-11")   # đủ 5 năm
        assert store.first_price_date("VNINDEX") == pd.Timestamp("2021-10-11")
    # mã mới niêm yết: tải đủ độ sâu một lần, lần sau không tải lại dù lịch sử vẫn ngắn
    cafef = MarketCafeF()
    _updater(cafef).ensure("NEWBIE")
    assert [c for c in cafef.calls if c[0] == "NEWBIE"]
    cafef = MarketCafeF()
    _updater(cafef).ensure("NEWBIE")
    assert not [c for c in cafef.calls if c[0] == "NEWBIE"]


def test_quick_update_with_adjusted_history_keeps_depth(market_db):
    _updater().ensure("DEMO_A")                              # 5 năm
    shifted = MarketCafeF(scale=1.02)                        # nguồn điều chỉnh lại giá cũ
    later = datetime(2026, 10, 12, 16, 0, tzinfo=VN_TZ)        # thứ Hai: cần phiên mới
    steps = Updater(shifted, MarketVn(), AUTO, now=lambda: later).quick_prices(["DEMO_A"], years=1)
    assert steps["DEMO_A"].status == "ok" and "tải lại toàn bộ" in steps["DEMO_A"].detail
    with Store(market_db) as store:
        assert store.first_price_date("DEMO_A") == pd.Timestamp("2021-10-11")
