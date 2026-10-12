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


def test_earnings_growth_prefers_same_period_then_ttm_then_annual():
    from types import SimpleNamespace

    from conftest import B, _fundamentals

    from vnsa.screen.deep import earnings_growth

    fin = _fundamentals("X")                                  # LNST mẹ 75 → 101,25 tỷ (2022–2025)
    q8 = pd.DataFrame({"year": [2024] * 4 + [2025] * 4, "quarter": [1, 2, 3, 4] * 2,
                       "net_income_parent": [20 * B] * 4 + [25 * B] * 4})
    g, basis = earnings_growth(fin, q8)
    assert g == pytest.approx(0.25) and basis == "TTM Q4/2025 so cùng kỳ"
    ttm = SimpleNamespace(basis="TTM Q2/2026", net_income_parent=81 * B)
    g, basis = earnings_growth(fin, q8.tail(4), ttm)
    assert g == pytest.approx(81 / 101.25 - 1) and basis == "TTM Q2/2026 so năm 2025"
    g, basis = earnings_growth(fin, None)
    assert g == pytest.approx(1.35 / 1.2 - 1) and basis == "năm 2025 so 2024"
    loss = fin.assign(net_income_parent=[10 * B, 5 * B, -3 * B, 4 * B])
    assert np.isnan(earnings_growth(loss, None)[0])           # năm gốc lỗ: không tính %
    gap = q8.drop(index=3)                                    # thiếu một quý: không đủ 8 quý liền
    assert earnings_growth(fin, gap)[1] == "năm 2025 so 2024"


def test_groups_follow_macro_map_then_icb_level_2():
    from vnsa.screen.groups import industry_group

    mapping = {"industry": [{"name": "Thép", "match": "thép"},
                            {"name": "Ngân hàng", "match": "ngân hàng", "model": "bank"}]}
    assert industry_group({"industry": "Tài nguyên Cơ bản", "industry4": "Thép và sản phẩm thép"},
                          mapping) == "Thép"
    assert industry_group({"industry": "Tài chính", "model": "bank"}, mapping) == "Ngân hàng"
    assert industry_group({"industry": "Truyền thông", "model": "general"}, mapping) == "Truyền thông"
    assert industry_group({}, mapping) == "Chưa rõ ngành"


def _members(group, n, rs, growth, expected, fa):
    return pd.DataFrame({"group_name": group, "symbol": [f"{group}{i}" for i in range(n)],
                         "rs_3m": rs, "rs_6m": rs, "above_ma50": 1.0 if rs > 0 else 0.0,
                         "above_ma200": 1.0 if rs > 0 else 0.0, "eps_growth": growth,
                         "expected_return": expected, "fa_score": fa, "value_20d": 5e9})


def test_industry_scores_rank_groups_and_reweight_missing_parts():
    from vnsa.screen.industry import industry_scores

    table = pd.concat([_members("Mạnh", 4, 0.10, 0.30, 0.25, 70),
                       _members("Vừa", 3, 0.00, -0.02, 0.10, 60),
                       _members("Yếu", 5, -0.10, -0.20, -0.05, 40),
                       _members("Ít mã", 2, 0.50, 0.50, 0.50, 90)])
    ind = industry_scores(table, {"Mạnh": (0.5, 1.0), "Yếu": (-0.5, 0.3)}, {"min_members": 3})
    ranked = ind.dropna(subset=["score"])
    assert list(ranked["group_name"]) == ["Mạnh", "Vừa", "Yếu"]
    assert ranked["rank"].tolist() == [1, 2, 3]
    top = ranked.iloc[0]
    assert top["pt_momentum"] == 100 and top["pt_macro"] == 75     # điểm vĩ mô 0,5 → 75
    assert top["score"] == pytest.approx(100 * 0.9 + 75 * 0.1)     # mọi phần khác hạng nhất
    assert np.isnan(ranked.iloc[2]["pt_macro"])                   # độ phủ 30% < 60%: bỏ vĩ mô
    last = ranked.iloc[2]                                         # hoà với "Vừa" ở độ rộng và tỷ lệ
    assert last["weight_used"] == pytest.approx(0.9)              # DN lãi tăng: chung hạng bình quân
    assert last["score"] == pytest.approx((0.10 * 25 + 0.25 * 12.5) / 0.9)
    few = ind[ind["group_name"] == "Ít mã"].iloc[0]
    assert np.isnan(few["score"]) and "dưới 3 mã" in few["note"]


def test_full_run_to_industry_stage(market_db, monkeypatch):
    from vnsa.screen import run as screen_run
    from vnsa.screen.report import format_industry

    cfg = {**screen_run.screen_cfg(), "min_members": 1, "top_industries": 1}
    monkeypatch.setattr(screen_run, "screen_cfg", lambda: cfg)
    monkeypatch.setattr("vnsa.screen.report.screen_cfg", lambda: cfg)
    res = screen_run.run("industry", updater=_updater(), log=None)
    t = res.table.set_index("symbol")
    assert t.loc["DEMO_A", "stage"] >= 2 and t.loc["DEMO_C", "stage"] >= 2
    assert t.loc["DEMO_A", "action"] in ("MUA", "KHẢ QUAN", "NẮM GIỮ", "KÉM KHẢ QUAN", "BÁN")
    assert not np.isnan(t.loc["DEMO_A", "fa_score"]) and t.loc["DEMO_A", "growth_basis"]
    assert t.loc["LOWVOL", "stage"] == 0 and np.isnan(t.loc["LOWVOL", "fa_score"])
    ind = res.industry
    assert set(ind["group_name"]) == {"Công nghệ", "Bán lẻ"} and ind["score"].notna().all()
    assert (t["stage"] == 3).sum() == 1                            # chỉ nhóm hạng 1
    with Store(market_db) as store:
        assert store.first_price_date("DEMO_A") == pd.Timestamp("2021-10-11")   # tầng 2 đủ 5 năm
        assert len(store.screen_industry()) == 2
        assert store.screen().set_index("symbol").loc["DEMO_A", "action"] == t.loc["DEMO_A", "action"]
    text = format_industry(res)
    assert "NHÓM NGÀNH" in text and "Công nghệ" in text and "phân tích được 2" in text


def test_rate_limiter_waits_below_vnstock_quota():
    from vnsa.data.providers.vnstock_provider import RateLimiter

    clock = [0.0]
    slept = []

    def sleep(s):
        slept.append(s)
        clock[0] += s

    lim = RateLimiter(per_minute=3, per_hour=5, clock=lambda: clock[0], sleep=sleep)
    for _ in range(3):
        lim.wait()
        clock[0] += 1
    assert not slept                                         # 3 lượt trong phút đầu: không chờ
    lim.wait()                                               # lượt 4: chờ hết cửa sổ 60 giây
    assert slept and clock[0] >= 60
    lim.wait()
    lim.wait()                                               # lượt 6 vượt 5/giờ: chờ tới giờ sau
    assert clock[0] >= 3600 and lim.waited == pytest.approx(sum(slept))


def test_vnstock_system_exit_becomes_error(monkeypatch):
    import vnsa.data.providers.vnstock_provider as vp

    monkeypatch.setattr(vp, "_LIMITER", vp.RateLimiter(100, 1000))
    monkeypatch.setattr(vp.time, "sleep", lambda s: None)
    calls = []

    def quota_exit():
        calls.append(1)
        raise SystemExit("Rate limit exceeded")

    with pytest.raises(RuntimeError, match="hết hạn mức"):
        vp._call(quota_exit)
    assert len(calls) == 2                                   # thử lại một lần sau khi chờ
    assert vp._call(lambda: "ok") == "ok"


def test_deep_fetch_stops_at_time_budget(market_db):
    from vnsa.screen.deep import fetch_all

    errors = fetch_all(["DEMO_A", "DEMO_C"], _updater(), log=None, minutes=-1)
    assert set(errors) == {"DEMO_A", "DEMO_C"}
    assert all(e.startswith("chưa tải BCTC lượt này") for e in errors.values())


def test_pick_excludes_weak_and_orders_within_top_groups():
    from vnsa.screen.pick import exclusion, pick

    cfg = {"per_industry": 2, "min_fa": 50, "min_z": 1.1, "min_confidence": "Thấp"}
    base = {"stage": 3, "group_name": "Ngân hàng", "model": "general", "fa_score": 70.0,
            "z_em": 3.0, "red_flags": "", "confidence": "Trung bình", "rs_6m": 0.1,
            "action": "MUA", "score": 60.0, "reason": ""}
    rows = [
        {**base, "symbol": "AAA", "action": "KHẢ QUAN", "score": 90.0},
        {**base, "symbol": "BBB", "score": 55.0},
        {**base, "symbol": "CCC", "score": 70.0},
        {**base, "symbol": "DDD", "action": "NẮM GIỮ"},
        {**base, "symbol": "EEE", "fa_score": 40.0},
        {**base, "symbol": "FFF", "z_em": 0.8},
        {**base, "symbol": "GGG", "red_flags": "CFO âm 3 năm"},
        {**base, "symbol": "OUT", "stage": 2, "group_name": "Thép"},
        {**base, "symbol": "OUT2", "stage": 2, "group_name": "Thép", "action": "KHẢ QUAN"},
        {**base, "symbol": "LOW", "stage": 1, "action": None},
    ]
    t, outside = pick(pd.DataFrame(rows).set_index("symbol"), cfg)
    assert t.loc["CCC", "pick_rank"] == 1 and t.loc["BBB", "pick_rank"] == 2   # MUA trước KHẢ QUAN
    assert t.loc["AAA", "stage"] == 3 and t.loc["AAA", "reason"] == "ngoài 2 mã đầu nhóm"
    assert t.loc["DDD", "reason"] == "xếp hạng NẮM GIỮ"
    assert t.loc["EEE", "reason"] == "FA 40 < 50"
    assert t.loc["FFF", "reason"].startswith("Altman Z''")
    assert t.loc["GGG", "reason"] == "cờ đỏ BCTC: CFO âm 3 năm"
    assert list(outside.index) == ["OUT"]                                       # chỉ MUA ngoài nhóm
    assert t.loc["OUT2", "reason"] == "đạt nhưng ngoài nhóm ngành dẫn đầu"
    assert t.loc["LOW", "stage"] == 1
    bank = {**base, "model": "bank", "z_em": float("nan")}
    assert exclusion(bank, cfg) == ""                                           # ngân hàng: không xét Z''
    strict = {**cfg, "min_confidence": "Cao"}
    assert exclusion(base, strict) == "độ tin cậy Trung bình < Cao"


def test_full_run_with_picks_report(market_db, monkeypatch):
    from vnsa.screen import run as screen_run
    from vnsa.screen.report import format_picks

    cfg = {**screen_run.screen_cfg(), "min_members": 1, "top_industries": 2, "min_fa": 0}
    monkeypatch.setattr(screen_run, "screen_cfg", lambda: cfg)
    monkeypatch.setattr("vnsa.screen.report.screen_cfg", lambda: cfg)
    res = screen_run.run("pick", updater=_updater(), log=None)
    t = res.table.set_index("symbol")
    good = t[t["action"].isin(["MUA", "KHẢ QUAN"])]
    assert (t.loc[good.index, "stage"] == 4).all()                  # 2 nhóm đều dẫn đầu
    assert t.loc[t["stage"] == 3, "reason"].str.len().gt(0).all()   # không chọn thì có lý do
    text = format_picks(res)
    assert "NHÓM NGÀNH TIỀM NĂNG" in text and "MÃ TIỀM NĂNG TRONG 2 NHÓM DẪN ĐẦU" in text
    assert "không phải lời khuyên đầu tư" in text
    with Store(market_db) as store:
        saved = store.screen().set_index("symbol")
    assert saved["pick_rank"].notna().sum() == len(good)


def test_quick_prices_parallel_prefetch_matches_sequential(market_db):
    cafef = MarketCafeF()
    steps = _updater(cafef).quick_prices([*LISTING["symbol"]], years=1, workers=3)
    assert {s: st.status for s, st in steps.items()} == {
        "VNINDEX": "ok", "DEMO_A": "ok", "DEMO_B": "ok", "DEMO_C": "ok", "LOWVOL": "ok",
        "NEWBIE": "ok", "OLD": "ok", "GONE": "error"}
    downloaded = [c[0] for c in cafef.calls]
    assert sorted(downloaded) == sorted(set(downloaded))          # mỗi mã đúng một lần gọi CafeF
    with Store(market_db) as store:
        assert store.first_price_date("DEMO_A") >= pd.Timestamp("2025-10-09")
        assert store.last_price_date("DEMO_C") == pd.Timestamp("2026-10-09")
