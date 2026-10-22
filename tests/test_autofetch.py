"""Tự lấy dữ liệu cho mã bất kỳ (data/autofetch.py), quy tắc theo sàn (market.py), lệnh analyze."""

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest
from conftest import _fundamentals  # BCTC giả lập dùng chung

from vnsa.data.autofetch import VN_TZ, Updater, last_session
from vnsa.data.providers import CSVProvider
from vnsa.data.providers.vnstock_provider import standardize_listing
from vnsa.data.store import Store

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "prices"
AUTO = {"years": 5, "overlap_days": 10, "adjust_tolerance": 0.005, "market_close_hour": 15,
        "bctc_max_age_days": 30, "listing_max_age_days": 30, "pause": 0}
LISTING = pd.DataFrame({
    "symbol": ["DEMO_A", "DEMO_B", "DEMO_C"], "name": ["Công ty A", "Ngân hàng B", "Công ty C"],
    "exchange": ["HOSE", "HNX", "UPCOM"], "industry": ["Công nghệ", "Ngân hàng", "Bán lẻ"],
    "model": ["general", "bank", "general"]})


class FakeCafeF:
    """Trả giá mẫu trong khoảng ngày được hỏi; `upto` giả lập nguồn mới có giá đến ngày đó."""
    name = "cafef"

    def __init__(self, upto: str | None = None, scale: float = 1.0):
        prov = CSVProvider(SAMPLE)
        self.data = {s: prov.ohlcv(s) for s in prov.symbols()}
        idx = self.data["DEMO_A"].assign(symbol="VNINDEX")
        idx[["open", "high", "low", "close"]] /= 40
        self.data["VNINDEX"] = idx
        self.upto, self.scale, self.calls, self.repaired = upto, scale, [], []

    def ohlcv(self, symbol, start, end):
        self.calls.append((symbol, start, end))
        if symbol not in self.data:
            raise ValueError(f"CafeF không có dữ liệu {symbol}. Mã có đúng không?")
        df = self.data[symbol]
        last = min(pd.Timestamp(end), pd.Timestamp(self.upto or end))
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= last)].copy()
        df[["open", "high", "low", "close"]] *= self.scale
        return df.reset_index(drop=True)


class FakeVnstock:
    name = "vnstock-vci"

    def __init__(self, listing_error=False, fin_error=False, history=None):
        self.missing: list[str] = []
        self.history, self.price_calls = history or {}, []
        self.listing_error, self.fin_error, self.fin_calls = listing_error, fin_error, 0

    def listing(self):
        if self.listing_error:
            raise RuntimeError("Chưa cài vnstock")
        return LISTING

    def ohlcv(self, symbol, start, end):
        self.price_calls.append((symbol, start, end))
        if symbol not in self.history:
            raise ValueError(f"vnstock không có giá {symbol}")
        df = self.history[symbol]
        return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]

    def fundamentals(self, symbol):
        self.fin_calls += 1
        if self.fin_error:
            raise RuntimeError("hết lượt gọi")
        return _fundamentals(symbol)

    def quarterly(self, symbol):
        if self.fin_error:
            raise RuntimeError("hết lượt gọi")
        return _quarters(symbol)

    def profile(self, symbol):
        if self.fin_error:
            raise RuntimeError("hết lượt gọi")
        return {"shares_out": 33e6, "foreign_pct": 0.2, "foreign_max_pct": 0.49,
                "avg_value_1m": 5e9, "analyst_rating": None}

    def dividends(self, symbol):
        return pd.DataFrame({"symbol": [symbol], "exright_date": [pd.Timestamp("2026-06-01")],
                             "cash_per_share": [1500.0], "stock_ratio": [None], "title": ["Cổ tức"]})


def _quarters(symbol):
    """4 quý 2025-Q3 → 2026-Q2: mỗi quý = 1/4 số năm 2025 (× 1,1 cho các khoản dòng chảy)."""
    last = _fundamentals(symbol).iloc[-1]
    rows = []
    for y, q in [(2025, 3), (2025, 4), (2026, 1), (2026, 2)]:
        r = last.copy()
        for f in ("revenue", "gross_profit", "net_income", "net_income_parent", "cfo", "pbt",
                  "tax_expense", "interest_expense", "capex", "equity_issued"):
            r[f] = last[f] * 1.1 / 4
        r["year"], r["quarter"] = y, q
        rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True)


@pytest.fixture
def empty_db(tmp_path, monkeypatch):
    db = tmp_path / "auto.duckdb"
    monkeypatch.setenv("VNSA_WAREHOUSE", str(db))
    monkeypatch.setenv("VNSA_RAW_PRICES_DIR", str(tmp_path / "prices"))
    monkeypatch.setenv("VNSA_RAW_BCTC_DIR", str(tmp_path / "bctc"))
    return db


def updater(cafef=None, vn=None, now=datetime(2026, 10, 9, 16, 0, tzinfo=VN_TZ)):
    return Updater(cafef or FakeCafeF(), vn or FakeVnstock(), AUTO, now=lambda: now)


def test_last_session_skips_weekend_and_waits_for_close():
    assert last_session(datetime(2026, 10, 10, 9, tzinfo=VN_TZ), 15) == pd.Timestamp("2026-10-09")   # thứ Bảy
    assert last_session(datetime(2026, 10, 12, 10, tzinfo=VN_TZ), 15) == pd.Timestamp("2026-10-09")  # T2 sáng
    assert last_session(datetime(2026, 10, 12, 15, tzinfo=VN_TZ), 15) == pd.Timestamp("2026-10-12")


def test_first_time_downloads_everything(empty_db, tmp_path):
    rep = updater().ensure("demo_a")
    assert rep.ok and not rep.errors, str(rep)
    with Store(empty_db) as store:
        df = store.prices("DEMO_A")
        assert df["date"].iloc[0] == pd.Timestamp("2021-10-11")    # đúng 5 năm tính lùi
        assert store.last_price_date("VNINDEX") == pd.Timestamp("2026-10-09")
        assert len(store.fundamentals("DEMO_A")) == 4
        assert store.company("DEMO_B")["exchange"] == "HNX"
        assert len(store.quarterly("DEMO_A")) == 4
        assert store.company("DEMO_A")["shares_out"] == 33e6
        assert store.dividends("DEMO_A")["cash_per_share"].iloc[0] == 1500
    assert (tmp_path / "prices" / "DEMO_A.csv").exists()
    assert (tmp_path / "bctc" / "DEMO_A.csv").exists()


def test_fresh_data_is_not_downloaded_again(empty_db):
    updater().ensure("DEMO_A")
    cafef, vn = FakeCafeF(), FakeVnstock()
    rep = updater(cafef, vn).ensure("DEMO_A")
    assert rep.ok and cafef.calls == [] and vn.fin_calls == 0
    assert {s.status for s in rep.steps} == {"skip", "ok"}   # "ok" = dòng thông tin doanh nghiệp


def test_incremental_download_only_fetches_new_sessions(empty_db):
    updater(FakeCafeF(upto="2026-09-30"), now=datetime(2026, 9, 30, 16, tzinfo=VN_TZ)).ensure("DEMO_A")
    cafef = FakeCafeF()
    rep = updater(cafef).ensure("DEMO_A")
    sym, start, _ = cafef.calls[0]
    assert sym == "DEMO_A" and start == date(2026, 9, 20)       # phiên cuối − 10 ngày
    assert "thêm 7 phiên mới" in str(rep)
    with Store(empty_db) as store:
        df = store.prices("DEMO_A")
    assert len(df) == 1311 and not df["date"].duplicated().any()   # 5 năm tính từ 30/09


def test_adjusted_history_triggers_full_reload(empty_db):
    updater(FakeCafeF(upto="2026-09-30"), now=datetime(2026, 9, 30, 16, tzinfo=VN_TZ)).ensure("DEMO_A")
    cafef = FakeCafeF(scale=0.9)                    # nguồn điều chỉnh lại giá sau chia cổ tức
    rep = updater(cafef).ensure("DEMO_A")
    assert "tải lại toàn bộ" in str(rep)
    with Store(empty_db) as store:
        new = store.prices("DEMO_A").set_index("date")["close"]
    sample = FakeCafeF().data["DEMO_A"].set_index("date")["close"]
    assert (new / sample.reindex(new.index)).round(6).eq(0.9).all()   # mọi phiên đều mới


def test_unknown_symbol_reports_error(empty_db):
    rep = updater().ensure("XYZ")
    assert not rep.ok
    assert any("không có trong danh sách" in e for e in rep.errors)
    assert any("Mã có đúng không" in e for e in rep.errors)


def test_missing_sources_do_not_block_price_analysis(empty_db):
    rep = updater(vn=FakeVnstock(listing_error=True, fin_error=True)).ensure("DEMO_A")
    assert rep.ok
    assert [s.what for s in rep.steps if s.status == "error"] == ["niêm yết", "BCTC", "BCTC quý",
                                                                   "hồ sơ"]
    assert "chỉ phân tích kỹ thuật" in str(rep)


def test_standardize_vnstock_listing():
    by_exchange = pd.DataFrame({
        "symbol": ["vcb", "shs", "acv", "fpt", "VCB2501"],
        "exchange": ["HSX", "HNX", "UPCOM", "HSX", "HSX"],
        "type": ["STOCK", "STOCK", "STOCK", "STOCK", "BOND"],
        "organ_name": ["Vietcombank", "Chứng khoán SHS", "ACV", "FPT", "Trái phiếu"]})
    by_industry = pd.DataFrame({
        "symbol": ["VCB", "SHS", "ACV", "FPT"],
        "icb_name2": ["Ngân hàng", "Dịch vụ tài chính", "Công nghiệp", "Công nghệ"],
        "icb_name4": ["Ngân hàng", "Môi giới chứng khoán", "Sân bay", "Phần mềm"],
        "com_type_code": ["NH", None, "CT", "CT"]})
    out = standardize_listing(by_exchange, by_industry).set_index("symbol")
    assert list(out.index) == ["VCB", "SHS", "ACV", "FPT"]
    assert out.loc["VCB", "exchange"] == "HOSE" and out.loc["ACV", "exchange"] == "UPCOM"
    assert out.loc["VCB", "model"] == "bank"
    assert out.loc["SHS", "model"] == "securities"       # suy ra từ tên ngành
    assert out.loc["FPT", "model"] == "general"
    assert out.loc["ACV", "industry4"] == "Sân bay"       # ICB cấp 4 cho ánh xạ vĩ mô
    assert out["industry3"].isna().all()                  # nguồn không có cấp 3: để trống


def test_standardize_listing_long_icb_format():
    """vnstock mới: mỗi mã 4 dòng icb_level/icb_name — phải lấy đúng cấp 2 (không lấy cấp 1)."""
    by_exchange = pd.DataFrame({"symbol": ["HPG", "SSI"], "exchange": ["HSX", "HSX"],
                                "type": ["STOCK", "STOCK"], "organ_name": ["Hòa Phát", "SSI"]})
    rows = [("HPG", "CT", 1, "Nguyên vật liệu"), ("HPG", "CT", 2, "Tài nguyên Cơ bản"),
            ("HPG", "CT", 3, "Kim loại"), ("HPG", "CT", 4, "Thép và sản phẩm thép"),
            ("SSI", "CK", 1, "Tài chính"), ("SSI", "CK", 2, "Dịch vụ tài chính"),
            ("SSI", "CK", 3, "Dịch vụ tài chính"), ("SSI", "CK", 4, "Môi giới chứng khoán")]
    by_industry = pd.DataFrame(rows, columns=["symbol", "com_type_code", "icb_level", "icb_name"])
    by_industry["icb_code"] = "0"
    out = standardize_listing(by_exchange, by_industry).set_index("symbol")
    assert out.loc["HPG", "industry"] == "Tài nguyên Cơ bản"
    assert out.loc["HPG", "industry3"] == "Kim loại"
    assert out.loc["HPG", "industry4"] == "Thép và sản phẩm thép"
    assert out.loc["SSI", "model"] == "securities"


def test_exchange_rules_band_ticks_and_financials(warehouse):
    from vnsa import analysis, market

    with Store(warehouse) as store:
        store.write_companies(LISTING, "test")
        store.write_fundamentals(_fundamentals("DEMO_B"), "test")
    assert market.band("DEMO_A") == 0.07
    assert market.band("DEMO_B") == 0.10 and market.band("DEMO_C") == 0.15
    assert market.band("CHUA_CO") == 0.07                  # chưa biết sàn: mặc định HOSE
    assert market.ticks("DEMO_C") == [[0, 100]]

    p = analysis.plan(analysis.indicators(analysis.load("DEMO_C", need_fin=False).prices))
    assert p.entry % 100 == 0 and p.stop % 100 == 0         # bước giá UPCoM 100 đồng

    d = analysis.load("DEMO_B")
    with pytest.raises(ValueError, match="ngân hàng"):
        analysis.valuation(d)
    rec, _ = analysis.recommendation("DEMO_B")
    assert rec.inputs.fa_score > 0 and rec.inputs.pe > 0 and rec.inputs.pb > 0
    assert rec.inputs.fair_value > 0 and pd.isna(rec.inputs.z_em)   # RI, không có Altman Z
    assert any("ngân hàng" in r for r in rec.reasons)


def test_store_upsert_keeps_older_sessions(tmp_path):
    df = CSVProvider(SAMPLE).ohlcv("DEMO_A")
    with Store(tmp_path / "x.duckdb") as store:
        store.write_prices(df, "a")
        tail = df.tail(3).assign(close=1.0)
        store.upsert_prices(tail, "b")
        out = store.prices("DEMO_A")
    assert len(out) == len(df)
    assert (out["close"].tail(3) == 1.0).all() and out["close"].iloc[-4] != 1.0


def test_cli_analyze(warehouse, monkeypatch, capsys):
    from vnsa import analysis, cli
    from vnsa.data.autofetch import Step, UpdateReport

    calls = []
    monkeypatch.setattr(analysis, "update", lambda sym, *a, **k: calls.append(sym) or
                        UpdateReport(sym, [Step("giá", "skip", "đã mới")], has_prices=True))
    assert cli.main(["analyze", "demo_a"]) == 0
    out = capsys.readouterr().out
    assert calls == ["DEMO_A"]
    for title in ("Xếp hạng:", "Thời điểm:", "Độ tin cậy:", "ĐỊNH GIÁ", "ĐIỂM SỐ", "CƠ BẢN", "KỸ THUẬT",
                  "DỰ BÁO 20 PHIÊN", "KẾ HOẠCH GIAO DỊCH"):
        assert title in out
    assert "chưa rõ sàn" in out and "── RỦI RO" not in out          # chi tiết chỉ khi --detail
    assert cli.main(["analyze", "demo_a", "--detail"]) == 0
    out = capsys.readouterr().out
    assert "── RỦI RO" in out and "── ĐỊNH GIÁ" in out

    monkeypatch.setattr(analysis, "update", lambda sym, *a, **k: UpdateReport(
        sym, [Step("giá", "error", "Mã có đúng không?")]))
    assert cli.main(["analyze", "XYZ", "--brief"]) == 1
    assert "không phân tích được" in capsys.readouterr().out


def test_falls_back_to_vnstock_when_cafef_lacks_symbol(empty_db):
    cafef = FakeCafeF()
    only_vn = cafef.data.pop("DEMO_C")                 # như SHS/ACV: CafeF không trả dữ liệu
    vn = FakeVnstock(history={"DEMO_C": only_vn})
    rep = updater(cafef, vn).ensure("DEMO_C")
    assert rep.ok and "nguồn vnstock-vci" in str(rep), str(rep)
    with Store(empty_db) as store:
        assert store.price_source("DEMO_C") == "vnstock-vci"
        assert store.last_price_date("DEMO_C") == pd.Timestamp("2026-10-09")

    cafef2, vn2 = FakeCafeF(), FakeVnstock(history={"DEMO_C": only_vn})
    cafef2.data.pop("DEMO_C")
    updater(cafef2, vn2, now=datetime(2026, 10, 13, 16, tzinfo=VN_TZ)).ensure("DEMO_C")
    assert vn2.price_calls and not [c for c in cafef2.calls if c[0] == "DEMO_C"]   # dùng lại nguồn


def test_vnstock_history_is_converted_to_dong():
    from vnsa.data.providers.vnstock_provider import VnstockProvider

    raw = pd.DataFrame({"time": pd.to_datetime(["2026-10-08", "2026-10-09"]),
                        "open": [56.5, 56.9], "high": [57.0, 57.2], "low": [56.1, 56.5],
                        "close": [56.9, 56.7], "volume": [4_000_000, 3_500_000]})
    prov = VnstockProvider(get_history=lambda s, a, b: raw)
    df = prov.ohlcv("VCB", date(2026, 10, 1), date(2026, 10, 9))
    assert df["close"].tolist() == [56900, 56700] and df["symbol"].iloc[0] == "VCB"
    bad = raw.assign(high=[56.8, 57.2])                 # nến VCI ghi sai: cao nhất < đóng cửa
    fixed = VnstockProvider(get_history=lambda s, a, b: bad)
    assert fixed.ohlcv("SHS", date(2026, 10, 1), date(2026, 10, 9))["high"].iloc[0] == 56900
    assert fixed.repaired == ["08/10/2026"]
    idx = VnstockProvider(get_history=lambda s, a, b: raw.assign(close=[1290.5, 1301.2]))
    assert idx.ohlcv("VNINDEX", date(2026, 10, 1), date(2026, 10, 9))["close"].iloc[-1] == 1301   # chỉ số: điểm, làm tròn như CafeF


def test_unfinished_session_is_dropped_not_rejected(empty_db):
    """Trong giờ giao dịch CafeF trả cả phiên hôm nay với giá 0: bỏ phiên đó, vẫn nạp phần còn lại."""
    class IntradayCafeF(FakeCafeF):
        def ohlcv(self, symbol, start, end):
            df = super().ohlcv(symbol, start, end)
            today = pd.DataFrame({"symbol": [symbol], "date": [pd.Timestamp("2026-10-12")],
                                  "open": [0.0], "high": [0.0], "low": [0.0], "close": [0.0],
                                  "volume": [0]})
            return pd.concat([df, today], ignore_index=True)

    up = updater(IntradayCafeF(), now=datetime(2026, 10, 12, 10, 20, tzinfo=VN_TZ))
    rep = up.ensure("DEMO_A")
    steps = {s.what: s for s in rep.steps}
    assert steps["giá"].status == "ok" and steps["VNINDEX"].status == "ok", rep
    with Store(empty_db) as store:
        assert store.last_price_date("DEMO_A") < pd.Timestamp("2026-10-12")
        assert (store.prices("VNINDEX")["close"] > 0).all()


def test_fetch_command_drops_unfinished_session(empty_db, monkeypatch):
    """Lệnh fetch (tải hàng loạt) cũng bỏ phiên chưa đóng cửa CafeF trả về với giá 0."""
    from vnsa import cli
    from vnsa.data.providers import cafef_provider

    fake = FakeCafeF()

    def ohlcv(self, symbol, start, end):
        df = fake.data[symbol].copy()
        future = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)   # phiên chưa đóng
        zero = pd.DataFrame({"symbol": [symbol], "date": [future], "open": [0.0], "high": [0.0],
                             "low": [0.0], "close": [0.0], "volume": [0]})
        return pd.concat([df, zero], ignore_index=True)

    monkeypatch.setattr(cafef_provider.CafeFProvider, "ohlcv", ohlcv)
    assert cli.main(["fetch", "DEMO_A", "--years", "1", "--pause", "0"]) == 0
    with Store(empty_db) as store:
        assert (store.prices("VNINDEX")["close"] > 0).all()
        assert store.last_price_date("DEMO_A") < pd.Timestamp.today().normalize() + pd.Timedelta(days=1)


def test_source_unfinished_last_session_after_close(empty_db):
    """Sau giờ đóng cửa CafeF vẫn có thể trả phiên hôm nay chưa đủ giá (thật: VNINDEX 12/10/2026
    lúc 18h, một giá bằng 0): bỏ phiên cuối đó thay vì từ chối cả lần tải; phiên lỗi ở giữa chuỗi
    vẫn bị chặn như cũ."""
    class LateCafeF(FakeCafeF):
        bad_middle = False

        def ohlcv(self, symbol, start, end):
            df = super().ohlcv(symbol, start, end)
            if symbol != "VNINDEX":
                return df
            if self.bad_middle:
                df.loc[len(df) // 2, "open"] = 0.0
                return df
            late = df.iloc[[-1]].assign(date=pd.Timestamp("2026-10-12"), open=0.0)
            return pd.concat([df, late], ignore_index=True)

    up = updater(LateCafeF(), now=datetime(2026, 10, 12, 18, 39, tzinfo=VN_TZ))
    rep = up.ensure("DEMO_A")
    steps = {s.what: s for s in rep.steps}
    assert steps["VNINDEX"].status == "ok", rep
    with Store(empty_db) as store:
        assert store.last_price_date("VNINDEX") == pd.Timestamp("2026-10-09")

    bad = LateCafeF()
    bad.bad_middle = True
    rep = updater(bad, now=datetime(2026, 10, 12, 18, 39, tzinfo=VN_TZ)).ensure("DEMO_C")
    assert "gia_khong_duong" in {s.what: s for s in rep.steps}["VNINDEX"].detail


def test_dividend_error_keeps_shares_and_retries(empty_db):
    """Lỗi tải cổ tức: vẫn ghi số cp lưu hành nhưng không đánh dấu hồ sơ đã tải, để lần sau thử lại
    (trước đây báo nhầm "số cổ phiếu tạm tính = vốn góp" và 7 ngày không tải lại cổ tức)."""
    class NoDividends(FakeVnstock):
        fail, div_calls = True, 0

        def dividends(self, symbol):
            self.div_calls += 1
            if self.fail:
                raise AttributeError("'numpy.float64' object has no attribute 'where'")
            return super().dividends(symbol)

    vn = NoDividends()
    rep = updater(vn=vn).ensure("DEMO_A")
    step = {s.what: s for s in rep.steps}["hồ sơ"]
    assert step.status == "error" and "cổ tức" in step.detail and "vốn góp" not in step.detail
    with Store(empty_db) as store:
        assert store.company("DEMO_A")["shares_out"] == 33e6
        assert store.company("DEMO_A")["profile_at"] is None
    vn.fail = False
    rep = updater(vn=vn).ensure("DEMO_A")
    assert vn.div_calls == 2 and {s.what: s for s in rep.steps}["hồ sơ"].status == "ok"
    with Store(empty_db) as store:
        assert store.dividends("DEMO_A")["cash_per_share"].iloc[0] == 1500


def test_expected_period_follows_reporting_season():
    from vnsa.data.autofetch import expected_period

    cfg = {"quarter_lag_days": 30, "annual_lag_days": 100}
    ts = pd.Timestamp
    assert expected_period(ts("2026-07-29"), True, cfg) == (2026, 1)   # Q2 chưa đến hạn
    assert expected_period(ts("2026-08-01"), True, cfg) == (2026, 2)
    assert expected_period(ts("2026-01-15"), True, cfg) == (2025, 3)
    assert expected_period(ts("2026-02-01"), True, cfg) == (2025, 4)
    assert expected_period(ts("2026-04-05"), False, cfg) == (2024, 0)  # BCTC 2025 chưa kiểm toán xong
    assert expected_period(ts("2026-04-15"), False, cfg) == (2025, 0)


def test_bctc_due_skips_until_new_period():
    from vnsa.data.autofetch import bctc_due

    cfg = {"bctc_season": True, "quarter_lag_days": 30, "bctc_retry_days": 3,
           "bctc_max_age_days_current": 90, "bctc_max_age_days": 30}
    now = pd.Timestamp("2026-10-21")
    assert bctc_due(None, None, now, cfg, True) is None                              # chưa có
    # đã có Q3? chưa đến hạn (Q3 kết thúc 30/9 + 30 ngày) -> kỳ kỳ vọng là Q2: có rồi thì bỏ qua
    assert "Q2/2026" in bctc_due((2026, 2), pd.Timestamp("2026-08-10"), now, cfg, True)
    # sau 90 ngày vẫn tải lại một lần (BCTC điều chỉnh)
    assert bctc_due((2026, 2), pd.Timestamp("2026-07-01"), now, cfg, True) is None
    # nộp muộn: chỉ có Q1 -> tải, nhưng không quá mỗi 3 ngày
    assert bctc_due((2026, 1), pd.Timestamp("2026-10-15"), now, cfg, True) is None
    assert "thử lại" in bctc_due((2026, 1), pd.Timestamp("2026-10-20"), now, cfg, True)
    # tắt theo mùa: quy tắc cũ 30 ngày
    old = {**cfg, "bctc_season": False}
    assert bctc_due((2026, 2), pd.Timestamp("2026-08-10"), now, old, True) is None


def test_trading_status_blocks_buying(empty_db):
    """v3: trạng thái giao dịch từ bảng giá (hạn chế giao dịch) -> chặn mua; bình thường -> không."""
    from vnsa import analysis
    from vnsa.news import apply_status, NewsCheck

    class WithBoard(FakeVnstock):
        status, calls = "TRADING_RESTRICTION_ACTIVATED", 0

        def board(self, symbols):
            self.calls += 1
            return pd.DataFrame({"symbol": symbols, "trading_status": [self.status] * len(symbols),
                                 "security_status": ["N"] * len(symbols),
                                 "trading_date": [pd.Timestamp("2026-10-09")] * len(symbols)})

    vn = WithBoard()
    now = datetime(2026, 10, 9, 16, 0, tzinfo=VN_TZ)
    up = lambda: Updater(FakeCafeF(), vn, {**AUTO, "status": True}, now=lambda: now)  # noqa: E731
    rep = up().ensure("DEMO_A")
    assert {s.what: s for s in rep.steps}["trạng thái"].detail == "TRADING_RESTRICTION_ACTIVATED"
    rep = up().ensure("DEMO_A")                               # còn mới: không gọi lại
    assert vn.calls == 1 and {s.what: s for s in rep.steps}["trạng thái"].status == "skip"
    nc = analysis.news_check("DEMO_A")
    assert nc is not None and "hạn chế giao dịch" in nc.blocked
    with Store(empty_db) as store:
        store.write_status(pd.DataFrame({"symbol": ["DEMO_A"], "trading_status": ["TRADING_ACTIVATED"],
                                         "security_status": ["N"], "trading_date": ["2026-10-09"]}))
    assert analysis.news_check("DEMO_A") is None
    old = {"trading_status": "TRADING_RESTRICTION_ACTIVATED", "security_status": "N",
           "trading_date": None, "loaded_at": pd.Timestamp("2026-01-01")}
    assert apply_status(NewsCheck(pd.DataFrame()), old).blocked == ""      # trạng thái quá cũ
