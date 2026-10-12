"""Lớp vĩ mô: danh mục chuỗi, số nhập tay, đọc nguồn (không cần mạng), kho, biến động."""

import io
import json

import numpy as np
import pandas as pd
import pytest

from vnsa.data.providers.macro_provider import (
    MacroProvider,
    parse_fred_csv,
    parse_pink_sheet,
    parse_yahoo,
    pink_sheet_url,
)
from vnsa.data.store import Store
from vnsa.macro.catalog import Series, catalog, manual_frame, parse_period
from vnsa.macro.signals import fmt_change, reading, readings

CATALOG = """
[series]
us_10y  = { name = "Lợi suất Mỹ 10 năm", source = "fred", code = "DGS10", freq = "D", kind = "diff", unit = "%" }
urea    = { name = "Urê", source = "worldbank", code = "Urea", freq = "M", kind = "change", unit = "$/t" }
rubber  = { name = "Cao su", source = "worldbank", code = "Rubber, TSR20", freq = "M", kind = "change", unit = "$/kg" }
usd_jpy = { name = "USD/JPY", source = "yahoo", code = "JPY=X", freq = "D", kind = "change", unit = "yên" }
jpy_vnd = { name = "JPY/VND", source = "derived", code = "usd_vnd/usd_jpy", freq = "D", kind = "change", unit = "đồng" }
usd_vnd = { name = "USD/VND", source = "yahoo", code = "VND=X", freq = "D", kind = "change", unit = "đồng" }
vn_cpi  = { name = "CPI so cùng kỳ", source = "manual", freq = "M", kind = "level", unit = "%", neutral = 3.5, scale = 1.0 }
vn_gdp  = { name = "GDP so cùng kỳ", source = "manual", freq = "Q", kind = "level", unit = "%", neutral = 6.0, scale = 1.0 }
"""


@pytest.fixture
def cat(tmp_path):
    f = tmp_path / "macro_map.toml"
    f.write_text(CATALOG, encoding="utf-8")
    return catalog(f)


def test_catalog_and_validation(cat, tmp_path):
    assert set(cat) == {"us_10y", "urea", "rubber", "usd_jpy", "jpy_vnd", "usd_vnd", "vn_cpi",
                        "vn_gdp"}
    assert not cat["jpy_vnd"].auto and cat["usd_jpy"].auto
    assert cat["vn_cpi"].manual and not cat["us_10y"].manual
    assert cat["urea"].months == 3                          # mặc định 3 tháng
    bad = tmp_path / "bad.toml"
    bad.write_text('[series]\nx = { name = "x", source = "abc", code = "X" }\n', encoding="utf-8")
    with pytest.raises(ValueError, match="nguồn"):
        catalog(bad)
    bad.write_text('[series]\nx = { name = "x", source = "manual", kind = "level" }\n',
                   encoding="utf-8")
    with pytest.raises(ValueError, match="neutral"):
        catalog(bad)


def test_shipped_catalog_is_valid():
    cat = catalog()                                        # config/macro_map.toml thật
    assert len(cat) >= 10
    assert any(s.manual for s in cat.values()) and any(not s.manual for s in cat.values())


def test_parse_period_and_manual_frame(cat, tmp_path):
    assert parse_period("2026-09") == pd.Timestamp(2026, 9, 1)
    assert parse_period("2026-q3") == pd.Timestamp(2026, 7, 1)
    with pytest.raises(ValueError):
        parse_period("09/2026")
    f = tmp_path / "manual.toml"
    f.write_text('[vn_cpi]\n"2026-08" = 3.2\n"2026-09" = 3.4\n\n[vn_gdp]\n"2026-Q2" = 7.1\n',
                 encoding="utf-8")
    df = manual_frame(f, cat)
    assert len(df) == 3 and set(df["series"]) == {"vn_cpi", "vn_gdp"}
    assert df.loc[df["series"] == "vn_gdp", "date"].iloc[0] == pd.Timestamp(2026, 4, 1)
    f.write_text('[vn_cpii]\n"2026-08" = 3.2\n', encoding="utf-8")   # gõ sai tên chuỗi
    with pytest.raises(ValueError, match="vn_cpii"):
        manual_frame(f, cat)


def test_parsers():
    s = parse_fred_csv(b"observation_date,DGS10\n2026-10-08,4.10\n2026-10-09,.\n2026-10-10,4.15\n")
    assert list(s) == [4.10, 4.15]
    raw = json.dumps({"chart": {"result": [{
        "timestamp": [1791504000, 1791590400],             # 09/10 và 10/10/2026 (UTC)
        "indicators": {"quote": [{"close": [26300.0, None]}]}}]}}).encode()
    y = parse_yahoo(raw)
    assert len(y) == 1 and y.iloc[0] == 26300.0
    html = '<a href="https://thedocs.worldbank.org/x/CMO-Historical-Data-Monthly.xlsx">Monthly</a>'
    assert pink_sheet_url(html).endswith("CMO-Historical-Data-Monthly.xlsx")
    with pytest.raises(ValueError):
        pink_sheet_url("<html></html>")


def _pink_bytes() -> bytes:
    rows = [["World Bank Commodity Price Data"], [None], [None], [None],
            [None, "Crude oil, Brent", "Urea ", "Rubber, TSR20 **"],
            [None, "($/bbl)", "($/mt)", "($/kg)"],
            ["2026M07", 70.1, 400.0, 1.7], ["2026M08", 71.2, 410.0, 1.8],
            ["2026M09", 69.5, "…", 1.75]]
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xw:
        pd.DataFrame([["About"]]).to_excel(xw, sheet_name="AFOSHEET", header=False, index=False)
        pd.DataFrame(rows).to_excel(xw, sheet_name="Monthly Prices", header=False, index=False)
    return buf.getvalue()


def test_pink_sheet_and_provider(cat):
    pink = parse_pink_sheet(_pink_bytes())
    assert list(pink.columns) == ["Crude oil, Brent", "Urea", "Rubber, TSR20 **"]
    assert pink.index[0] == pd.Timestamp(2026, 7, 1) and np.isnan(pink.loc["2026-09-01", "Urea"])

    def fake_get(url: str) -> bytes:
        if "fredgraph" in url:
            return b"DATE,DGS10\n2014-01-02,3.0\n2026-10-09,4.1\n"
        if "commodity-markets" in url:
            return b'<a href="https://x.org/CMO-Historical-Data-Monthly.xlsx">x</a>'
        return _pink_bytes()

    prov = MacroProvider(cat, get=fake_get, pause=0, log=None)
    assert set(prov.symbols()) == {"us_10y", "urea", "rubber", "usd_jpy", "usd_vnd"}
    us = prov.series("us_10y", years=10)
    assert len(us) == 1 and us["value"].iloc[0] == 4.1       # cắt còn 10 năm gần nhất
    urea = prov.series("urea", years=50)
    assert list(urea["value"]) == [400.0, 410.0] and (urea["source"] == "worldbank").all()
    assert list(prov.series("rubber", years=50)["value"]) == [1.7, 1.8, 1.75]   # bỏ "**" khi khớp tên
    with pytest.raises(ValueError, match="nguồn ngoài"):
        prov.series("vn_cpi")


def test_derived_series_and_yahoo_bad_tick():
    from vnsa.data.providers.macro_provider import derive

    spec = Series("jpy_vnd", "JPY/VND", "derived", "usd_vnd/usd_jpy", freq="D")
    d = pd.date_range("2026-10-01", periods=3)
    data = pd.concat([
        pd.DataFrame({"series": "usd_vnd", "date": d, "value": [26000.0, 26100.0, 26200.0]}),
        pd.DataFrame({"series": "usd_jpy", "date": d[1:], "value": [160.0, 162.0]})])
    out = derive(spec, data)
    assert list(out["value"]) == pytest.approx([26100 / 160, 26200 / 162])   # chỉ ngày chung
    ts = [1790000000 + 86400 * i for i in range(7)]
    raw = json.dumps({"chart": {"result": [{"timestamp": ts, "indicators": {"quote": [{
        "close": [26000.0, 26010.0, 26020.0, 2603.0, 26040.0, 26050.0, 26060.0]}]}}]}}).encode()
    y = parse_yahoo(raw)
    assert len(y) == 6 and y.min() > 25000                  # phiên lỗi 2.603 bị bỏ


def test_store_macro_upsert_and_replace(warehouse):
    with Store(warehouse) as store:
        old = pd.DataFrame({"series": "usd_vnd", "date": pd.date_range("2020-01-01", periods=3),
                            "value": [23000.0, 23010.0, 23020.0]})
        store.write_macro(old, "yahoo")
        new = pd.DataFrame({"series": "usd_vnd", "date": pd.date_range("2020-01-03", periods=2),
                            "value": [23500.0, 23600.0]})
        store.write_macro(new, "yahoo")
        df = store.macro("usd_vnd")
        assert list(df["value"]) == [23000.0, 23010.0, 23500.0, 23600.0]   # giữ kỳ cũ hơn
        assert "usd_vnd" in store.macro_loaded_at()
        man = pd.DataFrame({"series": "vn_cpi", "date": [pd.Timestamp(2026, 9, 1)], "value": [3.4]})
        store.replace_macro(man, ["vn_cpi", "vn_gdp"], "nhập tay")
        store.replace_macro(man.iloc[:0], ["vn_cpi"], "nhập tay")            # xoá hết số nhập tay
        assert store.macro("vn_cpi").empty and len(store.macro()) == 4


def test_reading_change_z_and_staleness():
    spec = Series("oil", "Dầu", "fred", "X", freq="M", kind="change", unit="$", months=3)
    dates = pd.date_range("2016-01-01", "2026-09-01", freq="MS")
    rng = np.random.default_rng(0)
    vals = 60 * np.exp(np.cumsum(rng.normal(0, 0.03, len(dates))))
    vals[-1] = vals[-4] * 1.30                             # tăng 30% trong 3 tháng: bất thường
    df = pd.DataFrame({"date": dates, "value": vals})
    r = reading(df, spec, today=pd.Timestamp("2026-10-12"))
    assert r.change == pytest.approx(30.0) and r.z == 2.0 and r.basis == "z 5 năm"
    assert r.period_end == pd.Timestamp("2026-09-30") and not r.stale
    assert fmt_change(r) == "+30,0%"
    late = reading(df, spec, today=pd.Timestamp("2027-03-01"))
    assert late.stale                                      # hơn 75 ngày sau cuối kỳ


def test_reading_level_diff_fallback_and_missing():
    cpi = Series("vn_cpi", "CPI", "manual", freq="M", kind="level", unit="%", neutral=3.5,
                 scale=1.0)
    r = reading(pd.DataFrame({"date": [pd.Timestamp(2026, 9, 1)], "value": [4.0]}), cpi,
                today=pd.Timestamp("2026-10-12"))
    assert r.z == pytest.approx(0.5) and r.basis == "ngưỡng cấu hình"
    ib = Series("vn_ib", "Liên NH", "manual", freq="M", kind="diff", unit="%", scale=1.0)
    df = pd.DataFrame({"date": pd.date_range("2026-05-01", periods=5, freq="MS"),
                       "value": [4.0, 4.1, 4.2, 4.5, 5.0]})
    r = reading(df, ib, today=pd.Timestamp("2026-10-12"))
    assert r.change == pytest.approx(0.9) and r.z == pytest.approx(0.9)   # chưa đủ lịch sử: theo scale
    assert fmt_change(r) == "+0,9 điểm %"
    none = readings(pd.DataFrame(columns=["series", "date", "value", "source"]), {"vn_ib": ib})
    assert none["vn_ib"].basis == "chưa có số" and not none["vn_ib"].ok
    short = Series("x", "x", "fred", "X", freq="M", kind="change")
    r = reading(df, short, today=pd.Timestamp("2026-10-12"))
    assert np.isnan(r.z) and r.basis == "thiếu lịch sử"


def _r(sid: str, z: float, stale: bool = False, missing: bool = False):
    from vnsa.macro.signals import Reading

    if missing:
        return Reading(sid, sid, "%", "change", None, None, np.nan, np.nan, np.nan, "chưa có số",
                       None, True)
    return Reading(sid, sid, "%", "change", pd.Timestamp("2026-09-01"),
                   pd.Timestamp("2026-09-30"), 1.0, 1.0, z, "z 5 năm", 12, stale)


def test_score_drivers_formula_coverage_and_labels():
    from vnsa.macro.score import score_drivers

    specs = [{"series": "a", "sign": 1, "weight": 2, "why": "a tốt"},
             {"series": "b", "sign": -1, "why": "b xấu"},
             {"series": "c", "sign": 1, "why": "thiếu số"},
             {"series": "d", "sign": 1, "why": "số cũ"}]
    rs = {"a": _r("a", 1.0), "b": _r("b", 2.0), "c": _r("c", 0, missing=True),
          "d": _r("d", 2.0, stale=True)}
    m = score_drivers("thử", specs, rs)
    # (2×1×1 − 1×1×2) / (2 × 3) = 0; độ phủ 3/5
    assert m.score == pytest.approx(0.0) and m.coverage == pytest.approx(0.6)
    assert m.label == "Trung tính"
    assert [d.series for d in m.missing] == ["c"] and [d.series for d in m.stale] == ["d"]
    assert sum(d.contribution for d in m.drivers if d.used) == pytest.approx(m.score)
    rs["b"] = _r("b", -2.0)                                  # b giảm mạnh -> tốt
    m = score_drivers("thử", specs, rs)
    assert m.score == pytest.approx((2 * 1 + 2) / 6) and m.label == "Thuận lợi"
    rs = {"a": _r("a", -2.0)}                                # chỉ a có số: độ phủ 2/5 < 0,5
    m = score_drivers("thử", specs, rs)
    assert m.label == "Chưa đủ số liệu" and m.score == pytest.approx(-1.0)


def test_drivers_for_symbol_industry_level_and_model():
    from vnsa.macro.score import drivers_for, industry_score, market_context

    mapping = {
        "symbol": {"FPT": {"name": "FPT riêng", "drivers": [{"series": "a"}]}},
        "industry": [
            {"name": "Thép", "match": "thép", "drivers": [{"series": "b", "sign": -1}]},
            {"name": "Ngân hàng", "match": "ngân hàng", "model": "bank",
             "drivers": [{"series": "a"}]},
        ],
        "market": {"x": {"name": "Nhóm X", "drivers": [{"series": "a"}]},
                   "y": {"name": "Nhóm Y", "drivers": [{"series": "zz"}]}},
    }
    assert drivers_for("fpt", {"industry4": "Thép"}, mapping)[0] == "FPT riêng"
    scope, specs = drivers_for("HPG", {"industry": "Tài nguyên cơ bản",
                                       "industry4": "Thép và sản phẩm thép"}, mapping)
    assert scope.startswith("Thép (ICB cấp 4") and specs[0]["series"] == "b"
    assert drivers_for("XYZ", {"industry": "Tài chính khác", "model": "bank"}, mapping)[0] \
        == "Ngân hàng"
    assert drivers_for("ABC", {"industry": "Du lịch"}, mapping) is None
    rs = {"a": _r("a", 1.0), "b": _r("b", 1.0)}
    hpg = industry_score("HPG", {"industry4": "Thép"}, rs, mapping)
    assert hpg.score == pytest.approx(-0.5) and hpg.label == "Bất lợi"
    overall, groups = market_context(rs, mapping)
    assert groups["y"].label == "Chưa đủ số liệu"
    assert overall.score == pytest.approx(0.5) and overall.coverage == pytest.approx(0.5)


def test_shipped_mapping_references_known_series():
    import tomllib

    from vnsa.config import path

    with path("macro_map").open("rb") as f:
        m = tomllib.load(f)
    known = set(m["series"])
    groups = ([g["drivers"] for g in m["market"].values()] + [i["drivers"] for i in m["industry"]]
              + [s["drivers"] for s in m["symbol"].values()])
    for drivers in groups:
        for d in drivers:
            assert d["series"] in known, d
            assert d.get("sign", 1) in (1, -1)


class FakeMacro:
    """Nguồn vĩ mô giả: đếm số lần tải."""

    def __init__(self):
        self.calls = 0

    def series(self, sid, years=10):
        self.calls += 1
        if sid == "urea":
            raise ValueError("Pink Sheet lỗi")
        d = pd.date_range("2026-07-01", periods=3, freq="MS")
        return pd.DataFrame({"series": sid, "date": d, "value": [1.0, 2.0, 3.0], "source": "fake"})


def test_refresh_fetches_when_stale_and_loads_manual(warehouse, tmp_path, monkeypatch):
    from vnsa.macro.update import refresh

    (tmp_path / "map.toml").write_text(CATALOG, encoding="utf-8")
    (tmp_path / "manual.toml").write_text('[vn_cpi]\n"2026-09" = 3.4\n', encoding="utf-8")
    monkeypatch.setenv("VNSA_MACRO_MAP", str(tmp_path / "map.toml"))
    monkeypatch.setenv("VNSA_MACRO_MANUAL", str(tmp_path / "manual.toml"))
    fake = FakeMacro()
    status, detail = refresh(20, provider=fake)
    assert status == "ok" and "lỗi: urea" in detail and fake.calls == 5
    with Store(warehouse) as store:
        data = store.macro()
    assert set(data["series"]) == {"us_10y", "rubber", "usd_jpy", "usd_vnd", "jpy_vnd", "vn_cpi"}
    status, detail = refresh(20, provider=fake)    # urea lỗi: không bắt mọi lượt tải lại ngay
    assert status == "skip" and fake.calls == 5 and "1 số nhập tay" in detail
    status, _ = refresh(0, provider=fake)              # đến hạn làm mới: thử lại cả urea
    assert status == "ok" and fake.calls == 10


def _monthly(sid: str, values, daily: bool = False) -> pd.DataFrame:
    """Một số mỗi tháng, kỳ cuối = tháng trước. daily=True: kỳ cuối là hôm nay (chuỗi ngày mới)."""
    today = pd.Timestamp.today().normalize()
    end = today.replace(day=1) - pd.DateOffset(months=0 if daily else 1)
    d = list(pd.date_range(end=end, periods=len(values), freq="MS"))
    if daily:
        d[-1] = today
    return pd.DataFrame({"series": sid, "date": d, "value": values, "source": "test"})


def test_risk_free_from_gov_bond_when_fresh(warehouse):
    from vnsa import analysis

    analysis._RF_CACHE.clear()
    assert analysis.risk_free() == (0.035, "settings.toml")          # chưa có số: settings
    with Store(warehouse) as store:
        store.write_macro(_monthly("vn_gov_10y", [3.1, 3.2]))
    analysis._RF_CACHE.clear()
    rf, src = analysis.risk_free()
    assert rf == pytest.approx(0.032) and "TPCP 10 năm" in src
    assert analysis.valuation_settings()["risk_free"] == pytest.approx(0.032)
    with Store(warehouse) as store:                                   # gõ nhầm 32 thay vì 3.2
        store.write_macro(_monthly("vn_gov_10y", [3.1, 32.0]))
    analysis._RF_CACHE.clear()
    assert analysis.risk_free()[0] == 0.035
    with Store(warehouse) as store:                                   # số đã cũ 5 tháng
        store.replace_macro(_monthly("vn_gov_10y", [3.0]).assign(
            date=pd.Timestamp.today().normalize() - pd.DateOffset(months=5)), ["vn_gov_10y"], "x")
    analysis._RF_CACHE.clear()
    rf, src = analysis.risk_free()
    assert rf == 0.035 and "đã cũ" in src
    analysis._RF_CACHE.clear()


def test_macro_view_and_card_section(warehouse):
    from vnsa import analysis
    from vnsa.report.card import format_card

    rng = np.random.default_rng(1)
    walk = lambda n: 100 * np.exp(np.cumsum(rng.normal(0, 0.03, n)))
    with Store(warehouse) as store:
        store.write_companies(pd.DataFrame({
            "symbol": ["DEMO_A"], "name": ["Thép Demo"], "exchange": ["HOSE"],
            "industry": ["Tài nguyên Cơ bản"], "model": ["general"], "industry3": ["Kim loại"],
            "industry4": ["Thép và sản phẩm thép"]}), "test")
        ore = walk(72)
        ore[-1] = ore[-4] * 1.4                                       # quặng sắt +40%/3 tháng
        store.write_macro(_monthly("iron_ore", ore, daily=True))
        store.write_macro(_monthly("steel_hrc", walk(72), daily=True))
        store.write_macro(_monthly("usd_vnd", 23000 * np.exp(np.cumsum(rng.normal(0.002, 0.005, 72))),
                                   daily=True))
    analysis._RF_CACHE.clear()
    mv = analysis.macro_view("DEMO_A")
    ind = mv.industry
    assert ind.scope.startswith("Thép (ICB cấp 4: Thép và sản phẩm thép)")
    ore_driver = next(d for d in ind.drivers if d.series == "iron_ore")
    assert ore_driver.reading.z == 2.0 and ore_driver.contribution > 0  # quặng tăng cùng giá thép
    assert {d.series for d in ind.missing} == {"coal", "vn_credit_ytd", "vn_public_invest_yoy"}
    assert mv.groups["ty_gia"].label != "Chưa đủ số liệu" and mv.risk_free == 0.035
    text = format_card(analysis.report_card("DEMO_A"))
    assert "VĨ MÔ & NGÀNH (tham khảo, chưa vào điểm)" in text
    assert "Quặng sắt 62% (CFR Trung Quốc)" in text and "Thiếu:" in text
    assert "Lãi suất phi rủi ro trong WACC: 3,5% (settings.toml)" in text


def test_macro_guard_lowers_confidence_only_when_clearly_adverse():
    from types import SimpleNamespace

    from vnsa.macro.score import MacroScore
    from vnsa.report.card import macro_guard

    cfg = {"guard": -0.5, "guard_coverage": 0.6}
    mv = lambda score, cov: SimpleNamespace(
        industry=MacroScore("Thép", score, "Bất lợi", cov))
    assert "giảm một bậc" in macro_guard(mv(-0.6, 0.8), cfg)
    assert macro_guard(mv(-0.4, 0.8), cfg) is None                    # chưa đủ bất lợi
    assert macro_guard(mv(-0.9, 0.5), cfg) is None                    # độ phủ thấp
    assert macro_guard(None, cfg) is None


def test_sensitivity_detects_sign_and_conflict():
    from vnsa.macro.sensitivity import sensitivity, verdict

    rng = np.random.default_rng(7)
    n = 96
    months = pd.date_range("2018-01-31", periods=n, freq="ME")
    oil = 60 * np.exp(np.cumsum(rng.normal(0, 0.08, n)))
    mkt = np.exp(np.cumsum(rng.normal(0.005, 0.05, n)))
    oil_ch = np.r_[0, np.diff(oil) / oil[:-1]]
    mkt_ch = np.r_[0, np.diff(mkt) / mkt[:-1]]
    stock_ret = 0.8 * mkt_ch + 0.6 * oil_ch + rng.normal(0, 0.02, n)   # dầu tăng -> mã tăng
    stock = 100 * np.cumprod(1 + stock_ret)
    prices = pd.DataFrame({"date": months, "close": stock})
    index = pd.DataFrame({"date": months, "close": 1000 * mkt})
    data = pd.DataFrame({"series": "brent", "date": months, "value": oil})
    cat = {"brent": Series("brent", "Dầu Brent", "yahoo", "BZ=F", freq="D", kind="change"),
           "x": Series("x", "Thiếu", "manual", freq="M", kind="level", neutral=0, scale=1)}
    res = {r.series: r for r in sensitivity(prices, index, [
        {"series": "brent", "sign": 1}, {"series": "x", "sign": 1}], data, cat)}
    assert res["brent"].verdict == "đúng dấu" and res["brent"].beta_sd > 0 and res["brent"].t > 2
    assert res["brent"].n == n - 1 and res["x"].verdict == "thiếu lịch sử"
    flipped = sensitivity(prices, index, [{"series": "brent", "sign": -1}], data, cat)[0]
    assert flipped.conflict
    assert verdict(1, 0.3, 1.5) == "yếu" and verdict(1, 0.1, 0.4) == "chưa thấy liên hệ"


def test_signals_store_macro_scores_and_review_groups(warehouse):
    from vnsa import analysis
    from vnsa.recommend.review import macro_label, summarize_macro

    with Store(warehouse) as store:
        store.write_macro(_monthly("usd_vnd", list(np.linspace(23000, 26000, 72)), daily=True))
    a = analysis.assess("DEMO_A")
    a.rec.inputs.macro_industry = -0.45
    analysis.record_signals([(a.rec, a.plan)])
    with Store(warehouse) as store:
        sig = store.signals("DEMO_A")
    assert sig["macro_industry"].iloc[-1] == pytest.approx(-0.45)
    assert not pd.isna(sig["macro_market"].iloc[-1])                 # nhóm tỷ giá có số
    assert macro_label(0.31) == "Thuận lợi" and macro_label(float("nan")) == "chưa có điểm"
    ev = pd.DataFrame({"horizon": [20, 20, 20], "macro_industry": [0.5, -0.5, -0.6],
                       "excess": [0.05, -0.02, -0.04], "ret": [0.06, -0.01, -0.03]})
    by = summarize_macro(ev).set_index("macro")
    assert by.loc["Thuận lợi", "n"] == 1 and by.loc["Bất lợi", "avg_excess"] == pytest.approx(-0.03)


def test_news_parse_store_and_card(warehouse):
    from vnsa import analysis
    from vnsa.data.providers.vnstock_provider import standardize_news
    from vnsa.report.card import format_card

    today = pd.Timestamp.today().normalize()
    raw = pd.DataFrame({
        "news_title": ["DEMO_A: Nghị quyết HĐQT", "DEMO_A: Báo cáo quý 3", "Tin cũ", ""],
        "public_date": [(today - pd.Timedelta(days=d)).isoformat() for d in (1, 3, 90, 2)],
        "news_source_link": [None, "https://x.vn/a", None, None], "news_id": [1, 2, 3, 4]})
    news = standardize_news(raw, "demo_a")
    assert list(news["title"]) == ["DEMO_A: Nghị quyết HĐQT", "DEMO_A: Báo cáo quý 3", "Tin cũ"]
    with pytest.raises(ValueError, match="đổi cấu trúc"):
        standardize_news(pd.DataFrame({"x": [1]}), "A")
    with Store(warehouse) as store:
        assert store.write_news(news, "DEMO_A") == 3
        assert len(store.news("DEMO_A", days=30)) == 2 and store.news_loaded_at("DEMO_A")
        store.write_macro(_monthly("usd_vnd", list(np.linspace(23000, 26000, 72)), daily=True))
    text = format_card(analysis.report_card("DEMO_A"))
    assert "Tin gần đây" in text and "Nghị quyết HĐQT" in text and "Tin cũ" not in text


def test_mcp_macro_context(warehouse):
    pytest.importorskip("mcp")
    from vnsa import mcp_server as srv

    assert "fetch-macro" in srv.macro_context()["error"]
    with Store(warehouse) as store:
        store.write_macro(_monthly("usd_vnd", list(np.linspace(23000, 26000, 72)), daily=True))
    out = srv.macro_context("DEMO_A")
    json.dumps(out, allow_nan=False)
    assert out["industry"] is None and "ty_gia" in out["groups"]       # DEMO_A chưa có ngành
    assert out["risk_free"]["source"] == "settings.toml"
    table = srv.macro_context()
    assert any(r["series"] == "usd_vnd" and r["value"] for r in table["series"])
