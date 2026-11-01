"""Chuyển sang giao diện vnstock thống nhất (Market/Reference/Fundamental, source="vci"), lớp cũ dự
phòng; bảng giá phẳng; không dùng nguồn mặc định "kbs" (BCTC đảo nhãn năm)."""

import sys
import types

import pandas as pd
import pytest

from vnsa.data.providers import vnstock_provider as vp

CALLS = []


def _fake_vnstock(new=True, new_fails=False, board_flat=True):
    mod = types.ModuleType("vnstock")

    class Obj:
        def __init__(self, kind, sym=None):
            self.kind, self.sym = kind, sym

        def __getattr__(self, name):
            def method(*a, **k):
                CALLS.append((self.kind, self.sym, name, k))
                if new_fails:
                    raise ValueError("nguồn mới lỗi")
                if name == "ohlcv":
                    d = pd.bdate_range("2026-09-01", "2026-10-30")
                    return pd.DataFrame({"time": d, "open": 1.0, "high": 1.0, "low": 1.0,
                                         "close": 1.0, "volume": 1})
                return pd.DataFrame({"src": [k.get("source")], "how": ["new"]})
            return method

    class Market:
        def equity(self, sym):
            return Obj("equity", sym)

        def index(self, sym):
            return Obj("index", sym)

        def quote(self, syms, source=None):
            CALLS.append(("quote", tuple(syms), "quote", {"source": source}))
            if board_flat:
                return pd.DataFrame({"symbol": syms, "close_price": [52700] * len(syms),
                                     "reference_price": [52000] * len(syms),
                                     "ceiling_price": 55600, "floor_price": 48400,
                                     "volume_accumulated": 100, "time": [1793349188195] * len(syms)})
            return pd.DataFrame()

    class Reference:
        def company(self, sym):
            return Obj("company", sym)

        def equity(self):
            return Obj("ref_equity")

    class Fundamental:
        def equity(self, sym):
            return Obj("fundamental", sym)

    class Old:
        def __init__(self, *a, source=None, symbol=None):
            self.symbol = symbol

        def __getattr__(self, name):
            def method(*a, **k):
                CALLS.append(("old", self.symbol, name, k))
                if name == "price_board":
                    cols = pd.MultiIndex.from_tuples([("listing", "symbol"),
                                                      ("listing", "trading_status"),
                                                      ("match", "match_price")])
                    return pd.DataFrame([[s, "TRADING_RESTRICTION_ACTIVATED", 1.0] for s in a[0]],
                                        columns=cols)
                return pd.DataFrame({"how": ["old"]})
            return method

    if new:
        mod.Market, mod.Reference, mod.Fundamental = Market, Reference, Fundamental
    mod.Company = mod.Finance = mod.Quote = mod.Trading = mod.Listing = Old
    return mod


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(vp, "_call", lambda fn, *a, **k: fn(*a, **k))
    CALLS.clear()

    def install(**kw):
        monkeypatch.setitem(sys.modules, "vnstock", _fake_vnstock(**kw))
    return install


def test_new_api_always_uses_vci(fake):
    fake()
    st = vp._vnstock_statements("fpt", "quarter")
    assert set(st) == set(vp.STATEMENTS)
    assert all(df["src"].iloc[0] == "vci" for df in st.values())
    assert ("fundamental", "FPT", "income_statement", {"period": "quarter", "source": "vci"}) in CALLS
    prof = vp._vnstock_company("fpt", "overview")
    assert prof["how"].iloc[0] == "new" and CALLS[-1][2] == "info"
    vp._vnstock_company("fpt", "events")
    assert CALLS[-1][2:] == ("events", {"source": "vci"})


def test_falls_back_to_old_classes(fake):
    fake(new_fails=True)
    assert vp._vnstock_company("FPT", "overview")["how"].iloc[0] == "old"
    assert CALLS[-1][:3] == ("old", "FPT", "overview")
    fake(new=False)
    assert vp._vnstock_statements("FPT")["balance_sheet"]["how"].iloc[0] == "old"


def test_quota_error_is_not_retried_with_old(fake, monkeypatch):
    fake()

    def boom(fn, *a, **k):
        raise RuntimeError("vnstock: hết hạn mức lượt gọi (vnai), thử lại sau")

    monkeypatch.setattr(vp, "_call", boom)
    with pytest.raises(RuntimeError, match="hạn mức"):
        vp._vnstock_company("FPT", "news")


def test_history_uses_index_object_count_and_dates(fake):
    fake()
    df = vp._vnstock_history("VNINDEX", "2026-10-01", "2026-10-15")
    kind, sym, name, k = CALLS[-1]
    assert (kind, sym, name) == ("index", "VNINDEX", "ohlcv") and k["source"] == "vci"
    assert k["count"] >= vp._sessions_back("2026-10-01") - 1
    assert df["time"].min() >= pd.Timestamp("2026-10-01")
    assert df["time"].max() <= pd.Timestamp("2026-10-16")
    vp._vnstock_history("FPT", "2026-10-01", "2026-10-15")
    assert CALLS[-1][0] == "equity"


def test_board_prefers_full_table_with_trading_status(fake):
    fake()
    raw = vp._vnstock_board(["fpt", "dgc"])
    b = vp.standardize_board(raw)
    assert list(b["trading_status"]) == ["TRADING_RESTRICTION_ACTIVATED"] * 2   # lớp cũ đủ cột


def test_flat_board_standardized_with_date_from_time():
    raw = pd.DataFrame({"symbol": ["FPT"], "close_price": [0], "reference_price": [52700],
                        "ceiling_price": [56300], "floor_price": [49050],
                        "volume_accumulated": [5661600], "high_price": [53700],
                        "low_price": [52400], "time": [1793349188195]})
    b = vp.standardize_board(raw)
    assert b.loc[0, "price"] == 52700 and b.loc[0, "floor"] == 49050
    assert b.loc[0, "volume"] == 5661600 and b.loc[0, "high"] == 53700
    assert b.loc[0, "trading_date"] == pd.Timestamp("2026-10-30")
