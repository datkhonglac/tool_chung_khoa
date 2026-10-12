"""Đợt 3: giao dịch khối ngoại / tự doanh từ CafeF (định dạng JSON đã dò thật ngày 11/10/2026)."""

import pandas as pd
import pytest

from vnsa.data.providers.cafef_provider import CafeFProvider, standardize_flows

FOREIGN = [
    {"Symbol": "FPT", "Ngay": "09/10/2026", "KLGDRong": -2062200, "GTDGRong": -120636110000,
     "KLMua": 885000, "GtMua": 51712990000, "KLBan": 2947200, "GtBan": 172349100000,
     "RoomConLai": 353561043, "DangSoHuu": 28.38},
    {"Symbol": "FPT", "Ngay": "08/10/2026", "KLGDRong": -977100, "GTDGRong": -59152590000,
     "KLMua": 820500, "GtMua": 49132180000, "KLBan": 1797600, "GtBan": 108284770000,
     "RoomConLai": 353073084, "DangSoHuu": 28.4},
]
PROP = [{"Symbol": "FPT", "Date": "09/10/2026", "KLcpMua": 934500, "KlcpBan": 3419800,
         "GtMua": 54697960000, "GtBan": 198788480000}]


def test_standardize_flows_merges_foreign_and_proprietary():
    df = standardize_flows(FOREIGN, PROP, "fpt")
    assert list(df["date"]) == [pd.Timestamp("2026-10-08"), pd.Timestamp("2026-10-09")]
    last = df.iloc[-1]
    assert last["symbol"] == "FPT" and last["foreign_net"] == -120636110000
    assert last["foreign_pct"] == pytest.approx(0.2838)
    assert last["prop_net"] == pytest.approx(54697960000 - 198788480000)
    assert pd.isna(df.iloc[0]["prop_net"])                       # ngày không có tự doanh


def test_provider_flows_reads_both_pages():
    urls = []

    def fake(url):
        urls.append(url)
        if "GDKhoiNgoai" in url:
            return {"Success": True, "Data": {"TotalCount": 2, "Data": FOREIGN}}
        return {"Success": True, "Data": {"TotalCount": 1, "Data": {"ListDataTudoanh": PROP}}}

    df = CafeFProvider(pause=0, get_json=fake, log=None).flows("FPT", sessions=60)
    assert len(df) == 2 and any("GDTuDoanh" in u for u in urls)


def test_flows_stored_and_shown_on_card(warehouse):
    from vnsa import analysis
    from vnsa.data.store import Store
    from vnsa.report.card import build_card, format_card

    d = analysis.load("DEMO_A", need_fin=False)
    dates = d.prices["date"].tail(20)
    flows = pd.DataFrame({"symbol": "DEMO_A", "date": dates, "foreign_net": -1e9,
                          "prop_net": 2e8, "foreign_pct": 0.1})
    with Store(warehouse) as store:
        assert store.write_flows(flows) == 20
        assert store.write_flows(flows.tail(1)) == 1             # ghi đè, không nhân đôi
        assert len(store.flows("DEMO_A")) == 20
    d = analysis.load("DEMO_A", need_fin=False)
    fs = analysis.flow_summary(d)
    assert fs["foreign_5"] == pytest.approx(-5e9) and fs["foreign_20"] == pytest.approx(-20e9)
    assert fs["prop_20"] == pytest.approx(4e9) and fs["foreign_20_pct"] < 0
    text = format_card(build_card(analysis.assess("DEMO_A")))
    assert "Khối ngoại 5 phiên bán ròng 5,0 tỷ" in text and "Tự doanh 20 phiên mua ròng 4,0 tỷ" in text
