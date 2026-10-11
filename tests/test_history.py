"""Lịch sử bền vững (history.py): khuyến nghị và BCTC tích luỹ qua nhánh `signals`."""

import pandas as pd

from vnsa import history
from vnsa.data.store import Store

B = 1_000_000_000


def _fin(years, ni=10):
    return pd.DataFrame({"symbol": "HIST", "year": years, "net_income_parent": ni * B,
                         "equity": 100 * B, "total_assets": 300 * B})


def test_fundamentals_keep_older_years(warehouse):
    with Store(warehouse) as store:
        store.write_fundamentals(_fin([2020, 2021, 2022, 2023]), "vci")
        store.write_fundamentals(_fin([2022, 2023, 2024, 2025], ni=12), "vci")   # nguồn chỉ 4 năm
        f = store.fundamentals("HIST")
    assert list(f["year"]) == [2020, 2021, 2022, 2023, 2024, 2025]
    assert f.loc[f["year"] == 2023, "net_income_parent"].iloc[0] == 12 * B      # năm trùng: số mới


def test_export_import_roundtrip_adds_only_missing(warehouse, tmp_path):
    q = pd.DataFrame({"symbol": "HIST", "year": [2025, 2025], "quarter": [3, 4],
                      "net_income_parent": [3 * B, 4 * B]})
    sig = pd.DataFrame({"as_of": ["2026-10-09"], "symbol": ["HIST"], "action": ["MUA"],
                        "score": [70.0], "price": [37_650.0]})
    with Store(warehouse) as store:
        store.write_fundamentals(_fin([2024, 2025]), "vci")
        store.write_quarterly(q, "vci")
        store.write_signals(sig)
    history.export_history(tmp_path)
    fin = pd.read_csv(tmp_path / "fundamentals.csv")
    assert list(fin.loc[fin["symbol"] == "HIST", "year"]) == [2024, 2025]
    assert len(pd.read_csv(tmp_path / "fundamentals_q.csv").query("symbol == 'HIST'")) == 2

    with Store(warehouse) as store:                      # kho mới hơn: năm 2025 đã sửa số
        store.con.execute("DELETE FROM fundamentals WHERE symbol = 'HIST' AND year = 2024")
        store.con.execute("DELETE FROM signals")
        store.con.execute("UPDATE fundamentals SET net_income_parent = 99 WHERE symbol = 'HIST' AND year = 2025")
    added = history.import_history(tmp_path)
    assert added == {"signals": 1, "fundamentals": 1, "fundamentals_q": 0}
    with Store(warehouse) as store:
        f = store.fundamentals("HIST")
        assert f.loc[f["year"] == 2025, "net_income_parent"].iloc[0] == 99   # không ghi đè số mới
        assert len(store.signals()) == 1
        old = store.con.execute("SELECT loaded_at FROM fundamentals WHERE symbol = 'HIST' AND year = 2024").fetchone()[0]
    assert pd.Timestamp(old).year == 2000                # đánh dấu cũ: autofetch vẫn tải số mới


def test_import_from_missing_git_ref_is_harmless(warehouse):
    assert history.import_history(git_ref="origin/khong-co-nhanh-nay") == {
        "signals": 0, "fundamentals": 0, "fundamentals_q": 0}
