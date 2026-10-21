"""Kho dùng chung giữa nhiều tiến trình: thử lại khi kho bận, LazyStore không giữ khoá."""

import subprocess
import sys
import textwrap
import time

import pandas as pd
import pytest

from vnsa.data.store import SCHEMA_VERSION, LazyStore, Store, WarehouseBusy


def _hold(db, seconds: float) -> subprocess.Popen:
    """Tiến trình khác mở kho để ghi và giữ `seconds` giây (như lệnh screen đang chạy)."""
    code = textwrap.dedent(f"""
        import duckdb, sys, time
        con = duckdb.connect({str(db)!r})
        print("held", flush=True)
        time.sleep({seconds})
        con.close()
    """)
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "held"
    return proc


def _can_open(db) -> bool:
    code = f"import duckdb; duckdb.connect({str(db)!r}).close()"
    return subprocess.run([sys.executable, "-c", code], capture_output=True).returncode == 0


def test_waits_for_busy_warehouse(tmp_path):
    db = tmp_path / "w.duckdb"
    Store(db).close()
    proc = _hold(db, 1.5)
    try:
        t0 = time.monotonic()
        with Store(db, wait=20) as store:          # chờ tiến trình kia đóng rồi mở được
            assert store.companies().empty
        assert time.monotonic() - t0 > 0.5
    finally:
        proc.wait()


def test_gives_up_with_clear_message(tmp_path):
    db = tmp_path / "w.duckdb"
    Store(db).close()
    proc = _hold(db, 3)
    try:
        with pytest.raises(WarehouseBusy, match="tiến trình khác"):
            Store(db, wait=0.3)
    finally:
        proc.wait()


def test_lazy_store_does_not_hold_lock(tmp_path):
    db = tmp_path / "w.duckdb"
    lazy = LazyStore(db)
    prices = pd.DataFrame({"symbol": "AAA", "date": pd.date_range("2026-01-05", periods=3),
                           "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 100.0})
    with lazy as store:
        store.write_prices(prices, source="test")
        assert _can_open(db)                       # giữa hai lời gọi kho không bị khoá
        assert len(store.prices("AAA")) == 3


def test_lazy_store_rejects_unknown_attribute(tmp_path):
    with pytest.raises(AttributeError):
        LazyStore(tmp_path / "w.duckdb").khong_co


def test_schema_version_recorded(tmp_path):
    db = tmp_path / "w.duckdb"
    with Store(db) as store:
        assert store.con.execute("SELECT value FROM meta WHERE key = 'schema'").fetchone() == (
            SCHEMA_VERSION,)
    with Store(db) as store:                       # mở lại: bỏ qua bước nâng cấp, bảng vẫn đủ
        assert store.signals().empty
