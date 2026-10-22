"""Theo dõi trong phiên (monitor.py), bảng giá VCI (standardize_board), Telegram (notify.py)."""

import io
import json
import urllib.error

import pandas as pd
import pytest

from vnsa import monitor, notify
from vnsa import portfolio as pf
from vnsa.data.providers.vnstock_provider import price_board, standardize_board

NOW = pd.Timestamp("2026-10-22 13:50")        # thứ Năm


def _raw_board(rows):
    """Bảng giá giả theo cấu trúc VCI thật: cột MultiIndex (listing|match, tên)."""
    cols = pd.MultiIndex.from_tuples([
        ("listing", "symbol"), ("listing", "ceiling"), ("listing", "floor"),
        ("listing", "ref_price"), ("listing", "exchange"), ("listing", "trading_status"),
        ("listing", "security_status"), ("listing", "trading_date"),
        ("match", "match_price"), ("match", "open_price"), ("match", "highest"),
        ("match", "lowest"), ("match", "accumulated_volume")])
    data = [[s, round(ref * 1.07, -2), round(ref * 0.93, -2), ref, "HSX", st, sec, date,
             px, ref, max(px, ref), min(px, ref), 1000]
            for s, ref, px, st, sec, date in rows]
    return pd.DataFrame(data, columns=cols)


def _board(*rows):
    full = [(*r, "TRADING_ACTIVATED", "N", "2026-10-22")[:6] if len(r) == 3 else r for r in rows]
    return standardize_board(_raw_board(full))


def test_standardize_board_reads_multiindex_and_zero_price():
    b = standardize_board(_raw_board([("FPT", 57_400, 57_500, "TRADING_ACTIVATED", "N", "2026-10-22"),
                                      ("DIG", 10_200, 0, "TRADING_ACTIVATED", "N", "2026-10-22")]))
    assert list(b["symbol"]) == ["FPT", "DIG"]
    assert b.loc[0, "price"] == 57_500 and b.loc[0, "floor"] == 53_400
    assert b.loc[1, "price"] == 10_200          # chưa khớp: dùng giá tham chiếu
    assert b.loc[0, "trading_date"] == pd.Timestamp("2026-10-22")


def test_standardize_board_flat_columns_and_chunks():
    raw = _raw_board([("FPT", 57_400, 57_500, "TRADING_ACTIVATED", "N", "2026-10-22")])
    raw.columns = [c[1] for c in raw.columns]
    raw = raw.rename(columns={"ceiling": "x"})                 # thiếu cột -> NaN, không lỗi
    b = standardize_board(raw)
    assert b.loc[0, "price"] == 57_500 and pd.isna(b.loc[0, "ceiling"])
    calls = []
    out = price_board([f"M{i}" for i in range(5)], fetch=lambda s: calls.append(s) or _raw_board(
        [(x, 10_000, 10_000, "TRADING_ACTIVATED", "N", "2026-10-22") for x in s]), chunk=2)
    assert len(calls) == 3 and len(out) == 5


def _ledger(tmp_path, rows):
    f = tmp_path / "trades.csv"
    for side, sym, qty, price, date in rows:
        pf.add(side, sym, qty, price, date, fee=0, tax=0, file=f, calendar=pd.DatetimeIndex([]))
    return pf.load(f)


def _signals(rows):
    return pd.DataFrame(rows, columns=["as_of", "symbol", "action", "buy_low", "buy_high", "stop",
                                       "target"])


def test_stop_breach_reports_sellable(tmp_path):
    trades = _ledger(tmp_path, [(pf.BUY, "HPG", 1_000, 25_000, "2026-10-01"),
                                (pf.STOP, "HPG", 0, 23_000, "2026-10-01"),
                                (pf.BUY, "HPG", 500, 24_000, "2026-10-21")])
    res = monitor.run(NOW, board_fn=lambda s: _board(("HPG", 23_500, 22_900)), trades=trades,
                      signals=_signals([]), actions={}, extra_watch=[])
    a = [x for x in res.alerts if x.kind == "cat_lo"]
    assert len(a) == 1 and a[0].level == monitor.URGENT and a[0].held
    assert "bán được 1.000 cp" in a[0].message          # lô mua 21/10 chưa về (T+2)


def test_near_stop_target_and_no_stop(tmp_path):
    trades = _ledger(tmp_path, [(pf.BUY, "FPT", 100, 100_000, "2026-10-01"),
                                (pf.STOP, "FPT", 0, 95_000, "2026-10-01"),
                                (pf.BUY, "VNM", 100, 60_000, "2026-10-01")])
    sig = _signals([("2026-10-21", "FPT", "NẮM GIỮ", 90_000, 92_000, 88_000, 96_000)])
    board = lambda s: _board(("FPT", 97_000, 96_500), ("VNM", 60_000, 53_000))  # noqa: E731
    res = monitor.run(NOW, board_fn=board, trades=trades, signals=sig, actions={})
    kinds = {(x.symbol, x.kind) for x in res.alerts}
    assert ("FPT", "gan_cat_lo") in kinds and ("FPT", "muc_tieu") in kinds
    assert ("VNM", "lo_khong_cat") in kinds               # lỗ 11,7% chưa đặt cắt lỗ
    assert res.alerts[0].level == monitor.URGENT          # KHẨN xếp trước


def test_floor_and_abnormal_status(tmp_path):
    trades = _ledger(tmp_path, [(pf.BUY, "DIG", 1_000, 11_000, "2026-10-01"),
                                (pf.STOP, "DIG", 0, 9_000, "2026-10-01")])
    board = lambda s: _board(("DIG", 10_200, 9_490, "HALT", "C", "2026-10-22"))  # noqa: E731
    res = monitor.run(NOW, board_fn=board, trades=trades, signals=_signals([]), actions={})
    kinds = {x.kind for x in res.alerts}
    assert kinds == {"nam_san", "trang_thai"}
    assert "HALT" in next(x.message for x in res.alerts if x.kind == "trang_thai")


def test_watch_zone_only_for_buy_ratings(tmp_path):
    sig = _signals([("2026-10-21", "MWG", "MUA", 60_000, 62_000, 56_000, 70_000),
                    ("2026-10-21", "VCB", "NẮM GIỮ", 60_000, 62_000, 56_000, 70_000),
                    ("2026-09-01", "SSI", "MUA", 30_000, 32_000, 28_000, 36_000)])   # quá cũ
    board = lambda s: _board(("MWG", 63_000, 61_000), ("VCB", 63_000, 61_000),  # noqa: E731
                             ("SSI", 31_000, 31_000))
    res = monitor.run(NOW, board_fn=board, trades=pd.DataFrame(columns=pf.COLUMNS), signals=sig,
                      extra_watch=["MWG", "VCB", "SSI"])
    zone = [x for x in res.alerts if x.kind == "vung_mua"]
    assert [x.symbol for x in zone] == ["MWG"] and not zone[0].held


def test_public_output_hides_positions(tmp_path):
    trades = _ledger(tmp_path, [(pf.BUY, "HPG", 1_000, 25_000, "2026-10-01"),
                                (pf.STOP, "HPG", 0, 23_000, "2026-10-01")])
    sig = _signals([("2026-10-21", "MWG", "MUA", 60_000, 62_000, 56_000, 70_000)])
    board = lambda s: _board(("HPG", 23_500, 22_900), ("MWG", 63_000, 61_000))  # noqa: E731
    res = monitor.run(NOW, board_fn=board, trades=trades, signals=sig, actions={},
                      extra_watch=["MWG"])
    private, public = monitor.format_alerts(res), monitor.format_alerts(res, public=True)
    assert "HPG" in private and "HPG" not in public and "MWG" in public
    assert "đang giữ" in private


def test_skips_weekend_and_holiday(tmp_path):
    empty = pd.DataFrame(columns=pf.COLUMNS)
    called = []
    res = monitor.run(pd.Timestamp("2026-10-24 10:00"), board_fn=lambda s: called.append(s),
                      trades=empty, signals=_signals([]), extra_watch=["FPT"])
    assert not res.trading_day and not called
    old = lambda s: _board(("FPT", 57_400, 57_500, "TRADING_ACTIVATED", "N", "2026-10-21"))  # noqa
    res = monitor.run(NOW, board_fn=old, trades=empty, signals=_signals([]), extra_watch=["FPT"])
    assert not res.trading_day and "không có phiên" in monitor.format_alerts(res)


def test_missing_price_listed(tmp_path):
    res = monitor.run(NOW, board_fn=lambda s: _board(("FPT", 57_400, 57_500)),
                      trades=pd.DataFrame(columns=pf.COLUMNS), signals=_signals([]),
                      extra_watch=["FPT", "XYZ"])
    assert "XYZ" in res.missing and "XYZ" in monitor.format_alerts(res)


# ---------- Telegram ----------

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_split_long_message():
    text = "\n".join(f"dòng {i} " + "x" * 50 for i in range(200))
    parts = notify.split(text, 1000)
    assert all(len(p) <= 1000 for p in parts) and "\n".join(parts) == text


def test_send_telegram_posts_json_and_scrubs_token():
    seen = []

    def ok(req, timeout):
        seen.append((req.full_url, json.loads(req.data)))
        return _Resp(b'{"ok": true}')

    assert notify.send_telegram("xin chào", token="123:ABC", chat_id="42", opener=ok) == 1
    url, body = seen[0]
    assert url.endswith("/bot123:ABC/sendMessage") and body == {
        "chat_id": "42", "text": "xin chào", "disable_web_page_preview": True}

    def bad(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {},
                                     io.BytesIO(b'{"description": "bot123:ABC invalid"}'))

    with pytest.raises(notify.NotifyError) as e:
        notify.send_telegram("x", token="123:ABC", chat_id="42", opener=bad)
    assert "123:ABC" not in str(e.value) and "401" in str(e.value)


def test_send_requires_config(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert not notify.configured()
    with pytest.raises(notify.NotifyError):
        notify.send_telegram("x")
