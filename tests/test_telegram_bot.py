"""Bot Telegram: tách mã trong tin nhắn, lệnh, chỉ trả lời chat được phép, báo đã đọc (offset)."""

import io
import json

import pytest

from vnsa import telegram_bot as tb


def test_symbols_in_message():
    assert tb.symbols_in("fpt") == ["FPT"]
    assert tb.symbols_in("FPT, hpg; dgc  fpt") == ["FPT", "HPG", "DGC"]
    assert tb.symbols_in("xem giá mua HPG cho tôi") == ["HPG"]          # bỏ từ thông dụng 3 chữ
    assert tb.symbols_in("ACB1 VN30 A1 hello") == []


def test_plan_commands_and_help():
    assert tb.plan("/start") == ("command", ["/start"])
    assert tb.plan("/canhbao@ba_be111bot") == ("command", ["/canhbao"])
    assert tb.plan("/FPT") == ("symbols", ["FPT"])
    assert tb.plan("chào bot") == ("help", [])


def test_handle_limits_symbols_and_survives_errors():
    sent = []

    def analyze(sym):
        if sym == "HPG":
            raise RuntimeError("hết hạn mức")
        return f"thẻ {sym}"

    n = tb.handle("FPT HPG DGC VNM", sent.append, analyze)
    assert n == 4 and "Bỏ qua VNM" in sent[0]
    assert sent[1] == "thẻ FPT" and "HPG: lỗi" in sent[2] and sent[3] == "thẻ DGC"
    sent.clear()
    tb.handle("/help", sent.append, analyze)
    assert sent == [tb.HELP]


def test_process_only_allowed_chats_and_offset():
    updates = [
        {"update_id": 11, "message": {"chat": {"id": 42}, "text": "fpt"}},
        {"update_id": 10, "message": {"chat": {"id": 999}, "text": "hpg"}},     # người lạ: bỏ qua
        {"update_id": 12, "message": {"chat": {"id": 42}}},                     # ảnh, không có chữ
    ]
    sent = []
    n, nxt = tb.process(updates, "T", {"42"},
                        send_fn=lambda t, token, chat_id: sent.append((chat_id, t)),
                        analyze=lambda s: f"thẻ {s}")
    assert n == 1 and nxt == 13
    assert all(c == "42" for c, _ in sent) and sent[-1][1] == "thẻ FPT"


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_run_once_acknowledges_updates(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    calls = []

    def opener(req, timeout):
        body = json.loads(req.data)
        calls.append((req.full_url.rsplit("/", 1)[1], body))
        if req.full_url.endswith("getUpdates") and "offset" not in body:
            return _Resp(json.dumps({"ok": True, "result": [
                {"update_id": 5, "message": {"chat": {"id": 42}, "text": "/help"}}]}).encode())
        return _Resp(b'{"ok": true, "result": []}')

    monkeypatch.setattr(tb.notify.urllib.request, "urlopen", opener)
    monkeypatch.setattr(tb.notify, "send_telegram",
                        lambda text, token=None, chat_id=None, opener=opener:
                        calls.append(("sendMessage", {"chat_id": chat_id, "text": text})) or 1)
    logs = []
    assert tb.run(once=True, log=logs.append, opener=opener) == 0
    assert ("sendMessage", {"chat_id": "42", "text": tb.HELP}) in calls
    assert calls[-1] == ("getUpdates", {"timeout": 0, "allowed_updates": ["message"], "offset": 6})
    assert "Xong: 1" in logs[-1]


def test_run_requires_config(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    assert tb.run(once=True, log=lambda m: None) == 1
