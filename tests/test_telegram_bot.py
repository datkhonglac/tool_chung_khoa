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


def test_run_update_from_worker(monkeypatch, tmp_path):
    """v3.2: Worker chuyển một update sang -> trả lời, không gửi "Đang phân tích" (Worker đã báo)."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    sent = []
    monkeypatch.setattr(tb.notify, "send_telegram",
                        lambda text, token=None, chat_id=None: sent.append((chat_id, text)) or 1)
    f = tmp_path / "update.json"
    f.write_text(json.dumps({"update_id": 7, "message": {"chat": {"id": 42}, "text": "fpt"}}),
                 encoding="utf-8")
    logs = []
    assert tb.run_update(str(f), log=logs.append, analyze=lambda s: f"thẻ {s}") == 0
    assert sent == [("42", "thẻ FPT")] and "Xong: 1" in logs[-1]
    sent.clear()
    stranger = json.dumps({"update_id": 8, "message": {"chat": {"id": 7}, "text": "hpg"}})
    assert tb.run_update(stranger, log=logs.append, analyze=lambda s: f"thẻ {s}") == 0
    assert sent == []
    assert tb.run_update("{không phải json", log=logs.append) == 1


def test_webhook_conflict_explains(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")

    def opener(req, timeout):
        return _Resp(b'{"ok": false, "description": "Conflict: can\'t use getUpdates method while '
                     b'webhook is active; use deleteWebhook to delete the webhook first"}')

    logs = []
    assert tb.run(once=True, log=logs.append, opener=opener) == 0
    assert "deleteWebhook" in logs[-1] and "123:ABC" not in logs[-1]


def test_sangloc_plan_and_dispatch(monkeypatch):
    """/sangloc: đọc lượt đã lưu; /sangloc moi: gọi workflow_dispatch screen.yml (notify=true)."""
    assert tb.plan("/sangloc") == ("command", ["/sangloc"])
    assert tb.plan("/sangloc@ba_be111bot moi") == ("command", ["/sangloc", "moi"])
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert "GITHUB_TOKEN" in tb.start_screen()
    monkeypatch.setenv("GITHUB_TOKEN", "ghs_x")
    monkeypatch.setenv("GITHUB_REPOSITORY", "me/repo")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")
    seen = []

    class Ok(_Resp):
        status = 204

    def opener(req, timeout):
        seen.append(req)
        return Ok(b"")

    assert "Đã bắt đầu" in tb.start_screen(opener=opener)
    req = seen[0]
    assert req.get_method() == "POST" and req.full_url.endswith(
        "/repos/me/repo/actions/workflows/screen.yml/dispatches")
    assert json.loads(req.data) == {"ref": "main", "inputs": {"notify": "true"}}
    assert req.get_header("Authorization") == "Bearer ghs_x"

    def denied(req, timeout):
        raise tb.urllib.error.HTTPError(req.full_url, 403, "Forbidden", {}, io.BytesIO(b""))

    msg = tb.start_screen(opener=denied)
    assert "403" in msg and "actions: write" in msg and "ghs_x" not in msg


def test_sangloc_routes_and_formats(monkeypatch):
    import pandas as pd

    from vnsa.screen import run as sr

    monkeypatch.setattr(tb, "start_screen", lambda: "chạy")
    monkeypatch.setattr(tb, "screen_text", lambda: "kết quả")
    sent = []
    tb.handle("/sangloc mới", sent.append)
    tb.handle("/sangloc", sent.append)
    assert sent == ["chạy", "kết quả"]
    monkeypatch.undo()
    monkeypatch.setattr(sr, "latest", lambda as_of=None: (pd.DataFrame(), pd.DataFrame()))
    assert "/sangloc moi" in tb.screen_text(import_history=False)
