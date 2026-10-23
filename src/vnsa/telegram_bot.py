"""Hỏi tool qua Telegram (nâng cấp v3.1, lệnh `bot`, workflow telegram-bot.yml).

Nhắn mã cổ phiếu (vd "FPT" hoặc "fpt hpg") cho bot -> bot trả THẺ BÁO CÁO như lệnh `analyze`.
Lệnh khác: /canhbao (theo dõi trong phiên), /danhmuc (danh mục), /nhatky (nhật ký giao dịch),
/help. Chỉ trả lời các chat trong TELEGRAM_CHAT_ID (nhiều chat cách nhau bằng dấu phẩy); tin từ
người lạ bị bỏ qua.

Hai cách chạy:
- `python -m vnsa.cli bot`         chạy liên tục (Codespaces / máy tính) — trả lời ngay, tắt máy là dừng;
- `python -m vnsa.cli bot --once`  xử lý các tin đang chờ rồi thoát — workflow GitHub Actions chạy
                                   mỗi ~10 phút nên trả lời chậm 5–30 phút nhưng không cần bật máy.
Telegram giữ tin chưa xử lý 24 giờ; sau khi trả lời, bot báo "đã đọc" (offset) để không trả lời lại.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import time
import urllib.request

from vnsa import notify

API = "https://api.telegram.org/bot{token}/{method}"
MAX_SYMBOLS = 3
TICKER = re.compile(r"^[A-Z][A-Z0-9]{2}$")
STOPWORDS = {"THE", "AND", "MUA", "BAN", "GIA", "CON", "XEM", "CHO", "HOI", "SAO", "NAO", "MAY",
             "HAY", "VOI", "CUA", "LAI", "LOI", "TOT", "XAU", "KHI", "THI", "BOT", "OKE", "ANH",
             "CHI", "TOI", "BAO"}
HELP = (
    "VNSA — tool phân tích cổ phiếu\n"
    "• Nhắn mã cổ phiếu, ví dụ: FPT  hoặc  fpt hpg dgc (tối đa 3 mã/tin) → thẻ báo cáo như lệnh analyze\n"
    "• /canhbao — theo dõi trong phiên (cắt lỗ, vùng mua, mục tiêu)\n"
    "• /danhmuc — danh mục đang giữ (cần sổ lệnh)\n"
    "• /nhatky — nhật ký giao dịch: lợi suất, thắng/thua, điểm kỷ luật\n"
    "Mỗi mã mất khoảng 1–2 phút (tải giá, BCTC). Không phải lời khuyên đầu tư."
)


def _api(token: str, method: str, params: dict | None = None, timeout: int = 70,
         opener=urllib.request.urlopen) -> dict:
    url = API.format(token=token, method=method)
    body = json.dumps(params or {}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with opener(req, timeout=timeout) as resp:
            data = json.loads(resp.read() or b"{}")
    except OSError as e:
        raise notify.NotifyError(notify._scrub(f"Telegram: {e}", token)) from None
    if not data.get("ok", False):
        raise notify.NotifyError(notify._scrub(f"Telegram từ chối: {data.get('description')}", token))
    return data


def get_updates(token: str, offset: int | None = None, wait: int = 0, opener=urllib.request.urlopen
                ) -> list[dict]:
    params = {"timeout": wait, "allowed_updates": ["message"]}
    if offset is not None:
        params["offset"] = offset
    return _api(token, "getUpdates", params, timeout=wait + 20, opener=opener).get("result", [])


def allowed_chats() -> set[str]:
    raw = os.environ.get("TELEGRAM_CHAT_ID", "")
    return {x.strip() for x in raw.split(",") if x.strip()}


def symbols_in(text: str) -> list[str]:
    """Các mã 3 ký tự trong tin nhắn (bỏ từ thông dụng), giữ thứ tự, tối đa MAX_SYMBOLS."""
    words = re.split(r"[\s,;./]+", text.upper().strip())
    out = [w for w in words if TICKER.match(w) and w not in STOPWORDS]
    return list(dict.fromkeys(out))


def _capture(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            fn(*args, **kwargs)
        except SystemExit:
            pass
    return buf.getvalue().strip()


def analyze_text(symbol: str) -> str:
    """Thẻ báo cáo của một mã (tự tải dữ liệu như analyze), ghi khuyến nghị vào lịch sử."""
    from vnsa import analysis, market
    from vnsa.report.card import build_card, format_card

    sym = symbol.upper()
    with contextlib.redirect_stdout(io.StringIO()):
        rep = analysis.update(sym)
    if not rep.ok:
        return f"{sym}: không lấy được giá — kiểm tra lại mã (HOSE/HNX/UPCoM)."
    prof = market.profile(sym)
    head = " · ".join(x for x in (sym, prof["name"], prof["exchange"], prof["industry"]) if x)
    try:
        a = analysis.assess(sym)
    except (analysis.DataMissing, ValueError) as e:
        return f"{head}\n✗ {e}"
    try:
        analysis.record_signals([(a.rec, a.plan)])
    except Exception:  # noqa: BLE001 - lưu lịch sử lỗi không chặn trả lời
        pass
    return f"{head}\n\n{format_card(build_card(a))}"


def _monitor_text() -> str:
    from vnsa import monitor

    with contextlib.redirect_stdout(io.StringIO()):
        res = monitor.run(force=True)
    return monitor.format_alerts(res)


def _cli_text(*argv: str) -> str:
    from vnsa.cli import main

    out = _capture(main, list(argv))
    return out or "(không có kết quả)"


COMMANDS = {
    "/start": lambda: HELP, "/help": lambda: HELP,
    "/canhbao": _monitor_text, "/monitor": _monitor_text,
    "/danhmuc": lambda: _cli_text("portfolio"), "/portfolio": lambda: _cli_text("portfolio"),
    "/nhatky": lambda: _cli_text("portfolio", "report"),
}


def plan(text: str) -> tuple[str, list[str]]:
    """("command", [lệnh]) | ("symbols", [mã]) | ("help", [])."""
    t = (text or "").strip()
    first = t.split()[0].lower().split("@")[0] if t else ""
    if first in COMMANDS:
        return "command", [first]
    syms = symbols_in(t.lstrip("/"))
    return ("symbols", syms) if syms else ("help", [])


def handle(text: str, send, analyze=analyze_text) -> int:
    """Trả lời một tin nhắn qua send(text); trả về số tin đã gửi."""
    kind, items = plan(text)
    if kind == "command":
        try:
            reply = COMMANDS[items[0]]()
        except Exception as e:  # noqa: BLE001
            reply = f"Không chạy được {items[0]}: {type(e).__name__}: {e}"
        send(reply)
        return 1
    if kind == "help":
        send("Không thấy mã cổ phiếu nào trong tin nhắn.\n\n" + HELP)
        return 1
    extra = items[MAX_SYMBOLS:]
    items = items[:MAX_SYMBOLS]
    send(f"Đang phân tích {', '.join(items)} (~1–2 phút/mã)…"
         + (f" Bỏ qua {', '.join(extra)} (tối đa {MAX_SYMBOLS} mã/tin)." if extra else ""))
    n = 1
    for sym in items:
        try:
            reply = analyze(sym)
        except Exception as e:  # noqa: BLE001 - một mã lỗi không dừng các mã khác
            reply = f"{sym}: lỗi khi phân tích — {type(e).__name__}: {e}"
        send(reply)
        n += 1
    return n


def process(updates: list[dict], token: str, chats: set[str], send_fn=None,
            analyze=analyze_text) -> tuple[int, int | None]:
    """Xử lý danh sách update; trả (số tin đã trả lời, offset tiếp theo)."""
    send_fn = send_fn or notify.send_telegram
    answered, next_offset = 0, None
    for u in sorted(updates, key=lambda x: x.get("update_id", 0)):
        next_offset = u["update_id"] + 1
        msg = u.get("message") or {}
        chat = str((msg.get("chat") or {}).get("id", ""))
        text = msg.get("text")
        if not text or chat not in chats:
            continue
        handle(text, lambda t, c=chat: send_fn(t, token=token, chat_id=c), analyze)
        answered += 1
    return answered, next_offset


def run(once: bool = False, log=print, opener=urllib.request.urlopen) -> int:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chats = allowed_chats()
    if not token or not chats:
        log("Chưa cấu hình TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID.")
        return 1
    offset, total = None, 0
    if not once:
        log("Bot đang chạy — nhắn mã cổ phiếu cho bot trên Telegram. Ctrl+C để dừng.")
    while True:
        try:
            updates = get_updates(token, offset, 0 if once else 50, opener)
            n, nxt = process(updates, token, chats)
            if nxt is not None:
                offset = nxt
                get_updates(token, offset, 0, opener)       # báo Telegram đã đọc tới đây
            total += n
            if n:
                log(f"Đã trả lời {n} tin nhắn.")
        except notify.NotifyError as e:
            log(f"Lỗi Telegram: {e}")
            if once:
                return 1
            time.sleep(10)
        except KeyboardInterrupt:
            log("Đã dừng bot.")
            return 0
        if once:
            log(f"Xong: {total} tin nhắn được trả lời.")
            return 0
