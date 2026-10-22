"""Gửi cảnh báo qua Telegram (nâng cấp v3).

Cấu hình bằng biến môi trường — đặt ở GitHub → Settings → Secrets and variables → Actions (và
Codespaces → Secrets nếu muốn gửi từ Codespaces), KHÔNG ghi vào repo:
  TELEGRAM_BOT_TOKEN  token bot do @BotFather cấp (dạng 123456:ABC...)
  TELEGRAM_CHAT_ID    mã cuộc trò chuyện của bạn với bot (xem README, mục Cảnh báo Telegram)
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

LIMIT = 4000          # Telegram tối đa 4096 ký tự / tin


class NotifyError(RuntimeError):
    pass


def configured() -> bool:
    return bool(os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
                and os.environ.get("TELEGRAM_CHAT_ID", "").strip())


def split(text: str, limit: int = LIMIT) -> list[str]:
    """Cắt tin dài theo dòng, mỗi phần ≤ limit ký tự."""
    parts, cur = [], ""
    for line in text.splitlines():
        while len(line) > limit:
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(line[:limit])
            line = line[limit:]
        if len(cur) + len(line) + 1 > limit:
            parts.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}" if cur else line
    if cur:
        parts.append(cur)
    return parts


def _scrub(msg: str, token: str) -> str:
    msg = msg.replace(token, "***") if token else msg
    return re.sub(r"bot\d+:[\w-]+", "bot***", msg)


def send_telegram(text: str, token: str | None = None, chat_id: str | None = None,
                  opener=urllib.request.urlopen) -> int:
    """Gửi `text` (văn bản thuần, tự cắt nếu dài); trả về số tin đã gửi. Lỗi -> NotifyError
    (không bao giờ in token)."""
    token = (token or os.environ.get("TELEGRAM_BOT_TOKEN", "")).strip()
    chat_id = (chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")).strip()
    if not token or not chat_id:
        raise NotifyError("chưa cấu hình TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    sent = 0
    for part in split(text):
        body = json.dumps({"chat_id": chat_id, "text": part,
                           "disable_web_page_preview": True}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with opener(req, timeout=20) as resp:
                data = json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="ignore")[:200] if hasattr(e, "read") else ""
            raise NotifyError(_scrub(f"Telegram HTTP {e.code}: {detail}", token)) from None
        except OSError as e:
            raise NotifyError(_scrub(f"không gửi được Telegram: {e}", token)) from None
        if not data.get("ok", False):
            raise NotifyError(_scrub(f"Telegram từ chối: {data.get('description', data)}", token))
        sent += 1
    return sent
