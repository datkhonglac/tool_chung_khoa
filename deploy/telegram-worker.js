// Cloudflare Worker nhận tin Telegram (webhook) rồi gọi GitHub Actions chạy NGAY bot (v3.2).
// Dán toàn bộ file này vào Worker (Cloudflare → Workers & Pages → Create → Hello World → Edit code).
// Biến cần đặt ở Worker → Settings → Variables and Secrets (KHÔNG ghi vào file này):
//   TELEGRAM_BOT_TOKEN  (Secret)  token bot từ @BotFather
//   TELEGRAM_CHAT_ID    (Text)    chat id được phép hỏi, nhiều người cách nhau dấu phẩy
//   GITHUB_REPO         (Text)    datkhonglac/tool_chung_khoa
//   GITHUB_TOKEN        (Secret)  token GitHub fine-grained, chỉ repo trên, quyền Contents: Read and write
//   WEBHOOK_SECRET      (Secret)  chuỗi ngẫu nhiên tự đặt (chỉ chữ, số, _ -), vd vnsa_8f3k2m9q
// Bật webhook: mở https://<tên-worker>.<tài-khoản>.workers.dev/setup?key=<WEBHOOK_SECRET>

async function tg(env, method, body) {
  const r = await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/${method}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return r.json();
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Bật webhook một lần: Telegram sẽ gửi mọi tin nhắn tới chính Worker này
    if (request.method === "GET" && url.pathname === "/setup") {
      if (url.searchParams.get("key") !== env.WEBHOOK_SECRET) {
        return new Response("Sai key", { status: 403 });
      }
      const res = await tg(env, "setWebhook", {
        url: `${url.origin}/telegram`,
        secret_token: env.WEBHOOK_SECRET,
        allowed_updates: ["message"],
        drop_pending_updates: true,
      });
      return new Response(res.ok ? "Đã bật webhook — nhắn mã cổ phiếu cho bot để thử."
                                 : `Lỗi: ${res.description}`,
                          { headers: { "Content-Type": "text/plain; charset=utf-8" } });
    }

    if (request.method !== "POST" || url.pathname !== "/telegram") {
      return new Response("VNSA Telegram worker");
    }
    // Chỉ nhận yêu cầu thật từ Telegram (mã bí mật đặt lúc setWebhook)
    if (request.headers.get("X-Telegram-Bot-Api-Secret-Token") !== env.WEBHOOK_SECRET) {
      return new Response("forbidden", { status: 403 });
    }
    const update = await request.json();
    const msg = update.message || {};
    const chat = String((msg.chat || {}).id || "");
    const allowed = (env.TELEGRAM_CHAT_ID || "").split(",").map((s) => s.trim()).filter(Boolean);
    if (!msg.text || !allowed.includes(chat)) {
      return new Response("ok"); // người lạ / không phải chữ: bỏ qua, không tốn lượt chạy GitHub
    }

    const r = await fetch(`https://api.github.com/repos/${env.GITHUB_REPO}/dispatches`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${env.GITHUB_TOKEN}`,
        Accept: "application/vnd.github+json",
        "User-Agent": "vnsa-telegram-worker",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ event_type: "telegram", client_payload: { update } }),
    });
    await tg(env, "sendMessage", {
      chat_id: chat,
      text: r.ok
        ? "Đã nhận — máy chủ GitHub đang chạy, kết quả sau khoảng 2–4 phút."
        : `Không gọi được GitHub (HTTP ${r.status}). Kiểm tra GITHUB_TOKEN / GITHUB_REPO trong Worker.`,
    });
    return new Response("ok");
  },
};
