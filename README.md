# VN Stock Analyzer

Tool phân tích cổ phiếu Việt Nam cho cá nhân (đủ 10 ngày của lộ trình):
tải giá (CafeF) và BCTC (vnstock) → kho DuckDB → chỉ số tài chính, định giá DCF, F-Score/Altman Z,
chỉ báo kỹ thuật, kế hoạch giao dịch, GARCH/Monte Carlo/VaR → khuyến nghị theo quy tắc, backtest →
dùng qua dòng lệnh, dashboard web hoặc hỏi Claude Desktop qua MCP server.

Mọi con số do Python tính; tham số nằm ở `config/settings.toml`. Kết quả là sàng lọc theo quy tắc,
**không phải lời khuyên đầu tư**.

Làm hằng ngày (Codespaces):

```bash
python -m vnsa.cli analyze SHS                              # MỌI mã HOSE/HNX/UPCoM: tự tải rồi phân tích
python -m vnsa.cli fetch && python -m vnsa.cli fetch-bctc   # cập nhật danh sách theo dõi
python -m vnsa.cli recommend                                # bảng khuyến nghị 5 mã
python -m vnsa.cli portfolio                                # danh mục đang giữ (sổ lệnh riêng tư)
streamlit run dashboard/app.py                              # mở dashboard (cổng 8501)
```

## Nâng cấp v2.0 (10/2026)

Theo đề xuất "Quy trình nâng cấp hệ thống v2.0" sau khi đã cân chỉnh (chỉ đưa vào điểm số những gì
kiểm định được; phần chưa kiểm định thì hiển thị để theo dõi):

| Bước | Thay đổi | Dùng thế nào |
|---|---|---|
| 1.1 | Dashboard, Claude Desktop (MCP) và dòng lệnh dùng chung kho cùng lúc: chờ khi kho bận (tối đa 60 giây), không giữ khoá trong lúc tải mạng | tự động |
| 1.2 | Kiểm tra `config/settings.toml` khi khởi động, báo lỗi tiếng Việt chỉ rõ dòng sai | `python -m vnsa.config` |
| 1.3 | Tải BCTC theo mùa báo cáo: đã có kỳ mới nhất thì không gọi vnstock nữa | `[auto] bctc_season` |
| 2.1 | Định giá theo nhóm ngành: bất động sản bỏ DCF, nhóm chu kỳ (thép, hoá chất – phân bón, dầu khí) neo P/B | `[valuation.groups]`, `[valuation.weights]` |
| 3.3 | Quy tắc thời điểm `vcp_pullback` (nền chặt + nhịp điều chỉnh) đưa vào kiểm định 10 năm — **không đạt** (giai đoạn 2022–2026 Sharpe −0,32 so với mua & giữ 0,33; xem `docs/kiem_dinh_thoi_diem.md`), trục thời điểm giữ `none` | `study-timing` |
| 4.1 | Monte Carlo/VaR dùng lợi suất kỳ vọng 0, không kéo dài đà quá khứ | `[risk] drift` |
| 5.1 | **Sổ lệnh** từng giao dịch: giá vốn bình quân, số bán được sau T+2, cắt lỗ | `portfolio buy/sell/stop/status` |
| 5.2 | Kế hoạch mua giảm rủi ro khi tương quan cao với mã đang giữ; trần 30% vốn mỗi nhóm ngành | `[portfolio]` |
| 2.3, 3.2 | Gia tốc LNST quý, độ nén giá, dòng tiền tổ chức 10 phiên — **chỉ hiển thị** trên thẻ | tự động |
| 3.1 | Cảnh báo sớm VNINDEX (phiên phân phối, biến động tăng vọt) — **tắt** tới khi kiểm định | `[regime] early` |

Không làm: đưa điểm vĩ mô vào chi phí vốn k_e (trái nguyên tắc vĩ mô chỉ vào điểm khi
`review-signals` chứng minh có ích); EPS chuẩn hoá theo ROE trung vị (4 năm BCTC ngắn hơn một chu
kỳ ngành — chờ nhánh `signals` tích luỹ thêm).

### Sổ lệnh và danh mục

```bash
python -m vnsa.cli portfolio buy FPT 500 98500 --date 2026-10-10   # giá bằng ĐỒNG; phí 0,15% tự tính
python -m vnsa.cli portfolio stop FPT 92000                         # đặt/đổi mức cắt lỗ
python -m vnsa.cli portfolio sell FPT 200 105000                    # từ chối nếu chưa qua T+2
python -m vnsa.cli portfolio                                        # danh mục: lãi/lỗ, tỷ trọng, cắt lỗ
python -m vnsa.cli portfolio list                                   # toàn bộ sổ lệnh
python -m vnsa.cli portfolio undo                                   # xoá dòng vừa nhập nhầm
python -m vnsa.cli portfolio backup                                 # sao lưu sang repo riêng tư
```

Sổ lệnh nằm ở `data/portfolio/trades.csv` và **không đưa lên Git** (repo này công khai). Khi sổ
lệnh có vị thế, `analyze`/`plan` tự giảm khối lượng gợi ý nếu mã mới đi cùng chiều mã đang giữ, chặn
tổng một nhóm ngành ở 30% vốn và chặn **tổng rủi ro mở** (lỗ thêm nếu mọi mã cùng về cắt lỗ) ở 5%
vốn. Cổ tức tiền, cổ tức/thưởng cổ phiếu được tự tính theo ngày GDKHQ (số lượng, giá vốn, mức cắt
lỗ). Claude Desktop đọc được danh mục qua công cụ `get_portfolio`.

**Sao lưu sổ lệnh** (dữ liệu không tải lại được — Codespace bị xoá là mất):

1. Tạo một repo **Private** trống trên GitHub, ví dụ `so-lenh`.
2. Tạo token: GitHub → Settings → Developer settings → Fine-grained tokens → Generate new token;
   Repository access: *Only select repositories* → `so-lenh`; Permissions → Contents: *Read and write*.
3. GitHub → Settings → Codespaces → Secrets: thêm `VNSA_BACKUP_REPO` = `ten_ban/so-lenh` và
   `VNSA_BACKUP_TOKEN` = token vừa tạo, cho phép repo `tool_chung_khoa` dùng; khởi động lại Codespace.
4. `python -m vnsa.cli portfolio backup` (từ đó mỗi lệnh buy/sell/stop tự sao lưu);
   Codespace mới: `python -m vnsa.cli portfolio restore`.

Tool từ chối sao lưu lên repo công khai hoặc lên chính repo tool.

## Nâng cấp v2.1 (10/2026)

| Hạng mục | Thay đổi | Dùng thế nào |
|---|---|---|
| Thang mua 3 lớp | 3 lệnh LO: lớp 1 thăm dò 30% (mức đỡ gần), lớp 2 chính 40% (hỗ trợ chạm nhiều lần / MA50, trung vị −7%), lớp 3 sâu 30% (trung vị −13%); xác suất khớp 20 phiên. Kiểm định 10 năm: giá vốn rẻ hơn ~4% nhưng chỉ khớp 43% vốn nên lãi trên tổng vốn **thấp hơn** vùng cũ → mặc định kế hoạch giữ vùng cũ, lớp 2–3 hiện thành "mức mua thêm nếu giá giảm sâu" | `[entry] ladder` (tắt), `ladder_show` (bật); `study-entry` |
| Sự kiện quyền trong sổ lệnh | Cổ tức tiền (sau thuế 5%) hạ mức cắt lỗ đúng bằng cổ tức; cổ phiếu thưởng cộng số lượng, giảm giá vốn | tự động |
| Phá nền + cảnh báo sớm | Quy tắc `breakout_early` kiểm định 10 năm: **không đạt** (Sharpe 0,21 so với mua & giữ 0,33) → giữ tắt | `study-timing` |
| Sao lưu sổ lệnh | Sang repo riêng tư | `portfolio backup` / `restore` |
| Nhiệt danh mục | Tổng rủi ro mở ≤ 5% vốn, lệnh mới tự giảm | `[portfolio] max_heat` |

## Nâng cấp v3.0 (10/2026) — quản trị quỹ cá nhân chủ động

| Hạng mục | Thay đổi | Dùng thế nào |
|---|---|---|
| Theo dõi trong phiên + Telegram | Giá trong phiên (bảng giá VCI) so với cắt lỗ trong sổ lệnh và vùng mua của khuyến nghị gần nhất: KHẨN khi chạm/thủng cắt lỗ (kèm số cp bán được theo T+2), nằm sàn, lỗ ≥ 10% chưa đặt cắt lỗ; CHÚ Ý khi sát cắt lỗ, chạm mục tiêu, mã theo dõi vào vùng mua, trạng thái giao dịch bất thường | `monitor`, workflow **Theo dõi trong phiên** (~9:47 và ~13:47) |
| Nhật ký giao dịch | Lợi suất phần vốn đã giải ngân (TWR) so với VNINDEX, tỷ lệ thắng, payoff, profit factor, **điểm kỷ luật 0–100** (mua trong vùng kế hoạch, đặt cắt lỗ, bán khi thủng cắt lỗ, không dời cắt lỗ xuống) | `portfolio report`, tab Danh mục |
| Chất lượng BCTC | Beneish M-Score 8 biến, phải thu khác/vốn chủ, người mua trả trước/doanh thu (BĐS), tiền ròng/vốn hoá — **chỉ cảnh báo**, độ tin cậy giảm một bậc, không đổi xếp hạng | thẻ báo cáo, `[forensic]` |
| Trạng thái giao dịch + tin | Bảng giá báo **hạn chế/tạm ngừng giao dịch** hoặc tin "diện kiểm soát, đình chỉ, huỷ niêm yết" → **chặn mua** (tối đa NẮM GIỮ). Tin tiêu cực (xử phạt, khởi tố, kiểm toán ngoại trừ, giải chấp...) → cảnh báo, giảm độ tin cậy; tin tích cực đánh dấu [+] | `[news]`, tự động |
| Xoay vòng ngành RRG | RS-Ratio / RS-Momentum theo tuần của các nhóm ngành so với VNINDEX, đuôi 8 tuần — **chỉ tham khảo** | `rrg`, tab Sàng lọc thị trường |

Không làm: quy tắc "tin xấu mà giá tăng → độ tin cậy CAO" (kho chỉ có tiêu đề ~30 ngày, chưa kiểm
định được); đưa M-Score/RRG vào điểm (chưa kiểm định trên dữ liệu Việt Nam).

```bash
python -m vnsa.cli monitor                 # theo dõi ngay (in ra màn hình)
python -m vnsa.cli monitor --send          # ... và gửi Telegram nếu có cảnh báo
python -m vnsa.cli portfolio report        # nhật ký: TWR, thắng/thua, điểm kỷ luật
python -m vnsa.cli rrg                     # xoay vòng ngành (cần chạy screen trước)
```

### Cảnh báo Telegram (cài một lần)

1. Telegram: nhắn **@BotFather** → `/newbot` → đặt tên → nhận **token** dạng `123456:ABC...`.
2. Nhắn một tin bất kỳ (ví dụ `hi`) cho bot vừa tạo, rồi mở trình duyệt:
   `https://api.telegram.org/bot<TOKEN>/getUpdates` → tìm `"chat":{"id":123456789` — đó là **chat id**.
3. GitHub → repo `tool_chung_khoa` → Settings → Secrets and variables → **Actions** → New repository
   secret: `TELEGRAM_BOT_TOKEN` = token, `TELEGRAM_CHAT_ID` = chat id. (Muốn gửi từ Codespaces thì
   thêm cả ở Settings → Codespaces → Secrets.)
4. Để workflow đọc được sổ lệnh: thêm tiếp `VNSA_BACKUP_REPO`, `VNSA_BACKUP_TOKEN` (như mục sao lưu
   sổ lệnh) vào **Actions** secrets. Không có thì workflow chỉ theo dõi vùng mua các mã `[signals] daily`.
5. Thử: Actions → **Theo dõi trong phiên** → Run workflow, tick "Gửi Telegram cả khi không có cảnh báo".

### Hỏi tool qua Telegram (v3.1)

Nhắn mã cổ phiếu cho bot (vd `FPT` hoặc `fpt hpg dgc`, tối đa 3 mã/tin) → bot trả **thẻ báo cáo**
như lệnh `analyze`. Lệnh: `/canhbao` (theo dõi trong phiên), `/danhmuc`, `/nhatky`, `/help`. Bot chỉ
trả lời chat id trong `TELEGRAM_CHAT_ID` (nhiều người: cách nhau bằng dấu phẩy).

**Trả lời nhanh 24/7 (v3.2, khuyên dùng):** lịch chạy định kỳ của GitHub Actions không tin được
(cron 10 phút thực tế chỉ chạy mỗi 3–5 giờ), nên dùng **Cloudflare Worker** miễn phí làm cầu nối:
Telegram gửi tin tới Worker → Worker báo "Đã nhận" và gọi GitHub chạy ngay → kết quả sau ~2–4 phút.

1. Token GitHub cho Worker: GitHub → Settings → Developer settings → Fine-grained tokens → Generate:
   *Only select repositories* → `tool_chung_khoa`; Permissions → **Contents: Read and write**.
2. Tạo tài khoản miễn phí ở dash.cloudflare.com → **Workers & Pages** → **Create** → **Start with
   Hello World** → đặt tên (vd `vnsa-bot`) → **Deploy** → **Edit code**: xoá hết, dán nội dung file
   `deploy/telegram-worker.js` → **Deploy**.
3. Worker → **Settings** → **Variables and Secrets** → **Add**: `TELEGRAM_BOT_TOKEN` (Secret),
   `TELEGRAM_CHAT_ID` (Text), `GITHUB_REPO` = `datkhonglac/tool_chung_khoa` (Text), `GITHUB_TOKEN`
   (Secret, token bước 1), `WEBHOOK_SECRET` (Secret, tự đặt chuỗi chữ/số, vd `vnsa_8f3k2m9q`) → Deploy.
4. Mở `https://vnsa-bot.<tên-tài-khoản>.workers.dev/setup?key=<WEBHOOK_SECRET>` → thấy "Đã bật
   webhook" là xong. Nhắn `FPT` cho bot.

Dự phòng: chạy `python -m vnsa.cli bot` trong Codespaces/máy tính (trả lời ngay, tắt là dừng; phải tắt
webhook trước bằng `https://api.telegram.org/bot<TOKEN>/deleteWebhook`, bật lại bằng link `/setup`).

Repo công khai nên nhật ký Actions ai cũng xem được: workflow chạy `monitor --public`, **không in
mã đang giữ** (chỉ gửi qua Telegram). GitHub hay chạy lịch trễ 5–30 phút, nên đây là nhắc việc chứ
không thay lệnh dừng lỗ đặt sẵn trên app công ty chứng khoán.

## Nâng cấp v4.0 (10/2026) — mô hình định lượng nâng cao

Mọi mô-đun viết bằng numpy/scipy, đọc kho DuckDB; kiểm định 45 mã × 10 năm trước khi dùng
(chi tiết: `docs/kiem_dinh_thoi_diem.md`, mục 28/10/2026).

| Mô-đun | Kiểm định | Dùng thế nào |
|---|---|---|
| Hồ sơ khối lượng (POC, vùng giá trị 70%) | Vùng mua neo theo vùng giá trị **không hơn** vùng cũ → giữ vùng cũ | dòng trên thẻ báo cáo |
| Bước nhảy Merton (VaR đuôi, xác suất phiên giảm sàn) | VaR **đánh giá thấp** lỗ so với GARCH đang dùng → không thay | lệnh `risk` (tham khảo) |
| HRP + phụ thuộc đuôi | **Đạt**: Sharpe cao hơn, sụt giảm nhỏ hơn chia đều cả hai giai đoạn | lệnh `hrp`, tab Danh mục (gợi ý tỷ trọng) |
| HMM trạng thái VNINDEX (4 trạng thái) | **Đạt sát nút**: sụt giảm −28% thay vì −40% nhưng lợi suất thấp hơn | dòng HMM trên thẻ; `[hmm] sizing = true` để dùng cho ngân sách rủi ro |
| Lan truyền biến động Diebold–Yilmaz | chưa kiểm định (không ra tín hiệu) | lệnh `spillover`, tab Sàng lọc |

Không làm "Mật độ xác suất lượng tử" (Schrödinger): không có cách ước lượng E, V(x) từ dữ liệu để
kiểm chứng; phần dùng được (mật độ thanh khoản theo giá) là hồ sơ khối lượng.

```bash
python -m vnsa.cli hrp                      # tỷ trọng HRP cho mã đang giữ (thêm mã: hrp FPT HPG)
python -m vnsa.cli spillover                # mã nào dẫn biến động nhóm [signals] daily
python -m vnsa.cli risk FPT                 # VaR lịch sử / GARCH / bước nhảy
python -m vnsa.cli study-hmm                # kiểm định lại HMM; study-var, study-hrp tương tự
```

## Phân tích một mã bất kỳ (tự lấy dữ liệu)

```bash
python -m vnsa.cli analyze HPG                # thẻ báo cáo một màn hình (xem mẫu dưới)
python -m vnsa.cli analyze SHS VCB --detail   # thêm chi tiết: chỉ báo, kế hoạch, rủi ro, BCTC, định giá
python -m vnsa.cli recommend SHS VCB --update # bảng khuyến nghị, cập nhật dữ liệu trước
```

Thẻ báo cáo (`src/vnsa/report/card.py`, cũng hiện ở dashboard và Claude Desktop):

```text
HPG — Khuyến nghị: THEO DÕI (Trung lập) · Độ tin cậy: Trung bình · Khung: trung hạn
Ngày phân tích: … · Giá hiện tại: …đ
ĐỊNH GIÁ      Giá trị nội tại (P10 / P50 / P90) · Giá mục tiêu 12 tháng → Upside · Phương pháp + trọng số
ĐIỂM SỐ       FA · TA · Quant → Tổng hợp (50% / 30% / 20%)
CƠ BẢN        ROE (so với ngành) · Tăng trưởng EPS 3 năm · D/E · CFO/LNST · F-Score · cờ đỏ
KỸ THUẬT      Xu hướng (giá/MA50/MA200), RSI · Hỗ trợ · Kháng cự
DỰ BÁO 20 PHIÊN  Dải giá P10–P50–P90 · Xác suất tăng · VaR 95% 1 ngày
KẾ HOẠCH      Vùng mua · Cắt lỗ · Chốt lời · Tỷ trọng gợi ý % NAV
```

**Khuyến nghị hai trục** (như các công ty chứng khoán): **xếp hạng** theo lợi nhuận kỳ vọng 12
tháng = upside so với giá trị hợp lý + tỷ suất cổ tức (MUA ≥ 15% · KHẢ QUAN ≥ 5% · NẮM GIỮ ≥ −5% ·
KÉM KHẢ QUAN ≥ −15% · BÁN), có rào chắn: VNINDEX xu hướng giảm thì cần thêm 5% và giảm khối lượng
một nửa, FA yếu hạ một bậc, nguy cơ tài chính tối đa NẮM GIỮ. **Thời điểm** chỉ quản trị rủi ro
("Trong vùng mua — giữ kỷ luật cắt lỗ" hoặc "Chờ giá về vùng mua"): kiểm định 10 năm × 45 mã cho
thấy không quy tắc kỹ thuật nào (kể cả điểm TA cũ) thắng mua & giữ ở cả hai giai đoạn — chi tiết và
cách đổi quy tắc ở [docs/kiem_dinh_thoi_diem.md](docs/kiem_dinh_thoi_diem.md). Khối lượng không quá 15% vốn mỗi mã và 10%
GTGD bình quân 20 phiên. Điểm FA chấm P/E, P/B so với lịch sử của chính mã (không dùng ngưỡng chung).

Độ tin cậy = số điều kiện đạt trong 6 (có BCTC, ≥ 2 phương pháp định giá, kịch bản định giá hẹp,
giá trị hợp lý không lệch quá ±60% so với giá, FA và TA cùng chiều, đủ ~2 năm giá): ≥ 5 Cao,
≥ 3 Trung bình. Điểm Quant = xác suất tăng (Monte Carlo), biến động năm, lãi/lỗ mục tiêu 1.
Ngưỡng ở `[report]`, `[recommend.quant]`; trọng số phương pháp định giá ở `[valuation.weights]`.

Với mỗi mã, tool tự kiểm tra kho và chỉ tải phần còn thiếu (tham số ở `[auto]` trong settings):
- **Sàn, ngành, loại doanh nghiệp** của mọi cổ phiếu (vnstock/VCI), tải lại sau 30 ngày. Sàn quyết
  định biên độ (HOSE ±7%, HNX ±10%, UPCoM ±15%) và bước giá (HNX/UPCoM 100 đồng).
- **Giá**: lần đầu tải 5 năm từ CafeF; lần sau chỉ tải phiên mới. Mã CafeF không có dữ liệu
  (đã gặp với SHS, ACV trên HNX/UPCoM) thì tự lấy từ vnstock/VCI và lần sau dùng lại nguồn đó. Nếu CafeF đã điều chỉnh lại giá cũ
  (cổ tức, chia tách) thì tự tải lại toàn bộ cho khớp. Trước 15h chưa tính phiên hôm nay.
- **BCTC năm và BCTC quý** (vnstock/VCI; bản miễn phí 4 quý gần nhất), tải lại sau 30 ngày. Từ 4 quý
  tool tính **TTM** (12 tháng gần nhất): P/E, P/B, ROE và giá trị hợp lý dùng TTM thay cho năm tài
  chính đã cũ. Lỗi BCTC không chặn phần kỹ thuật.
- **Hồ sơ doanh nghiệp và cổ tức** (tải lại sau 7 ngày): số cổ phiếu lưu hành hiện tại (đã tính cổ
  phiếu thưởng mới, trừ cổ phiếu quỹ), sở hữu nước ngoài / room, GTGD bình quân, lịch sử cổ tức.
- Giá lưu cả **giá chưa điều chỉnh** (giá khớp thật) để P/E, P/B lịch sử tính theo vốn hoá thật
  (giá điều chỉnh đã trừ cổ tức tiền nên làm P/E quá khứ thấp đi).

`--refresh` tải lại BCTC và danh sách sàn ngay; `--full` tải lại toàn bộ giá; `--offline` không tải gì.
**Ngân hàng, chứng khoán, bảo hiểm**: không dùng DCF-FCFF và Altman Z (thiết kế cho doanh nghiệp
thường); định giá bằng thu nhập thặng dư + P/B theo ROE, điểm FA bằng bộ chỉ tiêu riêng có chỉ số
ngành tính từ BCTC (`fa/sector.py`): ngân hàng — NIM (xấp xỉ), CIR, chi phí tín dụng, dự phòng/dư
nợ, cho vay/tiền gửi, tăng trưởng tín dụng; công ty chứng khoán — cho vay ký quỹ/vốn chủ, tự doanh
FVTPL/vốn chủ, tỷ trọng môi giới. Nguồn miễn phí **không có nợ xấu (NPL), CASA, CAR**: tool ghi rõ
là thiếu chứ không tự ước lượng. Bạn có thể **nhập tay** mỗi quý từ BCTC/thuyết minh vào
`config/bank_manual.toml` (có ví dụ trong file) — khi có, tool hiển thị và chấm thêm nhóm "Chất lượng
tài sản" (nợ xấu, bao phủ nợ xấu), CASA, CAR. Khi kho có ≥ 5 ngân hàng (đã phân tích), NIM, CIR, chi
phí tín dụng được chấm **so với trung vị các ngân hàng** thay cho ngưỡng cố định.

Trên dashboard: gõ mã vào ô **Mã bất kỳ** ở thanh bên rồi bấm **Lấy dữ liệu & phân tích**.

## Giá mua khuyến nghị (vùng mua) và giá trị hợp lý

Tool **không** lấy giá đóng cửa làm giá mua (giá ATC hay bị kéo/đạp, mua đuổi sau phiên tăng dễ kẹt
T+2). Thay vào đó (`src/vnsa/ta/entry.py`, tham số `[entry]`):

1. **Điểm neo kỹ thuật** = mức gần giá nhất (trong 2 ATR phía dưới) trong: hỗ trợ gần nhất,
   **VWAP20** (giá vốn bình quân theo khối lượng 20 phiên), SMA20.
2. **Vùng mua kỹ thuật** = [neo, neo + 0,5 ATR], không cao hơn giá hiện tại.
3. **Trần giá trị** = giá trị hợp lý × (1 − 15%) (thêm 10% nếu điểm FA < 50).
4. Vùng mua bị chặn bởi trần giá trị; giá đang quá cao thì **CHỜ GIÁ** về ≤ trần.
5. Gợi ý 2 lệnh LO (½ ở đỉnh vùng, ½ ở đáy vùng); kế hoạch tính cắt lỗ/khối lượng theo giá bình quân.

**Giá trị hợp lý** (`src/vnsa/fa/fair_value.py`) = trung vị các phương pháp ra số dương:
- Doanh nghiệp thường: DCF-FCFF, **P/E lịch sử** và **P/B lịch sử** của chính mã (giá bình quân năm
  chia EPS/BVPS theo số cổ phiếu hiện tại) — vì DCF không ra số khi FCFF âm (HPG, ACV...).
  Bội số **mục tiêu 12 tháng** không giả định quay hẳn về trung vị lịch sử (thị trường có thể đã
  định giá lại mã): mục tiêu = hiện tại + 50% × (trung vị − hiện tại), chỉnh ở
  `[valuation] multiple_reversion` (1 = quay hẳn về trung vị).
- **P/E, P/B ngành**: khi kho có ≥ 3 mã cùng ngành (đã phân tích), thêm trung vị bội số ngành làm
  phương pháp độc lập (`pe_peer`, `pb_peer`). Thẻ báo cáo in thêm **giá mục tiêu CTCK** (VCI tổng
  hợp) và độ lệch; lệch > 30% hoặc chỉ có bội số quá khứ của chính mã thì không xếp độ tin cậy "Cao".
- Cổ tức 12 tháng > 10% giá (thường do hai đợt dồn vào 12 tháng) chỉ tính bình quân 3 năm, tối đa 10%.
- F-Score "không phát hành cổ phiếu" so vốn góp công ty mẹ sau khi trừ cổ phiếu thưởng/cổ tức cổ
  phiếu (dòng tiền "nhận vốn góp" gồm cả cổ đông thiểu số góp vào công ty con).
- **Ngân hàng / chứng khoán / bảo hiểm**: ½ × trung bình của **thu nhập thặng dư** (ROE giảm dần về
  chi phí vốn chủ) và **P/B hợp lý = (ROE − g) / (k_e − g)**, cộng ½ × **P/B lịch sử**; ROE xuất phát
  là bình quân 3 năm có trọng số (năm gần nhất nặng nhất); điểm FA riêng (ROE, tăng trưởng, vốn chủ /
  tổng tài sản, định giá). Tham số `[valuation.financials]`, `[fa_score.financials]`.

- **Chất lượng lợi nhuận** (tránh mã "rẻ so với chính nó" chỉ vì lợi nhuận đã đi xuống):
  - P/B lịch sử nhân với ROE hiện tại ÷ ROE trung vị lịch sử, tối đa 1. Thị trường từng trả P/B cao
    cho ROE cao; ROE giảm thì P/B hợp lý cũng giảm.
  - LNST công ty mẹ giảm nhanh hơn 5%/năm: không kỳ vọng P/E tăng lại về mức cũ hay mức ngành, và
    DCF lấy dòng tiền năm gần nhất thay vì bình quân 3 năm (còn chứa các năm đỉnh).
  - Giá trị hợp lý tối đa bằng giá mục tiêu CTCK (VCI tổng hợp) × 1,3. Mô hình lạc quan hơn thế thì
    thẻ báo cáo ghi rõ giá trị mô hình và giá dùng để xếp hạng.
  - Tham số ở `[valuation]`: `roe_adjust`, `no_rerate_below`, `declining_base`, `consensus_cap`.

## Lịch sử khuyến nghị — đo độ chính xác của tool

Mỗi lần `recommend`, `analyze` (hoặc Claude Desktop) chạy, khuyến nghị được lưu vào bảng `signals`
và file **`data/signals.csv`** (file này đưa lên Git được — commit định kỳ để không mất khi đổi
Codespace; kho mới tự nạp lại từ file).

```bash
python -m vnsa.cli review-signals                 # chấm lại sau 5/20/60 phiên
python -m vnsa.cli review-signals --update FPT HPG --detail
```

Bảng kết quả theo từng xếp hạng (MUA … BÁN): lãi/lỗ bình quân, **vượt trội so với VNINDEX**,
tỷ lệ thắng, tỷ lệ chạm mục tiêu trước cắt lỗ, tỷ lệ khớp vùng mua. MUA tốt khi vượt trội dương.

**Tự động mỗi ngày:** workflow `.github/workflows/daily.yml` chạy 16:40 (giờ VN) các ngày giao dịch,
phân tích các mã ở `[signals] daily`, lưu khuyến nghị + BCTC năm/quý tích luỹ + kết quả chấm vào
nhánh **`signals`** (thư mục `history/`). Lịch chạy có hiệu lực khi workflow nằm ở nhánh mặc định.
Lấy về Codespaces:

```bash
python -m vnsa.cli history import --git origin/signals   # nạp lịch sử vào kho của bạn
python -m vnsa.cli review-signals
```
Cần vài tháng và vài chục mẫu mới có ý nghĩa.
Trên Claude Desktop: hỏi "phân tích mã SHS" — Claude gọi công cụ `analyze_symbol`, tự tải dữ liệu.

## Vĩ mô và ngành (chỉ tham khảo)

Lớp thứ ba bên cạnh FA và TA: số liệu vĩ mô, giá hàng hoá, tỷ giá và tin tức liên quan tới ngành
của từng mã. Thẻ báo cáo có thêm mục **VĨ MÔ & NGÀNH**; dashboard có tab **Vĩ mô**; Claude Desktop có
công cụ `macro_context`.

```bash
python -m vnsa.cli fetch-macro          # tải Yahoo, World Bank, FRED + nạp số nhập tay
python -m vnsa.cli macro                # bảng mọi chuỗi: giá trị, biến động 3 tháng, z so 5 năm
python -m vnsa.cli macro HPG FPT        # yếu tố vĩ mô của ngành từng mã + kiểm chứng độ nhạy
```

- **Nguồn tự động:**
  - Yahoo: USD/VND, USD/JPY, lợi suất Mỹ 10 năm, dầu Brent, chỉ số USD, quặng sắt, thép HRC (Mỹ,
    đại diện), bông, đường.
  - World Bank Pink Sheet: urê, DAP, đá phốt phát, than, cao su.
  - FRED: lãi suất Fed, bán lẻ quần áo Mỹ.
  - Phân tích (`analyze`) tự tải lại khi số đã cũ hơn 20 giờ.
- **Số Việt Nam nhập tay** ở `config/macro_manual.toml`: CPI, tín dụng, bán lẻ, xuất khẩu, sản xuất
  công nghiệp, PMI, khách quốc tế, đầu tư công, lãi suất liên ngân hàng, lợi suất TPCP 10 năm, GDP.
  - Khoảng 10 số mỗi tháng; cách lấy ghi ở đầu file.
  - Không có nguồn miễn phí tự động; thiếu số thì tool ghi rõ là thiếu.
- **Ánh xạ ngành → yếu tố** ở `config/macro_map.toml`.
  - Khớp tên ngành ICB cấp 4, 3, rồi 2; FPT và ACV được khai báo riêng.
  - Mỗi yếu tố có dấu (+ thuận, − nghịch), trọng số và lý do.
- **Điểm ngành** = Σ dấu × trọng số × z ÷ (2 × Σ trọng số), trong khoảng −1..+1.
  - z là biến động hiện tại so với 5 năm của chính chuỗi đó, chặn ±2.
  - ≥ +0,3 là Thuận lợi; ≤ −0,3 là Bất lợi.
  - Yếu tố có số chiếm dưới 50% trọng số thì ghi "chưa đủ số liệu".
- **Kiểm chứng độ nhạy:** hồi quy lợi nhuận tháng của mã theo VNINDEX và biến động từng yếu tố, cần
  ≥ 36 tháng.
  - Báo yếu tố "NGƯỢC dấu cấu hình" để bạn xem lại.
  - Ví dụ: quặng sắt là chi phí của HPG, nhưng 60 tháng giá cho thấy HPG tăng khi quặng tăng
    (t +3,0, cùng chu kỳ giá thép) nên cấu hình đã đổi sang dấu +.
- **Tác động lên khuyến nghị:** vĩ mô chưa vào điểm xếp hạng.
  - Ngành bất lợi rõ (≤ −0,5, độ phủ ≥ 60%) chỉ làm **độ tin cậy giảm một bậc**.
  - Điểm vĩ mô lúc khuyến nghị được lưu vào `signals`. `review-signals` chia kết quả theo
    Thuận lợi / Trung tính / Bất lợi. Chỉ khi nhóm Thuận lợi vượt VNINDEX rõ hơn nhóm Bất lợi (đủ
    mẫu) mới nên cho vĩ mô vào điểm.
- **WACC:** dùng lợi suất TPCP 10 năm nhập tay nếu kỳ đó không cũ quá 45 ngày, ngược lại dùng 3,5% ở
  `[valuation] risk_free`.
- **Tin tức:** tiêu đề tin theo mã (vnstock/VCI), chỉ để đọc; không chấm điểm cảm xúc tin.

## Sàng lọc toàn thị trường

Lọc mọi mã HOSE, HNX, UPCoM qua 5 tầng để ra nhóm ngành tiềm năng và mã tiềm năng trong từng
nhóm: tầng 0 vũ trụ → tầng 1 thanh khoản → tầng 2 phân tích từng mã → tầng 3 chấm nhóm ngành →
tầng 4 chọn mã.

```bash
python -m vnsa.cli screen                       # đủ 4 tầng: nhóm ngành tiềm năng + mã tiềm năng
python -m vnsa.cli screen --stage industry      # dừng ở tầng 3 (bảng nhóm ngành)
python -m vnsa.cli screen --stage liquidity     # chỉ tầng 1 (nhanh)
python -m vnsa.cli screen --exchanges HOSE HNX  # chỉ hai sàn
python -m vnsa.cli screen --no-update           # dùng dữ liệu đã có trong kho, không tải
python -m vnsa.cli screen --groups              # bảng ngành ICB → nhóm ngành
```

- **Tầng 1** chỉ cần giá 1 năm mỗi mã (1 lần gọi CafeF, 4 luồng song song). Giữ mã có GTGD bình quân 20 phiên ≥ 3 tỷ,
  giá ≥ 5.000đ, ≥ 200 phiên có giao dịch trong 1 năm, còn giao dịch trong 10 ngày gần nhất.
- **Tầng 2** tải đủ 5 năm giá + BCTC năm/quý + hồ sơ cho các mã qua tầng 1 (thanh khoản cao trước),
  rồi chạy đúng bộ phân tích của `analyze` cho từng mã: xếp hạng, giá trị hợp lý, FA, TA, độ tin cậy,
  vùng mua, cắt lỗ.
- **Tầng 3** chấm mỗi nhóm ngành 0–100: động lượng 30% (sức mạnh tương đối so VNINDEX), tăng trưởng
  lợi nhuận 25%, định giá 15% (lợi nhuận kỳ vọng), độ rộng 10% (tỷ lệ mã trên MA50/MA200), chất lượng
  10% (điểm FA), vĩ mô 10%. Mỗi phần là thứ hạng giữa các nhóm; phần thiếu số thì chia lại trọng số.
  Nhóm ngành lấy theo `config/macro_map.toml` (Ngân hàng, Chứng khoán, Bảo hiểm, Điện, Dầu khí, Thép,
  Bất động sản...); mã không thuộc nhóm nào xếp theo ngành ICB cấp 2.
- **Tầng 4** lấy 5 nhóm điểm cao nhất, mỗi nhóm tối đa 5 mã xếp MUA hoặc KHẢ QUAN, FA ≥ 50,
  Altman Z'' ≥ 1,1, không cờ đỏ BCTC; xếp MUA trước, rồi điểm FA 60% + TA 40%, rồi sức mạnh giá.
  Mỗi mã in giá, giá trị hợp lý, lợi nhuận kỳ vọng, vùng mua, cắt lỗ, độ tin cậy và lý do chọn.
  Thêm mục mã MUA ngoài 5 nhóm dẫn đầu. Ngưỡng ở `[screen]`.
- Xem kết quả: tab **Sàng lọc thị trường** trên dashboard; Claude Desktop có công cụ
  `screen_market` (đọc lượt gần nhất, không chạy lại).
- **Hạn mức vnstock:** bản miễn phí chỉ 20 lượt gọi/phút, 1.200/giờ (khoảng 140 mã/giờ khi tải BCTC).
  Vượt hạn mức thì thư viện tự thoát chương trình, nên tool tự chờ cho đủ dưới hạn mức. Mỗi lượt chỉ
  tải BCTC tối đa 150 phút (`[screen] deep_minutes`); mã chưa kịp tải sẽ được tải ở lượt sau.
  - Muốn nhanh gấp 3: đăng ký khoá API miễn phí ở vnstocks.com/login, rồi đặt biến môi trường
    `VNSTOCK_API_KEY` (trên GitHub: Settings → Secrets and variables → Actions → New repository
    secret, tên `VNSTOCK_API_KEY`).
- Mã chỉ có 1 năm giá trong kho: `analyze` tự tải bổ sung đủ 5 năm khi cần.
- Kết quả: bảng `screen`, `screen_industry` trong kho và `data/screen/screen_YYYY-MM-DD.csv`,
  `industry_YYYY-MM-DD.csv`.
- **Kiểm chứng:**
  - `python -m vnsa.cli review-screen`: chấm lại các lượt đã lưu sau 20 và 60 phiên — mã được chọn,
    mọi mã trong nhóm dẫn đầu, so với rổ mọi mã đủ thanh khoản và VNINDEX. Cần khoảng 10 lượt
    (10 tuần) mới bắt đầu có ý nghĩa.
  - `python -m vnsa.cli study-screen`: kiểm định quá khứ 10 năm phần dựa trên giá (động lượng nhóm
    ngành, độ rộng, sức mạnh giá), mỗi tháng chọn lại, trừ phí và thuế, so với rổ chia đều và
    VNINDEX. Phần BCTC và vĩ mô không có lịch sử miễn phí để kiểm định ngược.
- Chạy trên GitHub: Actions → **Sàng lọc thị trường** → Run workflow (tick `study` để kiểm định 10
  năm). Sau khi gộp vào nhánh chính, workflow tự chạy mỗi sáng thứ Bảy và lưu lịch sử vào nhánh
  `signals`. Trên Codespaces nạp lịch sử về: `python -m vnsa.cli history import --git origin/signals`.

## 0. Chạy trên GitHub Codespaces (khuyến nghị)

Trên trang repo GitHub: **Code → Codespaces → Create codespace on main**. Lần đầu mất vài phút để
cài Python, thư viện và extension **Claude Code** (cấu hình trong `.devcontainer/`). Khi xong:

```bash
pytest                                   # kiểm tra
python -m vnsa.cli fetch                 # tải 5 năm giá từ CafeF và nạp vào kho
python -m vnsa.cli check FPT
```

Mở Claude Code bằng biểu tượng ✻ và đăng nhập bằng tài khoản Claude. **Dừng Codespace khi không dùng**
(menu ☰ → Codespaces → Stop Current Codespace) để không tốn giờ miễn phí.
Dữ liệu (`data/raw/`, `warehouse.duckdb`) không đưa lên Git — tải lại khi tạo Codespace mới.

## 1. Cài đặt trên máy Windows (PowerShell)

Cần Python 3.11 trở lên. Mở thư mục `vn-stock-analyzer` bằng VS Code, mở Terminal và chạy:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

macOS/Linux: thay dòng 2 bằng `source .venv/bin/activate`.

VS Code sẽ gợi ý cài các extension cần thiết (Python, Jupyter, Ruff…) — bấm Install.

## 2. Chạy thử với dữ liệu mẫu

```powershell
python scripts/make_sample_data.py      # tạo 3 mã giả lập DEMO_A, DEMO_B, DEMO_C
python -m vnsa.cli load --sample        # nạp vào data/warehouse.duckdb
python -m vnsa.cli list                 # xem kho có gì
python -m vnsa.cli check DEMO_A         # kiểm tra chất lượng
python -m vnsa.cli chart DEMO_A         # tạo data/charts/DEMO_A.html, mở bằng trình duyệt
pytest                                  # chạy test
```

> Dữ liệu mẫu là **giả lập**, không phải giá thật của cổ phiếu nào. Chỉ dùng để kiểm tra pipeline.

Sau đó mở `notebooks/01_kiem_tra_du_lieu.ipynb`, chọn kernel `.venv` và chạy từng ô.

## 3. Lấy dữ liệu giá thật (việc chính của ngày 1)

### Cách A — Tự động tải từ CafeF (khuyến nghị)

Một lệnh tải 5 năm giá cho cả danh sách theo dõi trong `config/settings.toml`, không cần bấm
"Xuất Excel" từng năm:

```powershell
python -m vnsa.cli fetch                  # FPT, HPG, MWG, VNM, DGC, 5 năm
python -m vnsa.cli fetch FPT --years 3    # một mã, số năm tuỳ chọn
python -m vnsa.cli check FPT
```

Lệnh tự chia từng đoạn 1 năm (CafeF chỉ trả tối đa 1 năm mỗi lần), ghép, bỏ ngày trùng, lấy
giá điều chỉnh và khối lượng khớp lệnh, ghi `data/raw/prices/<MÃ>.csv` rồi nạp vào kho.
Đây là nguồn không chính thức: nếu CafeF đổi cấu trúc, lệnh báo lỗi và bạn dùng Cách B.

### Cách B — Excel từ CafeF (thủ công)

1. Mở trang lịch sử giao dịch của mã, ví dụ `cafef.vn/du-lieu/lich-su-giao-dich-fpt-1.chn`.
2. Mỗi lần xuất được khoảng 1 năm: chọn từng khoảng 1 năm (5 lần cho 5 năm), bấm **Xuất Excel**.
3. Tạo **thư mục** `data/raw/excel/FPT/` và chép tất cả file `.xlsx` vào đó (giữ nguyên tên file).
4. Chạy:

```powershell
python -m vnsa.cli import-excel FPT
python -m vnsa.cli check FPT
python -m vnsa.cli chart FPT
```

`import-excel` ghép các file, bỏ ngày trùng, lấy khối lượng **khớp lệnh**, đổi giá từ
nghìn đồng sang đồng, và dùng cột **Giá điều chỉnh** để điều chỉnh cả nến
(mở cửa, cao nhất, thấp nhất, đóng cửa). Kết quả ghi ra `data/raw/prices/FPT.csv` rồi nạp vào kho.
Nhiều mã một lúc: `python -m vnsa.cli import-excel FPT HPG MWG VNM DGC`.

### Cách C — file CSV có sẵn

Chép file vào `data/raw/prices/` với tên theo mã (`FPT.csv`) rồi chạy `python -m vnsa.cli load`.
Tool tự nhận tiêu đề tiếng Anh lẫn tiếng Việt, ngày `dd/mm/yyyy` hoặc `yyyy-mm-dd`,
số có dấu phẩy, giá tính bằng nghìn đồng. Nếu báo "thiếu cột", thêm tên cột của file
vào `COLUMN_ALIASES` trong `src/vnsa/data/providers/csv_provider.py`.

**Lưu ý giá điều chỉnh:** nếu `check` báo `vuot_bien_do` đúng ngày giao dịch không hưởng quyền,
file của bạn đang là giá chưa điều chỉnh — dùng Cách A/B hoặc tải lại bản đã điều chỉnh.

## 4. Báo cáo tài chính và chỉ số (ngày 2)

```bash
python -m vnsa.cli fetch-bctc          # BCTC năm hợp nhất của 5 mã (vnstock, nguồn VCI)
python -m vnsa.cli ratios FPT          # ROE, ROA, biên gộp/ròng, vay/VCSH, CFO/LNST, tăng trưởng
```

Cần vnstock (Codespace đã cài sẵn; máy khác: `pip install -U --extra-index-url
https://vnstocks.com/api/simple vnstock vnai`). Bản miễn phí của vnstock chỉ trả 4 năm gần nhất.
Cách tính từng chỉ số ghi ở đầu file `src/vnsa/fa/ratios.py` (ROE dùng vốn chủ công ty mẹ bình quân).

## 5. Định giá (ngày 3)

```bash
python -m vnsa.cli value FPT                 # P/E, P/B, WACC, DCF-FCFF, ma trận độ nhạy
python -m vnsa.cli value FPT --growth 0.10   # tự đặt tăng trưởng năm đầu g1
python -m vnsa.cli value FPT --wacc 0.11     # tự đặt WACC
```

Cần giá của mã và của VNINDEX (để tính beta; `fetch` tự tải kèm) cùng BCTC (`fetch-bctc`).
Giả định (lãi suất phi rủi ro, phần bù rủi ro, tăng trưởng dài hạn…) nằm ở mục `[valuation]` của
`config/settings.toml`; cách tính ghi ở đầu file `src/vnsa/fa/valuation.py`. Kết quả DCF rất nhạy
với giả định, hãy xem ma trận độ nhạy thay vì chỉ một con số.

## 6. Điểm FA, kỹ thuật, kế hoạch giao dịch, rủi ro (ngày 4–6)

```bash
python -m vnsa.cli fa-score FPT   # Piotroski F-Score, Altman Z, điểm FA 0–100
python -m vnsa.cli ta FPT         # SMA, RSI, MACD, Bollinger, ATR, ADX; hỗ trợ/kháng cự; điểm TA
python -m vnsa.cli plan FPT       # giá vào, cắt lỗ theo ATR, mục tiêu, khối lượng theo 1% rủi ro
python -m vnsa.cli risk FPT       # GARCH(1,1), Monte Carlo P10–P90, VaR/CVaR
python -m vnsa.cli risk FPT --days 40 --paths 20000
```

Ngưỡng chấm điểm, vốn, % rủi ro mỗi lệnh, hệ số ATR, số đường mô phỏng... nằm ở các mục
`[fa_score]`, `[ta]`, `[ta_score]`, `[plan]`, `[risk]` trong `config/settings.toml`. Công thức ghi ở
đầu từng file trong `src/vnsa/fa/scores.py`, `src/vnsa/ta/` và `src/vnsa/risk/`.
Kế hoạch giao dịch là kết quả tính theo quy tắc, không phải khuyến nghị đầu tư.

## 7. Khuyến nghị và backtest (ngày 7)

```bash
python -m vnsa.cli recommend                  # MUA / THEO DÕI / TRÁNH kèm lý do, cho danh sách theo dõi
python -m vnsa.cli recommend FPT MWG
python -m vnsa.cli backtest FPT               # điểm TA vs phá nền 55 phiên vs mua & giữ
python -m vnsa.cli backtest FPT --entry 70 --exit 40
```

Xếp hạng theo lợi nhuận kỳ vọng, thời điểm theo kỹ thuật (xem mục "Khuyến nghị hai trục" ở trên).
Backtest kiểm tra trục **thời điểm** (BCTC miễn phí quá ngắn để kiểm tra giá trị hợp lý không "nhìn
trước" — trục xếp hạng theo dõi bằng `review-signals`): khớp giá mở cửa phiên sau tín hiệu, không
mua ở giá trần / không bán khi nằm sàn, T+2, phí 0,15%/chiều, thuế bán 0,1% (`[backtest]`). So sánh
**điểm TA** (cách cũ), **phá nền 55 phiên** và quy tắc đang dùng, cùng mua & giữ.

Kiểm định nhiều mã, nhiều năm (chọn tham số 2016–2021, kiểm tra độc lập 2022 → nay):

```bash
python -m vnsa.cli study-timing                 # 45 mã [study] universe, tự tải 10 năm giá
python -m vnsa.cli study-timing FPT HPG VCB     # rổ tuỳ chọn
```

## 8. Hỏi Claude Desktop qua MCP server (ngày 8)

MCP server cho Claude Desktop gọi thẳng các phép tính của tool (12 công cụ: phân tích/cập nhật mã bất
kỳ, giá, chỉ số tài chính, định giá, điểm FA, kỹ thuật, kế hoạch, rủi ro, khuyến nghị, backtest,
danh sách mã). Claude Desktop
chạy trên **máy Windows**, nên dữ liệu phải có trong thư mục dự án trên máy đó:

```powershell
cd "C:\Users\admin\Documents\tool chứng khoan\vn-stock-analyzer\vn-stock-analyzer"
git pull
.venv\Scripts\activate
pip install -e ".[dev,mcp]"
pip install -U --extra-index-url https://vnstocks.com/api/simple vnstock vnai
python -m vnsa.cli fetch
python -m vnsa.cli fetch-bctc
```

Trong Claude Desktop: **Settings → Developer → Edit Config**, dán nội dung
`docs/claude_desktop_config.example.json` (sửa đường dẫn nếu thư mục khác), lưu rồi khởi động lại
Claude Desktop. Hỏi thử: "Dùng vn-stock-analyzer, cho tôi khuyến nghị 5 mã và giải thích".
Kiểm tra server chạy được: `python -m vnsa.mcp_server` (đứng chờ, không in gì là đúng; Ctrl+C để thoát).

## 9. Dashboard (ngày 9)

```bash
streamlit run dashboard/app.py
```

Codespaces tự mở cổng 8501 (tab **Ports** → biểu tượng 🌐 nếu trình duyệt chưa mở). Trên Windows:
`pip install -e ".[dashboard]"` rồi chạy cùng lệnh, mở http://localhost:8501. Dashboard có 6 tab:
tổng quan khuyến nghị, kỹ thuật & kế hoạch, tài chính, định giá (chỉnh g1/WACC), rủi ro (biểu đồ quạt
Monte Carlo), backtest (thanh trượt ngưỡng). Ô **Mã bất kỳ** ở thanh bên tự tải dữ liệu cho mã mới.

## 10. Kiểm tra toàn bộ trên dữ liệu thật

```bash
python scripts/smoke_test.py      # gọi mọi công cụ MCP và mở dashboard cho từng mã
```

GitHub Actions (tab **Actions → Kiểm tra và tải dữ liệu → Run workflow**) chạy toàn bộ: tải dữ liệu,
mọi lệnh phân tích, khuyến nghị, backtest và bước kiểm tra này; kết quả in trong log.

## Ý nghĩa các kết quả kiểm tra

| Mã | Mức | Ý nghĩa |
| --- | --- | --- |
| `gia_khong_duong` | LỖI | Có giá ≤ 0 |
| `high_nho_hon_open_close`, `low_lon_hon_open_close` | LỖI | Dữ liệu OHLC mâu thuẫn |
| `trung_ngay`, `chua_sap_xep` | LỖI | Trùng hoặc sai thứ tự ngày |
| `thieu_phien` | CẢNH BÁO | Khoảng trống > 10 ngày (Tết thì bình thường) |
| `vuot_bien_do` | CẢNH BÁO | Thay đổi > biên độ sàn, thường do chưa điều chỉnh quyền |
| `thanh_khoan_thap` | CẢNH BÁO | > 5% số phiên không có khối lượng |

Mã có LỖI sẽ không được nạp (dùng `--force` nếu muốn nạp vẫn).

## Cấu trúc

```text
config/settings.toml          cấu hình: đường dẫn, đơn vị giá, biên độ, danh sách mã
src/vnsa/
  config.py                   đọc và kiểm tra cấu hình (python -m vnsa.config)
  portfolio.py                sổ lệnh, vị thế (T+2, giá vốn), giới hạn theo tương quan và nhóm ngành
  journal.py                  nhật ký giao dịch: TWR, thắng/thua, điểm kỷ luật (v3)
  monitor.py, notify.py       theo dõi trong phiên, cảnh báo Telegram (v3)
  news.py                     phân loại tiêu đề tin theo từ khoá, trạng thái giao dịch chặn mua (v3)
  data/providers/base.py      DataProvider: giao diện chung cho mọi nguồn dữ liệu
  data/providers/csv_provider.py   nguồn CSV thủ công
  data/providers/cafef_provider.py tải giá tự động từ CafeF
  data/excel_import.py        ghép file Excel tải từ web
  data/cleaning.py            kiểm tra chất lượng
  data/store.py               kho DuckDB
  data/providers/vnstock_provider.py  BCTC năm, danh sách niêm yết (sàn, ngành) qua vnstock (VCI)
  data/autofetch.py           tự tải phần dữ liệu còn thiếu/cũ cho một mã bất kỳ
  data/providers/macro_provider.py số liệu vĩ mô: Yahoo, World Bank Pink Sheet, FRED
  macro/                      lớp vĩ mô: danh mục + nhập tay (catalog), biến động/z (signals),
                              điểm ngành/bối cảnh (score), độ nhạy (sensitivity), cập nhật (update)
  market.py                   quy tắc theo sàn: biên độ, bước giá; nhận diện ngân hàng/CK/bảo hiểm
  fa/ratios.py                bộ chỉ số tài chính
  fa/valuation.py             WACC, DCF-FCFF, P/E, P/B, độ nhạy
  fa/scores.py                Piotroski F-Score, Altman Z, điểm FA
  ta/indicators.py            chỉ báo kỹ thuật
  ta/levels.py, ta/score.py   hỗ trợ/kháng cự, điểm TA
  ta/plan.py                  kế hoạch giao dịch (cắt lỗ ATR, khối lượng, bước giá theo sàn)
  risk/garch.py               GARCH(1,1)
  risk/simulate.py            Monte Carlo, VaR, CVaR
  recommend/engine.py         khuyến nghị MUA / THEO DÕI / TRÁNH
  recommend/backtest.py       backtest theo điểm TA
  analysis.py                 lớp dịch vụ dùng chung cho CLI, MCP, dashboard
  mcp_server.py               MCP server cho Claude Desktop
dashboard/app.py              dashboard Streamlit
docs/                         mẫu cấu hình Claude Desktop
scripts/smoke_test.py         kiểm tra MCP + dashboard trên dữ liệu thật
  report/charts.py            biểu đồ nến
  cli.py                      dòng lệnh
scripts/make_sample_data.py   sinh dữ liệu mẫu
notebooks/                    notebook kiểm tra
tests/                        pytest
```

Thêm nguồn mới (vnstock, DNSE, SSI…) = viết một lớp con của `DataProvider`
trong `data/providers/`, không phải sửa phần còn lại.

## Hướng phát triển tiếp

- Đối chiếu BCTC với số công bố gốc; mua gói vnstock trả phí để có hơn 4 năm BCTC.
- Duyệt lại giả định định giá (`[valuation]`) và ngưỡng chấm điểm theo kinh nghiệm của bạn.
- Thêm mã vào `[watchlist]`; backtest nhiều mã cùng lúc; cảnh báo tự động khi khuyến nghị đổi.
- Kế hoạch chi tiết: `docs/KIEN_TRUC_NANG_CAP_V2.md`.
