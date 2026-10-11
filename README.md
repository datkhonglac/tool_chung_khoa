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
streamlit run dashboard/app.py                              # mở dashboard (cổng 8501)
```

## Phân tích một mã bất kỳ (tự lấy dữ liệu)

```bash
python -m vnsa.cli analyze SHS VCB ACV        # đầy đủ: khuyến nghị, kỹ thuật, kế hoạch, rủi ro, FA, định giá
python -m vnsa.cli analyze HPG --brief        # chỉ khuyến nghị
python -m vnsa.cli recommend SHS VCB --update # bảng khuyến nghị, cập nhật dữ liệu trước
```

Với mỗi mã, tool tự kiểm tra kho và chỉ tải phần còn thiếu (tham số ở `[auto]` trong settings):
- **Sàn, ngành, loại doanh nghiệp** của mọi cổ phiếu (vnstock/VCI), tải lại sau 30 ngày. Sàn quyết
  định biên độ (HOSE ±7%, HNX ±10%, UPCoM ±15%) và bước giá (HNX/UPCoM 100 đồng).
- **Giá**: lần đầu tải 5 năm từ CafeF; lần sau chỉ tải phiên mới. Mã CafeF không có dữ liệu
  (đã gặp với SHS, ACV trên HNX/UPCoM) thì tự lấy từ vnstock/VCI và lần sau dùng lại nguồn đó. Nếu CafeF đã điều chỉnh lại giá cũ
  (cổ tức, chia tách) thì tự tải lại toàn bộ cho khớp. Trước 15h chưa tính phiên hôm nay.
- **BCTC năm** (vnstock/VCI), tải lại sau 30 ngày. Lỗi BCTC không chặn phần kỹ thuật.

`--refresh` tải lại BCTC và danh sách sàn ngay; `--full` tải lại toàn bộ giá; `--offline` không tải gì.
**Ngân hàng, chứng khoán, bảo hiểm**: DCF-FCFF, Altman Z và điểm FA không áp dụng (thiết kế cho doanh
nghiệp thường), tool chỉ dùng kỹ thuật cùng P/E, P/B và ghi rõ lý do.

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
- **Ngân hàng / chứng khoán / bảo hiểm**: **thu nhập thặng dư** (ROE giảm dần về chi phí vốn chủ),
  **P/B hợp lý = (ROE − g) / (k_e − g)**, P/B lịch sử; điểm FA riêng (ROE, tăng trưởng, vốn chủ /
  tổng tài sản, định giá). Tham số `[valuation.financials]`, `[fa_score.financials]`.

## Lịch sử khuyến nghị — đo độ chính xác của tool

Mỗi lần `recommend`, `analyze` (hoặc Claude Desktop) chạy, khuyến nghị được lưu vào bảng `signals`
và file **`data/signals.csv`** (file này đưa lên Git được — commit định kỳ để không mất khi đổi
Codespace; kho mới tự nạp lại từ file).

```bash
python -m vnsa.cli review-signals                 # chấm lại sau 5/20/60 phiên
python -m vnsa.cli review-signals --update FPT HPG --detail
```

Bảng kết quả theo từng loại MUA/THEO DÕI/TRÁNH: lãi/lỗ bình quân, **vượt trội so với VNINDEX**,
tỷ lệ thắng, tỷ lệ chạm mục tiêu trước cắt lỗ, tỷ lệ khớp vùng mua. MUA tốt khi vượt trội dương.
Cần vài tháng và vài chục mẫu mới có ý nghĩa.
Trên Claude Desktop: hỏi "phân tích mã SHS" — Claude gọi công cụ `analyze_symbol`, tự tải dữ liệu.

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
python -m vnsa.cli backtest FPT               # chiến lược theo điểm TA so với mua và nắm giữ
python -m vnsa.cli backtest FPT --entry 70 --exit 40
```

Điểm tổng = 60% điểm FA + 40% điểm TA. MUA cần điểm tổng ≥ 65, điểm TA ≥ 45 và giá không cao hơn
giá trị DCF quá 10%; TRÁNH khi điểm tổng < 40 hoặc Altman Z'' ở vùng nguy cơ (mục `[recommend]`).
Backtest chỉ dùng điểm TA (BCTC miễn phí quá ngắn để kiểm tra FA không "nhìn trước"); lệnh khớp ở
giá mở cửa phiên sau tín hiệu, có cắt lỗ ATR, T+2, phí 0,15%/chiều và thuế bán 0,1% (`[backtest]`).

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
  config.py                   đọc cấu hình
  data/providers/base.py      DataProvider: giao diện chung cho mọi nguồn dữ liệu
  data/providers/csv_provider.py   nguồn CSV thủ công
  data/providers/cafef_provider.py tải giá tự động từ CafeF
  data/excel_import.py        ghép file Excel tải từ web
  data/cleaning.py            kiểm tra chất lượng
  data/store.py               kho DuckDB
  data/providers/vnstock_provider.py  BCTC năm, danh sách niêm yết (sàn, ngành) qua vnstock (VCI)
  data/autofetch.py           tự tải phần dữ liệu còn thiếu/cũ cho một mã bất kỳ
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
