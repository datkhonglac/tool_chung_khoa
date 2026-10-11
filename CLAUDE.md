# VN Stock Analyzer — ghi chú cho Claude Code

Tool phân tích và dự báo cổ phiếu Việt Nam, dùng cá nhân. Người dùng học tài chính,
mới làm quen Python/VS Code: giải thích ngắn gọn bằng tiếng Việt, đưa lệnh cụ thể.

## Môi trường
- Chính: GitHub Codespaces (Linux, Python 3.12, cấu hình ở `.devcontainer/`). Thư viện đã cài sẵn
  khi tạo Codespace; không cần `.venv`.
- Phụ: máy Windows của người dùng (PowerShell, Python 3.13, môi trường ảo `.venv`; đường dẫn có
  dấu cách và chữ có dấu — luôn đặt trong ngoặc kép).
- Cài/cập nhật: `pip install -e ".[dev,mcp,dashboard]"`. Test: `pytest`.
- `data/raw/` và `*.duckdb` không đưa lên Git (xem `.gitignore`): dữ liệu thật phải tải lại trong
  từng môi trường (`python -m vnsa.cli fetch`, hoặc vnstock: `python scripts/fetch_vnstock.py FPT HPG MWG VNM DGC --years 5`).
- vnstock cài từ kho gói của nhà phát triển (PyPI đang cách ly gói này):
  `pip install -U --extra-index-url https://vnstocks.com/api/simple vnstock vnai`.

## Kiến trúc (giữ nguyên nguyên tắc)
- Mọi nguồn dữ liệu là lớp con của `DataProvider` (`src/vnsa/data/providers/base.py`).
- Phân tích chỉ đọc từ kho DuckDB `data/warehouse.duckdb` qua `Store`, không gọi nguồn trực tiếp.
- Giá lưu bằng **đồng**, đã **điều chỉnh** cổ tức/chia tách; khối lượng là **khớp lệnh**.
- Cấu hình (biên độ, danh sách mã, đường dẫn) nằm ở `config/settings.toml`, không viết cứng trong code.
- Mọi con số do Python tính; LLM chỉ diễn giải. Không bịa dữ liệu thị trường.

## Lệnh
- `python -m vnsa.cli analyze MÃ... [--brief] [--refresh] [--full] [--offline]` — **mã bất kỳ** HOSE/HNX/
  UPCoM: `data/autofetch.py` tự tải phần thiếu/cũ (sàn+ngành và BCTC qua vnstock, giá CafeF tải tăng dần — CafeF
  không trả một số mã HNX/UPCoM nên tự dự phòng sang vnstock `Quote` (VCI),
  tự tải lại toàn bộ khi CafeF điều chỉnh giá cũ) rồi in toàn bộ phân tích. Tham số `[auto]`.
  Biên độ/bước giá theo sàn qua `src/vnsa/market.py` (bảng `companies`; chưa biết sàn → HOSE), không
  dùng thẳng `band_hose`. Ngân hàng/CK/bảo hiểm (`model` ≠ general): `analysis.fa/valuation` báo
  ValueError, khuyến nghị chỉ dùng TA + P/E, P/B. MCP: `analyze_symbol`, `update_symbol` (chuyển
  stdout sang stderr khi tải). Dashboard: ô "Mã bất kỳ" ở thanh bên.
- `python -m vnsa.cli fetch [MÃ...] [--years 5]` — tải giá từ CafeF (API JSON không chính thức,
  ≤1 năm mỗi lần gọi, ngày dạng tháng/ngày/năm) → `data/raw/prices/` → kho. Cách chính để lấy giá.
- `python -m vnsa.cli fetch-bctc [MÃ...]` — BCTC năm hợp nhất qua vnstock nguồn VCI → `data/raw/bctc/`
  và bảng `fundamentals` trong kho. Bản miễn phí chỉ 4 năm gần nhất. KHÔNG dùng nguồn KBS (nhãn năm
  bị đảo ngược). `python -m vnsa.cli ratios FPT` — bộ chỉ số (`src/vnsa/fa/ratios.py`, quy ước ở đầu file).
- `python -m vnsa.cli value FPT [--growth 0.1] [--wacc 0.11]` — WACC (beta tuần so với VNINDEX), DCF-FCFF
  hai giai đoạn (g1 giảm dần về g), P/E, P/B, ma trận độ nhạy. Giả định ở `[valuation]` trong settings.
- `fa-score FPT` (F-Score, Altman Z, điểm FA) · `ta FPT` (chỉ báo, hỗ trợ/kháng cự, điểm TA) ·
  `plan FPT` (cắt lỗ ATR, mục tiêu, khối lượng theo % rủi ro, bước giá HOSE) · `risk FPT [--days N]`
  (GARCH(1,1) tự viết bằng scipy, Monte Carlo lọc lịch sử chặn biên độ ±7%, VaR/CVaR).
- `recommend [MÃ...]` (60% FA + 40% TA → MUA/THEO DÕI/TRÁNH) · `backtest FPT` (chỉ điểm TA, khớp mở
  cửa phiên sau, T+2, phí + thuế). Mọi phép tính dùng chung qua `src/vnsa/analysis.py`.
- MCP server: `python -m vnsa.mcp_server` (extra `[mcp]`, chạy được mcp 1.x và 2.x); cấu hình mẫu cho
  Claude Desktop ở `docs/`. Dashboard: `streamlit run dashboard/app.py` (extra `[dashboard]`, cổng 8501).
- `python scripts/smoke_test.py` gọi mọi công cụ MCP + dashboard trên kho thật. Test dùng fixture
  `warehouse` (tests/conftest.py) và biến môi trường `VNSA_WAREHOUSE` để không đụng kho thật.
- `python -m vnsa.cli import-excel FPT` — ghép `data/raw/excel/FPT/*.xlsx` (CafeF) → CSV → kho
- `python -m vnsa.cli load [--sample]` · `check FPT` · `list` · `chart FPT`

## Lộ trình sprint 10 ngày (đã xong cả 10 ngày)
1. Dữ liệu giá ✔ — 5 năm cho FPT, HPG, MWG, VNM, DGC (lệnh `fetch`)
2. BCTC + bộ chỉ số tài chính ✔ (4 năm 2022–2025; còn đối chiếu thêm với BCTC gốc)
3. WACC, DCF-FCFF, P/E, P/B, ma trận độ nhạy ✔ (lệnh `value`; giả định cần người dùng duyệt)
4. F-Score, Altman Z, điểm FA; chỉ báo TA ✔ (`fa-score`, `ta`)
5. Hỗ trợ/kháng cự, điểm TA, kế hoạch giao dịch (cắt lỗ theo ATR) ✔ (`ta`, `plan`)
6. GARCH, Monte Carlo P10–P90, VaR ✔ (`risk`)
7. Engine khuyến nghị + backtest đơn giản ✔ (`recommend`, `backtest`)
8. MCP server cho Claude Desktop ✔ (`src/vnsa/mcp_server.py`)
9. Dashboard Streamlit ✔ (`dashboard/app.py`)
10. Dự phòng, sửa lỗi ✔ (`scripts/smoke_test.py`, chạy thật trên GitHub Actions)

Mỗi bước: viết test cho công thức tài chính, chạy `pytest` trước khi báo xong.
Chạy thật trên máy chủ GitHub: workflow `.github/workflows/data.yml` (Actions → Run workflow; tuỳ chọn
bỏ qua bước tải giá). Kết quả in trong log, file CSV/kho ở mục Artifacts.
