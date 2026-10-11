# VN Stock Analyzer — ghi chú cho Claude Code

Tool phân tích và dự báo cổ phiếu Việt Nam, dùng cá nhân. Người dùng học tài chính,
mới làm quen Python/VS Code: giải thích ngắn gọn bằng tiếng Việt, đưa lệnh cụ thể.

## Môi trường
- Chính: GitHub Codespaces (Linux, Python 3.12, cấu hình ở `.devcontainer/`). Thư viện đã cài sẵn
  khi tạo Codespace; không cần `.venv`.
- Phụ: máy Windows của người dùng (PowerShell, Python 3.13, môi trường ảo `.venv`; đường dẫn có
  dấu cách và chữ có dấu — luôn đặt trong ngoặc kép).
- Cài/cập nhật: `pip install -e ".[dev]"`. Test: `pytest`.
- `data/raw/` và `*.duckdb` không đưa lên Git (xem `.gitignore`): dữ liệu thật phải tải lại trong
  từng môi trường (vnstock: `python scripts/fetch_vnstock.py FPT HPG MWG VNM DGC --years 5`).
- vnstock cài từ kho gói của nhà phát triển (PyPI đang cách ly gói này):
  `pip install -U --extra-index-url https://vnstocks.com/api/simple vnstock vnai`.

## Kiến trúc (giữ nguyên nguyên tắc)
- Mọi nguồn dữ liệu là lớp con của `DataProvider` (`src/vnsa/data/providers/base.py`).
- Phân tích chỉ đọc từ kho DuckDB `data/warehouse.duckdb` qua `Store`, không gọi nguồn trực tiếp.
- Giá lưu bằng **đồng**, đã **điều chỉnh** cổ tức/chia tách; khối lượng là **khớp lệnh**.
- Cấu hình (biên độ, danh sách mã, đường dẫn) nằm ở `config/settings.toml`, không viết cứng trong code.
- Mọi con số do Python tính; LLM chỉ diễn giải. Không bịa dữ liệu thị trường.

## Lệnh
- `python -m vnsa.cli import-excel FPT` — ghép `data/raw/excel/FPT/*.xlsx` (CafeF) → CSV → kho
- `python -m vnsa.cli load [--sample]` · `check FPT` · `list` · `chart FPT`

## Lộ trình sprint 10 ngày (đang ở ngày 1–2)
1. Dữ liệu giá ✔ (khung) → nạp đủ 5 năm cho FPT, HPG, MWG, VNM, DGC
2. BCTC + bộ chỉ số tài chính, đối chiếu số công bố
3. WACC, DCF-FCFF, P/E, P/B, ma trận độ nhạy
4. F-Score, Altman Z, điểm FA; chỉ báo TA
5. Hỗ trợ/kháng cự, điểm TA, kế hoạch giao dịch (cắt lỗ theo ATR)
6. GARCH, Monte Carlo P10–P90, VaR
7. Engine khuyến nghị + backtest đơn giản
8. MCP server cho Claude Desktop
9. Dashboard Streamlit
10. Dự phòng, sửa lỗi

Mỗi bước: viết test cho công thức tài chính, chạy `pytest` trước khi báo xong.
