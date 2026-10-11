# VN Stock Analyzer — Ngày 1: tầng dữ liệu giá

Tool phân tích cổ phiếu Việt Nam cho cá nhân. Bản này mới có **tầng dữ liệu**:
đọc file giá CSV → kiểm tra chất lượng → lưu vào kho DuckDB → vẽ biểu đồ nến.
Các phần FA, TA, Quant, Claude sẽ thêm vào các ngày sau, đều đọc từ cùng một kho này.

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
  data/providers/vnstock_provider.py  BCTC năm qua vnstock (VCI)
  fa/ratios.py                bộ chỉ số tài chính
  report/charts.py            biểu đồ nến
  cli.py                      dòng lệnh
scripts/make_sample_data.py   sinh dữ liệu mẫu
notebooks/                    notebook kiểm tra
tests/                        pytest
```

Thêm nguồn mới (vnstock, DNSE, SSI…) = viết một lớp con của `DataProvider`
trong `data/providers/`, không phải sửa phần còn lại.

## Việc tiếp theo (ngày 3)

WACC, định giá DCF-FCFF, P/E, P/B và ma trận độ nhạy.
