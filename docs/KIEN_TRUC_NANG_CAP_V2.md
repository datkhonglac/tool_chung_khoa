# Kiến trúc nâng cấp v2: VN Stock Analyzer

> Bản v1 đã làm xong lộ trình 10 ngày. Tài liệu này mô tả **những gì cần nâng cấp** để tool
> dùng cá nhân được lâu dài, tin được và tiện dùng hằng ngày. Tool vẫn chạy trên Codespaces hoặc
> máy riêng, **không đưa lên web công khai**.
>
> Ngày viết: 11/10/2026. Tài liệu dựa trên mã nguồn ở commit `14f433d`.

---

## 1. Hiện trạng v1: điểm mạnh và chỗ còn thiếu

### Điểm mạnh (giữ nguyên)
- Kiến trúc đã rõ ràng: `DataProvider` → kho DuckDB (`Store`) → `analysis.py` → CLI, MCP và dashboard.
  Ba giao diện dùng chung một lớp dịch vụ, nên cùng một con số ở mọi nơi.
- Tham số nằm ở `config/settings.toml`. Mỗi công thức tài chính có test riêng.
- Backtest không nhìn trước dữ liệu tương lai, đã tính T+2, phí và thuế.

### Chỗ còn thiếu (đã kiểm tra trong code)

| # | Vấn đề | Ở đâu | Hậu quả |
|---|--------|-------|---------|
| 1 | Biên độ luôn lấy theo HOSE (`band_hose`), kể cả khi mã niêm yết ở HNX/UPCoM | `cli.py` (lệnh `check`, `risk`, `chart`) | Mã HNX/UPCoM bị báo lỗi sai, còn Monte Carlo chặn biên sai |
| 2 | Không có bảng thông tin doanh nghiệp (sàn, ngành, số cổ phiếu lưu hành) | `store.py` chỉ có `prices` và `fundamentals` | Không so được với ngành, không lọc được theo sàn |
| 3 | Chỉ có BCTC **năm**, tối đa 4 năm, và không lưu ngày công bố | `vnstock_provider.py`, bảng `fundamentals` | Số liệu chậm tới 12 tháng, không có TTM, không backtest được phần FA |
| 4 | Mô hình FA dùng chung cho mọi loại doanh nghiệp | `fa/ratios.py`, `fa/valuation.py`, `fa/scores.py` | Ngân hàng, chứng khoán, bảo hiểm cho kết quả vô nghĩa (DCF-FCFF và Altman Z không áp dụng được) |
| 5 | `fetch` mỗi lần đều xoá rồi ghi lại toàn bộ giá của mã | `Store.write_prices` | Chậm khi có nhiều mã; lỗi mạng giữa chừng phải tải lại từ đầu |
| 6 | Chỉ có 5 mã, phải gõ từng mã | `[watchlist]` | Chưa sàng lọc được cả thị trường (VN30, VN100) |
| 7 | Không có danh mục thật: chưa ghi lệnh mua bán, lãi lỗ, tiền mặt | — | Tool phân tích từng mã nhưng không biết bạn đang cầm gì |
| 8 | Khuyến nghị không được lưu lại | `recommend/engine.py` | Không đánh giá được sau 3–6 tháng quy tắc có đúng không |
| 9 | Backtest chỉ chạy một mã với toàn bộ vốn | `recommend/backtest.py` | Chưa kiểm tra được chiến lược trên rổ nhiều mã; dễ chọn tham số "đẹp" cho riêng một mã |
| 10 | Chưa có bộ lọc trạng thái thị trường | `recommend/engine.py` | Vẫn báo MUA khi VNINDEX đang giảm mạnh |
| 11 | Kho không đưa lên Git: mỗi Codespace mới phải tải lại từ đầu | `.gitignore` | Mất thời gian; lịch sử khuyến nghị và nhật ký lệnh sẽ mất theo |
| 12 | `cli.py` dài 648 dòng; chưa chạy `ruff` trong CI; kho chưa có phiên bản schema | — | Khó sửa tiếp khi thêm nhiều lệnh |

---

## 2. Kiến trúc đích v2

```
                ┌──────────────────────── Nguồn ─────────────────────────┐
                │ CafeF (giá) · vnstock/VCI (BCTC năm+quý, hồ sơ DN,     │
                │ sự kiện quyền) · CSV/Excel thủ công (dự phòng)         │
                └───────────────┬────────────────────────────────────────┘
                                │  DataProvider (+ provider dự phòng)
                                ▼
┌─────────────────────── Tầng dữ liệu: data/ ─────────────────────────────┐
│ ingest.py   : tải tăng dần, thử lại, ghi nhật ký tải (fetch_log)        │
│ quality.py  : kiểm tra độ mới, khoảng trống, nhảy giá vượt biên độ     │
│ store.py    : DuckDB + migrations/ (schema_version)                     │
│   prices · companies · fundamentals (năm) · fundamentals_q (quý)        │
│   corporate_actions · fetch_log                                         │
└───────────────┬─────────────────────────────────────────────────────────┘
                ▼
┌─────────────────────── Tầng phân tích (thuần, có test) ─────────────────┐
│ fa/   ratios · ttm · valuation (DCF | RI cho ngân hàng) · scores · sector│
│ ta/   indicators · levels · score · regime (VNINDEX)                    │
│ risk/ garch · simulate · portfolio_risk (tương quan, VaR danh mục)      │
│ recommend/ engine · screener · backtest (1 mã) · portfolio_backtest     │
│ portfolio/ ledger (lệnh) · positions · pnl · rules (T+2, lô, bước giá)  │
└───────────────┬─────────────────────────────────────────────────────────┘
                ▼
          analysis.py  (lớp dịch vụ duy nhất, như v1)
                │
   ┌────────────┼──────────────┬──────────────┬────────────────┐
   ▼            ▼              ▼              ▼                ▼
 cli/        dashboard/     mcp_server     report/          jobs/
 (tách       (thêm trang    (thêm công cụ  (báo cáo tuần   (chạy hằng ngày:
 theo nhóm)  danh mục,      danh mục,      Markdown/HTML)  tải → chấm điểm →
             sàng lọc)      lịch sử)                       lưu → cảnh báo)
                │
                ▼
 Lưu bền: kho + nhật ký lệnh sao lưu sang một repo/Release RIÊNG TƯ
```

**Nguyên tắc v1 vẫn giữ nguyên.** Phân tích chỉ đọc từ `Store`. Giá tính bằng đồng, đã điều chỉnh.
Tham số ở `settings.toml`. Mọi con số do Python tính, LLM chỉ diễn giải.

**Thêm 3 nguyên tắc mới:**
1. **Biết thời điểm (point-in-time):** mỗi dòng BCTC lưu cả `publish_date`. Phân tích "tại ngày D"
   chỉ được dùng số liệu đã công bố trước D.
2. **Theo loại doanh nghiệp:** mỗi mã có một `model` (`general` | `bank` | `securities` |
   `insurance`). Hàm FA nào không áp dụng được cho loại đó thì trả về "không áp dụng" thay vì một
   con số sai.
3. **Có nhật ký:** mỗi lần tải dữ liệu, mỗi khuyến nghị và mỗi lệnh đều ghi vào kho kèm thời điểm,
   để xem lại và đo độ chính xác sau này.

---

## 3. Mô hình dữ liệu mới (DuckDB)

| Bảng | Cột chính | Ghi chú |
|------|-----------|---------|
| `schema_version` | `version`, `applied_at` | Migration đánh số `src/vnsa/data/migrations/001_*.sql`… thay cho `ALTER TABLE` rải rác |
| `companies` | `symbol` PK, `name`, `exchange` (HOSE/HNX/UPCOM), `industry_icb`, `model`, `shares_out`, `updated_at` | Lấy biên độ, bước giá và nhóm ngành theo từng mã |
| `fundamentals_q` | `symbol`, `year`, `quarter`, FIELDS…, `publish_date`, `source` | PK (`symbol`, `year`, `quarter`). Dùng để tính TTM |
| `fundamentals` | thêm `publish_date` | Giữ nguyên bảng năm; ước lượng ngày công bố nếu nguồn không có (năm + 90 ngày, ghi trong config) |
| `bank_fundamentals` | `symbol`, `year`, `quarter`, NII, NIM, NPL, LDR, CASA, CIR, chi phí dự phòng, vốn chủ… | Khoản mục riêng của ngân hàng, không ép vào FIELDS chung |
| `corporate_actions` | `symbol`, `ex_date`, `type` (cổ tức tiền/cp, phát hành), `ratio`, `cash` | Dùng để điều chỉnh EPS và kiểm tra các phiên giá nhảy |
| `fetch_log` | `run_id`, `symbol`, `kind`, `start`, `end`, `rows`, `status`, `error`, `at` | Biết lần tải nào lỗi và dữ liệu cũ đến mức nào |
| `signals` | `as_of`, `symbol`, `fa`, `ta`, `total`, `action`, `price`, `fair_value`, `stop`, `target`, `params_hash` | Lưu khuyến nghị mỗi ngày, sau này đo tỷ lệ đúng |
| `transactions` | `id`, `date`, `symbol`, `side`, `qty`, `price`, `fee`, `tax`, `note` | Nhật ký lệnh **thật** do bạn nhập |
| `cash_flows` | `date`, `amount`, `note` | Nộp/rút tiền, cổ tức tiền mặt nhận về |

`Store.write_prices` đổi thành **upsert theo khoảng ngày**: chỉ thay các ngày được tải lại, không
xoá cả mã. Riêng khi phát hiện sự kiện quyền mới thì tải lại toàn bộ chuỗi đã điều chỉnh.

---

## 4. Các hạng mục nâng cấp (xếp theo ưu tiên)

### Giai đoạn A: Nền móng dữ liệu (làm trước, các phần sau phụ thuộc vào đây)

**A1. Bảng `companies` và quy tắc theo sàn**
- `fetch-profile [MÃ...]` lấy sàn, ngành ICB và số cổ phiếu lưu hành qua vnstock.
- Thêm hàm `market.band(symbol)` và `market.tick(symbol, price)`. Bước giá HNX/UPCoM khác HOSE
  (100 đồng cho mọi mức giá), nên chuyển `[plan].ticks` thành `[market.ticks.HOSE]`,
  `[market.ticks.HNX]`, `[market.ticks.UPCOM]`.
- Sửa mọi chỗ đang dùng `band_hose` (`cli.py` dòng 45, 419, 521) cho gọi `market.band(symbol)`.
- *Test:* mã HNX bị chặn ±10%; bước giá UPCoM đúng.

**A2. Migration và phiên bản schema**
- Thư mục `data/migrations/` gồm các file SQL đánh số. `Store.__init__` chạy những file chưa áp
  dụng. Kho tạo từ v1 nâng cấp lên tự động.
- *Test:* tạo kho schema v1 → mở bằng v2 → đủ bảng mới, dữ liệu cũ còn nguyên.

**A3. Tải tăng dần và nhật ký tải**
- `fetch` chỉ tải từ ngày cuối cùng trong kho trừ 5 phiên (để bắt các phiên CafeF sửa lại). Có cờ
  `--full` để tải lại toàn bộ.
- Thử lại 3 lần, chờ tăng dần. Lỗi của một mã không làm dừng cả lượt tải. Mọi kết quả ghi vào `fetch_log`.
- Lệnh `status` in bảng: mã · phiên cuối · số ngày cũ · lần tải lỗi gần nhất.
- *Test:* giả lập provider lỗi ở mã thứ 2 → mã 1 và 3 vẫn vào kho, `fetch_log` có dòng lỗi.

**A4. BCTC quý, TTM và ngày công bố**
- `fetch-bctc --period quarter` ghi vào `fundamentals_q`, kèm `publish_date`. Nếu nguồn không có
  ngày công bố thì dùng hạn nộp theo quy định (quý: +30 ngày; công ty mẹ có công ty con: +45
  ngày; năm kiểm toán: +90 ngày). Các khoảng này đặt trong `[fundamentals.publish_lag_days]`.
- `fa/ttm.py`: cộng 4 quý gần nhất cho các khoản dòng chảy (doanh thu, LNST, CFO, capex). Khoản
  tồn (tài sản, vốn chủ) lấy quý cuối. Hàm nhận tham số `as_of` để tính đúng tại một ngày quá khứ.
- P/E, ROE và điểm FA chuyển sang dùng TTM; nếu thiếu quý thì quay về số năm.
- *Test:* 4 quý cộng lại khớp số năm (sai số < 0,5%); `as_of` trước ngày công bố quý 3 thì
  không được thấy số quý 3.

**A5. Sự kiện quyền**
- `corporate_actions` lấy từ vnstock (lịch sự kiện). `cleaning.py` dùng bảng này để giải thích
  các phiên giá nhảy thay vì chỉ cảnh báo chung chung.
- Tính EPS điều chỉnh: chia LNST cty mẹ cho số cổ phiếu quy đổi theo hệ số chia tách. Cách này
  sửa được ghi chú "EPS không điều chỉnh" trong `ratios.py`.

### Giai đoạn B: Phân tích đúng hơn

**B1. Mô hình riêng cho ngân hàng và chứng khoán**
- `fa/models/bank.py`:
  - Chỉ số: NIM, NPL, chi phí tín dụng, CIR, CASA, LDR, ROE, ROA.
  - Định giá: **thu nhập thặng dư** (Residual Income), V = B₀ + Σ (ROE − ke)·B_{t−1}/(1+ke)^t,
    cùng P/B mục tiêu = (ROE − g)/(ke − g).
  - Điểm FA dùng trọng số và ngưỡng riêng trong `[fa_score.bank]`.
  - Bỏ Altman Z và DCF-FCFF.
- `fa/models/securities.py`: P/B kèm ROE, tỷ trọng margin/vốn chủ, tỷ trọng tự doanh.
- `analysis.fa()` chọn mô hình theo `companies.model`.
- *Test:* trên số liệu ngân hàng tự dựng, RI ra đúng P/B mục tiêu theo công thức đóng; mã có
  `model = bank` không có Altman Z.

**B2. So sánh với ngành**
- `fa/sector.py` tính trung vị P/E, P/B, ROE và biên lợi nhuận theo `industry_icb` trên các mã có trong kho.
- Điểm định giá có thêm thành phần "rẻ hay đắt so với ngành". Ngưỡng ở `[fa_score.thresholds]`
  chuyển sang dạng tương đối (ví dụ P/E ÷ trung vị ngành nằm trong [1,5; 0,7]).
- Cần ít nhất 5 mã cùng ngành; thiếu thì dùng ngưỡng tuyệt đối như v1 và ghi chú lại.

**B3. Bộ lọc trạng thái thị trường**
- `ta/regime.py` xét VNINDEX:
  - giá so với SMA200 và độ dốc SMA50;
  - độ rộng thị trường: tỷ lệ mã trong kho nằm trên SMA50;
  - biến động GARCH của chỉ số.
- Kết quả có 3 trạng thái: `tăng` / `trung tính` / `giảm`. Khi `giảm`, engine nâng `buy_above` thêm
  `[recommend.regime_penalty]` điểm và giảm `risk_per_trade` đi một nửa.
- Kiểm chứng bằng backtest: có và không có bộ lọc.

**B4. Thanh khoản**
- Thêm GTGD bình quân 20 phiên (giá × khối lượng). `plan` cảnh báo khi vị thế lớn hơn
  `[plan].max_adv_pct` (mặc định 5%) GTGD ngày.
- Backtest thêm trượt giá theo thanh khoản. Phiên trần/sàn (giá mở cửa = trần khi mua, = sàn khi
  bán) coi như **không khớp**.

### Giai đoạn C: Danh mục và kỷ luật giao dịch

**C1. Sổ lệnh và danh mục**
- `portfolio/ledger.py`: lệnh `trade add FPT buy 500 98500 --date 2026-10-10`, nhập từ CSV, và
  `trade list`. Phí và thuế tự tính theo `[backtest]`, có thể ghi đè.
- `portfolio/positions.py`: số lượng đang giữ, **số lượng bán được** (đã qua T+2), giá vốn bình
  quân, lãi/lỗ đã chốt và chưa chốt, cổ tức tiền mặt nhận về (lấy từ `corporate_actions`).
- Lệnh `portfolio`: bảng vị thế, tỷ trọng, khoảng cách tới cắt lỗ và mục tiêu đã lập trong `plan`.
- *Test:* chuỗi mua–mua–bán ra giá vốn bình quân và lãi lỗ đúng; bán trước T+2 thì báo lỗi.

**C2. Rủi ro danh mục**
- `risk/portfolio_risk.py`: ma trận tương quan, VaR/CVaR danh mục (lịch sử và Monte Carlo có
  tương quan), đóng góp rủi ro của từng mã, mức tập trung theo ngành.
- Quy tắc trong `[portfolio]`: `max_position = 0.25`, `max_sector = 0.40`, `max_open_risk = 0.06`
  (tổng rủi ro tới cắt lỗ / vốn). `plan` tính khối lượng theo **phần rủi ro còn trống**, không chỉ
  theo 1% mỗi lệnh.

**C3. Nhật ký khuyến nghị và đo độ chính xác**
- Mỗi lần chạy `recommend` ghi vào bảng `signals` kèm `params_hash`, tức mã băm của phần cấu hình
  đang dùng.
- Lệnh `review-signals --horizon 20,60`: lợi nhuận sau 20 và 60 phiên của từng nhóm MUA, THEO DÕI,
  TRÁNH, so với VNINDEX; tỷ lệ chạm mục tiêu trước hay chạm cắt lỗ trước.
- Đây là thước đo **thật** của tool. Sau 3–6 tháng mới có ý nghĩa, nên bắt đầu ghi càng sớm càng tốt.

### Giai đoạn D: Mở rộng và kiểm chứng

**D1. Sàng lọc nhiều mã**
- Danh sách rổ trong `[universe]`: `vn30`, `vn100` hoặc tự định nghĩa. Lệnh `screen --universe vn30
  --min-adv 10e9 --sort total` chỉ đọc từ kho.
- Tải dữ liệu cho 100 mã: CafeF cần khoảng 0,3 giây mỗi lần gọi, tải tăng dần mất vài phút. Tải đầy
  đủ lần đầu nên chạy trên GitHub Actions.

**D2. Backtest danh mục**
- `recommend/portfolio_backtest.py`:
  - nhiều mã, tối đa N vị thế, khối lượng theo rủi ro ATR;
  - tái cân bằng hằng tuần;
  - dùng FA point-in-time (từ A4) cùng TA;
  - có bộ lọc trạng thái thị trường (B3).
- **Walk-forward:** chọn tham số trên giai đoạn 1, kiểm tra trên giai đoạn 2, rồi trượt cửa sổ.
  Chỉ số báo cáo: CAGR, Sharpe, max drawdown, tỷ lệ thắng, số lệnh, mức vượt VNINDEX.
- Có **lưới độ nhạy** tham số (entry/exit score): nếu kết quả chỉ tốt ở một điểm thì là quá khớp.
- *Lưu ý:* mỗi lần backtest chỉ dùng danh sách mã hiện tại, nên bị **thiên lệch sống sót** (các mã
  đã huỷ niêm yết không có trong danh sách). Báo cáo phải ghi rõ điều này.

### Giai đoạn E: Vận hành hằng ngày (dùng cá nhân, riêng tư)

**E1. Job hằng ngày**
- `jobs/daily.py` chạy theo thứ tự: tải tăng dần → kiểm tra chất lượng → `recommend` (ghi `signals`)
  → kiểm tra danh mục (chạm cắt lỗ hoặc mục tiêu?) → xuất báo cáo.
- Có hai cách chạy:
  1. **Thủ công:** gõ `python -m vnsa.cli daily` trên Codespaces mỗi tối.
  2. **Tự động:** workflow `daily.yml` chạy lúc 16:00 giờ Việt Nam (cron `0 9 * * 1-5` UTC).

**E2. Lưu kho bền vững mà vẫn riêng tư**
- Repo chính vẫn không chứa dữ liệu. Có hai cách:
  - **(Khuyến nghị)** Một repo **private** riêng, ví dụ `tool_chung_khoa-data`. Job hằng ngày
    đẩy `warehouse.duckdb` (nén) và `transactions.csv` vào đó. Lệnh `restore` tải về khi mở Codespace mới.
  - Hoặc dùng artifact của GitHub Actions (giữ tối đa 90 ngày): đơn giản hơn nhưng có thể mất dữ liệu.
- **Nhật ký lệnh mua bán là dữ liệu không tải lại được**, nên phải có bản sao lưu. Ngoài kho, xuất
  thêm `transactions.csv` (dạng văn bản, xem được thay đổi qua Git).

**E3. Cảnh báo (tuỳ chọn)**
- Gửi qua bot Telegram cá nhân: token đặt ở GitHub Secrets hoặc `.env`, không đưa lên Git.
- Chỉ gửi khi có sự kiện: chạm cắt lỗ, mã chuyển sang MUA hoặc TRÁNH, dữ liệu cũ hơn 2 phiên, lần
  tải bị lỗi.

**E4. Báo cáo tuần**
- `report/weekly.py` tạo một file HTML tĩnh (Plotly, giống `chart`) gồm: danh mục, các khuyến nghị
  thay đổi trong tuần, trạng thái thị trường và kết quả `review-signals`. File này mở trên máy hoặc
  tải từ artifact, không đưa lên web.

### Giai đoạn F: Chất lượng mã nguồn
- Tách `cli.py` thành gói `cli/` (`data.py`, `fa.py`, `ta.py`, `portfolio.py`, `jobs.py`), mỗi file
  dưới 250 dòng.
- Kiểm tra cấu hình khi khởi động (`config.validate()`): trọng số cộng lại bằng 100, các ngưỡng
  đúng chiều, mọi khoá bắt buộc đều có. Báo lỗi bằng tiếng Việt, chỉ rõ dòng cần sửa.
- CI chạy thêm `ruff check` và `pytest --cov`; đặt mức tối thiểu cho thư mục `fa/`, `risk/`, `portfolio/`.
- **Test vàng (golden):** chụp kết quả `value`, `fa-score` và `recommend` trên một kho cố định
  (`tests/fixtures/golden.duckdb`, chỉ vài mã). Khi sửa code mà số thay đổi thì test báo, và bạn
  quyết định đó là sửa lỗi hay lỗi mới.
- MCP: thêm `get_portfolio`, `get_signal_history`, `screen`, `get_market_regime`. Dashboard: thêm
  trang **Danh mục**, **Sàng lọc** và **Lịch sử khuyến nghị**.

---

## 5. Thứ tự làm đề xuất

| Đợt | Hạng mục | Lý do | Ước lượng |
|-----|----------|-------|-----------|
| 1 | A1, A2, A3, C3 | Sửa lỗi biên độ sai sàn, kho nâng cấp an toàn; **bắt đầu ghi lịch sử khuyến nghị ngay** để có dữ liệu đo sau này | 2–3 ngày |
| 2 | C1, E2 | Tool biết bạn đang cầm gì; sao lưu sổ lệnh | 2 ngày |
| 3 | A4, A5 | BCTC quý, TTM, point-in-time: nền cho FA chính xác và backtest FA | 3 ngày |
| 4 | B1, B2 | Ngân hàng và chứng khoán chiếm tỷ trọng lớn trong VN30; so sánh ngành | 3 ngày |
| 5 | B3, B4, C2 | Bộ lọc thị trường, thanh khoản, rủi ro danh mục | 2–3 ngày |
| 6 | D1, D2 | Sàng lọc VN30/VN100, backtest danh mục, walk-forward | 3–4 ngày |
| 7 | E1, E3, E4, F | Tự động hoá hằng ngày, báo cáo, dọn mã nguồn | 2–3 ngày |

Mỗi đợt kết thúc khi: `pytest` xanh, `scripts/smoke_test.py` chạy được trên kho thật, và README
cùng CLAUDE.md đã cập nhật lệnh mới.

---

## 6. Việc cần bạn quyết định trước khi làm

1. **Rổ mã:** giữ 5 mã hay mở rộng ra VN30 / VN100? Câu trả lời ảnh hưởng tới thời gian tải và
   việc có làm D1 sớm hay không.
2. **Có nhập lệnh thật không?** Nếu có, cần làm C1 và E2 sớm. Nếu chỉ phân tích, có thể bỏ qua C1 và C2.
3. **Sao lưu kho:** dùng repo private riêng hay artifact của Actions?
4. **Cảnh báo:** có muốn nhận tin qua Telegram không, hay chỉ cần báo cáo khi tự mở?
5. **Giả định định giá** (`risk_free`, `equity_risk_premium`, `terminal_growth`) và giả định cho
   ngân hàng (ke, g) cần bạn duyệt lại theo lãi suất hiện tại.

---

## 7. Những gì *không* làm (để giữ tool gọn)
- Không đưa lên web công khai, không làm đăng nhập hay nhiều người dùng.
- Không đặt lệnh tự động qua API công ty chứng khoán: tool chỉ gợi ý, bạn tự quyết.
- Không dùng LLM để dự báo giá hay sinh con số. LLM chỉ diễn giải số do Python tính.
- Không dùng dữ liệu theo phút (intraday): tool hướng tới nắm giữ nhiều ngày đến nhiều tháng.

> Kết quả của tool là sàng lọc theo quy tắc, **không phải lời khuyên đầu tư**.
