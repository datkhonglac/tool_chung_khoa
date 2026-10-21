# Kiểm định quy tắc thời điểm (11/10/2026)

Câu hỏi: trục **"Thời điểm"** của tool (khi nào mua) có giúp tốt hơn việc mua rồi nắm giữ không?

## Cách làm

- **Rổ:** 45 mã gồm VN30 và một số mã vừa (`[study] universe` trong `config/settings.toml`).
- **Dữ liệu:** giá ngày 10 năm (12/10/2016 → 09/10/2026) từ CafeF; PVS lấy từ vnstock.
- **Khớp lệnh sát thực tế** (`src/vnsa/recommend/strategies.py`):
  - tín hiệu lấy ở giá đóng cửa, khớp ở giá mở cửa phiên sau;
  - không mua được khi mở cửa ở giá trần, không bán được khi cả phiên nằm sàn;
  - T+2, phí 0,15% mỗi chiều, thuế bán 0,1%.
- **Danh mục:** chia đều vốn cho các mã, so với danh mục mua & giữ cùng các mã đó.
- **Hai giai đoạn:**
  - chọn tham số trên 2016–2021 (hệ số cắt lỗ động k = 2,5 / 3 / 4 ATR);
  - kiểm tra độc lập trên 2022 → 10/2026, giữ nguyên tham số đã chọn.
- **Tiêu chí "đạt"** (đặt trước khi chạy): Sharpe cao hơn **và** sụt giảm tối đa nhỏ hơn mua & giữ ở **cả hai** giai đoạn.

Chạy lại: `python -m vnsa.cli study-timing`, hoặc trên GitHub: Actions → "Kiểm định quy tắc thời điểm".

## Kết quả

| Quy tắc | Giai đoạn | k ATR | CAGR | Sharpe | Sụt giảm tối đa | Thời gian cầm cp | Số lệnh | Tỷ lệ thắng |
|---|---|---|---|---|---|---|---|---|
| Điểm TA (cách cũ) | 2016–2021 | 2 | 15,2% | 1,27 | −23,6% | 44% | 861 | 31% |
| | 2022–2026 | 2 | −0,7% | −0,01 | −18,5% | 47% | 1.086 | 26% |
| Điểm TA + VNINDEX | 2016–2021 | 2 | 14,2% | 1,28 | −23,8% | 42% | 807 | 31% |
| | 2022–2026 | 2 | −0,5% | 0,00 | −16,2% | 44% | 1.008 | 25% |
| Theo xu hướng MA200 | 2016–2021 | 4 | 11,2% | 1,36 | −11,7% | 24% | 355 | 51% |
| | 2022–2026 | 4 | −0,3% | −0,01 | −10,6% | 24% | 371 | 33% |
| Phá nền 55 phiên | 2016–2021 | 4 | 9,3% | 1,24 | −11,9% | 21% | 365 | 42% |
| | 2022–2026 | 4 | 2,3% | 0,37 | −7,9% | 21% | 345 | 37% |
| Xu hướng + VNINDEX | 2016–2021 | 4 | 10,4% | 1,28 | −11,7% | 23% | 342 | 51% |
| | 2022–2026 | 4 | −0,5% | −0,04 | −11,1% | 22% | 355 | 33% |
| Phá nền + VNINDEX | 2016–2021 | 4 | 8,3% | 1,13 | −15,3% | 19% | 332 | 40% |
| | 2022–2026 | 4 | 2,3% | 0,38 | −9,2% | 19% | 320 | 38% |
| **Mua & giữ** | 2016–2021 | | **30,2%** | **1,37** | −48,6% | 100% | | |
| | 2022–2026 | | 4,6% | 0,31 | −46,4% | 100% | | |

## Kết luận

1. **Không quy tắc nào đạt.** Giai đoạn 2016–2021 là thị trường tăng mạnh, nên không quy tắc nào có Sharpe vượt mua & giữ (1,37). Trục thời điểm của tool chuyển sang chỉ quản trị rủi ro: `[recommend] timing_rule = "none"`.
   - Thẻ báo cáo chỉ còn "Chờ giá về vùng mua" (giá cao hơn giá trị hợp lý trừ biên an toàn) hoặc "Trong vùng mua (giữ kỷ luật cắt lỗ)".
   - Thẻ không còn đoán "Vào được" hay "Chờ xu hướng" nữa.
2. **Điểm TA cũ không có lợi thế.**
   - Giai đoạn 2022–2026: lỗ, thắng 26%, hơn 1.000 lệnh nên phí và thuế ăn mòn.
   - Lọc VNINDEX gần như không thay đổi kết quả.
3. **Các quy tắc theo xu hướng và phá nền cắt mạnh sụt giảm**, từ khoảng −47% còn −8% đến −12%. Đổi lại, lợi nhuận thấp hơn nhiều vì chỉ cầm cổ phiếu khoảng 20% thời gian.
   - "Phá nền 55 phiên" gần đạt nhất: giai đoạn 2022–2026 có Sharpe 0,37 so với 0,31 và sụt giảm −8% so với −46%.
   - Chênh lệch Sharpe này nhỏ, có thể do ngẫu nhiên.
   - Nếu bạn ưu tiên tránh sụt giảm lớn hơn là tối đa lợi nhuận, đặt `timing_rule = "breakout"`. Thẻ báo cáo khi đó hiện "Vào được — xu hướng tăng xác nhận, cắt lỗ động …" hoặc "Chờ tín hiệu xu hướng".
4. **Hạn chế:**
   - Rổ là các mã lớn đang niêm yết, nên có thiên lệch sống sót: mã đã huỷ niêm yết không có trong mẫu.
   - Trần/sàn được xấp xỉ từ giá đã điều chỉnh.
   - Chưa tính trượt giá ngoài phí và thuế.
   - Kết quả quá khứ không đảm bảo tương lai.

## Bổ sung 21/10/2026: quy tắc `vcp_pullback` (nâng cấp v2.0, bước 3.3)

Quy tắc mua nhịp điều chỉnh trong nền giá chặt của xu hướng tăng (`src/vnsa/recommend/strategies.py`). Mua khi đủ 4 điều kiện:

1. Giá > MA200 và MA50 > MA200.
2. Độ nén Bollinger `bb_width_pct` < 0,40, hoặc giá cách MA20 không quá 1 ATR.
3. 45 ≤ RSI14 ≤ 62.
4. Phiên xác nhận: đóng cửa > mở cửa và khối lượng > 1,2 × trung bình 20 phiên.

Bán bằng cắt lỗ động k ATR, hoặc khi đóng cửa dưới MA50 với khối lượng > 1,5 × trung bình 20 phiên.

Tham số được đặt **trước** khi chạy. Kiểm định dùng cùng rổ 45 mã, cùng cách khớp lệnh và cùng tiêu chí "đạt" như trên. Dữ liệu đến 21/10/2026, chạy trên GitHub Actions.

| Quy tắc | Giai đoạn | k ATR | CAGR | Sharpe | Sụt giảm tối đa | Thời gian cầm cp | Số lệnh | Tỷ lệ thắng |
|---|---|---|---|---|---|---|---|---|
| Nền chặt + điều chỉnh (VCP) | 2016–2021 | 2,5 | 6,2% | 1,34 | −9,3% | 16% | 578 | 44% |
| | 2022–10/2026 | 2,5 | −1,6% | −0,32 | −10,5% | 19% | 708 | 32% |
| Phá nền 55 phiên (tham chiếu) | 2022–10/2026 | 4 | 2,2% | 0,36 | −7,9% | 21% | 346 | 37% |
| **Mua & giữ** | 2016–2021 | | 29,4% | 1,33 | −49,4% | 100% | | |
| | 2022–10/2026 | | 5,1% | 0,33 | −46,4% | 100% | | |

**Kết luận: không đạt.**

- Giai đoạn chọn tham số trông tốt: Sharpe 1,34 nhỉnh hơn mua & giữ (1,33), sụt giảm chỉ −9%.
- Giai đoạn kiểm tra độc lập thì lỗ: CAGR −1,6%, Sharpe −0,32, trong khi mua & giữ đạt 0,33.
- Quy tắc nhiều điều kiện khớp tốt với quá khứ nhưng không giữ được ở dữ liệu mới. Đây đúng là rủi ro "khớp quá mức" đã nêu khi nhận xét đề xuất.
- Trục thời điểm giữ nguyên `timing_rule = "none"`.
- Các quy tắc cũ chạy lại với dữ liệu thêm đến 21/10/2026 cho kết luận như lần trước. Phá nền 55 phiên vẫn gần đạt nhất.

## Bổ sung 22/10/2026: phá nền 55 phiên + cảnh báo sớm VNINDEX (nâng cấp v2.1)

Quy tắc `breakout_early` giống "Phá nền 55 phiên", nhưng chỉ MUA MỚI khi VNINDEX không ở trạng thái "giảm". Trạng thái này tính **có** cảnh báo sớm (`ta/regime.early_warning`), kích hoạt khi có một trong hai điều kiện:

- ≥ 5 phiên phân phối trong 20 phiên, đóng cửa dưới MA50;
- ATR/giá vượt phân vị 85% của 250 phiên, đóng cửa dưới MA20.

Cùng rổ 45 mã, cùng tiêu chí "đạt". Kho giá được tải lại từ đầu, dữ liệu đến 21/10/2026.

| Quy tắc | Giai đoạn | k ATR | CAGR | Sharpe | Sụt giảm tối đa | Thời gian cầm cp |
|---|---|---|---|---|---|---|
| Phá nền + cảnh báo sớm | 2016–2021 | 3 | 6,0% | 1,06 | −12,6% | 14% |
| | 2022–10/2026 | 3 | 0,9% | 0,21 | −6,5% | 14% |
| Phá nền + VNINDEX (MA) | 2022–10/2026 | 3 | 0,9% | 0,20 | −6,3% | 14% |
| Phá nền 55 phiên | 2022–10/2026 | 2,5 | 0,9% | 0,23 | −5,8% | 13% |
| **Mua & giữ** | 2016–2021 | | 30,2% | 1,36 | −49,7% | 100% |
| | 2022–10/2026 | | 5,1% | 0,33 | −46,4% | 100% |

**Kết luận: không đạt.**

- Cảnh báo sớm gần như không đổi kết quả so với bộ lọc MA50/MA200 cũ: Sharpe 0,21 so với 0,20.
- Cả hai đều thấp hơn mua & giữ ở cả hai giai đoạn.
- Giữ nguyên `[regime] early = false` và `timing_rule = "none"`.

Một lưu ý quan trọng: lần chạy này tải lại kho giá từ đầu, và "Phá nền 55 phiên" chọn k = 2,5 thay cho k = 4. Kết quả giai đoạn kiểm tra vì thế đổi từ Sharpe 0,36 xuống 0,23. Chỉ khác dữ liệu một chút mà kết quả đã dịch chuyển như vậy, tức là lợi thế "gần đạt" của quy tắc này mong manh. Đó thêm một lý do để không bật trục thời điểm.

## Bổ sung 22/10/2026: kiểm định ĐIỂM MUA — vùng cũ so với thang mua 3 lớp (v2.1)

Câu hỏi: vùng mua cũ chỉ cách giá 0–2%; đặt lệnh sâu hơn (thang 3 lớp 30% / 40% / 30%) có tốt hơn không?

Cách làm (lệnh `study-entry`, `src/vnsa/recommend/entry_study.py`, workflow "Kiểm định điểm mua"):

- Rổ 45 mã, 10 năm. Cứ 5 phiên lại lập vùng mua một lần, chỉ dùng dữ liệu đến phiên đó.
- Lệnh LO chờ khớp 20 phiên. Kết quả đo sau 60 phiên; phần vốn không khớp giữ tiền mặt.
- Không có trần giá trị hợp lý (BCTC không có theo ngày công bố). Phần này giống nhau ở mọi cách nên không làm lệch so sánh.

| Giai đoạn | Cách | Khớp vốn | Giá vốn so với giá lúc lập vùng | Lãi trên vốn đã khớp | **Lãi trên tổng vốn** |
|---|---|---|---|---|---|
| 2016–2021 | Mua ngay (mở cửa phiên sau) | 100% | 0,0% | +7,2% | **+7,2%** |
| | Vùng cũ | 82% | −1,6% | +7,1% | **+5,3%** |
| | Thang 3 lớp | 43% | −4,2% | +8,4% | **+2,7%** |
| 2022–10/2026 | Mua ngay | 100% | +0,2% | +1,7% | **+1,7%** |
| | Vùng cũ | 84% | −1,5% | +1,8% | **+1,1%** |
| | Thang 3 lớp | 43% | −4,1% | +3,4% | **+0,8%** |

Từng lệnh (toàn kỳ, ~18.800 lần lập vùng):

| Lệnh | Độ sâu trung vị | Khớp trong 20 phiên | Lãi 60 phiên khi khớp |
|---|---|---|---|
| Vùng cũ — đỉnh | −0,6% | 87% | +3,8% |
| Vùng cũ — đáy | −1,7% | 78% | +3,6% |
| Thang — lớp 1 | −1,9% | 76% | +3,5% |
| Thang — lớp 2 | −7,0% | 38% | +4,0% |
| Thang — lớp 3 | −12,8% | 16% | +5,5% |

**Kết luận.**

1. **Mua sâu cho giá vốn tốt hơn thật.** Thang mua rẻ hơn khoảng 4%, và phần vốn khớp được lãi nhiều hơn (+5,6% so với +4,2% toàn kỳ). Lớp 3 khớp được thì lãi +5,5%.
2. **Nhưng lớp càng sâu càng ít khớp.** Lớp 2 chỉ khớp 38%, lớp 3 chỉ 16%. Thang 3 lớp chỉ giải ngân được 43% vốn, nên **lãi trên tổng vốn thấp hơn vùng cũ ở cả hai giai đoạn**. Đây là cái giá của việc lỡ mua: những mã không lùi về vùng sâu thường là những mã tăng tiếp.
3. **Chính vùng cũ cũng kém "mua ngay"** (+3,0% so với +4,2% trên tổng vốn toàn kỳ). Chờ giá lùi luôn tốn một phần lợi nhuận. Lợi ích của vùng mua nằm ở chỗ khác: điểm cắt lỗ rõ ràng và tránh mua đuổi sau phiên tăng mạnh.

**Quyết định:**

- `[entry] ladder = false`: kế hoạch mua và khối lượng tiếp tục dùng vùng cũ.
- `ladder_show = true`: thẻ báo cáo, lệnh `plan` và dashboard in thêm lớp 2, lớp 3 thành **"mức mua thêm nếu giá giảm sâu (tham khảo)"**, kèm xác suất khớp.
- Ai ưu tiên giá vốn thấp hơn số lần mua được thì đặt `ladder = true`. Lúc đó cắt lỗ đặt dưới lớp 3 và khối lượng tính theo giá vốn bình quân của 3 lớp.

Hạn chế: rổ là mã lớn hiện tại (thiên lệch sống sót); giá điều chỉnh; chưa trừ phí và thuế (như nhau ở mọi cách); khớp theo giá thấp nhất trong ngày là lạc quan, nhưng lạc quan như nhau cho mọi lệnh LO.

Trục **xếp hạng** (giá trị hợp lý) không kiểm định được theo cách này, vì BCTC miễn phí chỉ có 4 năm. Thay vào đó, workflow hằng ngày lưu khuyến nghị vào nhánh `signals`, và lệnh `review-signals` chấm kết quả thật sau 5/20/60 phiên.
