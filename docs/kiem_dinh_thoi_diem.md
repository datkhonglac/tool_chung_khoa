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

Trục **xếp hạng** (giá trị hợp lý) không kiểm định được theo cách này, vì BCTC miễn phí chỉ có 4 năm. Thay vào đó, workflow hằng ngày lưu khuyến nghị vào nhánh `signals`, và lệnh `review-signals` chấm kết quả thật sau 5/20/60 phiên.
