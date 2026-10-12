"""Sàng lọc toàn thị trường: từ mọi mã niêm yết đến nhóm ngành và mã tiềm năng.

Các tầng (ngưỡng ở mục [screen] trong config/settings.toml):
  T0 Vũ trụ          — danh sách niêm yết (bảng companies), lọc theo sàn.
  T1 Thanh khoản     — chỉ cần giá 1 năm (1 lần gọi CafeF mỗi mã): GTGD bình quân 20 phiên, giá,
                       số phiên giao dịch, còn giao dịch gần đây (screen/liquidity.py).
  T2 Phân tích từng mã, T3 Chấm nhóm ngành, T4 Chọn mã trong ngành — các bước sau.

Mỗi lượt ghi vào bảng `screen` (một dòng mỗi mã: tầng cao nhất đã qua, lý do bị loại, số liệu) và
file CSV ở `[data] screen_dir`. Mọi số tính từ kho; thiếu số thì ghi lý do, không ước lượng.
"""
