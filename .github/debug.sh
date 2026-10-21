#!/bin/bash
# TẠM THỜI (nhánh nang-cap-v2): chạy lệnh trên dữ liệu thật, đưa log vào annotation để đọc qua API
pytest -q -p no:cacheprovider --tb=line -rf 2>&1 | grep -v "^\.\+" | tail -15
python -m vnsa.cli analyze HPG VHM 2>&1 | grep -E "Xếp hạng|Phương pháp|nhóm ngành|quỹ đất|Tổ chức|Độ nén|Gia tốc|Xác suất|Lỗi|Error|Traceback" | head -40
python -m vnsa.cli portfolio buy HPG 1000 27000 --date 2026-10-15 2>&1 | tail -3
python -m vnsa.cli portfolio sell HPG 500 28000 --date 2026-10-16 2>&1 | tail -2
python -m vnsa.cli portfolio stop HPG 25000 2>&1 | tail -2
python -m vnsa.cli portfolio 2>&1 | tail -12
python -m vnsa.cli plan VHM 2>&1 | tail -15
python -m vnsa.config 2>&1 | tail -2
