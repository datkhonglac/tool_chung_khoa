#!/bin/bash
# TẠM THỜI (nhánh nang-cap-v2): chạy lệnh trên dữ liệu thật, đưa log vào annotation để đọc qua API
pytest -q -p no:cacheprovider --tb=line -rf 2>&1 | grep -v "^\.\+" | tail -60
python -m vnsa.cli analyze FPT 2>&1 | tail -60
