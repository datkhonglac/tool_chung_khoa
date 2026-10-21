#!/bin/bash
# TẠM THỜI (nhánh nang-cap-v2): chạy lệnh trên dữ liệu thật, đưa log vào annotation để đọc qua API
python -m vnsa.cli analyze FPT HPG 2>&1 | tail -120
