"""Tải giá ngày bằng vnstock rồi ghi ra data/raw/prices/<MÃ>.csv để tool nạp như CSV thường.

Chạy trong môi trường ảo RIÊNG đã cài vnstock (xem README, mục vnstock):

    .venv-vnstock\\Scripts\\activate
    python scripts/fetch_vnstock.py FPT HPG MWG VNM DGC --years 5

Sau đó quay lại môi trường chính (.venv) và chạy:

    python -m vnsa.cli load
    python -m vnsa.cli check FPT

Script chỉ dùng pandas + vnstock, không import gì từ dự án, để môi trường vnstock tách biệt.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "prices"


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("symbols", nargs="+")
    p.add_argument("--years", type=int, default=5)
    p.add_argument("--pause", type=float, default=3.5,
                   help="Giây nghỉ giữa các mã (bản miễn phí giới hạn lượt gọi/phút)")
    args = p.parse_args()

    try:
        from vnstock import Market
    except ImportError:
        print("Chưa cài vnstock trong môi trường này. Xem README, mục vnstock.")
        return 1

    end = date.today()
    start = end - timedelta(days=365 * args.years + 10)
    market = Market()
    OUT.mkdir(parents=True, exist_ok=True)
    failed = []
    for i, sym in enumerate(s.upper() for s in args.symbols):
        if i:
            time.sleep(args.pause)
        try:
            df = market.equity.ohlcv(symbol=sym, start=start.isoformat(), end=end.isoformat())
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ {sym}: {e}")
            failed.append(sym)
            continue
        if df is None or len(df) == 0:
            print(f"  ✗ {sym}: không có dữ liệu")
            failed.append(sym)
            continue
        df = pd.DataFrame(df).reset_index()
        df = df.drop(columns=[c for c in df.columns if str(c).lower() in {"index", "level_0"}])
        file = OUT / f"{sym}.csv"
        df.to_csv(file, index=False, encoding="utf-8")
        print(f"  ✓ {sym}: {len(df)} dòng → {file.name}  (cột: {', '.join(map(str, df.columns))})")
    print("Xong." + (f" Lỗi: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
