"""Sinh dữ liệu giá GIẢ LẬP để chạy thử pipeline khi chưa có dữ liệu thật.

Đây KHÔNG phải giá thật của bất kỳ cổ phiếu nào. Mã đặt tên DEMO_* để không nhầm.
Một file được ghi theo kiểu file tải từ web tiếng Việt (ngày dd/mm/yyyy, giá nghìn đồng,
khối lượng có dấu phẩy) để kiểm tra phần đọc tiêu đề tiếng Việt.

    python scripts/make_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "sample" / "prices"

# mã, giá đầu, lợi suất kỳ vọng năm, biến động năm, KL trung bình
SPECS = [
    ("DEMO_A", 60_000, 0.15, 0.28, 2_000_000),
    ("DEMO_B", 25_000, 0.05, 0.40, 15_000_000),
    ("DEMO_C", 80_000, -0.05, 0.25, 800_000),
]


def simulate(start_price, mu, sigma, avg_vol, days, rng) -> pd.DataFrame:
    dt = 1 / 250
    rets = rng.standard_t(df=4, size=len(days)) * sigma * np.sqrt(dt) / np.sqrt(2)
    rets = np.clip(rets + (mu - 0.5 * sigma**2) * dt, -0.068, 0.068)  # trong biên độ ±7%
    close = start_price * np.exp(np.cumsum(rets))
    prev = np.concatenate([[start_price], close[:-1]])
    open_ = prev * (1 + rng.normal(0, 0.004, len(days)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.008, len(days))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.008, len(days))))
    vol = (avg_vol * rng.lognormal(0, 0.5, len(days))).round(-2)
    # Bước giá HOSE: <10.000đ: 10đ; 10.000–49.950đ: 50đ; ≥50.000đ: 100đ
    def r(x):
        tick = np.where(x < 10_000, 10, np.where(x < 50_000, 50, 100))
        return np.round(x / tick) * tick
    o, h, lo, c = r(open_), r(high), r(low), r(close)
    h = np.maximum.reduce([h, o, c])
    lo = np.minimum.reduce([lo, o, c])
    return pd.DataFrame({"date": days, "open": o, "high": h, "low": lo, "close": c, "volume": vol})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    days = pd.bdate_range("2021-10-01", "2026-10-09")
    for i, (sym, p0, mu, sig, vol) in enumerate(SPECS):
        df = simulate(p0, mu, sig, vol, days, rng)
        if i == 1:  # Định dạng kiểu tiếng Việt
            vn = pd.DataFrame({
                "Ngày": df["date"].dt.strftime("%d/%m/%Y"),
                "Giá mở cửa": df["open"] / 1000,
                "Giá cao nhất": df["high"] / 1000,
                "Giá thấp nhất": df["low"] / 1000,
                "Giá đóng cửa": df["close"] / 1000,
                "Khối lượng khớp lệnh": df["volume"].map(lambda v: f"{int(v):,}"),
            }).iloc[::-1]  # nhiều trang xuất mới nhất lên đầu
            vn.to_csv(OUT / f"{sym}.csv", index=False, encoding="utf-8-sig")
        else:
            df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).to_csv(OUT / f"{sym}.csv", index=False)
        print(f"Đã tạo {OUT / (sym + '.csv')} ({len(df)} phiên)")


if __name__ == "__main__":
    main()
