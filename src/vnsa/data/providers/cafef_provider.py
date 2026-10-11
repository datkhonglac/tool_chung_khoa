"""Tải giá ngày trực tiếp từ CafeF, thay cho việc bấm "Xuất Excel" từng năm.

Dùng đúng nguồn dữ liệu mà trang lịch sử giá của CafeF gọi tới (trả JSON). Mỗi lần gọi CafeF
chỉ trả tối đa khoảng 1 năm, nên lớp này tự chia khoảng thời gian thành từng đoạn 1 năm,
lật trang, ghép lại và bỏ ngày trùng. Giá đã điều chỉnh, khối lượng khớp lệnh, tính bằng đồng.

Đây là API không chính thức: nếu CafeF đổi cấu trúc, lệnh sẽ báo lỗi rõ ràng và vẫn có thể
quay về cách tải Excel thủ công (import-excel).
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import date, timedelta

import pandas as pd

from vnsa.data.cleaning import repair_ohlc
from vnsa.data.providers.base import DataProvider
from vnsa.data.providers.csv_provider import normalize_prices

URL = "https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Referer": "https://cafef.vn/du-lieu/lich-su-giao-dich-fpt-1.chn",
    "Accept": "application/json, text/javascript, */*",
}
# Tên trường JSON của CafeF -> tiêu đề tiếng Việt mà normalize_prices đã nhận diện.
FIELDS = {
    "Ngay": "Ngày",
    "GiaMoCua": "Giá mở cửa",
    "GiaCaoNhat": "Giá cao nhất",
    "GiaThapNhat": "Giá thấp nhất",
    "GiaDongCua": "Giá đóng cửa",
    "GiaDieuChinh": "Giá điều chỉnh",
    "KhoiLuongKhopLenh": "Khối lượng khớp lệnh",
}
INDEX_SYMBOLS = {"VNINDEX", "VN30", "HNXINDEX", "HNX30", "UPCOMINDEX"}
WINDOW_DAYS = 365   # CafeF cắt mỗi lần gọi còn khoảng 1 năm tính lùi từ ngày kết thúc
PAGE_SIZE = 30      # CafeF trả tối đa 30 dòng mỗi trang, ~9 trang cho 1 năm
MAX_PAGES = 100     # chặn vòng lặp vô hạn nếu CafeF đổi cách phân trang


def _http_get_json(url: str, timeout: float = 20) -> dict:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8-sig"))


class CafeFProvider(DataProvider):
    name = "cafef"

    def __init__(
        self,
        pause: float = 0.3,
        get_json: Callable[[str], dict] = _http_get_json,
        retries: int = 2,
        log: Callable[[str], None] | None = print,
    ):
        self.pause = pause          # giây nghỉ giữa các lần gọi, để không bị chặn
        self.get_json = get_json    # thay được trong test, không cần mạng
        self.retries = retries
        self.log = log              # in tiến trình; None = im lặng (MCP server dùng stdout)
        self.repaired: list[str] = []   # ngày có nến mâu thuẫn đã được sửa ở lần tải gần nhất

    def symbols(self) -> list[str]:
        return []  # CafeF có mọi mã niêm yết; danh sách theo dõi nằm ở config/settings.toml

    def _call(self, symbol: str, start: date, end: date, page: int) -> dict:
        query = urllib.parse.urlencode({
            "Symbol": symbol,
            "StartDate": start.strftime("%m/%d/%Y"),   # CafeF dùng tháng/ngày/năm
            "EndDate": end.strftime("%m/%d/%Y"),
            "PageIndex": page,
            "PageSize": PAGE_SIZE,
        })
        last_err: Exception | None = None
        for attempt in range(self.retries):
            try:
                payload = self.get_json(f"{URL}?{query}")
                if not payload.get("Success", False):
                    raise RuntimeError(f"CafeF trả lỗi: {payload.get('Message')}")
                return payload["Data"]
            except Exception as e:  # noqa: BLE001 - thử lại vài lần rồi mới báo lỗi
                last_err = e
                time.sleep(self.pause * (attempt + 1))
        raise RuntimeError(f"không tải được từ CafeF ({last_err})") from last_err

    def _window(self, symbol: str, start: date, end: date) -> list[dict]:
        rows: list[dict] = []
        seen: set[str] = set()
        for page in range(1, MAX_PAGES + 1):
            data = self._call(symbol, start, end, page)
            batch = [r for r in data.get("Data") or [] if r.get("Ngay") not in seen]
            seen.update(r.get("Ngay") for r in batch)
            rows.extend(batch)
            # Dừng khi hết trang, đủ số phiên, hoặc CafeF trả lại trang cũ (không có ngày mới)
            if not batch or len(rows) >= int(data.get("TotalCount") or 0):
                break
            time.sleep(self.pause)
        if self.log:
            self.log(f"      {start} → {end}: {len(rows)} phiên ({page} trang)")
        return rows

    def raw_rows(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Bảng thô (tiêu đề tiếng Việt như file Excel CafeF), mọi đoạn 1 năm đã ghép."""
        symbol = symbol.upper()
        rows: list[dict] = []
        win_end = end
        while win_end >= start:
            win_start = max(start, win_end - timedelta(days=WINDOW_DAYS - 1))
            rows.extend(self._window(symbol, win_start, win_end))
            win_end = win_start - timedelta(days=1)
            if win_end >= start:
                time.sleep(self.pause)
        if not rows:
            raise ValueError(f"CafeF không có dữ liệu từ {start} đến {end}. Mã có đúng không?")
        df = pd.DataFrame(rows)
        missing = [k for k in FIELDS if k not in df]
        if missing:
            raise ValueError(f"CafeF đổi cấu trúc, thiếu trường {missing}. "
                             "Tạm dùng cách tải Excel (import-excel).")
        return df[list(FIELDS)].rename(columns=FIELDS)

    def ohlcv(self, symbol: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        end = end or date.today()
        start = start or end - timedelta(days=365 * 5)
        # Cổ phiếu: CafeF ghi nghìn đồng. Chỉ số (VNINDEX...): điểm, giữ nguyên.
        unit = "dong" if symbol.upper() in INDEX_SYMBOLS else "nghin"
        df = normalize_prices(self.raw_rows(symbol, start, end), symbol, price_unit=unit)
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        return self._repair_ohlc(df.reset_index(drop=True))

    def _repair_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sửa nến mâu thuẫn (xem cleaning.repair_ohlc); ngày đã sửa ghi vào self.repaired."""
        df, self.repaired = repair_ohlc(df)
        return df
