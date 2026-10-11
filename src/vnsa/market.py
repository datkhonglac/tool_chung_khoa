"""Quy tắc thị trường theo từng mã: sàn niêm yết → biên độ, bước giá; loại doanh nghiệp.

Thông tin sàn/ngành lấy từ bảng `companies` trong kho (nạp bằng lệnh analyze hoặc fetch-listing).
Mã chưa có trong bảng thì coi như sàn `default_exchange` (HOSE) và doanh nghiệp thường.
"""

from __future__ import annotations

from vnsa.config import path, settings

EXCHANGES = ("HOSE", "HNX", "UPCOM")
FINANCIAL_MODELS = {"bank": "ngân hàng", "securities": "công ty chứng khoán",
                    "insurance": "công ty bảo hiểm"}


def profile(symbol: str) -> dict:
    """{symbol, name, exchange, industry, model, known}; known=False nếu kho chưa có thông tin."""
    from vnsa.data.store import Store

    sym = symbol.upper()
    row = None
    if path("warehouse").exists():
        with Store(path("warehouse")) as store:
            row = store.company(sym)
    default = settings()["market"].get("default_exchange", "HOSE")
    row = row or {}
    exchange = row.get("exchange") if row.get("exchange") in EXCHANGES else default
    return {"symbol": sym, "name": row.get("name"), "exchange": exchange,
            "industry": row.get("industry"), "model": row.get("model") or "general",
            "known": bool(row)}


def band(symbol: str | None = None, exchange: str | None = None) -> float:
    """Biên độ dao động trong phiên của sàn niêm yết (HOSE 7%, HNX 10%, UPCoM 15%)."""
    ex = exchange or (profile(symbol)["exchange"] if symbol else
                      settings()["market"].get("default_exchange", "HOSE"))
    return float(settings()["market"][f"band_{ex.lower()}"])


def ticks(symbol: str | None = None, exchange: str | None = None) -> list[list[float]]:
    """Bảng bước giá của sàn: [[ngưỡng giá, bước giá], ...]."""
    ex = exchange or (profile(symbol)["exchange"] if symbol else
                      settings()["market"].get("default_exchange", "HOSE"))
    return settings()["market"]["ticks"][ex]


def financial_note(symbol: str) -> str | None:
    """Câu giải thích nếu mã là ngân hàng/chứng khoán/bảo hiểm (mô hình FA hiện tại không áp dụng)."""
    kind = FINANCIAL_MODELS.get(profile(symbol)["model"])
    if not kind:
        return None
    return (f"{symbol.upper()} là {kind}: không dùng DCF-FCFF, Altman Z; định giá bằng thu nhập "
            "thặng dư và P/B theo ROE, điểm FA theo bộ chỉ tiêu riêng cho định chế tài chính")
