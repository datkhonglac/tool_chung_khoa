"""Phân loại tiêu đề tin theo từ khoá (nâng cấp v3) — miễn phí, không cần mô hình ngôn ngữ.

Tiêu đề lấy từ vnstock (Company.news, VCI) vào bảng `news`. Mỗi tiêu đề được gắn một nhãn:
  NGHIÊM TRỌNG  trạng thái giao dịch khách quan: diện kiểm soát, hạn chế giao dịch, tạm ngừng/đình
                chỉ giao dịch, huỷ niêm yết cổ phiếu -> CHẶN MUA (xếp hạng tối đa NẮM GIỮ) cho tới
                khi có tin "ra khỏi diện ..." mới hơn.
  TIÊU CỰC      xử phạt, khởi tố, điều tra, kiểm toán ngoại trừ, giải chấp, chậm thanh toán, thua lỗ,
                diện cảnh báo... -> cảnh báo trên thẻ, độ tin cậy giảm một bậc (không đổi xếp hạng).
  TÍCH CỰC      lãi kỷ lục, vượt kế hoạch, người nội bộ đăng ký mua, mua cổ phiếu quỹ, ra khỏi diện
                cảnh báo/kiểm soát... -> chỉ đánh dấu [+].
  NỘI BỘ BÁN    người nội bộ/liên quan đăng ký bán -> đánh dấu [−], không đổi gì.
Từ khoá trong `[news]` của config/settings.toml (biểu thức chính quy, không phân biệt hoa thường).
Không dùng quy tắc "tin xấu mà giá tăng -> tin cậy cao" (phân kỳ) vì chưa kiểm định được: kho chỉ
có tiêu đề ~30 ngày gần nhất.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

SEVERE, NEGATIVE, POSITIVE, INSIDER_SELL, NEUTRAL = (
    "NGHIÊM TRỌNG", "TIÊU CỰC", "TÍCH CỰC", "NỘI BỘ BÁN", "")
MARK = {SEVERE: "[!!]", NEGATIVE: "[−]", POSITIVE: "[+]", INSIDER_SELL: "[−]", NEUTRAL: ""}

DEFAULTS = {
    # "ra khỏi diện ..." xét TRƯỚC: tin tích cực, đồng thời gỡ trạng thái nghiêm trọng
    "exit": [r"(ra|thoát|đưa ra) khỏi diện", r"(bỏ|gỡ|chấm dứt).{0,20}(kiểm soát|cảnh báo|hạn chế)",
             r"(giao dịch|niêm yết) trở lại", r"tiếp tục giao dịch"],
    "severe": [r"diện (bị )?kiểm soát", r"kiểm soát đặc biệt", r"hạn chế giao dịch",
               r"(tạm )?ngừng giao dịch", r"đình chỉ giao dịch", r"hủy niêm yết|huỷ niêm yết"],
    "severe_exclude": [r"trái phiếu", r"chứng quyền", r"\bCW\b"],
    "negative": [r"diện cảnh báo", r"xử phạt|bị phạt|phạt vi phạm", r"truy thu", r"khởi tố",
                 r"tạm giam|bắt giam|bị bắt", r"điều tra", r"ngoại trừ", r"từ chối (đưa ra )?ý kiến",
                 r"giải chấp", r"chậm (công bố|nộp|thanh toán|trả)", r"nợ quá hạn",
                 r"mất khả năng thanh toán", r"thua lỗ|lỗ ròng|báo lỗ|lỗ lũy kế|lỗ luỹ kế",
                 r"cắt margin|(loại|ra) khỏi danh (sách|mục).{0,25}ký quỹ", r"vỡ nợ"],
    "positive": [r"kỷ lục", r"vượt (kế hoạch|chỉ tiêu)", r"(lãi|lợi nhuận).{0,20}tăng",
                 r"(chủ tịch|tổng giám đốc|tgđ|thành viên hđqt|người nội bộ|cổ đông lớn|vợ|chồng)"
                 r".{0,40}đăng ký mua", r"(mua|mua lại) cổ phiếu quỹ", r"trúng thầu"],
    "insider_sell": [r"đăng ký bán"],
    "block_status": ["RESTRICT", "SUSPEND", "HALT", "DELIST", "CONTROL", "STOP"],
    "status_max_age_days": 5,
    "severe_days": 180,         # tin trạng thái nghiêm trọng còn hiệu lực tối đa 180 ngày
    "negative_days": 30,        # tin tiêu cực trong 30 ngày gần nhất mới giảm độ tin cậy
}


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", str(text)).lower()


def cfg() -> dict:
    try:
        from vnsa.config import settings

        return {**DEFAULTS, **settings().get("news", {})}
    except Exception:  # noqa: BLE001
        return dict(DEFAULTS)


def _any(patterns, text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def classify(title: str, c: dict | None = None) -> str:
    c = c or cfg()
    t = _norm(title)
    if _any(c["exit"], t):
        return POSITIVE
    if _any(c["severe"], t) and not _any(c["severe_exclude"], t):
        return SEVERE
    if _any(c["negative"], t):
        return NEGATIVE
    if _any(c["positive"], t):
        return POSITIVE
    if _any(c["insider_sell"], t):
        return INSIDER_SELL
    return NEUTRAL


@dataclass
class NewsCheck:
    tagged: pd.DataFrame                          # date, title, label, mark
    blocked: str = ""                             # lý do chặn mua ("" = không chặn)
    negatives: list[str] = field(default_factory=list)   # tiêu đề tiêu cực gần đây
    positives: int = 0
    status_note: str = ""                         # trạng thái giao dịch bất thường (không chặn)


STATUS_NAMES = {"TRADING_RESTRICTION_ACTIVATED": "hạn chế giao dịch"}


def apply_status(res: NewsCheck, status: dict | None, c: dict | None = None,
                 now: pd.Timestamp | None = None) -> NewsCheck:
    """Trạng thái giao dịch từ bảng giá (bảng trading_status): chứa chữ trong `block_status` ->
    chặn mua; khác TRADING_ACTIVATED / mã chứng khoán khác "N" -> chỉ ghi chú. Bỏ qua bản cũ."""
    c = c or cfg()
    if not status or status.get("loaded_at") is None:
        return res
    now = pd.Timestamp(now or pd.Timestamp.now())
    if (now - pd.Timestamp(status["loaded_at"])).days > int(c["status_max_age_days"]):
        return res
    ts = str(status.get("trading_status") or "").upper()
    sec = str(status.get("security_status") or "").upper()
    when = pd.Timestamp(status.get("trading_date") or status["loaded_at"])
    if ts and any(k in ts for k in c["block_status"]):
        res.blocked = (f"bảng giá {when:%d/%m/%Y}: {STATUS_NAMES.get(ts, ts)}"
                       + (f"; {res.blocked}" if res.blocked else ""))
    elif ts and ts not in ("TRADING_ACTIVATED", "NONE", "NAN"):
        res.status_note = f"trạng thái giao dịch {ts} ({when:%d/%m/%Y})"
    if sec and sec not in ("N", "NONE", "NAN"):
        res.status_note = (res.status_note + "; " if res.status_note else "") + \
            f"mã trạng thái chứng khoán {sec} (bình thường là N)"
    return res


def check(news: pd.DataFrame, as_of: pd.Timestamp | None = None, c: dict | None = None) -> NewsCheck:
    """Gắn nhãn và tổng hợp: trạng thái nghiêm trọng còn hiệu lực, tin tiêu cực gần đây."""
    c = c or cfg()
    if news is None or news.empty:
        return NewsCheck(pd.DataFrame(columns=["date", "title", "label", "mark"]))
    as_of = pd.Timestamp(as_of or pd.Timestamp.now()).normalize()
    df = news.assign(date=pd.to_datetime(news["date"]))
    df = df.assign(label=[classify(t, c) for t in df["title"]])
    df["mark"] = df["label"].map(MARK)
    df["is_exit"] = [_any(c["exit"], _norm(t)) for t in df["title"]]
    res = NewsCheck(df.drop(columns="is_exit").sort_values("date", ascending=False)
                    .reset_index(drop=True))
    status = ""
    for r in df.sort_values("date").itertuples():
        if r.is_exit:
            status = ""
        elif r.label == SEVERE and r.date >= as_of - pd.Timedelta(days=int(c["severe_days"])):
            status = f"{r.date:%d/%m/%Y} {r.title}"
    res.blocked = status
    recent = df[df["date"] >= as_of - pd.Timedelta(days=int(c["negative_days"]))]
    res.negatives = [f"{r.date:%d/%m} {r.title}" for r in
                     recent[recent["label"] == NEGATIVE].sort_values("date", ascending=False)
                     .itertuples()]
    res.positives = int((recent["label"] == POSITIVE).sum())
    return res
