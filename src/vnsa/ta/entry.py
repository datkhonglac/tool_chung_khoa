"""Vùng giá mua khuyến nghị — thay cho việc lấy luôn giá đóng cửa làm giá vào.

Vì sao không dùng giá đóng cửa: giá đóng cửa VN là giá khớp phiên ATC, thường bị kéo/đạp cuối phiên;
mua ngay giá đó dễ thành "đuổi giá" sau phiên tăng mạnh. Lại thêm T+2 (mua xong 2,5 ngày chưa bán
được) nên cần mua ở vùng có đỡ giá và còn biên an toàn về giá trị. Lệnh giới hạn (LO) đặt sẵn trong
vùng sẽ khớp khi giá lùi về, không phụ thuộc giá đóng cửa.

Công thức (tham số ở mục [entry] trong settings):
1. Điểm neo kỹ thuật = mức CAO NHẤT (gần giá nhất) trong các mức nằm dưới giá hiện tại và cách giá
   không quá `max_pullback_atr` × ATR14:
     - hỗ trợ gần nhất (đáy cục bộ được kiểm nghiệm nhiều lần),
     - VWAP20 = Σ(giá điển hình × khối lượng) / Σ khối lượng của 20 phiên, giá điển hình = (cao +
       thấp + đóng) / 3 — giá vốn bình quân của người mua trong 1 tháng, vùng hay có lực đỡ,
     - SMA20 (đường giá trung bình 1 tháng).
   Không có mức nào: giá đã rơi dưới mọi mức đỡ (xu hướng yếu) → neo = giá − 1 ATR.
2. Vùng mua kỹ thuật = [neo, neo + `zone_atr` × ATR], không cao hơn giá hiện tại.
3. Trần giá trị = giá trị hợp lý × (1 − biên an toàn). Biên an toàn = `margin_of_safety`, cộng thêm
   `weak_fa_extra` khi điểm FA < 50 (doanh nghiệp yếu cần chiết khấu sâu hơn).
4. Vùng mua cuối = vùng kỹ thuật bị chặn trên bởi trần giá trị:
     - trần ≥ đỉnh vùng: giữ nguyên ("MUA TRONG VÙNG");
     - trần nằm trong vùng: hạ đỉnh vùng xuống trần;
     - trần < đáy vùng: "CHỜ GIÁ" — chỉ mua khi giá về ≤ trần giá trị.
   Giá đã cách SMA20 hơn `chase_atr` × ATR thì cảnh báo không đuổi giá.
5. Làm tròn theo bước giá của sàn: đáy vùng làm tròn lên, đỉnh vùng làm tròn xuống.
6. Giải ngân 2 lệnh LO: một nửa ở đỉnh vùng, một nửa ở đáy vùng. Kế hoạch giao dịch (plan.py) dùng
   giá mua bình quân dự kiến = giữa vùng (làm tròn bước giá) để tính cắt lỗ và khối lượng.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from vnsa.ta.levels import Level
from vnsa.ta.plan import round_tick


@dataclass
class EntryZone:
    status: str                  # "MUA TRONG VÙNG" | "CHỜ GIÁ"
    low: float                   # đáy vùng mua (đồng)
    high: float                  # đỉnh vùng mua = giá đặt lệnh giới hạn tối đa
    anchor: float
    anchor_source: str           # "hỗ trợ" | "VWAP20" | "SMA20" | "giá − 1 ATR"
    close: float
    vwap20: float
    value_cap: float             # trần giá trị (NaN nếu không có giá trị hợp lý)
    fair_value: float
    margin_of_safety: float
    mid: float = float("nan")    # giá mua bình quân dự kiến khi chia 2 lệnh (giữa vùng)
    notes: list[str] = field(default_factory=list)

    @property
    def entry(self) -> float:
        """Giá dùng để lập kế hoạch: giá mua bình quân dự kiến (giữa vùng)."""
        return self.mid if not pd.isna(self.mid) else self.high

    @property
    def discount(self) -> float:
        """Đỉnh vùng thấp hơn giá hiện tại bao nhiêu (âm = phải chờ giá lùi)."""
        return self.high / self.close - 1


def vwap(df: pd.DataFrame, n: int = 20) -> float:
    tail = df.tail(n)
    typical = (tail["high"] + tail["low"] + tail["close"]) / 3
    vol = tail["volume"].astype(float)
    return float((typical * vol).sum() / vol.sum()) if vol.sum() > 0 else float(typical.mean())


def entry_zone(ind: pd.DataFrame, supports: list[Level], cfg: dict,
               ticks: list[list[float]] | None = None, fair_value: float = float("nan"),
               fa_score: float = float("nan")) -> EntryZone:
    last = ind.iloc[-1]
    close, atr = float(last["close"]), float(last["atr"])
    if pd.isna(atr) or atr <= 0:
        raise ValueError("chưa đủ dữ liệu để tính ATR")
    vw = vwap(ind, int(cfg.get("vwap_sessions", 20)))
    sma20 = float(last.get("sma20", float("nan")))
    floor = close - cfg["max_pullback_atr"] * atr
    candidates = [(s.price, "hỗ trợ") for s in supports[:1]]
    candidates += [(vw, "VWAP20"), (sma20, "SMA20")]
    near = [(p, src) for p, src in candidates if not pd.isna(p) and floor <= p <= close]
    notes = []
    if near:
        anchor, source = max(near)
    else:
        anchor, source = close - atr, "giá − 1 ATR"
        notes.append("giá đã ở dưới hỗ trợ, VWAP20 và SMA20 (xu hướng yếu): vùng mua tạm tính "
                     "lùi 1 ATR, nên chờ tín hiệu tạo đáy")
    if not pd.isna(sma20) and close > sma20 + cfg["chase_atr"] * atr:
        notes.append(f"giá đã cách SMA20 hơn {cfg['chase_atr']:g} ATR: không đuổi giá, chờ lùi về vùng")
    low = anchor
    high = min(anchor + cfg["zone_atr"] * atr, close)

    mos = float(cfg["margin_of_safety"])
    if not pd.isna(fa_score) and fa_score < 50:
        mos += float(cfg.get("weak_fa_extra", 0.0))
    cap = fair_value * (1 - mos) if not pd.isna(fair_value) and fair_value > 0 else float("nan")
    status = "MUA TRONG VÙNG"
    if pd.isna(cap):
        notes.append("chưa có giá trị hợp lý: vùng mua chỉ dựa vào kỹ thuật")
    elif cap < low:
        status = "CHỜ GIÁ"
        low = high = cap
        notes.append(f"giá đang cao hơn trần giá trị (giá trị hợp lý {fair_value:,.0f} trừ biên an "
                     f"toàn {mos:.0%}): chỉ mua khi giá về ≤ {round_tick(cap, 'down', ticks):,.0f}")
    elif cap < high:
        high = cap
        notes.append("đỉnh vùng mua bị hạ xuống trần giá trị")
    low, high = round_tick(low, "up", ticks), round_tick(high, "down", ticks)
    low = min(low, high)          # vùng hẹp hơn 1 bước giá: còn một mức giá
    mid = round_tick((low + high) / 2, "nearest", ticks)
    return EntryZone(status=status, low=low, high=high, anchor=anchor, anchor_source=source,
                     close=close, vwap20=vw, value_cap=cap, fair_value=fair_value,
                     margin_of_safety=mos, mid=mid, notes=notes)
