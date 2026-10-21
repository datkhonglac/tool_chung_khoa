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

THANG MUA 3 LỚP (nâng cấp v2.1, `[entry] ladder = true`): vùng trên chỉ cách giá 0–2% vì neo vào mức
đỡ GẦN NHẤT và rộng 0,5 ATR. Thang mua chia vốn thành 3 lệnh LO ở 3 độ sâu, tính từ giá tham chiếu
ref = giá hiện tại (hoặc trần giá trị nếu giá đang cao hơn trần — trường hợp "CHỜ GIÁ"):
  Lớp 1 thăm dò (30%): điểm neo như trên (mức đỡ gần nhất trong `max_pullback_atr` ATR dưới ref);
        "CHỜ GIÁ": mức đỡ cao nhất trong 1 ATR dưới trần, không có thì đúng trần.
  Lớp 2 chính (40%): mức đỡ MẠNH nhất trong khoảng [ref − `l2_range_atr` ATR, lớp 1 − `ladder_gap_atr`
        ATR]: hỗ trợ chạm ≥ 2 lần (nhiều lần chạm nhất, rồi gần giá nhất), không có thì MA50/MA200
        cao nhất trong khoảng, không có nữa thì lớp 1 − `ladder_fallback_atr` ATR.
  Lớp 3 sâu (30%): như lớp 2 trong khoảng [ref − `l3_range_atr` ATR, lớp 2 − `ladder_gap_atr` ATR].
Mọi lớp ≤ trần giá trị. low = lớp 3, high = lớp 1, mid = giá bình quân có trọng số khi khớp đủ 3 lớp
(plan.py dùng mid làm giá vào và đặt cắt lỗ DƯỚI lớp 3). Xác suất khớp từng lớp trong 20 phiên tính
từ Monte Carlo (analysis.risk, theo giá đóng cửa nên thận trọng). Hiệu quả so với vùng cũ: lệnh
study-entry (recommend/entry_study.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from vnsa.ta.levels import Level
from vnsa.ta.plan import round_tick


@dataclass
class Tier:
    name: str                    # "Lớp 1 thăm dò" | "Lớp 2 chính" | "Lớp 3 sâu"
    price: float                 # giá đặt lệnh LO (đã làm tròn bước giá)
    weight: float                # phần vốn của lệnh (0–1)
    source: str                  # hỗ trợ (n lần chạm) | SMA50 | ... | "lớp trên − k ATR" | "trần giá trị"
    depth: float                 # so với giá hiện tại (âm = thấp hơn)
    fill_prob: float = float("nan")   # xác suất khớp trong 20 phiên (Monte Carlo), điền ở analysis.risk


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
    mid: float = float("nan")    # giá mua bình quân dự kiến (giữa vùng / bình quân thang mua)
    notes: list[str] = field(default_factory=list)
    tiers: list[Tier] = field(default_factory=list)   # thang mua 3 lớp (rỗng nếu tắt)
    stop_cap: float = float("nan")     # cắt lỗ phải thấp hơn mức này (dưới lớp sâu nhất)

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


def _pick(cands: list[tuple[float, str, int]], lo: float, hi: float) -> tuple[float, str] | None:
    """Mức đỡ mạnh nhất trong [lo, hi]: hỗ trợ ≥ 2 lần chạm (nhiều lần chạm nhất, rồi giá cao nhất),
    không có thì đường trung bình/VWAP cao nhất trong khoảng."""
    inside = [(p, src, n) for p, src, n in cands if not pd.isna(p) and lo <= p <= hi]
    strong = [c for c in inside if c[2] >= 2]
    if strong:
        p, src, n = max(strong, key=lambda c: (c[2], c[0]))
        return p, f"{src} ({n} lần chạm)"
    lines = [c for c in inside if c[2] == 0]
    if lines:
        p, src, _ = max(lines)
        return p, src
    return None


def build_ladder(ind: pd.DataFrame, supports: list[Level], cfg: dict, ref: float, top: float,
                 top_source: str, cap: float, ticks) -> tuple[list[Tier], float]:
    """(3 lớp, mức cắt lỗ tối đa) — xem docstring module."""
    last = ind.iloc[-1]
    close, atr = float(last["close"]), float(last["atr"])
    # lớp 2/3 chỉ neo vào hỗ trợ và MA50/MA200: MA20, VWAP20 là mốc ngắn hạn đã dùng cho lớp 1
    cands = [(s.price, "hỗ trợ", s.touches) for s in supports]
    for col, name in (("sma50", "MA50"), ("sma200", "MA200")):
        if col in ind:
            cands.append((float(last[col]), name, 0))
    gap, fb = cfg.get("ladder_gap_atr", 0.75) * atr, cfg.get("ladder_fallback_atr", 1.5) * atr
    p1 = top
    got = _pick(cands, ref - cfg.get("l2_range_atr", 4.0) * atr, p1 - gap)
    p2, s2 = got if got else (p1 - fb, f"lớp 1 − {cfg.get('ladder_fallback_atr', 1.5):g} ATR")
    got = _pick(cands, ref - cfg.get("l3_range_atr", 7.0) * atr, p2 - gap)
    p3, s3 = got if got else (p2 - fb, f"lớp 2 − {cfg.get('ladder_fallback_atr', 1.5):g} ATR")
    w = cfg.get("ladder_weights", [0.3, 0.4, 0.3])
    names = ("Lớp 1 thăm dò", "Lớp 2 chính", "Lớp 3 sâu")
    tiers = []
    for name, price, weight, src in zip(names, (p1, p2, p3), w, (top_source, s2, s3), strict=True):
        if not pd.isna(cap):
            price = min(price, cap)
        price = round_tick(max(price, 0.0), "down", ticks)
        tiers.append(Tier(name, price, float(weight), src, price / close - 1))
    stop_cap = tiers[-1].price - cfg.get("ladder_stop_atr", 1.0) * atr
    return tiers, stop_cap


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
    zone = EntryZone(status=status, low=low, high=high, anchor=anchor, anchor_source=source,
                     close=close, vwap20=vw, value_cap=cap, fair_value=fair_value,
                     margin_of_safety=mos, mid=mid, notes=notes)
    if cfg.get("ladder", False):
        if status == "CHỜ GIÁ":
            under = [(p, src) for p, src in [(s.price, "hỗ trợ") for s in supports]
                     + [(vw, "VWAP20"), (sma20, "MA20")] if not pd.isna(p) and cap - atr <= p <= cap]
            ref, (top, top_src) = cap, (max(under) if under else (cap, "trần giá trị"))
        else:
            ref, top, top_src = close, anchor, source
        tiers, stop_cap = build_ladder(ind, supports, cfg, ref, top, top_src, cap, ticks)
        zone.tiers, zone.stop_cap = tiers, stop_cap
        zone.high, zone.low = tiers[0].price, tiers[-1].price
        total = sum(t.weight for t in tiers)
        zone.mid = round_tick(sum(t.price * t.weight for t in tiers) / total, "nearest", ticks)
    return zone
