"""Kế hoạch giao dịch mua (chỉ chiều mua, vì nhà đầu tư cá nhân VN không bán khống được).

Quy tắc (tham số ở mục [plan] trong settings):
- Giá vào = giá đóng cửa phiên gần nhất (làm tròn theo bước giá HOSE).
- Cắt lỗ = giá vào − `atr_stop` × ATR14. Nếu có hỗ trợ dưới giá vào mà cắt lỗ đặt dưới nó
  `support_buffer_atr` × ATR (để không bị quét ngay trên hỗ trợ) cách giá vào từ `min_stop_atr` đến
  `max_stop_atr` × ATR, dùng hỗ trợ gần nhất như vậy. Cắt lỗ sát hơn `min_stop_atr` × ATR dễ bị dao
  động bình thường trong phiên quét mất. Làm tròn XUỐNG theo bước giá.
- Rủi ro mỗi cp R = giá vào − cắt lỗ. Mục tiêu: các kháng cự phía trên có lãi/lỗ ≥ 1R (tối đa 2),
  thiếu thì bổ sung giá vào + k × R với k trong `reward_multiples`.
- Khối lượng: rủi ro tối đa = vốn × `risk_per_trade` × hệ số thị trường (thị trường chung giảm thì
  nhỏ hơn); số cp = rủi ro tối đa / R, làm tròn xuống lô 100 cp, rồi chặn bởi: số vốn (không margin),
  trần tỷ trọng `max_position_pct` × vốn, và `max_adv_pct` × giá trị giao dịch bình quân 20 phiên
  (mua nhiều hơn thế thì khó khớp và khó thoát hàng).
- Danh mục đang giữ (sổ lệnh, portfolio.py): tương quan cao với mã đang giữ thì giảm ngân sách rủi
  ro; trần tổng giá trị mỗi nhóm ngành ([portfolio], xem portfolio.sizing_limits).
- Lưu ý T+2: cổ phiếu mua hôm nay chiều T+2 mới về tài khoản, trước đó không bán (cắt lỗ) được.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from vnsa.ta.levels import Level

DEFAULT_TICKS = [[0, 10], [10_000, 50], [50_000, 100]]   # HOSE: giá ≥ ngưỡng -> bước giá


def tick_size(price: float, ticks: list[list[float]] | None = None) -> float:
    size = 10.0
    for threshold, step in ticks or DEFAULT_TICKS:
        if price >= threshold:
            size = step
    return float(size)


def round_tick(price: float, mode: str = "nearest", ticks: list[list[float]] | None = None) -> float:
    step = tick_size(price, ticks)
    f = {"nearest": round, "down": math.floor, "up": math.ceil}[mode]
    return float(f(price / step) * step)


@dataclass
class Target:
    price: float
    reward_risk: float
    source: str        # "kháng cự" hoặc "k×R"


@dataclass
class TradePlan:
    entry: float
    stop: float
    stop_source: str
    atr: float
    targets: list[Target]
    shares: int
    capital: float
    risk_budget: float
    notes: list[str] = field(default_factory=list)
    zone: object = None          # EntryZone (ta/entry.py) nếu giá vào lấy từ vùng mua khuyến nghị

    @property
    def risk_per_share(self) -> float:
        return self.entry - self.stop

    @property
    def position_value(self) -> float:
        return self.shares * self.entry

    @property
    def max_loss(self) -> float:
        return self.shares * self.risk_per_share

    @property
    def stop_pct(self) -> float:
        return self.stop / self.entry - 1


def trade_plan(close: float, atr: float, supports: list[Level], resistances: list[Level],
               cfg: dict, ticks: list[list[float]] | None = None, adv_value: float = float("nan"),
               risk_scale: float = 1.0, risk_adjust: tuple[float, str] | None = None,
               extra_caps: list[tuple[float, str]] | None = None,
               max_stop: float | None = None) -> TradePlan:
    """risk_adjust: (hệ số ≤ 1, lý do) giảm thêm ngân sách rủi ro (vd tương quan cao với mã đang giữ);
    extra_caps: [(giá trị tối đa đồng, lý do)] chặn thêm khối lượng (vd trần nhóm ngành)."""
    if pd.isna(atr) or atr <= 0:
        raise ValueError("chưa đủ dữ liệu để tính ATR")
    entry = round_tick(close, "nearest", ticks)
    stop, source = entry - cfg["atr_stop"] * atr, f"{cfg['atr_stop']:g} × ATR"
    buffer, lo, hi = (cfg["support_buffer_atr"] * atr, cfg["max_stop_atr"] * atr,
                      cfg.get("min_stop_atr", 1.0) * atr)
    near = [s for s in supports if s.price < entry and hi <= entry - (s.price - buffer) <= lo]
    if near:
        stop = near[0].price - buffer
        source = f"dưới hỗ trợ {near[0].price:,.0f}"
    if max_stop is not None and not pd.isna(max_stop) and stop > max_stop:
        # thang mua: cắt lỗ phải nằm dưới lớp sâu nhất (không thì lớp 3 vừa khớp đã chạm cắt lỗ)
        stop, source = max_stop, "dưới lớp mua sâu nhất"
        below = [s for s in supports if s.price < max_stop + buffer and
                 max_stop - lo <= s.price - buffer]
        if below:
            stop, source = below[0].price - buffer, f"dưới hỗ trợ {below[0].price:,.0f} (dưới lớp 3)"
    stop = round_tick(stop, "down", ticks)
    risk = entry - stop
    notes = []

    targets = [Target(round_tick(r.price, "nearest", ticks), (r.price - entry) / risk, "kháng cự")
               for r in resistances if r.price - entry >= risk][:2]
    if not resistances:
        notes.append("giá đang ở vùng cao nhất của giai đoạn xét, chưa có kháng cự phía trên")
    for k in cfg["reward_multiples"]:
        if len(targets) >= 2:
            break
        price = round_tick(entry + k * risk, "nearest", ticks)
        if all(abs(price - t.price) > tick_size(price, ticks) for t in targets):
            targets.append(Target(price, k, f"{k:g}R"))
    targets.sort(key=lambda t: t.price)
    if targets and targets[0].reward_risk < cfg.get("min_reward_risk", 2):
        notes.append(f"mục tiêu đầu chỉ {targets[0].reward_risk:.1f}R, thấp hơn mức tối thiểu "
                     f"{cfg.get('min_reward_risk', 2):g}R")

    capital = float(cfg["capital"])
    budget = capital * cfg["risk_per_trade"] * risk_scale
    if risk_scale < 1:
        notes.append(f"thị trường chung đang giảm: rủi ro mỗi lệnh còn {risk_scale:.0%} mức thường")
    if risk_adjust is not None and risk_adjust[0] < 1:
        budget *= max(risk_adjust[0], 0.0)
        notes.append(risk_adjust[1])
    lot = int(cfg.get("lot_size", 100))
    shares = int(budget / risk // lot * lot)
    caps = [(capital, "số vốn (không dùng margin)")]
    if cfg.get("max_position_pct"):
        caps.append((capital * cfg["max_position_pct"],
                     f"trần tỷ trọng {cfg['max_position_pct']:.0%} vốn mỗi mã"))
    if cfg.get("max_adv_pct") and not pd.isna(adv_value) and adv_value > 0:
        caps.append((adv_value * cfg["max_adv_pct"],
                     (f"thanh khoản ({cfg['max_adv_pct']:.0%} GTGD bình quân 20 phiên "
                      f"{adv_value / 1e9:,.1f} tỷ)")))
    caps += list(extra_caps or [])
    for value, why in caps:
        limit = int(value / entry // lot * lot)
        if shares > limit:
            shares = limit
            notes.append(f"khối lượng bị giới hạn bởi {why}")
    if shares == 0:
        notes.append("vốn/rủi ro cho phép không đủ mua 1 lô 100 cp")
    notes.append("T+2: cổ phiếu mua hôm nay chiều T+2 mới về, trước đó chưa bán cắt lỗ được")
    return TradePlan(entry=entry, stop=stop, stop_source=source, atr=atr, targets=targets,
                     shares=shares, capital=capital, risk_budget=budget, notes=notes)
