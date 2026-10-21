"""Quy tắc thời điểm và máy mô phỏng chung để kiểm định (đợt 4).

Mọi quy tắc dùng chung một cách khớp lệnh sát thực tế Việt Nam, không nhìn trước:
- Tín hiệu lấy ở giá đóng cửa phiên t, đặt lệnh khớp giá MỞ CỬA phiên t+1.
- Không mua được khi phiên đó mở cửa ở giá TRẦN (trắng bên bán); không bán được khi cả phiên nằm ở
  giá SÀN (cao nhất = sàn, trắng bên mua) — lệnh bán (kể cả cắt lỗ) dời sang phiên sau.
  Trần/sàn = giá đóng cửa phiên trước × (1 ± biên độ của sàn: HOSE 7%, HNX 10%, UPCoM 15%).
- T+2: chỉ bán từ phiên thứ 2 sau ngày mua.
- Cắt lỗ: "fixed" = giá mua − k × ATR14 (cố định); "trail" = cắt lỗ động (chandelier): đóng cửa cao
  nhất từ khi mua − k × ATR14, chỉ dịch lên. Giá thấp nhất chạm cắt lỗ thì bán ở giá cắt lỗ (hoặc giá
  mở cửa nếu mở cửa đã thấp hơn).
- Phí môi giới mỗi chiều + thuế bán; phần vốn không dùng giữ tiền mặt (lãi 0).

Các quy tắc (`rule_set`):
- ta          : điểm TA ≥ 65 thì mua, < 45 thì bán, cắt lỗ cố định 2 ATR (công thức ban đầu).
- ta_vni      : như trên; VNINDEX "giảm" thì cần điểm ≥ 75 và chỉ dùng 50% vốn (đợt 2).
- trend       : xu hướng tăng (giá > MA200, MA50 > MA200, MA50 cao hơn 20 phiên trước) VÀ giá đóng
                cửa vượt đỉnh đóng cửa 20 phiên trước (xác nhận đà, tránh mua lại ngay sau khi vừa
                bị cắt lỗ) thì mua; đóng cửa dưới MA200 thì bán; cắt lỗ động k ATR.
- breakout    : giá đóng cửa vượt đỉnh cao nhất 55 phiên trước, khối lượng > 1,5 × TB 20 phiên và
                giá > MA200 thì mua; bán bằng cắt lỗ động k ATR (hoặc thủng MA200).
- trend_vni, breakout_vni: như trên, chỉ MUA MỚI khi VNINDEX không ở trạng thái "giảm".
- vcp_pullback: mua nhịp điều chỉnh trong nền giá chặt của xu hướng tăng (đề xuất nâng cấp v2.0,
                bước 3.3). Mua khi CẢ BỐN điều kiện: (1) giá > MA200 và MA50 > MA200; (2) nền chặt
                (độ nén bb_width_pct < 0,40) HOẶC giá cách MA20 không quá 1 ATR; (3) 45 ≤ RSI14 ≤ 62;
                (4) phiên xác nhận: đóng cửa > mở cửa và khối lượng > 1,2 × TB 20 phiên. Bán: cắt lỗ
                động k ATR, hoặc đóng cửa dưới MA50 với khối lượng > 1,5 × TB 20 phiên.
                Tham số đặt TRƯỚC khi chạy kiểm định, không chỉnh theo kết quả.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vnsa.recommend.backtest import BacktestResult, Trade

LOCK_TOL = 0.002        # sai số khi so giá với trần/sàn (giá điều chỉnh, bước giá)


@dataclass
class Rule:
    name: str
    entry: pd.Series            # bool theo phiên: mua ở mở cửa phiên sau
    exit: pd.Series             # bool theo phiên: bán ở mở cửa phiên sau
    size: pd.Series             # phần vốn dùng khi mua (0–1)
    stop: str = "fixed"         # fixed | trail
    atr_k: float = 2.0


def simulate(df: pd.DataFrame, rule: Rule, band: float, fee: float, tax: float,
             start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> BacktestResult:
    """Mô phỏng một quy tắc trên bảng giá + chỉ báo (date, open, high, low, close, atr). Chỉ giao
    dịch trong [start, end] (chỉ báo vẫn tính từ toàn bộ lịch sử trước đó — không nhìn trước)."""
    d = df.reset_index(drop=True)
    dates = pd.to_datetime(d["date"])
    window = np.ones(len(d), dtype=bool)
    if start is not None:
        window &= (dates >= start).to_numpy()
    if end is not None:
        window &= (dates <= end).to_numpy()
    idx = np.flatnonzero(window)
    if len(idx) < 30:
        raise ValueError("chưa đủ dữ liệu trong giai đoạn kiểm định (cần ≥ 30 phiên)")
    o, h, lo, c = (d[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = d["atr"].to_numpy(float)
    entry, exit_ = rule.entry.to_numpy(bool), rule.exit.to_numpy(bool)
    size = rule.size.to_numpy(float)

    cash, shares, entry_i, entry_px, stop, peak = 1.0, 0.0, -1, 0.0, 0.0, 0.0
    pend_buy, pend_sell, frac = False, False, 0.0
    trades: list[Trade] = []
    equity = np.empty(len(idx))

    def sell(i: int, px: float, reason: str) -> None:
        nonlocal cash, shares
        cash += shares * px * (1 - fee - tax)
        trades.append(Trade(dates[entry_i], entry_px, dates[i], px, reason))
        shares = 0.0

    for j, i in enumerate(idx):
        prev_c = c[i - 1] if i > 0 else c[i]
        ceiling, floor = prev_c * (1 + band), prev_c * (1 - band)
        floor_locked = h[i] <= floor * (1 + LOCK_TOL)
        if pend_buy and shares == 0:
            if o[i] < ceiling * (1 - LOCK_TOL) and not np.isnan(atr[i - 1]):
                spend = cash * frac
                shares, cash = spend * (1 - fee) / o[i], cash - spend
                entry_i, entry_px, peak = i, o[i], o[i]
                stop = o[i] - rule.atr_k * atr[i - 1]
            pend_buy = False
        if pend_sell and shares > 0 and i - entry_i >= 2 and not floor_locked:
            sell(i, o[i], "tín hiệu bán")
            pend_sell = False
        if shares > 0 and i - entry_i >= 2 and lo[i] <= stop and not floor_locked:
            sell(i, min(o[i], stop), "cắt lỗ")
            pend_sell = False
        if shares > 0 and rule.stop == "trail" and not np.isnan(atr[i]):
            peak = max(peak, c[i])
            stop = max(stop, peak - rule.atr_k * atr[i])
        last = j == len(idx) - 1
        if shares == 0 and not last and entry[i] and size[i] > 0:
            pend_buy, frac = True, min(float(size[i]), 1.0)
        elif shares > 0 and exit_[i]:
            pend_sell = True
        equity[j] = cash + shares * c[i]
    if shares > 0:
        trades.append(Trade(dates[entry_i], entry_px, dates[idx[-1]], c[idx[-1]], "cuối kỳ"))
    sel = dates[idx]
    held = pd.Series(equity, index=sel)
    bench = pd.Series(c[idx] / c[idx[0]], index=sel)
    in_market = np.zeros(len(idx), dtype=bool)
    for t in trades:
        in_market |= ((sel >= t.entry_date) & (sel < t.exit_date)).to_numpy()
    return BacktestResult(equity=held, benchmark=bench, trades=trades,
                          exposure=float(in_market.mean()),
                          open_stop=float(stop) if trades and trades[-1].reason == "cuối kỳ"
                          else float("nan"))


def rule_state(ind: pd.DataFrame, score: pd.Series, regime: pd.Series | None, name: str,
               atr_k: float, cfg: dict, band: float, sessions: int = 750) -> dict | None:
    """Trạng thái HIỆN TẠI của quy tắc `name` (mô phỏng `sessions` phiên gần nhất, cùng cách khớp
    lệnh như kiểm định): đang giữ hay không, từ ngày nào, cắt lỗ hiện tại, tín hiệu mua mới hôm nay.
    Dùng cho trục "Thời điểm" khi `[recommend] timing_rule` là quy tắc đã qua kiểm định."""
    rule = rule_set(ind, score, regime, atr_k, cfg)[name]
    d = ind.reset_index(drop=True)
    start = pd.Timestamp(d["date"].iloc[max(0, len(d) - sessions)])
    try:
        res = simulate(d, rule, band, cfg["fee"], cfg["sell_tax"], start)
    except ValueError:
        return None
    last = res.trades[-1] if res.trades else None
    holding = last is not None and last.reason == "cuối kỳ"
    return {"holding": holding, "since": last.entry_date if holding else None,
            "stop": res.open_stop, "signal_today": bool(rule.entry.iloc[-1]),
            "rule": name}


def _regime_on(df: pd.DataFrame, regime: pd.Series | None) -> pd.Series:
    """Trạng thái VNINDEX theo từng phiên của mã (ffill); không có thì "trung tính"."""
    if regime is None or regime.empty:
        return pd.Series("trung tính", index=df.index)
    r = regime.copy()
    r.index = pd.to_datetime(r.index)
    out = r.reindex(pd.to_datetime(df["date"]), method="ffill").fillna("trung tính")
    return pd.Series(out.to_numpy(), index=df.index)


def rule_set(ind: pd.DataFrame, score: pd.Series, regime: pd.Series | None, atr_k: float,
             cfg: dict) -> dict[str, Rule]:
    """Các quy tắc thời điểm trên cùng bảng chỉ báo. cfg = mục [backtest]."""
    d = ind.reset_index(drop=True)
    s = pd.Series(score.to_numpy(float), index=d.index)
    reg = _regime_on(d, regime)
    bear = reg == "giảm"
    one = pd.Series(1.0, index=d.index)
    entry_score, exit_score = cfg["entry_score"], cfg["exit_score"]
    sma50, sma200 = d["sma50"], d["sma200"]
    up_trend = ((d["close"] > sma200) & (sma50 > sma200) & (sma50 > sma50.shift(20))
                & (d["close"] > d["close"].shift(1).rolling(20).max()))
    below_200 = d["close"] < sma200
    hh55 = d["high"].shift(1).rolling(55).max()
    breakout = ((d["close"] > hh55) & (d["volume"] > 1.5 * d["vol_ma20"])
                & (d["close"] > sma200))
    sma20 = d["sma20"] if "sma20" in d else d["close"].rolling(20).mean()
    tight = (d["bb_width_pct"] < 0.40) if "bb_width_pct" in d else pd.Series(False, index=d.index)
    vcp = ((d["close"] > sma200) & (sma50 > sma200)
           & (tight | ((d["close"] - sma20).abs() <= 1.0 * d["atr"]))
           & d["rsi"].between(45, 62)
           & (d["close"] > d["open"]) & (d["volume"] > 1.2 * d["vol_ma20"]))
    vcp_exit = (d["close"] < sma50) & (d["volume"] > 1.5 * d["vol_ma20"])
    extra = cfg.get("bear_entry_extra", 10)
    bear_size = cfg.get("bear_position", 0.5)
    return {
        "ta": Rule("ta", s >= entry_score, s < exit_score, one, "fixed", cfg.get("atr_stop", 2.0)),
        "ta_vni": Rule("ta_vni", (s >= entry_score) & (~bear | (s >= entry_score + extra)),
                       s < exit_score, one.where(~bear, bear_size), "fixed",
                       cfg.get("atr_stop", 2.0)),
        "trend": Rule("trend", up_trend.fillna(False), below_200.fillna(False), one, "trail", atr_k),
        "breakout": Rule("breakout", breakout.fillna(False), below_200.fillna(False), one, "trail",
                         atr_k),
        "trend_vni": Rule("trend_vni", (up_trend & ~bear).fillna(False), below_200.fillna(False),
                          one, "trail", atr_k),
        "breakout_vni": Rule("breakout_vni", (breakout & ~bear).fillna(False),
                             below_200.fillna(False), one, "trail", atr_k),
        "vcp_pullback": Rule("vcp_pullback", vcp.fillna(False), vcp_exit.fillna(False), one,
                             "trail", atr_k),
    }


NAMES = {"ta": "Điểm TA (hiện tại)", "ta_vni": "Điểm TA + VNINDEX",
         "trend": "Theo xu hướng MA200", "breakout": "Phá nền 55 phiên",
         "trend_vni": "Xu hướng + VNINDEX", "breakout_vni": "Phá nền + VNINDEX",
         "vcp_pullback": "Nền chặt + điều chỉnh (VCP)", "hold": "Mua & giữ"}
