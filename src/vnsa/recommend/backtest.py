"""Backtest đơn giản cho chiến lược theo điểm TA (chỉ mua, một mã, dùng toàn bộ vốn).

Vì sao chỉ TA: BCTC miễn phí chỉ có 4 năm và công bố chậm sau kỳ, dùng điểm FA cho từng ngày quá
khứ dễ "nhìn trước" số liệu chưa công bố. Điểm TA tính từ giá đến đúng ngày đó nên kiểm tra được.

Quy tắc (tham số ở mục [backtest] trong settings), không nhìn trước:
- Tín hiệu lấy ở giá đóng cửa phiên t, khớp lệnh ở giá MỞ CỬA phiên t+1.
- Mua khi điểm TA ≥ `entry_score` và đang không giữ cổ phiếu.
- Cắt lỗ cố định = giá mua − `atr_stop` × ATR14 (ATR của phiên tín hiệu), làm tròn xuống theo bước
  giá. Nếu giá thấp nhất trong
  phiên chạm cắt lỗ thì bán ở giá cắt lỗ (hoặc giá mở cửa nếu mở cửa đã thấp hơn).
- Bán khi điểm TA < `exit_score` (khớp mở cửa phiên sau).
- T+2: chỉ được bán từ phiên thứ 2 sau ngày mua (cổ phiếu về chiều T+2).
- Phí: `fee` mỗi chiều mua/bán (môi giới), thêm `sell_tax` khi bán (thuế 0,1% giá trị bán).
Kết quả so với mua và nắm giữ trong cùng giai đoạn (tính từ phiên đầu có điểm TA).

Công thức mới (đợt 2) — truyền `regime` (ta/regime.regime_series, chỉ dùng VNINDEX đến phiên tín
hiệu) và/hoặc `capital` trong cfg:
- VNINDEX "giảm" ở phiên tín hiệu: cần điểm TA ≥ entry_score + `bear_entry_extra` và chỉ dùng
  `bear_position` phần vốn (như kế hoạch giao dịch giảm rủi ro/lệnh 50%).
- Thanh khoản: giá trị lệnh ≤ `max_adv_pct` × GTGD bình quân 20 phiên; với vốn `capital` đồng,
  phần vốn dùng = min(…, max_adv_pct × ADV20 / capital); dưới `min_position` thì bỏ lệnh.
Phần vốn không dùng giữ tiền mặt (lãi 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vnsa.ta.plan import round_tick

TRADING_DAYS = 252


@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry: float
    exit_date: pd.Timestamp
    exit: float
    reason: str               # "điểm TA" | "cắt lỗ" | "cuối kỳ"

    @property
    def ret(self) -> float:
        return self.exit / self.entry - 1


@dataclass
class BacktestResult:
    equity: pd.Series         # giá trị tài sản theo ngày (vốn ban đầu = 1)
    benchmark: pd.Series      # mua và nắm giữ (= 1 ở phiên đầu)
    trades: list[Trade] = field(default_factory=list)
    exposure: float = 0.0     # tỷ lệ số phiên đang giữ cổ phiếu

    @staticmethod
    def _stats(curve: pd.Series) -> dict:
        years = len(curve) / TRADING_DAYS
        total = curve.iloc[-1] / curve.iloc[0] - 1
        cagr = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else float("nan")
        dd = (curve / curve.cummax() - 1).min()
        daily = curve.pct_change().dropna()
        sharpe = (daily.mean() / daily.std() * np.sqrt(TRADING_DAYS)
                  if daily.std() > 0 else float("nan"))
        return {"total": float(total), "cagr": float(cagr), "max_drawdown": float(dd),
                "sharpe": float(sharpe)}

    @property
    def stats(self) -> dict:
        s = self._stats(self.equity)
        rets = [t.ret for t in self.trades]
        s.update(trades=len(rets), win_rate=float(np.mean([r > 0 for r in rets])) if rets else
                 float("nan"), avg_trade=float(np.mean(rets)) if rets else float("nan"),
                 exposure=self.exposure)
        return s

    @property
    def benchmark_stats(self) -> dict:
        return self._stats(self.benchmark)


def backtest(ind: pd.DataFrame, score: pd.Series, cfg: dict,
             regime: pd.Series | None = None) -> BacktestResult:
    """ind: bảng giá + chỉ báo (cần open, low, close, atr; volume nếu chặn thanh khoản);
    score: điểm TA từng phiên; regime: trạng thái VNINDEX theo ngày (tuỳ chọn)."""
    df = ind.assign(score=score.values).reset_index(drop=True)
    if regime is not None and len(regime):
        r = regime.copy()
        r.index = pd.to_datetime(r.index)
        df["regime"] = r.reindex(pd.to_datetime(df["date"]), method="ffill").to_numpy()
    else:
        df["regime"] = None
    capital = cfg.get("capital")
    df["adv"] = ((df["close"] * df["volume"]).rolling(20).mean()
                 if capital and "volume" in df else np.nan)
    start = (df["score"].notna() & df["atr"].notna()).idxmax() if df["score"].notna().any() \
        else None
    if start is None or len(df) - start < 30:
        raise ValueError("chưa đủ dữ liệu để backtest (cần điểm TA cho ít nhất 30 phiên)")
    df = df.iloc[start:].reset_index(drop=True)
    fee, tax = cfg["fee"], cfg["sell_tax"]
    cash, shares = 1.0, 0.0
    entry_i, entry_px, stop = -1, 0.0, 0.0
    trades: list[Trade] = []
    equity = np.empty(len(df))
    pending: str | None = None        # lệnh chờ khớp ở giá mở cửa phiên này
    frac = 1.0                        # phần vốn dùng cho lệnh mua đang chờ

    def size(i: int) -> tuple[bool, float]:
        """(có mua không, phần vốn) theo điểm TA, trạng thái VNINDEX và thanh khoản phiên i."""
        need, f = cfg["entry_score"], 1.0
        if df.at[i, "regime"] == "giảm":
            need += cfg.get("bear_entry_extra", 0)
            f = cfg.get("bear_position", 1.0)
        if not df.at[i, "score"] >= need:          # NaN = chưa đủ dữ liệu
            return False, 0.0
        if capital and not pd.isna(df.at[i, "adv"]):
            nav = cash + shares * df.at[i, "close"]
            f = min(f, cfg.get("max_adv_pct", 0.1) * df.at[i, "adv"] / (capital * nav))
        return f >= cfg.get("min_position", 0.0), f

    def sell(i: int, px: float, reason: str) -> None:
        nonlocal cash, shares
        cash += shares * px * (1 - fee - tax)
        trades.append(Trade(df.at[entry_i, "date"], entry_px, df.at[i, "date"], px, reason))
        shares = 0.0

    for i in range(len(df)):
        o, lo, c = df.at[i, "open"], df.at[i, "low"], df.at[i, "close"]
        if pending == "buy" and shares == 0:
            spend = cash * frac
            shares, cash = spend * (1 - fee) / o, cash - spend
            entry_i, entry_px = i, o
            stop = round_tick(o - cfg["atr_stop"] * df.at[i - 1, "atr"], "down", cfg.get("ticks"))
        elif pending == "sell" and shares > 0:
            sell(i, o, "điểm TA")
        pending = None
        if shares > 0 and i - entry_i >= 2 and lo <= stop:      # T+2 mới bán được
            sell(i, min(o, stop), "cắt lỗ")
        s = df.at[i, "score"]
        if shares == 0 and i + 1 < len(df):
            ok, frac = size(i)
            pending = "buy" if ok else None
        elif shares > 0 and s < cfg["exit_score"] and i + 1 - entry_i >= 2:
            pending = "sell"
        equity[i] = cash + shares * c
    if shares > 0:
        last = len(df) - 1
        trades.append(Trade(df.at[entry_i, "date"], entry_px, df.at[last, "date"],
                            df.at[last, "close"], "cuối kỳ"))
    dates = pd.to_datetime(df["date"])
    held = pd.Series(equity, index=dates)
    bench = pd.Series(df["close"].to_numpy() / df.at[0, "close"], index=dates)
    in_market = np.zeros(len(df), dtype=bool)
    for t in trades:
        in_market |= ((dates >= t.entry_date) & (dates < t.exit_date)).to_numpy()
    return BacktestResult(equity=held, benchmark=bench, trades=trades,
                          exposure=float(in_market.mean()))
