"""Sổ lệnh và danh mục đang nắm giữ (nâng cấp v2.0, bước 5.1).

Sổ lệnh là dữ liệu KHÔNG tải lại được, nên lưu ở dạng văn bản: một file CSV ghi thêm từng dòng
(`[data] trades`, mặc định data/portfolio/trades.csv). File này nằm trong .gitignore: repo đang để
CÔNG KHAI, đẩy sổ lệnh lên Git (kể cả nhánh `signals`) là công khai danh mục thật. Sao lưu bằng cách
tải file về máy hoặc đặt biến môi trường VNSA_TRADES trỏ tới thư mục của một repo RIÊNG TƯ.

Mỗi dòng là MỘT lệnh (không phải một mã), để theo dõi đúng T+2 khi mua một mã nhiều lần:
    date, symbol, side (MUA | BÁN | CẮT LỖ), quantity, price, fee, tax, note
- MUA/BÁN: giá khớp (đồng), phí môi giới và thuế bán tính sẵn theo [backtest] fee/sell_tax nếu không
  nhập (ghi đè được bằng số thật trên sao kê).
- CẮT LỖ: đặt/đổi mức cắt lỗ của mã (quantity 0, price = mức cắt lỗ).

Quy ước tính:
- Giá vốn BÌNH QUÂN gia quyền (như các công ty chứng khoán VN), gồm phí mua. Bán làm giảm số lượng
  theo giá vốn bình quân; lãi/lỗ đã chốt = tiền bán sau phí, thuế − giá vốn phần đã bán.
- T+2: cổ phiếu mua phiên T về tài khoản chiều phiên T+2 và bán được từ phiên đó. Số phiên đếm theo
  lịch giao dịch = các ngày có giá VNINDEX trong kho, sau ngày cuối cùng thì đếm ngày thường (thứ
  Hai–Sáu; chưa trừ ngày lễ chưa có trong kho).
- Lệnh bán vượt số lượng bán được tại ngày đó bị từ chối.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vnsa.config import path, settings

COLUMNS = ["date", "symbol", "side", "quantity", "price", "fee", "tax", "note"]
BUY, SELL, STOP = "MUA", "BÁN", "CẮT LỖ"
SIDES = (BUY, SELL, STOP)


def ledger_path() -> Path:
    return path("trades")


def load(file: Path | None = None) -> pd.DataFrame:
    """Sổ lệnh (sắp theo ngày, giữ thứ tự nhập trong cùng ngày); rỗng nếu chưa có file."""
    file = Path(file or ledger_path())
    if not file.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(file, dtype={"symbol": str, "side": str, "note": str}, keep_default_na=False)
    df["date"] = pd.to_datetime(df["date"])
    for c in ("quantity", "price", "fee", "tax"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["_order"] = range(len(df))
    return df.sort_values(["date", "_order"]).drop(columns="_order").reset_index(drop=True)


def sessions_after(start: pd.Timestamp, end: pd.Timestamp, calendar: pd.DatetimeIndex) -> int:
    """Số phiên giao dịch trong (start, end]: theo lịch `calendar` (ngày có giá VNINDEX), phần sau
    ngày cuối của lịch đếm ngày thường."""
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if end <= start:
        return 0
    cal = pd.DatetimeIndex(calendar).normalize() if len(calendar) else pd.DatetimeIndex([])
    last = cal.max() if len(cal) else start
    n = int(((cal > start) & (cal <= min(end, last))).sum())
    tail_from = max(start, last)
    if end > tail_from:
        n += int(np.busday_count((tail_from + pd.Timedelta(days=1)).date(),
                                 (end + pd.Timedelta(days=1)).date()))
    return n


@dataclass
class Position:
    symbol: str
    quantity: float
    sellable: float               # đã qua T+2 tại ngày tính
    avg_cost: float               # giá vốn bình quân / cp, gồm phí mua
    cost: float                   # tổng giá vốn phần đang giữ
    realized: float               # lãi/lỗ đã chốt (sau phí, thuế)
    first_buy: pd.Timestamp | None
    last_buy: pd.Timestamp | None
    stop: float                   # mức cắt lỗ đang đặt (NaN nếu chưa)


def positions(trades: pd.DataFrame, as_of: pd.Timestamp | None = None,
              calendar: pd.DatetimeIndex | None = None) -> dict[str, Position]:
    """Vị thế từng mã tại ngày `as_of` (mặc định hôm nay), kể cả mã đã bán hết (để có lãi đã chốt).
    ValueError nếu có lệnh bán vượt số lượng bán được lúc đó."""
    as_of = pd.Timestamp(as_of or pd.Timestamp.today()).normalize()
    cal = calendar if calendar is not None else pd.DatetimeIndex([])
    out: dict[str, Position] = {}
    for sym, rows in trades[trades["date"] <= as_of].groupby("symbol", sort=True):
        lots: list[tuple[pd.Timestamp, float]] = []      # (ngày mua, số lượng còn) để tính T+2
        qty = cost = realized = 0.0
        stop, first, last = float("nan"), None, None
        for r in rows.itertuples(index=False):
            if r.side == BUY:
                qty += r.quantity
                cost += r.quantity * r.price + r.fee
                lots.append((r.date, r.quantity))
                first, last = first or r.date, r.date
            elif r.side == SELL:
                ready = sum(q for d, q in lots if sessions_after(d, r.date, cal) >= 2)
                if r.quantity > ready + 1e-9:
                    raise ValueError(f"{sym} ngày {r.date:%d/%m/%Y}: bán {r.quantity:,.0f} cp nhưng "
                                     f"chỉ có {ready:,.0f} cp đã về tài khoản (T+2)")
                avg = cost / qty
                realized += r.quantity * r.price - r.fee - r.tax - avg * r.quantity
                cost -= avg * r.quantity
                qty -= r.quantity
                left = r.quantity                          # trừ vào lô đã về, lô cũ trước
                for i, (d, q) in enumerate(lots):
                    if left <= 0:
                        break
                    if sessions_after(d, r.date, cal) >= 2:
                        take = min(q, left)
                        lots[i], left = (d, q - take), left - take
                lots = [(d, q) for d, q in lots if q > 1e-9]
                if qty <= 1e-9:
                    qty, cost, stop = 0.0, 0.0, float("nan")
            elif r.side == STOP:
                stop = r.price
        sellable = sum(q for d, q in lots if sessions_after(d, as_of, cal) >= 2)
        out[sym] = Position(sym, qty, sellable, cost / qty if qty > 0 else float("nan"), cost,
                            realized, first, last, stop)
    return out


def _costs(side: str, quantity: float, price: float, fee: float | None,
           tax: float | None) -> tuple[float, float]:
    bt = settings()["backtest"]
    value = quantity * price
    fee = value * bt.get("fee", 0.0015) if fee is None else fee
    tax = (value * bt.get("sell_tax", 0.001) if side == SELL else 0.0) if tax is None else tax
    return round(fee), round(tax)


def add(side: str, symbol: str, quantity: float, price: float, date=None,
        fee: float | None = None, tax: float | None = None, note: str = "",
        file: Path | None = None, calendar: pd.DatetimeIndex | None = None) -> dict:
    """Ghi một lệnh vào sổ (kiểm tra trước: số lượng > 0, bán không vượt số bán được). Trả dòng đã ghi."""
    side = side.upper()
    if side not in SIDES:
        raise ValueError(f"loại lệnh phải là một trong {', '.join(SIDES)}")
    symbol = symbol.upper().strip()
    if price <= 0:
        raise ValueError("giá phải > 0 (đơn vị đồng, ví dụ 98500)")
    if price < 1000 and side != STOP:
        raise ValueError(f"giá {price} quá nhỏ — nhập bằng đồng (98500), không phải nghìn đồng (98.5)")
    if side == STOP:
        quantity, fee, tax = 0, 0, 0
    elif quantity <= 0:
        raise ValueError("số lượng phải > 0")
    date = pd.Timestamp(date or pd.Timestamp.today()).normalize()
    fee, tax = (0, 0) if side == STOP else _costs(side, quantity, price, fee, tax)
    row = {"date": date, "symbol": symbol, "side": side, "quantity": float(quantity),
           "price": float(price), "fee": float(fee), "tax": float(tax), "note": note}
    file = Path(file or ledger_path())
    trades = load(file)
    if side == STOP and symbol not in set(trades.loc[trades["side"] == BUY, "symbol"]):
        raise ValueError(f"chưa có lệnh mua {symbol} trong sổ")
    trial = (pd.DataFrame([row]) if trades.empty
             else pd.concat([trades, pd.DataFrame([row])], ignore_index=True))
    positions(trial, max(date, pd.Timestamp.today().normalize()), calendar)   # báo lỗi T+2 nếu có
    file.parent.mkdir(parents=True, exist_ok=True)
    new = not file.exists()
    with file.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        w.writerow([date.date().isoformat(), symbol, side, f"{quantity:g}", f"{price:g}",
                    f"{fee:g}", f"{tax:g}", note])
    return row


def undo(file: Path | None = None) -> str | None:
    """Xoá dòng nhập CUỐI CÙNG của sổ (sửa lỗi nhập nhầm). Trả nội dung dòng đã xoá."""
    file = Path(file or ledger_path())
    if not file.exists():
        return None
    lines = file.read_text(encoding="utf-8").splitlines()
    if len(lines) <= 1:
        return None
    file.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    return lines[-1]


def status(trades: pd.DataFrame, prices: dict[str, float], groups: dict[str, str],
           capital: float, as_of: pd.Timestamp | None = None,
           calendar: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Bảng vị thế đang giữ: số lượng, bán được (T+2), giá vốn, giá hiện tại, lãi/lỗ tạm tính,
    tỷ trọng / vốn, khoảng cách tới cắt lỗ, nhóm ngành."""
    rows = []
    for sym, p in positions(trades, as_of, calendar).items():
        if p.quantity <= 0:
            continue
        px = prices.get(sym, float("nan"))
        value = p.quantity * px
        rows.append({"Mã": sym, "Nhóm ngành": groups.get(sym, ""), "SL": p.quantity,
                     "Bán được": p.sellable, "Giá vốn": p.avg_cost, "Giá": px,
                     "Giá trị": value, "Lãi/lỗ": value - p.cost,
                     "Lãi/lỗ %": value / p.cost - 1 if p.cost > 0 else float("nan"),
                     "Tỷ trọng": value / capital if capital > 0 else float("nan"),
                     "Cắt lỗ": p.stop,
                     "Cách cắt lỗ": px / p.stop - 1 if p.stop > 0 else float("nan"),
                     "Đã chốt": p.realized})
    return pd.DataFrame(rows)


def realized_total(trades: pd.DataFrame, as_of=None, calendar=None) -> float:
    return float(sum(p.realized for p in positions(trades, as_of, calendar).values()))


def market_context(symbols: list[str]) -> tuple[pd.DatetimeIndex, dict[str, float], dict[str, str]]:
    """(lịch giao dịch = ngày có giá VNINDEX, giá đóng cửa gần nhất, nhóm ngành) từ kho."""
    from vnsa.analysis import macro_mapping
    from vnsa.data.store import Store
    from vnsa.screen.groups import industry_group

    mapping = macro_mapping()
    prices, groups = {}, {}
    with Store(path("warehouse")) as store:
        idx = store.prices(settings()["valuation"]["index"])
        for s in symbols:
            p = store.prices(s)
            if not p.empty:
                prices[s] = float(p["close"].iloc[-1])
            groups[s] = industry_group(store.company(s), mapping)
    cal = pd.DatetimeIndex(pd.to_datetime(idx["date"])) if not idx.empty else pd.DatetimeIndex([])
    return cal, prices, groups


def correlation_factor(rho: float, threshold: float = 0.60, max_cut: float = 0.50) -> float:
    """Hệ số ngân sách rủi ro theo tương quan cực đại ρ với các mã đang giữ (bước 5.2):
    ρ ≤ ngưỡng: 1; trên ngưỡng giảm tuyến tính, ρ = 1 thì còn 1 − max_cut."""
    if pd.isna(rho) or rho <= threshold:
        return 1.0
    return 1.0 - min((rho - threshold) / (1.0 - threshold), 1.0) * max_cut


def max_correlation(returns: pd.Series, held: dict[str, pd.Series],
                    sessions: int = 120) -> tuple[float, str | None]:
    """(ρ cực đại, mã) giữa lợi suất ngày của mã mới và từng mã đang giữ, `sessions` phiên chung
    gần nhất (cần ≥ 60 phiên chung)."""
    best, who = float("nan"), None
    for sym, r in held.items():
        both = pd.concat([returns, r], axis=1, join="inner").dropna().tail(sessions)
        if len(both) < 60:
            continue
        rho = float(both.iloc[:, 0].corr(both.iloc[:, 1]))
        if not pd.isna(rho) and (pd.isna(best) or rho > best):
            best, who = rho, sym
    return best, who


def _returns(prices: pd.DataFrame) -> pd.Series:
    s = prices.set_index(pd.to_datetime(prices["date"]))["close"].astype(float)
    return s.pct_change()


def sizing_limits(symbol: str, prices: pd.DataFrame, capital: float, cfg: dict | None = None,
                  trades: pd.DataFrame | None = None, context=None
                  ) -> tuple[tuple[float, str] | None, list[tuple[float, str]]]:
    """Giới hạn khối lượng của lệnh mua mới `symbol` theo danh mục đang giữ:
    (risk_adjust cho trade_plan hoặc None, extra_caps). Sổ lệnh trống thì không đổi gì.
    context: (lịch, giá gần nhất, nhóm ngành, {mã: bảng giá}) để test; mặc định đọc kho."""
    cfg = {**settings().get("portfolio", {}), **(cfg or {})}
    trades = load() if trades is None else trades
    if trades.empty:
        return None, []
    symbol = symbol.upper()
    syms = sorted(set(trades["symbol"]) | {symbol})
    if context is None:
        cal, last, groups = market_context(syms)
        from vnsa.data.store import Store

        with Store(path("warehouse")) as store:
            hist = {s: store.prices(s) for s in syms if s != symbol}
    else:
        cal, last, groups, hist = context
    held = {s: p for s, p in positions(trades, calendar=cal).items() if p.quantity > 0}
    if not held:
        return None, []
    others = {s: _returns(hist[s]) for s in held if s != symbol and s in hist and len(hist[s])}
    rho, who = max_correlation(_returns(prices), others, int(cfg.get("corr_sessions", 120)))
    factor = correlation_factor(rho, cfg.get("corr_threshold", 0.60), cfg.get("corr_max_cut", 0.50))
    adjust = None
    if factor < 1:
        adjust = (factor, f"tương quan {rho:.2f} với {who} đang giữ: rủi ro lệnh này còn "
                          f"{factor:.0%} (tránh đa dạng hoá giả)")
    caps = []
    group = groups.get(symbol, "")
    max_pct = cfg.get("max_sector_pct", 0.30)
    if group and max_pct:
        used = sum(p.quantity * last.get(s, p.avg_cost) for s, p in held.items()
                   if groups.get(s) == group)
        caps.append((max(0.0, capital * max_pct - used),
                     f"trần nhóm {group} {max_pct:.0%} vốn (đang giữ {used / capital:.0%})"))
    return adjust, caps
