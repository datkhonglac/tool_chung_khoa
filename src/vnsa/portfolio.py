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

Sự kiện quyền (nâng cấp v2.1, bảng `dividends` trong kho, theo ngày GDKHQ): người đang giữ cổ phiếu
hết phiên TRƯỚC ngày GDKHQ được hưởng.
- Cổ tức tiền: ghi nhận tiền nhận = số cp × cổ tức/cp × (1 − thuế TNCN `[portfolio] dividend_tax`,
  5%); giá vốn không đổi; mức cắt lỗ đặt trước ngày GDKHQ hạ đúng bằng cổ tức/cp (giá thị trường điều
  chỉnh giảm bằng chừng ấy — không thì báo chạm cắt lỗ giả).
- Cổ tức/thưởng bằng cổ phiếu tỷ lệ r: nhận thêm floor(số cp × r) cp (phần lẻ bị bỏ như thực tế),
  tổng giá vốn không đổi nên giá vốn/cp giảm; mức cắt lỗ chia (1 + r). Cổ phiếu mới coi như bán được
  sau `[portfolio] bonus_sellable_sessions` phiên (thời gian chờ niêm yết bổ sung, ước lượng).
- Chưa xử lý: quyền mua phát hành thêm (cần nhập lệnh MUA thật khi thực hiện quyền).
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


def _cfg() -> dict:
    try:
        return settings().get("portfolio", {})
    except Exception:  # noqa: BLE001 - test không có settings đầy đủ
        return {}


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
    stop: float                   # mức cắt lỗ đang đặt (NaN nếu chưa), đã điều chỉnh theo sự kiện quyền
    dividends: float = 0.0        # cổ tức tiền đã nhận (sau thuế) từ khi bắt đầu giữ
    events: list = None           # mô tả các sự kiện quyền đã áp dụng


def positions(trades: pd.DataFrame, as_of: pd.Timestamp | None = None,
              calendar: pd.DatetimeIndex | None = None,
              actions: dict[str, pd.DataFrame] | None = None) -> dict[str, Position]:
    """Vị thế từng mã tại ngày `as_of` (mặc định hôm nay), kể cả mã đã bán hết (để có lãi đã chốt).
    actions: {mã: bảng cổ tức (exright_date, cash_per_share, stock_ratio)} — sự kiện quyền.
    ValueError nếu có lệnh bán vượt số lượng bán được lúc đó."""
    as_of = pd.Timestamp(as_of or pd.Timestamp.today()).normalize()
    cal = calendar if calendar is not None else pd.DatetimeIndex([])
    cfg = _cfg()
    tax_rate = float(cfg.get("dividend_tax", 0.05))
    wait_bonus = int(cfg.get("bonus_sellable_sessions", 15))
    out: dict[str, Position] = {}
    for sym, rows in trades[trades["date"] <= as_of].groupby("symbol", sort=True):
        # lô: (ngày, số lượng còn, số phiên chờ trước khi bán được)
        lots: list[tuple[pd.Timestamp, float, int]] = []
        qty = cost = realized = divs = 0.0
        stop, first, last = float("nan"), None, None
        notes: list[str] = []
        ev = (actions or {}).get(sym)
        events = []
        if ev is not None and not ev.empty and len(rows):
            start = rows["date"].min()
            ev = ev[(ev["exright_date"] > start) & (ev["exright_date"] <= as_of)]
            events = [("ACT", r.exright_date, r) for r in ev.itertuples(index=False)]
        # sự kiện quyền ngày D xử lý TRƯỚC lệnh cùng ngày D (hưởng quyền = giữ hết phiên D − 1)
        seq = sorted([("ACT", d, r) for _, d, r in events]
                     + [("TRD", r.date, r) for r in rows.itertuples(index=False)],
                     key=lambda x: (x[1], 0 if x[0] == "ACT" else 1))
        for kind, day, r in seq:
            if kind == "ACT":
                if qty <= 0:
                    continue
                cash = getattr(r, "cash_per_share", float("nan"))
                ratio = getattr(r, "stock_ratio", float("nan"))
                if not pd.isna(cash) and cash > 0:
                    got = qty * cash * (1 - tax_rate)
                    divs += got
                    if not pd.isna(stop):
                        stop -= cash
                    notes.append(f"{day:%d/%m/%Y}: cổ tức tiền {cash:,.0f}đ/cp → nhận {got:,.0f}đ")
                if not pd.isna(ratio) and ratio > 0:
                    new = float(np.floor(qty * ratio))
                    if new > 0:
                        qty += new
                        lots.append((day, new, wait_bonus))
                    if not pd.isna(stop):
                        stop /= 1 + ratio
                    notes.append(f"{day:%d/%m/%Y}: cổ tức/thưởng cổ phiếu {ratio:.0%} → thêm "
                                 f"{new:,.0f} cp")
                continue
            if r.side == BUY:
                qty += r.quantity
                cost += r.quantity * r.price + r.fee
                lots.append((r.date, r.quantity, 2))
                first, last = first or r.date, r.date
            elif r.side == SELL:
                ready = sum(q for d, q, w in lots if sessions_after(d, r.date, cal) >= w)
                if r.quantity > ready + 1e-9:
                    raise ValueError(f"{sym} ngày {r.date:%d/%m/%Y}: bán {r.quantity:,.0f} cp nhưng "
                                     f"chỉ có {ready:,.0f} cp đã về tài khoản (T+2)")
                avg = cost / qty
                realized += r.quantity * r.price - r.fee - r.tax - avg * r.quantity
                cost -= avg * r.quantity
                qty -= r.quantity
                left = r.quantity                          # trừ vào lô đã về, lô cũ trước
                for i, (d, q, w) in enumerate(lots):
                    if left <= 0:
                        break
                    if sessions_after(d, r.date, cal) >= w:
                        take = min(q, left)
                        lots[i], left = (d, q - take, w), left - take
                lots = [(d, q, w) for d, q, w in lots if q > 1e-9]
                if qty <= 1e-9:
                    qty, cost, stop = 0.0, 0.0, float("nan")
            elif r.side == STOP:
                stop = r.price
        sellable = sum(q for d, q, w in lots if sessions_after(d, as_of, cal) >= w)
        out[sym] = Position(sym, qty, sellable, cost / qty if qty > 0 else float("nan"), cost,
                            realized, first, last, stop, divs, notes)
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
        file: Path | None = None, calendar: pd.DatetimeIndex | None = None,
        actions: dict[str, pd.DataFrame] | None = None) -> dict:
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
    positions(trial, max(date, pd.Timestamp.today().normalize()), calendar, actions)   # lỗi T+2
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
           calendar: pd.DatetimeIndex | None = None,
           actions: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Bảng vị thế đang giữ: số lượng, bán được (T+2), giá vốn, giá hiện tại, lãi/lỗ tạm tính,
    tỷ trọng / vốn, khoảng cách tới cắt lỗ, cổ tức tiền đã nhận, nhóm ngành."""
    rows = []
    for sym, p in positions(trades, as_of, calendar, actions).items():
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
                     "Cổ tức nhận": p.dividends, "Đã chốt": p.realized})
    return pd.DataFrame(rows)


def realized_total(trades: pd.DataFrame, as_of=None, calendar=None, actions=None) -> float:
    """Lãi/lỗ đã chốt + cổ tức tiền đã nhận (kể cả mã đã bán hết)."""
    return float(sum(p.realized + p.dividends
                     for p in positions(trades, as_of, calendar, actions).values()))


def corporate_actions(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """{mã: bảng cổ tức trong kho} cho các mã trong sổ lệnh."""
    from vnsa.data.store import Store

    with Store(path("warehouse")) as store:
        return {s: store.dividends(s) for s in symbols}


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
    acts = corporate_actions(syms) if context is None else None
    held = {s: p for s, p in positions(trades, calendar=cal, actions=acts).items()
            if p.quantity > 0}
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


# ---- sao lưu sổ lệnh sang repo RIÊNG TƯ (v2.1) ----------------------------------------------------
# Repo chính đang CÔNG KHAI nên sổ lệnh nằm trong .gitignore; Codespace bị xoá là mất. Lệnh
# `portfolio backup` / `portfolio restore` chép data/portfolio/trades.csv sang/từ một repo riêng tư.
# Cấu hình bằng biến môi trường (đặt ở GitHub → Settings → Codespaces → Secrets):
#   VNSA_BACKUP_REPO  = "chu_so_huu/ten_repo" (repo RIÊNG TƯ, tạo trước, có thể trống)
#   VNSA_BACKUP_TOKEN = token GitHub fine-grained chỉ có quyền Contents: Read and write với repo đó
#                       (không có thì dùng quyền git sẵn có của máy)
# Từ chối sao lưu nếu repo đích là repo công khai hoặc chính repo tool.

BACKUP_FILE = "trades.csv"


class BackupError(RuntimeError):
    pass


def _backup_target(repo: str | None = None) -> tuple[str, str | None, str | None]:
    """(URL git, "owner/repo" nếu là GitHub, token)."""
    import os
    import re

    repo = (repo or os.environ.get("VNSA_BACKUP_REPO") or _cfg().get("backup_repo") or "").strip()
    if not repo:
        raise BackupError("chưa cấu hình repo sao lưu: đặt biến môi trường VNSA_BACKUP_REPO = "
                          "\"ten_ban/ten_repo_rieng_tu\" (xem README, mục Sổ lệnh)")
    token = os.environ.get("VNSA_BACKUP_TOKEN") or None
    m = re.fullmatch(r"(?:https://github\.com/)?([\w.-]+/[\w.-]+?)(?:\.git)?/?", repo)
    if m and "/" in repo and not Path(repo).exists():
        slug = m.group(1)
        auth = f"x-access-token:{token}@" if token else ""
        return f"https://{auth}github.com/{slug}.git", slug, token
    return repo, None, token                       # đường dẫn/URL git khác (vd repo nội bộ, test)


def _check_private(slug: str, token: str | None) -> None:
    """Chặn sao lưu lên repo công khai hoặc chính repo của tool (GitHub API)."""
    import json
    import subprocess
    import urllib.error
    import urllib.request

    origin = subprocess.run(["git", "config", "--get", "remote.origin.url"], capture_output=True,
                            text=True, cwd=Path(__file__).resolve().parents[2]).stdout
    if slug.lower() in origin.lower():
        raise BackupError(f"{slug} là repo của tool (công khai) — chọn một repo RIÊNG TƯ khác")
    req = urllib.request.Request(f"https://api.github.com/repos/{slug}",
                                 headers={"Accept": "application/vnd.github+json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            info = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:          # không thấy: repo riêng tư (không có token) hoặc chưa tạo
            return
        raise BackupError(f"không kiểm tra được repo {slug}: HTTP {e.code}") from e
    except OSError:
        return                     # không có mạng tới API: để git tự báo lỗi quyền
    if not info.get("private", False):
        raise BackupError(f"repo {slug} đang CÔNG KHAI — không sao lưu sổ lệnh lên đó. Đổi repo sang "
                          "Private (Settings → Danger Zone → Change visibility) rồi chạy lại.")


def _git(*args, cwd=None) -> str:
    import subprocess

    import re

    res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        msg = (res.stderr or res.stdout).strip().splitlines()
        text = re.sub(r"x-access-token:[^@\s]+@", "x-access-token:***@", msg[-1] if msg else "lỗi")
        raise BackupError(f"git {args[0]}: {text}")   # không in token ra màn hình
    return res.stdout


def _clone(url: str, where: Path) -> None:
    import shutil

    shutil.rmtree(where, ignore_errors=True)
    where.parent.mkdir(parents=True, exist_ok=True)
    _git("clone", "-q", "--depth", "1", url, str(where))


def backup(file: Path | None = None, repo: str | None = None) -> str:
    """Chép sổ lệnh lên repo riêng tư (một commit nếu có thay đổi). Trả mô tả kết quả."""
    import shutil
    import tempfile

    file = Path(file or ledger_path())
    if not file.exists():
        raise BackupError(f"chưa có sổ lệnh {file}")
    url, slug, token = _backup_target(repo)
    if slug:
        _check_private(slug, token)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "backup"
        _clone(url, work)
        shutil.copyfile(file, work / BACKUP_FILE)
        _git("add", BACKUP_FILE, cwd=work)
        if not _git("status", "--porcelain", cwd=work).strip():
            return "sổ lệnh trên repo sao lưu đã giống bản trên máy — không cần sao lưu"
        _git("-c", "user.name=vnsa", "-c", "user.email=vnsa@localhost", "commit", "-q", "-m",
             f"Sao lưu sổ lệnh {pd.Timestamp.now():%d/%m/%Y %H:%M}", cwd=work)
        _git("push", "-q", "origin", "HEAD", cwd=work)
    return f"đã sao lưu {file.name} ({len(load(file))} lệnh) lên {slug or 'repo sao lưu'}"


def restore(file: Path | None = None, repo: str | None = None, force: bool = False) -> str:
    """Lấy sổ lệnh từ repo riêng tư về máy. Không ghi đè sổ đang có trừ khi force (bản cũ được giữ
    thành trades.csv.bak)."""
    import shutil
    import tempfile

    file = Path(file or ledger_path())
    url, slug, token = _backup_target(repo)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "backup"
        _clone(url, work)
        src = work / BACKUP_FILE
        if not src.exists():
            raise BackupError(f"repo {slug or url} chưa có {BACKUP_FILE}")
        if file.exists() and not force:
            if file.read_bytes() == src.read_bytes():
                return "sổ lệnh trên máy đã giống bản sao lưu"
            raise BackupError(f"đã có sổ lệnh {file} khác bản sao lưu — chạy lại với --force để thay "
                              "(bản trên máy được giữ thành trades.csv.bak)")
        file.parent.mkdir(parents=True, exist_ok=True)
        if file.exists():
            shutil.copyfile(file, file.with_suffix(".csv.bak"))
        shutil.copyfile(src, file)
    return f"đã lấy sổ lệnh ({len(load(file))} lệnh) từ {slug or 'repo sao lưu'} về {file}"


def auto_backup() -> str | None:
    """Sao lưu sau mỗi lệnh ghi nếu đã cấu hình VNSA_BACKUP_REPO; lỗi chỉ cảnh báo, không chặn."""
    import os

    if not (os.environ.get("VNSA_BACKUP_REPO") or _cfg().get("backup_repo")):
        return None
    try:
        return backup()
    except BackupError as e:
        return f"chưa sao lưu được: {e}"
