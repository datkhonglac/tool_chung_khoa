"""Theo dõi trong phiên và cảnh báo (nâng cấp v3, lệnh `monitor`, workflow monitor.yml).

Mỗi lần chạy (GitHub Actions ~9:47 và ~13:47 giờ VN, hoặc tự chạy trên máy):
1. Giá trong phiên của các mã đang giữ (sổ lệnh) + mã theo dõi (`[signals] daily` + `[monitor] watch`)
   từ bảng giá VCI (`vnstock_provider.price_board`).
2. Vùng mua / cắt lỗ / mục tiêu lấy từ khuyến nghị gần nhất đã lưu (bảng signals trong kho, hoặc
   history/signals.csv trên nhánh `signals` do workflow hằng ngày ghi) — không tính lại gì trong phiên.
3. Cảnh báo:
   KHẨN    mã đang giữ chạm/thủng cắt lỗ (kèm số cp bán được theo T+2), nằm sàn, hoặc chưa đặt cắt
           lỗ mà đã lỗ ≥ `[portfolio] default_risk_pct`;
   CHÚ Ý   còn ≤ `near_stop_pct` tới cắt lỗ; chạm mục tiêu 1; trạng thái giao dịch bất thường;
           mã theo dõi xếp hạng MUA/KHẢ QUAN đi vào vùng mua.
4. Gửi Telegram (`notify.send_telegram`) nếu có TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID.

Riêng tư: repo công khai nên nhật ký Actions ai cũng xem được. Với `public=True` phần in ra màn hình
bỏ hết cảnh báo về mã ĐANG GIỮ (chỉ gửi qua Telegram), chỉ in cảnh báo mã theo dõi (vốn đã công khai
trên nhánh `signals`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from vnsa.config import settings

URGENT, WATCH = "KHẨN", "CHÚ Ý"
LEVEL_ORDER = {URGENT: 0, WATCH: 1}
NORMAL_TRADING = "TRADING_ACTIVATED"


@dataclass
class Alert:
    level: str          # KHẨN | CHÚ Ý
    symbol: str
    kind: str           # cat_lo | gan_cat_lo | nam_san | lo_khong_cat | muc_tieu | trang_thai | vung_mua
    message: str
    held: bool          # liên quan mã đang giữ (riêng tư)


@dataclass
class MonitorResult:
    as_of: pd.Timestamp | None
    trading_day: bool
    alerts: list[Alert] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)      # mã không lấy được giá
    held: list[str] = field(default_factory=list)
    watched: list[str] = field(default_factory=list)


def cfg() -> dict:
    try:
        s = settings()
    except Exception:  # noqa: BLE001 - test
        s = {}
    out = {"watch": [], "near_stop_pct": 0.02, "zone_actions": ["MUA", "KHẢ QUAN"],
           "max_zone_age_days": 7, "signals_ref": "origin/signals"}
    out.update(s.get("monitor", {}))
    out.setdefault("daily", s.get("signals", {}).get("daily", []))
    out.setdefault("default_risk_pct", s.get("portfolio", {}).get("default_risk_pct", 0.10))
    return out


def _fmt(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:,.0f}".replace(",", ".")


def latest_zones(signals: pd.DataFrame, as_of: pd.Timestamp, max_age_days: int = 7) -> pd.DataFrame:
    """Khuyến nghị gần nhất của mỗi mã (index = mã), bỏ bản cũ hơn `max_age_days` ngày lịch."""
    if signals is None or signals.empty:
        return pd.DataFrame()
    s = signals.assign(as_of=pd.to_datetime(signals["as_of"]))
    s = s[s["as_of"] >= pd.Timestamp(as_of).normalize() - pd.Timedelta(days=max_age_days)]
    if s.empty:
        return pd.DataFrame()
    return s.sort_values("as_of").groupby("symbol").tail(1).set_index("symbol")


def check(positions: dict, board: pd.DataFrame, zones: pd.DataFrame, watch: list[str],
          c: dict | None = None) -> list[Alert]:
    """Danh sách cảnh báo (KHẨN trước). positions: {mã: portfolio.Position}; board: price_board
    chuẩn hoá; zones: latest_zones; watch: mã theo dõi."""
    c = {**cfg(), **(c or {})}
    near = float(c["near_stop_pct"])
    by = board.set_index("symbol") if not board.empty else pd.DataFrame()
    alerts: list[Alert] = []

    def z(sym, key):
        if zones is None or zones.empty or sym not in zones.index or key not in zones.columns:
            return float("nan")
        v = zones.at[sym, key]
        return float(v) if not isinstance(v, str) and not pd.isna(v) else float("nan")

    def status_alert(sym, q, held):
        ts, ss = str(q.get("trading_status") or ""), str(q.get("security_status") or "")
        bad = [x for x in (ts if ts and ts not in (NORMAL_TRADING, "None", "nan") else "",
                           f"mã trạng thái {ss}" if ss not in ("", "N", "None", "nan") else "") if x]
        if bad:
            alerts.append(Alert(WATCH, sym, "trang_thai",
                                f"{sym}: trạng thái giao dịch bất thường ({', '.join(bad)}) — kiểm tra "
                                "tin công bố (diện cảnh báo/kiểm soát/tạm ngừng?)", held))

    held = [s for s, p in positions.items() if p.quantity > 0]
    for sym in held:
        p = positions[sym]
        if sym not in by.index:
            continue
        q = by.loc[sym]
        px, floor = float(q["price"]), float(q.get("floor", float("nan")))
        pnl = px / p.avg_cost - 1 if p.avg_cost > 0 else float("nan")
        sell = (f"bán được {_fmt(p.sellable)} cp" if p.sellable > 0
                else "CHƯA bán được (cổ phiếu chưa về, T+2)")
        stop = p.stop if not pd.isna(p.stop) and p.stop > 0 else float("nan")
        stop_src = ""
        if pd.isna(stop) and not pd.isna(z(sym, "stop")):
            stop, stop_src = z(sym, "stop"), " (theo khuyến nghị gần nhất)"
        if not pd.isna(floor) and floor > 0 and px <= floor:
            alerts.append(Alert(URGENT, sym, "nam_san",
                                f"{sym} NẰM SÀN {_fmt(px)} ({pnl:+.1%} so với giá vốn) — {sell}", True))
        if not pd.isna(stop):
            if px <= stop:
                alerts.append(Alert(URGENT, sym, "cat_lo",
                                    f"{sym} {_fmt(px)} đã chạm/thủng cắt lỗ {_fmt(stop)}{stop_src} "
                                    f"({pnl:+.1%} so với giá vốn) — {sell}", True))
            elif px <= stop * (1 + near):
                alerts.append(Alert(WATCH, sym, "gan_cat_lo",
                                    f"{sym} {_fmt(px)} còn {px / stop - 1:.1%} tới cắt lỗ "
                                    f"{_fmt(stop)}{stop_src}", True))
        elif not pd.isna(pnl) and pnl <= -float(c["default_risk_pct"]):
            alerts.append(Alert(URGENT, sym, "lo_khong_cat",
                                f"{sym} {_fmt(px)} lỗ {pnl:.1%} so với giá vốn mà CHƯA đặt cắt lỗ "
                                f"(portfolio stop {sym} GIÁ) — {sell}", True))
        tgt = z(sym, "target")
        if not pd.isna(tgt) and px >= tgt:
            alerts.append(Alert(WATCH, sym, "muc_tieu",
                                f"{sym} {_fmt(px)} chạm mục tiêu 1 ({_fmt(tgt)}) — cân nhắc chốt một "
                                f"phần, nâng cắt lỗ ({pnl:+.1%} so với giá vốn)", True))
        status_alert(sym, q, True)

    actions = set(c["zone_actions"])
    for sym in watch:
        if sym in held or sym not in by.index:
            continue
        q = by.loc[sym]
        px = float(q["price"])
        lo, hi = z(sym, "buy_low"), z(sym, "buy_high")
        act = zones.at[sym, "action"] if zones is not None and not zones.empty and \
            sym in zones.index and "action" in zones.columns else None
        if act in actions and not pd.isna(lo) and not pd.isna(hi) and lo <= px <= hi:
            stop = z(sym, "stop")
            alerts.append(Alert(WATCH, sym, "vung_mua",
                                f"{sym} {_fmt(px)} vào vùng mua {_fmt(lo)}–{_fmt(hi)} (xếp hạng {act}"
                                f"{', cắt lỗ ' + _fmt(stop) if not pd.isna(stop) else ''}) — chạy "
                                f"analyze {sym} --refresh để xem lại trước khi đặt lệnh", False))
        status_alert(sym, q, False)
    return sorted(alerts, key=lambda a: (LEVEL_ORDER.get(a.level, 9), a.symbol))


def format_alerts(res: MonitorResult, public: bool = False) -> str:
    """Tin nhắn cảnh báo. public=True: bỏ cảnh báo mã đang giữ (in ra nhật ký công khai)."""
    when = f"{res.as_of:%d/%m/%Y %H:%M}" if res.as_of is not None else ""
    if not res.trading_day:
        return f"VNSA theo dõi {when}: hôm nay không có phiên giao dịch — bỏ qua."
    alerts = [a for a in res.alerts if not (public and a.held)]
    head = f"VNSA theo dõi {when}"
    if public:
        hidden = sum(a.held for a in res.alerts)
        head += f" · mã theo dõi: {len(res.watched)} (cảnh báo mã đang giữ chỉ gửi Telegram: {hidden})"
    else:
        head += f" · đang giữ {len(res.held)} mã, theo dõi {len(res.watched)} mã"
    lines = [head]
    if not alerts:
        lines.append("Không có cảnh báo.")
    for level in (URGENT, WATCH):
        group = [a for a in alerts if a.level == level]
        if group:
            lines.append(f"\n[{level}]")
            lines += [f"- {a.message}" for a in group]
    missing = [s for s in res.missing if not (public and s in res.held)]
    if missing:
        lines.append(f"\nKhông lấy được giá: {', '.join(missing)}")
    lines.append("\nGiá trong phiên từ bảng giá VCI; vùng/cắt lỗ theo khuyến nghị đã lưu và sổ lệnh. "
                 "Chỉ là nhắc việc — tự kiểm tra trước khi đặt lệnh.")
    return "\n".join(lines)


def _load_signals(ref: str) -> pd.DataFrame:
    """Bảng signals trong kho; kho trống/không có thì đọc history/signals.csv trên nhánh `ref`."""
    from vnsa.config import path

    try:
        if path("warehouse").exists():
            from vnsa.data.store import Store

            with Store(path("warehouse")) as store:
                df = store.signals()
            if not df.empty:
                return df
    except Exception:  # noqa: BLE001 - kho bận/lỗi: dùng bản trên Git
        pass
    from vnsa.history import _read_git

    df = _read_git(ref, "signals.csv")
    return df if df is not None else pd.DataFrame()


def _actions(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Sự kiện quyền cho mã đang giữ: từ kho, thiếu thì hỏi vnstock (Company.events)."""
    out: dict[str, pd.DataFrame] = {}
    try:
        from vnsa.portfolio import corporate_actions

        out = {s: d for s, d in corporate_actions(symbols).items() if d is not None and not d.empty}
    except Exception:  # noqa: BLE001
        pass
    missing = [s for s in symbols if s not in out]
    if missing:
        try:
            from vnsa.data.providers.vnstock_provider import VnstockProvider

            prov = VnstockProvider()
            for s in missing:
                try:
                    out[s] = prov.dividends(s)
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            pass
    return out


def run(now: pd.Timestamp | None = None, board_fn=None, trades: pd.DataFrame | None = None,
        signals: pd.DataFrame | None = None, actions: dict | None = None, extra_watch=None,
        force: bool = False) -> MonitorResult:
    """Một lượt theo dõi. Các tham số *_fn / bảng để test; mặc định đọc sổ lệnh, signals, vnstock."""
    from vnsa import portfolio as pf

    c = cfg()
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").tz_localize(None)
    trades = pf.load() if trades is None else trades
    held_pos = pf.positions(trades, now, None, None) if len(trades) else {}
    held = sorted(s for s, p in held_pos.items() if p.quantity > 0)
    if held and actions is None:
        actions = _actions(held)
    pos = pf.positions(trades, now, None, actions) if len(trades) else {}
    watch = list(dict.fromkeys([*c["daily"], *c["watch"], *(extra_watch or [])]))
    watch = [s.upper() for s in watch]
    res = MonitorResult(now, True, held=held, watched=[s for s in watch if s not in held])
    if now.weekday() >= 5 and not force:
        res.trading_day = False
        return res
    symbols = list(dict.fromkeys(held + watch))
    if not symbols:
        return res
    if board_fn is None:
        from vnsa.data.providers.vnstock_provider import price_board as board_fn
    board = board_fn(symbols)
    dates = pd.to_datetime(board["trading_date"], errors="coerce").dropna() if len(board) else []
    if len(dates) and dates.max().normalize() < now.normalize() and not force:
        res.trading_day = False           # bảng giá vẫn là phiên trước: nghỉ lễ
        return res
    res.missing = [s for s in symbols if board.empty or s not in set(board["symbol"])]
    sig = _load_signals(c["signals_ref"]) if signals is None else signals
    zones = latest_zones(sig, now, int(c["max_zone_age_days"]))
    res.alerts = check(pos, board, zones, watch, c)
    return res
