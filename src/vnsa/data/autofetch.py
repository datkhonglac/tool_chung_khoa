"""Tự lấy dữ liệu cho một mã bất kỳ trên HOSE, HNX, UPCoM khi người dùng hỏi tới.

Thứ tự cho mỗi mã (tham số ở mục [auto] trong settings):
1. Danh sách niêm yết (vnstock/VCI): sàn, ngành, loại doanh nghiệp của mọi cổ phiếu. Tải lại khi
   cũ hơn `listing_max_age_days`. Sàn quyết định biên độ và bước giá của mã.
2. Giá ngày (CafeF; mã CafeF không có, thường là HNX/UPCoM, thì dùng vnstock/VCI và lần sau ưu
   tiên nguồn đó): chưa có thì tải `years` năm. Đã có nhưng thiếu phiên mới thì chỉ tải từ
   (phiên cuối − `overlap_days`). Nếu giá đóng cửa các phiên chồng lấn lệch hơn `adjust_tolerance`
   thì nguồn đã điều chỉnh lại lịch sử (cổ tức/chia tách), nên tải lại toàn bộ cho khớp.
3. Chỉ số thị trường (VNINDEX, để tính beta): như bước 2.
4. BCTC năm và BCTC quý (vnstock/VCI; bản miễn phí 4 quý gần nhất, dùng tính TTM): chưa có hoặc
   cũ hơn `bctc_max_age_days` thì tải lại.
5. Hồ sơ doanh nghiệp (số cổ phiếu lưu hành, sở hữu nước ngoài, free float, GTGD bình quân) và
   lịch sử cổ tức: cũ hơn `profile_max_age_days` thì tải lại.

Lỗi ở bước 1, 3, 4 không chặn phân tích (ghi chú lại); thiếu giá thì không phân tích được.
Phiên "đã có giá" tính theo giờ Việt Nam: trước `market_close_hour` giờ chưa có giá phiên hôm nay.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from vnsa import market
from vnsa.config import path, settings
from vnsa.data.cleaning import validate_prices
from vnsa.data.store import Store

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")


@dataclass
class Step:
    what: str        # "niêm yết" | "giá" | chỉ số | "BCTC"
    status: str      # "ok" | "skip" | "error"
    detail: str

    def __str__(self) -> str:
        mark = {"ok": "✓", "skip": "·", "error": "✗"}[self.status]
        return f"  {mark} {self.what}: {self.detail}"


@dataclass
class UpdateReport:
    symbol: str
    steps: list[Step] = field(default_factory=list)
    has_prices: bool = False

    @property
    def ok(self) -> bool:
        """Đủ giá để phân tích (BCTC có thể thiếu)."""
        return self.has_prices

    @property
    def errors(self) -> list[str]:
        return [f"{s.what}: {s.detail}" for s in self.steps if s.status == "error"]

    def __str__(self) -> str:
        return "\n".join(str(s) for s in self.steps)


def last_session(now: datetime, close_hour: int) -> pd.Timestamp:
    """Phiên gần nhất lẽ ra đã có giá (bỏ thứ Bảy, Chủ nhật; ngày lễ thì lần tải chỉ không có gì mới)."""
    d = now.date()
    if now.hour < close_hour:
        d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return pd.Timestamp(d)


def _fmt(d) -> str:
    return pd.Timestamp(d).strftime("%d/%m/%Y")


class Updater:
    """Cập nhật kho cho từng mã. Nguồn dữ liệu truyền vào được (test không cần mạng)."""

    def __init__(self, price_provider=None, vnstock_provider=None, cfg: dict | None = None,
                 now: Callable[[], datetime] | None = None, log: Callable[[str], None] | None = None,
                 macro_provider=None):
        from vnsa.data.providers.cafef_provider import CafeFProvider
        from vnsa.data.providers.vnstock_provider import VnstockProvider

        self.cfg = cfg or settings()["auto"]
        self.cafef = price_provider or CafeFProvider(pause=self.cfg["pause"], log=log)
        self.vn = vnstock_provider or VnstockProvider()
        self.now = now or (lambda: datetime.now(VN_TZ))
        self.index = settings()["valuation"]["index"]
        self._listing_step: Step | None = None     # mỗi lượt chỉ kiểm tra danh sách một lần
        self._index_step: Step | None = None
        self._macro_step: Step | None = None       # số vĩ mô dùng chung mọi mã: kiểm tra một lần
        self.macro_provider = macro_provider

    # ---- các bước ----------------------------------------------------------------------------
    def _listing(self, store: Store, refresh: bool) -> Step:
        if self._listing_step:
            return self._listing_step
        loaded = store.companies_loaded_at()
        age = (pd.Timestamp.now() - loaded).days if loaded is not None else None
        # kho cũ chưa có ngành ICB cấp 3–4 (vnstock đổi định dạng 10/2026): tải lại, mỗi ngày tối đa
        # một lần (nguồn vẫn không trả cấp 4 thì không tải lại liên tục)
        if (age is not None and age < self.cfg["listing_max_age_days"] and not refresh
                and (store.has_icb_detail() or age < 1)):
            step = Step("niêm yết", "skip", f"danh sách sàn/ngành tải {age} ngày trước")
        else:
            try:
                df = self.vn.listing()
                n = store.write_companies(df, self.vn.name)
                counts = df["exchange"].value_counts()
                step = Step("niêm yết", "ok", f"{n} cổ phiếu (" + ", ".join(
                    f"{ex} {counts.get(ex, 0)}" for ex in market.EXCHANGES) + ")")
            except Exception as e:  # noqa: BLE001 - thiếu danh sách vẫn phân tích được
                step = Step("niêm yết", "error", f"không tải được danh sách sàn/ngành ({e}); "
                            "coi như sàn " + settings()["market"].get("default_exchange", "HOSE"))
        self._listing_step = step
        return step

    def _prices(self, store: Store, sym: str, label: str, full: bool) -> Step:
        end_ts = last_session(self.now(), self.cfg["market_close_hour"])
        today = self.now().date()
        last = store.last_price_date(sym)
        if last is not None and last >= end_ts and not full:
            return Step(label, "skip", f"đã mới (phiên cuối {_fmt(last)})")
        prof = store.company(sym) or {}
        exchange = prof.get("exchange") if prof.get("exchange") in market.EXCHANGES else None
        band = 1.0 if sym == self.index else market.band(exchange=exchange or settings()["market"]
                                                         .get("default_exchange", "HOSE"))
        self._order = [self.cafef, self.vn]
        if store.price_source(sym) == self.vn.name and not full:
            self._order.reverse()                       # mã này lần trước lấy được từ vnstock
        try:
            if last is None or full:
                df = self._download(sym, today - timedelta(days=round(365.25 * self.cfg["years"])),
                                    today)
                note = f"tải {len(df)} phiên ({_fmt(df['date'].iloc[0])} → {_fmt(df['date'].iloc[-1])})"
                write = store.write_prices
            else:
                start = (last - timedelta(days=self.cfg["overlap_days"])).date()
                df = self._download(sym, start, today)
                old = store.prices(sym, start=start.isoformat())
                both = old.merge(df, on="date", suffixes=("_old", ""))
                drift = (both["close"] / both["close_old"] - 1).abs().max() if len(both) else 0.0
                if drift > self.cfg["adjust_tolerance"]:
                    df = self._download(
                        sym, today - timedelta(days=round(365.25 * self.cfg["years"])), today)
                    note = (f"giá lịch sử đã được nguồn điều chỉnh (lệch {drift:.1%}, thường do "
                            f"cổ tức/chia tách) → tải lại toàn bộ {len(df)} phiên")
                    write = store.write_prices
                else:
                    new = int((df["date"] > last).sum())
                    note = (f"thêm {new} phiên mới, đến {_fmt(df['date'].iloc[-1])}" if new else
                            f"chưa có phiên mới (phiên cuối {_fmt(last)}, có thể là ngày nghỉ)")
                    write = store.upsert_prices
        except Exception as e:  # noqa: BLE001
            have = f"; dùng dữ liệu cũ đến {_fmt(last)}" if last is not None else ""
            return Step(label, "error", f"{e}{have}")
        errors = [i for i in validate_prices(df, band=band) if i.level == "LOI"]
        if errors:
            return Step(label, "error", "dữ liệu tải về có lỗi, không nạp: "
                        + "; ".join(str(i) for i in errors))
        write(df, source=self._used.name)
        self._save_csv(store, sym)
        repaired = getattr(self._used, "repaired", None)
        if repaired:
            note += f"; sửa cao/thấp cho {len(repaired)} phiên nguồn ghi sai"
        if self._used is not self.cafef:
            note += f" — nguồn {self._used.name}" + (f" (CafeF: {self._fallback_reason})"
                                                     if self._fallback_reason else "")
        return Step(label, "ok", note)

    def _download(self, sym: str, start, end) -> pd.DataFrame:
        """Thử lần lượt các nguồn giá trong self._order; ghi nhận nguồn đã dùng vào self._used."""
        errors = []
        self._fallback_reason = None
        for prov in self._order:
            try:
                df = prov.ohlcv(sym, start, end)
                if df.empty:
                    raise ValueError(f"không có giá {sym} từ {start} đến {end}")
            except Exception as e:  # noqa: BLE001 - thử nguồn kế tiếp
                errors.append(f"{prov.name}: {e}")
                if prov is self.cafef:
                    self._fallback_reason = "không có dữ liệu mã này"
                continue
            self._used = prov
            return df
        raise ValueError("; ".join(errors) + ". Mã có đúng không?")

    def _save_csv(self, store: Store, sym: str) -> None:
        """Giữ bản CSV trong data/raw/prices (như lệnh fetch) để nạp lại được bằng lệnh load."""
        out = path("raw_prices_dir")
        out.mkdir(parents=True, exist_ok=True)
        df = store.prices(sym)
        df.assign(date=df["date"].dt.strftime("%Y-%m-%d")).drop(columns="symbol").to_csv(
            out / f"{sym}.csv", index=False)

    def _fundamentals(self, store: Store, sym: str, refresh: bool) -> Step:
        loaded = store.fundamentals_loaded_at(sym)
        age = (pd.Timestamp.now() - loaded).days if loaded is not None else None
        if age is not None and age < self.cfg["bctc_max_age_days"] and not refresh:
            return Step("BCTC", "skip", f"đã có (tải {age} ngày trước)")
        try:
            df = self.vn.fundamentals(sym)
            fields = df.drop(columns=["symbol", "year"])
            if df.empty or fields.isna().all().all():
                raise ValueError("nguồn không trả số liệu")
        except Exception as e:  # noqa: BLE001 - thiếu BCTC vẫn phân tích kỹ thuật được
            keep = "; dùng bản cũ trong kho" if loaded is not None else "; chỉ phân tích kỹ thuật"
            return Step("BCTC", "error", f"{e}{keep}")
        store.write_fundamentals(df, source=self.vn.name)
        out = path("raw_bctc_dir")
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / f"{sym}.csv", index=False)
        years = df["year"].tolist()
        missing = getattr(self.vn, "missing", [])
        return Step("BCTC", "ok", f"năm {years[0]}–{years[-1]}"
                    + (f", thiếu {len(missing)} khoản mục" if missing else ""))

    def _quarterly(self, store: Store, sym: str, refresh: bool) -> Step:
        loaded = store.quarterly_loaded_at(sym)
        age = (pd.Timestamp.now() - loaded).days if loaded is not None else None
        if age is not None and age < self.cfg["bctc_max_age_days"] and not refresh:
            return Step("BCTC quý", "skip", f"đã có (tải {age} ngày trước)")
        try:
            df = self.vn.quarterly(sym)
            if df.empty or df.drop(columns=["symbol", "year", "quarter"]).isna().all().all():
                raise ValueError("nguồn không trả số liệu")
        except Exception as e:  # noqa: BLE001 - thiếu BCTC quý thì dùng BCTC năm
            return Step("BCTC quý", "error", f"{e}; dùng BCTC năm")
        store.write_quarterly(df, source=self.vn.name)
        first, last = df.iloc[0], df.iloc[-1]
        return Step("BCTC quý", "ok", f"Q{int(first['quarter'])}/{int(first['year'])} → "
                    f"Q{int(last['quarter'])}/{int(last['year'])} ({len(df)} quý)")

    def _profile(self, store: Store, sym: str, refresh: bool) -> Step:
        prof = store.company(sym) or {}
        at = prof.get("profile_at")
        age = (pd.Timestamp.now() - pd.Timestamp(at)).days if at is not None else None
        if age is not None and age < self.cfg.get("profile_max_age_days", 7) and not refresh:
            return Step("hồ sơ", "skip", f"đã có (tải {age} ngày trước)")
        try:
            p = self.vn.profile(sym)
            if not p or pd.isna(p.get("shares_out", float("nan"))):
                raise ValueError("nguồn không trả số cổ phiếu lưu hành")
            store.write_profile(sym, p)
            n_div = store.write_dividends(self.vn.dividends(sym), sym)
        except Exception as e:  # noqa: BLE001 - thiếu hồ sơ thì dùng vốn góp / mệnh giá
            return Step("hồ sơ", "error", f"{e}; số cổ phiếu tạm tính = vốn góp / mệnh giá")
        return Step("hồ sơ", "ok", f"{p['shares_out'] / 1e6:,.1f} triệu cp lưu hành, "
                    f"{n_div} lần chia cổ tức")

    def _flows(self, store: Store, sym: str) -> Step | None:
        """Khối ngoại + tự doanh (CafeF). Nguồn giá không có hàm flows (test) thì bỏ qua."""
        get = getattr(self.cafef, "flows", None)
        if get is None:
            return None
        have = store.flows(sym)
        last_px = store.last_price_date(sym)
        if len(have) and last_px is not None and have["date"].max() >= pd.Timestamp(last_px):
            return Step("khối ngoại", "skip", "đã mới")
        try:
            n = store.write_flows(get(sym, self.cfg.get("flow_sessions", 60)))
        except Exception as e:  # noqa: BLE001 - không có dòng tiền vẫn phân tích được
            return Step("khối ngoại", "error", f"{e}; bỏ qua dòng tiền khối ngoại/tự doanh")
        if not n:
            return Step("khối ngoại", "error", "CafeF không có dữ liệu mã này")
        return Step("khối ngoại", "ok", f"{n} phiên (khối ngoại, tự doanh — CafeF)")

    def _news(self, store: Store, sym: str) -> Step | None:
        """Tiêu đề tin theo mã (vnstock/VCI), tải lại khi cũ hơn `[auto] news_max_age_hours`.
        Tắt bằng `[auto] news = false`; nguồn không có hàm news (test) thì bỏ qua."""
        get = getattr(self.vn, "news", None)
        if not self.cfg.get("news", False) or get is None:
            return None
        loaded = store.news_loaded_at(sym)
        age = (pd.Timestamp.now() - loaded).total_seconds() / 3600 if loaded is not None else None
        if age is not None and age < self.cfg.get("news_max_age_hours", 12):
            return Step("tin tức", "skip", f"tải {age:.0f} giờ trước")
        try:
            n = store.write_news(get(sym), sym)
        except Exception as e:  # noqa: BLE001 - không có tin vẫn phân tích được
            return Step("tin tức", "error", f"{e}; bỏ qua tiêu đề tin")
        return Step("tin tức", "ok", f"{n} tiêu đề (vnstock/VCI, chỉ tham khảo)")

    def _macro(self) -> Step | None:
        """Số vĩ mô (lớp vĩ mô): nạp số nhập tay, tải lại chuỗi tự động nếu đã cũ. Tắt bằng
        `[auto] macro = false`."""
        if not self.cfg.get("macro", False):
            return None
        if self._macro_step is None:
            from vnsa.macro.update import refresh

            mc = settings().get("macro", {})
            try:
                status, detail = refresh(mc.get("max_age_hours", 20), mc.get("years", 10),
                                         self.macro_provider)
            except Exception as e:  # noqa: BLE001 - thiếu vĩ mô vẫn phân tích được
                status, detail = "error", f"không cập nhật được số vĩ mô ({e})"
            self._macro_step = Step("vĩ mô", status, detail)
        return self._macro_step

    def long_prices(self, symbols: list[str], years: int) -> list[Step]:
        """Chỉ tải GIÁ (kèm VNINDEX và danh sách sàn để biết biên độ) đủ `years` năm cho kiểm định
        nhiều mã (lệnh study-timing); mã đã có đủ thì chỉ tải phần mới."""
        steps = []
        cfg_years = self.cfg.get("years")
        self.cfg = {**self.cfg, "years": years}
        want = pd.Timestamp(self.now().date()) - pd.Timedelta(days=round(365.25 * years) - 30)
        try:
            with Store(path("warehouse")) as store:
                steps.append(self._listing(store, False))
                for sym in [*symbols, self.index]:
                    have = store.prices(sym)
                    short = have.empty or have["date"].min() > want
                    steps.append(self._prices(store, sym, sym, full=short))
        finally:
            self.cfg = {**self.cfg, "years": cfg_years}
        return steps

    # ---- điểm vào ----------------------------------------------------------------------------
    def ensure(self, symbol: str, refresh: bool = False, full: bool = False) -> UpdateReport:
        """Bảo đảm kho có dữ liệu mới cho `symbol`. refresh: tải lại BCTC và danh sách dù chưa cũ;
        full: tải lại toàn bộ giá."""
        sym = symbol.upper().strip()
        rep = UpdateReport(sym)
        with Store(path("warehouse")) as store:
            rep.steps.append(self._listing(store, refresh))
            prof = store.company(sym)
            if prof is None and store.companies_loaded_at() is not None and sym != self.index:
                rep.steps.append(Step("niêm yết", "error", f"{sym} không có trong danh sách cổ "
                                      "phiếu HOSE/HNX/UPCoM; vẫn thử tải giá"))
            elif prof:
                rep.steps.append(Step("thông tin", "ok", " · ".join(
                    str(x) for x in (prof["name"], prof["exchange"], prof["industry"]) if x)))
            rep.steps.append(self._prices(store, sym, "giá", full))
            if sym != self.index:
                if self._index_step is None:
                    self._index_step = self._prices(store, self.index, self.index, full)
                rep.steps.append(self._index_step)
            rep.has_prices = store.last_price_date(sym) is not None
            if rep.has_prices and sym != self.index:
                rep.steps.append(self._fundamentals(store, sym, refresh))
                rep.steps.append(self._quarterly(store, sym, refresh))
                rep.steps.append(self._profile(store, sym, refresh))
                flow = self._flows(store, sym)
                if flow is not None:
                    rep.steps.append(flow)
                news = self._news(store, sym)
                if news is not None:
                    rep.steps.append(news)
        macro = self._macro() if rep.has_prices and sym != self.index else None
        if macro is not None:
            rep.steps.append(macro)
        return rep


def ensure(symbol: str, refresh: bool = False, full: bool = False,
           log: Callable[[str], None] | None = None) -> UpdateReport:
    return Updater(log=log).ensure(symbol, refresh, full)
