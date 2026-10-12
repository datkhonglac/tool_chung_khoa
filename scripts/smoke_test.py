"""Kiểm tra nhanh MCP server và dashboard trên kho dữ liệu THẬT (sau khi fetch + fetch-bctc).

    python scripts/smoke_test.py [MÃ...]

Gọi mọi công cụ MCP cho từng mã, rồi chạy dashboard ở chế độ không giao diện (Streamlit AppTest)
cho từng mã và từng tab. In một dòng cho mỗi bước; mã thoát 1 nếu có lỗi.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from vnsa.config import settings

ROOT = Path(__file__).resolve().parents[1]


def check_mcp(symbols: list[str]) -> list[str]:
    from vnsa import mcp_server as srv

    errors = []
    calls = [("get_prices", {"sessions": 5}), ("get_ratios", {}), ("get_valuation", {}),
             ("get_fa_score", {}), ("get_technical", {}), ("get_trade_plan", {}),
             ("get_risk", {}), ("run_backtest", {})]
    for sym in symbols:
        for name, kw in calls:
            out = getattr(srv, name)(symbol=sym, **kw)
            json.dumps(out, allow_nan=False)
            if "error" in out:
                errors.append(f"MCP {name}({sym}): {out['error']}")
        print(f"  MCP {sym}: {len(calls)} công cụ", flush=True)
        mc = srv.macro_context(sym)          # kho chưa tải vĩ mô thì chỉ báo, không tính là lỗi
        json.dumps(mc, allow_nan=False)
        ind = (mc.get("industry") or {}).get("label", mc.get("error", "chưa có ánh xạ ngành"))
        print(f"  MCP macro_context {sym}: {ind}", flush=True)
    sc = srv.screen_market()                 # chưa chạy sàng lọc thì chỉ báo, không tính là lỗi
    json.dumps(sc, allow_nan=False)
    print(f"  MCP screen_market: {sc.get('error') or sc['funnel']}", flush=True)
    rec = srv.get_recommendations(symbols)
    errors += [f"MCP get_recommendations: {e}" for e in rec["errors"]]
    for r in rec["recommendations"]:
        print(f"  MCP khuyến nghị {r['symbol']}: {r['action']} ({r['score']:.0f})")
    return errors


def check_dashboard(symbols: list[str]) -> list[str]:
    from streamlit.testing.v1 import AppTest

    errors = []
    at = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=300).run()
    for sym in symbols:
        if sym not in at.selectbox[0].options:
            errors.append(f"dashboard: không có {sym} trong danh sách")
            continue
        at.selectbox[0].select(sym).run()
        if at.exception:
            errors.append(f"dashboard {sym}: {at.exception[0].message}")
        labels = [m.label for m in at.metric]
        print(f"  dashboard {sym}: {len(labels)} chỉ số, {len(at.error)} lỗi hiển thị", flush=True)
        errors += [f"dashboard {sym}: {e.value}" for e in at.error]
    return errors


def main() -> int:
    symbols = [s.upper() for s in sys.argv[1:]] or settings()["watchlist"]["symbols"]
    errors = check_mcp(symbols) + check_dashboard(symbols)
    for e in errors:
        print(f"  ✗ {e}")
    print("Không có lỗi." if not errors else f"{len(errors)} lỗi.")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
