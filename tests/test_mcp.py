import asyncio
import json

import pytest

pytest.importorskip("mcp")

from vnsa import mcp_server as srv

EXPECTED = {"list_symbols", "get_prices", "get_ratios", "get_valuation", "get_fa_score",
            "get_technical", "get_trade_plan", "get_risk", "get_recommendations", "run_backtest",
            "update_symbol", "analyze_symbol", "review_signals", "macro_context",
            "screen_market"}


def test_all_tools_registered_with_schemas():
    tools = asyncio.run(srv.mcp.list_tools())
    names = {t.name for t in tools}
    assert EXPECTED <= names
    prices = next(t for t in tools if t.name == "get_prices")
    schema = getattr(prices, "input_schema", None) or prices.inputSchema   # mcp 2.x / 1.x
    assert "symbol" in schema["properties"]
    assert "phiên" in prices.description


def test_tools_return_json_safe_results(warehouse):
    for name, kwargs in [("list_symbols", {}), ("get_prices", {"symbol": "demo_a", "sessions": 5}),
                         ("get_ratios", {"symbol": "DEMO_A"}),
                         ("get_valuation", {"symbol": "DEMO_A"}),
                         ("get_fa_score", {"symbol": "DEMO_A"}),
                         ("get_technical", {"symbol": "DEMO_A"}),
                         ("get_trade_plan", {"symbol": "DEMO_A"}),
                         ("get_risk", {"symbol": "DEMO_A", "sessions": 10}),
                         ("get_recommendations", {"symbols": ["DEMO_A", "DEMO_B", "NOPE"]}),
                         ("run_backtest", {"symbol": "DEMO_A"})]:
        out = getattr(srv, name)(**kwargs)
        json.dumps(out, allow_nan=False)            # không còn NaN/numpy/Timestamp
        assert "error" not in out, (name, out)
    assert len(srv.get_prices("DEMO_A", 5)["prices"]) == 5
    rec = srv.get_recommendations(["DEMO_A", "NOPE"])
    assert rec["recommendations"][0]["action"] in {"MUA", "KHẢ QUAN", "NẮM GIỮ", "KÉM KHẢ QUAN", "BÁN", "KHÔNG XẾP HẠNG"}
    assert rec["errors"] and "NOPE" in rec["errors"][0]


def test_missing_data_returns_error_message(warehouse):
    out = srv.get_valuation("DEMO_B")                 # có giá nhưng chưa có BCTC
    assert "fetch-bctc DEMO_B" in out["error"]


def test_call_through_protocol(warehouse):
    result = asyncio.run(srv.mcp.call_tool("get_prices", {"symbol": "DEMO_A", "sessions": 3}))
    text = json.dumps(result, default=str)
    assert "DEMO_A" in text


def test_analyze_symbol_updates_first_and_keeps_stdout_clean(warehouse, monkeypatch, capsys):
    from vnsa import analysis
    from vnsa.data.autofetch import Step, UpdateReport

    def fake_update(sym, refresh=False, **kw):
        print("tiến trình tải")                      # stdout là kênh MCP: phải bị chuyển hướng
        return UpdateReport(sym.upper(), [Step("giá", "ok", "thêm 1 phiên")], has_prices=True)

    monkeypatch.setattr(analysis, "update", fake_update)
    out = srv.analyze_symbol("demo_a")
    json.dumps(out, allow_nan=False)
    assert out["action"] in {"MUA", "KHẢ QUAN", "NẮM GIỮ", "KÉM KHẢ QUAN", "BÁN", "KHÔNG XẾP HẠNG"} and out["profile"]["exchange"] == "HOSE"
    assert out["data_update"] == ["✓ giá: thêm 1 phiên"]
    assert "ĐỊNH GIÁ" in out["report"] and out["confidence"] in {"Cao", "Trung bình", "Thấp"}
    up = srv.update_symbol("DEMO_A")
    assert up["ready"] and up["steps"][0]["status"] == "ok"
    captured = capsys.readouterr()
    assert "tiến trình tải" not in captured.out and "tiến trình tải" in captured.err

    monkeypatch.setattr(analysis, "update", lambda sym, refresh=False, **kw: UpdateReport(
        sym.upper(), [Step("giá", "error", "Mã có đúng không?")]))
    assert "Mã có đúng không" in srv.analyze_symbol("XYZ")["error"]


def test_screen_market_reads_latest_run(warehouse):
    import json

    import pandas as pd

    from vnsa.data.store import Store

    out = srv.screen_market()
    assert "error" in out and "screen" in out["error"]
    day = pd.Timestamp("2026-10-09")
    table = pd.DataFrame([
        {"as_of": day, "symbol": "AAA", "group_name": "Ngân hàng", "stage": 4, "action": "MUA",
         "pick_rank": 1.0, "score": 60.0, "reason": ""},
        {"as_of": day, "symbol": "CCC", "group_name": "Thép", "stage": 2, "action": "MUA",
         "score": 50.0, "reason": "đạt nhưng ngoài nhóm ngành dẫn đầu"},
        {"as_of": day, "symbol": "DDD", "group_name": "Thép", "stage": 0, "reason": "giá thấp: …"}])
    ind = pd.DataFrame([{"as_of": day, "group_name": "Ngân hàng", "rank": 1.0, "score": 80.0,
                         "members": 1}])
    with Store(warehouse) as store:
        store.write_screen(table, ind)
    out = srv.screen_market()
    json.dumps(out, allow_nan=False)
    assert out["as_of"] == "2026-10-09" and out["funnel"]["picked"] == 1
    assert [p["symbol"] for p in out["picks"]] == ["AAA"]
    assert [p["symbol"] for p in out["outside_top_industries"]] == ["CCC"]
    assert out["industries"][0]["group_name"] == "Ngân hàng"
    grp = srv.screen_market(group="thép")
    assert {p["symbol"] for p in grp["group"]} == {"CCC", "DDD"}
