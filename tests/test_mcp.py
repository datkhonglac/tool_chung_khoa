import asyncio
import json

import pytest

pytest.importorskip("mcp")

from vnsa import mcp_server as srv

EXPECTED = {"list_symbols", "get_prices", "get_ratios", "get_valuation", "get_fa_score",
            "get_technical", "get_trade_plan", "get_risk", "get_recommendations", "run_backtest"}


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
    assert rec["recommendations"][0]["action"] in {"MUA", "THEO DÕI", "TRÁNH"}
    assert rec["errors"] and "NOPE" in rec["errors"][0]


def test_missing_data_returns_error_message(warehouse):
    out = srv.get_valuation("DEMO_B")                 # có giá nhưng chưa có BCTC
    assert "fetch-bctc DEMO_B" in out["error"]


def test_call_through_protocol(warehouse):
    result = asyncio.run(srv.mcp.call_tool("get_prices", {"symbol": "DEMO_A", "sessions": 3}))
    text = json.dumps(result, default=str)
    assert "DEMO_A" in text
