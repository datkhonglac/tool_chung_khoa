from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "prices"
B = 1_000_000_000


def _fundamentals(symbol: str) -> pd.DataFrame:
    years = [2022, 2023, 2024, 2025]
    g = np.array([1.0, 1.1, 1.2, 1.35])
    return pd.DataFrame({
        "symbol": symbol, "year": years,
        "revenue": 1000 * B * g, "gross_profit": 300 * B * g, "net_income": 80 * B * g,
        "net_income_parent": 75 * B * g, "eps": 2500 * g, "total_assets": 1500 * B * g,
        "total_liabilities": 600 * B * g, "equity": 900 * B * g, "minority_interest": 30 * B * g,
        "short_debt": 150 * B * g, "long_debt": 100 * B * g, "cfo": 110 * B * g,
        "pbt": 100 * B * g, "tax_expense": 20 * B * g, "interest_expense": 15 * B * g,
        "capex": 40 * B * g, "cash": 120 * B * g, "short_term_investments": 50 * B * g,
        "paid_in_capital": [300 * B] * 4, "current_assets": 700 * B * g,
        "current_liabilities": 400 * B * g, "retained_earnings": 250 * B * g,
        "equity_issued": [0.0] * 4,
    })


@pytest.fixture
def warehouse(tmp_path, monkeypatch):
    """Kho tạm có giá mẫu DEMO_A/B/C, chỉ số VNINDEX giả lập và BCTC giả lập cho DEMO_A."""
    from vnsa.data.providers import CSVProvider
    from vnsa.data.store import Store

    db = tmp_path / "test.duckdb"
    monkeypatch.setenv("VNSA_WAREHOUSE", str(db))
    monkeypatch.setenv("VNSA_SIGNALS_CSV", str(tmp_path / "signals.csv"))
    monkeypatch.setenv("VNSA_TRADES", str(tmp_path / "trades.csv"))   # sổ lệnh riêng của test
    prov = CSVProvider(SAMPLE)
    with Store(db) as store:
        frames = [prov.ohlcv(s) for s in prov.symbols()]
        for df in frames:
            store.write_prices(df, "test")
        idx = frames[0].copy()
        idx["symbol"] = "VNINDEX"
        idx[["open", "high", "low", "close"]] = idx[["open", "high", "low", "close"]] / 40
        store.write_prices(idx, "test")
        store.write_fundamentals(_fundamentals("DEMO_A"), "test")
    return db
