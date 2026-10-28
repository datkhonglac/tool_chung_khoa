"""HRP: hai cụm tương quan nhận rủi ro cân bằng; phụ thuộc đuôi; kiểm định cân lại hằng tháng."""

import numpy as np
import pandas as pd
import pytest

from vnsa.risk import hrp
from vnsa.risk import hrp_study as hs


def _returns(n=600, seed=0):
    rng = np.random.default_rng(seed)
    f1, f2 = rng.normal(0, 0.02, n), rng.normal(0, 0.01, n)
    cols = {}
    for i in range(3):            # cụm A: 3 mã biến động mạnh, cùng nhân tố
        cols[f"A{i}"] = f1 + rng.normal(0, 0.005, n)
    cols["B0"] = f2 + rng.normal(0, 0.003, n)        # cụm B: 1 mã ít biến động
    idx = pd.bdate_range("2016-01-01", periods=n)
    return pd.DataFrame(cols, index=idx)


def test_hrp_balances_clusters_not_names():
    r = _returns()
    w = hrp.hrp_weights(r)
    assert w.sum() == pytest.approx(1.0) and (w > 0).all()
    assert w["B0"] > 0.5                    # cụm 1 mã ít rủi ro nhận hơn nửa vốn, không phải 1/4
    assert w[["A0", "A1", "A2"]].sum() < w["B0"]   # (thuật toán gốc chia đôi theo thứ tự lá)
    assert hrp.hrp_weights(r[["A0"]]).iloc[0] == 1.0


def test_tail_dependence():
    r = _returns()
    lam = hrp.tail_dependence(r, 0.05)
    assert lam.loc["A0", "A1"] > 0.6 and lam.loc["A0", "B0"] < 0.3
    pairs = hrp.top_pairs(lam, 2)
    assert {pairs[0][0], pairs[0][1]} <= {"A0", "A1", "A2"}
    v = hrp.HrpView(hrp.hrp_weights(r), pd.Series({"A0": 1.0}), pairs, len(r))
    assert "HRP" in hrp.format_hrp(v)


def test_monthly_rebalance_study():
    r = _returns(900)
    table, rets = hs.study(r, {"fee": 0.0015, "tax": 0.001}, split="2018-01-01", start="2017-01-01",
                           sessions=250)
    assert set(table["method"]) == set(hs.METHODS)
    hrp_vol = table[(table.period == "Toàn kỳ") & (table.method == "HRP")]["vol"].iloc[0]
    eq_vol = table[(table.period == "Toàn kỳ") & (table.method == "Chia đều")]["vol"].iloc[0]
    assert hrp_vol < eq_vol                 # HRP dồn vào cụm ít biến động
    assert "Sharpe" in hs.format_hrp_study(hs.HrpStudy(table, list(r.columns)))
