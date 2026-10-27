"""HMM trạng thái VNINDEX: Baum–Welch tìm lại tham số, Viterbi, đặt tên, kiểm định walk-forward."""

import numpy as np
import pandas as pd
import pytest

from vnsa.recommend import hmm_study as hs
from vnsa.ta import hmm_regime as hr


def _simulate(T=1200, seed=0):
    rng = np.random.default_rng(seed)
    A = np.array([[0.97, 0.03], [0.05, 0.95]])
    mu = np.array([[1.0, 15.0], [-2.0, 35.0]])
    sd = np.array([[1.0, 2.0], [1.5, 4.0]])
    s = np.zeros(T, dtype=int)
    for t in range(1, T):
        s[t] = rng.choice(2, p=A[s[t - 1]])
    X = mu[s] + sd[s] * rng.normal(size=(T, 2))
    return X, s


def test_baum_welch_recovers_two_states():
    X, s = _simulate()
    m = hr.fit(X, k=2, seeds=2)
    assert sorted(m.names) == ["Suy thoái", "Tăng tốc"]
    bad = m.names.index("Suy thoái")
    assert m.mu[bad, 0] == pytest.approx(-2.0, abs=0.4) and m.mu[bad, 1] == pytest.approx(35, abs=2)
    assert np.diag(m.A).min() > 0.85
    path = hr.viterbi(m, X)
    truth_bad = s == 1
    assert ((path == bad) == truth_bad).mean() > 0.9
    p = hr.filtered(m, X)
    assert p.shape == (len(X), 2) and np.allclose(p.sum(axis=1), 1)


def test_name_states_four():
    mu = np.array([[0.5, 12], [-3, 40], [2.5, 20], [-0.5, 30]])
    assert hr.name_states(mu, mu) == ["Tích lũy", "Suy thoái", "Tăng tốc", "Phân phối"]


def _index(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    r = np.r_[rng.normal(0.0008, 0.008, n // 2), rng.normal(-0.002, 0.02, n // 4),
              rng.normal(0.001, 0.009, n - n // 2 - n // 4)]
    return pd.DataFrame({"date": pd.bdate_range("2019-01-01", periods=n),
                         "close": 1000 * np.exp(np.cumsum(r))})


def test_current_view_and_walk_forward_study():
    idx = _index()
    v = hr.current(idx, {"states": 3, "train_sessions": 1500, "seeds": 2})
    assert v is not None and abs(sum(v.tomorrow.values()) - 1) < 1e-9 and "HMM VNINDEX" in str(v)
    ma = pd.Series("tăng", index=pd.to_datetime(idx["date"]))
    res = hs.study(idx, ma, {"states": 3, "train_sessions": 500, "seeds": 1, "threshold": 0.7},
                   {"fee": 0.0015, "tax": 0.001}, split="2021-06-01", start="2020-06-01")
    assert set(res.table["method"]) == {"Mua & giữ", "MA200", "HMM"}
    assert res.signal.index.min() >= pd.Timestamp("2020-06-01")
    text = hs.format_hmm_study(res, {"threshold": 0.7})
    assert "Sharpe" in text


def test_strategy_returns_costs():
    idx = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=4),
                        "close": [100.0, 110.0, 99.0, 99.0]})
    e = pd.Series([1.0, 0.5, 0.5, 0.5], index=pd.to_datetime(idx["date"]))
    r = hs.strategy_returns(idx, e, fee=0.001, tax=0.001)
    assert r.iloc[0] == pytest.approx(0.10)                       # phiên đầu: 100% × +10%
    assert r.iloc[1] == pytest.approx(0.5 * -0.10 - 0.5 * 0.002)  # giảm 50%: phí + thuế bán
