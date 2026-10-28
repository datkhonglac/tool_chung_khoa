"""Trạng thái ẩn của VNINDEX bằng Mô hình Markov ẩn (HMM) Gauss — nâng cấp v4.

Quan sát mỗi phiên (2 chiều): lợi suất 5 phiên (%) và độ biến động 20 phiên (% năm hoá) của VNINDEX.
Trạng thái ẩn S_t ∈ {1..K} (mặc định K = 4) theo xích Markov:
    P(S_t = j | S_{t−1} = i) = A_ij,     Y_t | S_t = k  ~  N(μ_k, diag σ_k²)
- Ước lượng A, μ, σ bằng Baum–Welch (EM), forward–backward có chuẩn hoá (Rabiner) để không tràn số;
  chạy vài điểm khởi tạo, lấy nghiệm có log-likelihood cao nhất. Không cần thư viện hmmlearn.
- Viterbi: chuỗi trạng thái khả dĩ nhất (để vẽ/giải thích).
- Xác suất lọc P(S_T | Y_1..T) (chỉ dùng dữ liệu đến hôm nay) và dự báo ngày mai P(S_{T+1}) = P(S_T)·A.
Đặt tên trạng thái sau khi ước lượng (thứ tự trạng thái của EM là tuỳ ý), theo lợi suất bình quân:
thấp nhất = "Suy thoái", cao nhất = "Tăng tốc"; hai trạng thái giữa: biến động cao hơn = "Phân phối",
thấp hơn = "Tích lũy". K = 3: Suy thoái / Tích lũy / Tăng tốc.

Blueprint đề xuất "xác suất sang Phân phối > 70% thì siết rủi ro" — CHƯA dùng trong kế hoạch: phải đạt
kiểm định walk-forward (lệnh study-hmm, recommend/hmm_study.py) trước.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NAMES4 = ("Suy thoái", "Phân phối", "Tích lũy", "Tăng tốc")
BAD = ("Suy thoái", "Phân phối")


def features(index: pd.DataFrame, ret_days: int = 5, vol_days: int = 20) -> pd.DataFrame:
    """date, ret (lợi suất `ret_days` phiên, %), vol (độ lệch chuẩn `vol_days` phiên, % năm)."""
    c = index.sort_values("date").set_index(pd.to_datetime(index["date"]))["close"].astype(float)
    lr = np.log(c).diff()
    out = pd.DataFrame({"ret": (np.log(c).diff(ret_days)) * 100,
                        "vol": lr.rolling(vol_days).std() * np.sqrt(252) * 100})
    return out.dropna()


@dataclass
class HMM:
    pi: np.ndarray            # (K,) xác suất ban đầu
    A: np.ndarray             # (K, K) ma trận chuyển
    mu: np.ndarray            # (K, D)
    var: np.ndarray           # (K, D)
    loglik: float = float("nan")
    names: list[str] = field(default_factory=list)

    @property
    def k(self) -> int:
        return len(self.pi)

    def log_emission(self, X: np.ndarray) -> np.ndarray:
        """(T, K) log mật độ Gauss chéo — vector hoá, không vòng lặp."""
        d = X[:, None, :] - self.mu[None, :, :]
        return -0.5 * (np.log(2 * np.pi * self.var)[None] + d ** 2 / self.var[None]).sum(axis=2)


def _emission(model: "HMM", X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(B, c): B = mật độ đã chia cho hằng số mỗi dòng (tránh tràn số), log c = hằng số đó."""
    logB = model.log_emission(X)
    c = logB.max(axis=1, keepdims=True)
    return np.exp(logB - c), c[:, 0]


def _forward(B: np.ndarray, pi: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward có chuẩn hoá: alpha_t = P(S_t | Y_1..t); scale_t = P(Y_t | Y_1..t−1) (trên B)."""
    T, K = B.shape
    alpha = np.empty((T, K))
    scale = np.empty(T)
    a = pi * B[0]
    scale[0] = a.sum()
    alpha[0] = a / scale[0]
    for t in range(1, T):          # đệ quy theo thời gian (bản chất tuần tự), mỗi bước là phép ma trận
        a = (alpha[t - 1] @ A) * B[t]
        scale[t] = a.sum()
        alpha[t] = a / scale[t]
    return alpha, scale


def _backward(B: np.ndarray, A: np.ndarray, scale: np.ndarray) -> np.ndarray:
    T, K = B.shape
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / scale[t + 1]
    return beta


def _em(X: np.ndarray, model: HMM, iters: int, tol: float, floor: np.ndarray) -> HMM:
    prev = -np.inf
    for _ in range(iters):
        B, c = _emission(model, X)
        alpha, scale = _forward(B, model.pi, model.A)
        beta = _backward(B, model.A, scale)
        ll = float(np.log(scale).sum() + c.sum())
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        # xi tổng theo t: A_ij × Σ_t alpha_t(i) B_{t+1}(j) beta_{t+1}(j) / scale_{t+1}
        xi = model.A * (alpha[:-1].T @ (B[1:] * beta[1:] / scale[1:, None]))
        model.pi = np.clip(gamma[0], 1e-6, None)
        model.pi /= model.pi.sum()
        model.A = np.clip(xi / xi.sum(axis=1, keepdims=True), 1e-6, None)
        model.A /= model.A.sum(axis=1, keepdims=True)
        w = gamma.sum(axis=0)[:, None]
        model.mu = gamma.T @ X / w
        model.var = np.maximum((gamma.T @ X ** 2) / w - model.mu ** 2, floor)
        model.loglik = ll
        if ll - prev < tol:
            break
        prev = ll
    return model


def name_states(mu: np.ndarray, var: np.ndarray) -> list[str]:
    """Tên theo lợi suất bình quân (cột 0) và biến động (cột 1) của từng trạng thái."""
    k = len(mu)
    order = list(np.argsort(mu[:, 0]))
    names = [""] * k
    if k == 2:
        names[order[0]], names[order[1]] = "Suy thoái", "Tăng tốc"
    elif k == 3:
        for s, n in zip(order, ("Suy thoái", "Tích lũy", "Tăng tốc")):
            names[s] = n
    else:
        names[order[0]], names[order[-1]] = "Suy thoái", "Tăng tốc"
        mid = sorted(order[1:-1], key=lambda s: mu[s, 1])          # theo biến động
        for i, s in enumerate(mid):
            names[s] = "Tích lũy" if i == 0 else ("Phân phối" if i == len(mid) - 1
                                                  else f"Trung gian {i}")
    return names


def fit(X: np.ndarray, k: int = 4, iters: int = 200, tol: float = 1e-3, seeds: int = 4,
        seed: int = 0, init: HMM | None = None) -> HMM:
    """Baum–Welch nhiều điểm khởi tạo; `init` (vd mô hình tháng trước) thêm một điểm khởi tạo ấm."""
    X = np.asarray(X, float)
    if len(X) < 50 * k:
        raise ValueError(f"cần ít nhất {50 * k} phiên để ước lượng HMM {k} trạng thái")
    rng = np.random.default_rng(seed)
    floor = X.var(axis=0) * 1e-3
    best = None
    if init is not None and init.k == k:
        best = _em(X, HMM(init.pi.copy(), init.A.copy(), init.mu.copy(), init.var.copy()), iters,
                   tol, floor)
    for s in range(seeds):
        # khởi tạo: chia theo phân vị lợi suất (lần đầu) hoặc chọn ngẫu nhiên các điểm làm tâm
        if s == 0:
            qs = np.quantile(X[:, 0], np.linspace(0.1, 0.9, k))
            mu = np.column_stack([qs] + [np.full(k, X[:, j].mean()) for j in range(1, X.shape[1])])
        else:
            mu = X[rng.choice(len(X), k, replace=False)]
        A = np.full((k, k), 0.05 / max(k - 1, 1))
        np.fill_diagonal(A, 0.95)
        m = HMM(np.full(k, 1 / k), A, mu.astype(float), np.tile(X.var(axis=0), (k, 1)))
        m = _em(X, m, iters, tol, floor)
        if best is None or m.loglik > best.loglik:
            best = m
    best.names = name_states(best.mu, best.var)
    return best


def filtered(model: HMM, X: np.ndarray) -> np.ndarray:
    """(T, K) P(S_t | Y_1..t) — chỉ dùng dữ liệu đến t (không nhìn trước)."""
    B, _ = _emission(model, np.asarray(X, float))
    return _forward(B, model.pi, model.A)[0]


def viterbi(model: HMM, X: np.ndarray) -> np.ndarray:
    logB = model.log_emission(np.asarray(X, float))
    T, K = logB.shape
    logA = np.log(model.A)
    delta = np.log(model.pi) + logB[0]
    back = np.zeros((T, K), dtype=int)
    for t in range(1, T):
        cand = delta[:, None] + logA
        back[t] = cand.argmax(axis=0)
        delta = cand.max(axis=0) + logB[t]
    path = np.empty(T, dtype=int)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


@dataclass
class HmmView:
    date: pd.Timestamp
    state: str
    prob: float                     # P(trạng thái hiện tại)
    tomorrow: dict[str, float]      # P(S_{T+1} = tên)
    bad_tomorrow: float             # P(ngày mai ∈ Phân phối/Suy thoái)
    model: HMM

    def __str__(self) -> str:
        top = sorted(self.tomorrow.items(), key=lambda kv: -kv[1])
        return (f"HMM VNINDEX: {self.state} (xác suất {self.prob:.0%}) · ngày mai: "
                + " · ".join(f"{n} {p:.0%}" for n, p in top[:3])
                + " (tham khảo, chưa vào điểm)")


def current(index: pd.DataFrame, cfg: dict | None = None) -> HmmView | None:
    cfg = cfg or {}
    f = features(index, int(cfg.get("ret_days", 5)), int(cfg.get("vol_days", 20)))
    f = f.tail(int(cfg.get("train_sessions", 1500)))
    if len(f) < 50 * int(cfg.get("states", 4)):
        return None
    m = fit(f.to_numpy(), int(cfg.get("states", 4)), seeds=int(cfg.get("seeds", 4)))
    p = filtered(m, f.to_numpy())[-1]
    nxt = p @ m.A
    s = int(p.argmax())
    tomorrow = {m.names[i]: float(nxt[i]) for i in range(m.k)}
    return HmmView(f.index[-1], m.names[s], float(p[s]), tomorrow,
                   float(sum(v for n, v in tomorrow.items() if n in BAD)), m)
