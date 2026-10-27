"""Mạng lan truyền biến động giữa các mã — chỉ số Diebold–Yilmaz (2012), nâng cấp v4. CHỈ THAM KHẢO.

1. VAR(p) cho lợi suất ngày của N mã (OLS):  Y_t = c + Σ_l Φ_l Y_{t−l} + ε_t,  Σ = cov(ε).
2. Biểu diễn trung bình trượt: A_0 = I, A_h = Σ_{l=1..p} Φ_l A_{h−l}.
3. Phân rã phương sai dự báo tổng quát (không phụ thuộc thứ tự biến), tầm H phiên:
     θ_ij(H) = σ_jj⁻¹ Σ_h (e_iᵀ A_h Σ e_j)² / Σ_h (e_iᵀ A_h Σ A_hᵀ e_i),  rồi chuẩn hoá mỗi dòng tổng = 1.
   θ_ij = phần biến động dự báo của mã i do cú sốc ở mã j gây ra (mã j → mã i).
4. Tổng lan truyền = Σ_{i≠j} θ_ij / N (mức độ "cả thị trường đi cùng nhau");
   Lan ra (to) của j = Σ_{i≠j} θ_ij, Nhận vào (from) của i = Σ_{j≠i} θ_ij, Ròng = lan ra − nhận vào.
   Mã ròng dương lớn nhất = mã "dẫn" biến động của nhóm trong cửa sổ đang xét.

Lưu ý: đây là phần biến động được GIẢI THÍCH bởi cú sốc của mã khác, không phải "xác suất dòng tiền
chảy sang mã khác" (blueprint diễn giải như vậy là quá tay) và chưa kiểm định khả năng dự báo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def fit_var(Y: np.ndarray, p: int = 1) -> tuple[list[np.ndarray], np.ndarray]:
    """(Φ_1..Φ_p mỗi cái N×N, Σ phần dư) bằng OLS, một phép lstsq cho cả hệ."""
    T, N = Y.shape
    if T <= N * p + 10:
        raise ValueError(f"cần nhiều hơn {N * p + 10} phiên cho VAR({p}) {N} mã")
    X = np.hstack([np.ones((T - p, 1))] + [Y[p - l: T - l] for l in range(1, p + 1)])
    B, *_ = np.linalg.lstsq(X, Y[p:], rcond=None)              # ((1 + N p), N)
    resid = Y[p:] - X @ B
    sigma = resid.T @ resid / (len(resid) - X.shape[1])
    phis = [B[1 + (l - 1) * N: 1 + l * N].T for l in range(1, p + 1)]
    return phis, sigma


def ma_coefficients(phis: list[np.ndarray], H: int) -> np.ndarray:
    N = phis[0].shape[0]
    A = np.zeros((H, N, N))
    A[0] = np.eye(N)
    for h in range(1, H):
        for l, phi in enumerate(phis, start=1):
            if h - l >= 0:
                A[h] += phi @ A[h - l]
    return A


def gfevd(phis: list[np.ndarray], sigma: np.ndarray, H: int = 10) -> np.ndarray:
    """Ma trận θ (N×N) đã chuẩn hoá dòng."""
    A = ma_coefficients(phis, H)
    AS = A @ sigma                                              # (H, N, N)
    num = (AS ** 2).sum(axis=0) / np.diag(sigma)[None, :]       # Σ_h (e_iᵀ A_h Σ e_j)² / σ_jj
    den = np.einsum("hij,hkj,hik->i", A, A, np.broadcast_to(sigma, A.shape))  # e_iᵀ A_h Σ A_hᵀ e_i
    theta = num / den[:, None]
    return theta / theta.sum(axis=1, keepdims=True)


@dataclass
class Spillover:
    theta: pd.DataFrame          # dòng i (nhận), cột j (phát)
    total: float
    to: pd.Series
    frm: pd.Series
    net: pd.Series
    sessions: int
    end: pd.Timestamp | None = None

    def edges(self, n: int = 8) -> list[tuple[str, str, float]]:
        """Các cạnh có hướng mạnh nhất j → i (bỏ đường chéo)."""
        t = self.theta
        out = [(j, i, float(t.at[i, j])) for i in t.index for j in t.columns if i != j]
        return sorted(out, key=lambda x: -x[2])[:n]


def spillover(returns: pd.DataFrame, p: int = 1, H: int = 10) -> Spillover:
    r = returns.dropna(axis=1, thresh=int(len(returns) * 0.9)).dropna()
    cols = list(r.columns)
    if len(cols) < 3:
        raise ValueError("cần ít nhất 3 mã có đủ giá")
    Y = (r.to_numpy() - r.to_numpy().mean(axis=0)) / r.to_numpy().std(axis=0)
    phis, sigma = fit_var(Y, p)
    th = pd.DataFrame(gfevd(phis, sigma, H), index=cols, columns=cols)
    off = th.to_numpy() - np.diag(np.diag(th.to_numpy()))
    to = pd.Series(off.sum(axis=0), index=cols)
    frm = pd.Series(off.sum(axis=1), index=cols)
    return Spillover(th, float(off.sum() / len(cols)), to, frm, (to - frm).sort_values(ascending=False),
                     len(r), r.index[-1] if len(r) else None)


def rolling_total(returns: pd.DataFrame, window: int = 250, step: int = 20, p: int = 1,
                  H: int = 10) -> pd.Series:
    """Tổng lan truyền theo cửa sổ trượt (để xem thị trường 'đồng pha' tăng hay giảm)."""
    vals = {}
    for end in range(window, len(returns) + 1, step):
        try:
            sp = spillover(returns.iloc[end - window: end], p, H)
            vals[returns.index[end - 1]] = sp.total
        except (ValueError, np.linalg.LinAlgError):
            continue
    return pd.Series(vals, dtype=float)


def format_spillover(sp: Spillover, top: int = 5) -> str:
    end = f" đến {sp.end:%d/%m/%Y}" if sp.end is not None else ""
    lines = [f"LAN TRUYỀN BIẾN ĐỘNG (Diebold–Yilmaz, {len(sp.theta)} mã, {sp.sessions} phiên{end})",
             f"Tổng lan truyền: {sp.total:.0%} — phần biến động của mỗi mã do cú sốc từ mã khác "
             "(cao = cả nhóm đi cùng nhau, đa dạng hoá kém hiệu quả)",
             "", "Dẫn biến động (ròng dương):  " + " · ".join(
                 f"{s} {v:+.0%}" for s, v in sp.net.head(top).items()),
             "Chạy theo (ròng âm):        " + " · ".join(
                 f"{s} {v:+.0%}" for s, v in sp.net.tail(top)[::-1].items()),
             "", "Cạnh mạnh nhất (mã phát → mã nhận, % biến động dự báo 10 phiên của mã nhận):"]
    lines += [f"  {j} → {i}: {v:.1%}" for j, i, v in sp.edges(top)]
    lines += ["", "Chỉ tham khảo: đo mã nào 'dẫn' biến động trong quá khứ gần, không phải xác suất dòng "
              "tiền chảy sang mã khác; chưa kiểm định khả năng dự báo."]
    return "\n".join(lines)
