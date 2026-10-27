"""Phân bổ vốn cân bằng rủi ro theo cây phân cụm — HRP (López de Prado 2016) và phụ thuộc đuôi
(nâng cấp v4). CHỈ GỢI Ý, giữ nguyên các trần đang dùng (30% nhóm ngành, nhiệt danh mục 5%) cho tới
khi kiểm định (lệnh study-hrp) cho thấy HRP tốt hơn ở cả hai giai đoạn.

HRP, trên lợi suất ngày `sessions` phiên gần nhất:
1. Khoảng cách tương quan d_ij = √(½(1 − ρ_ij)); phân cụm phân cấp (liên kết đơn) → cây.
2. Xếp các mã theo thứ tự lá của cây (mã giống nhau đứng cạnh nhau).
3. Chia đôi đệ quy: mỗi nửa có phương sai cụm V = wᵀΣw với w ∝ 1/phương sai từng mã; tỷ trọng chia
   theo α = 1 − V_trái/(V_trái + V_phải). Không cần nghịch đảo ma trận hiệp phương sai (ổn định hơn
   tối ưu Markowitz khi ít dữ liệu).

Phụ thuộc đuôi dưới thực nghiệm (thay cho ước lượng copula Student-t, ít giả định hơn):
    λ_L(i, j) ≈ P(U_i ≤ q, U_j ≤ q) / q,  U = hạng lợi suất / (n + 1),  q = 5%
λ_L ≈ q nếu hai mã độc lập ở vùng đuôi; càng gần 1 thì càng hay cùng giảm mạnh một ngày.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    c = cov[np.ix_(idx, idx)]
    w = 1 / np.diag(c)
    w /= w.sum()
    return float(w @ c @ w)


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Tỷ trọng HRP (tổng 1) cho các cột của `returns` (lợi suất ngày, đã bỏ dòng thiếu)."""
    r = returns.dropna(axis=1, how="all").dropna()
    cols = list(r.columns)
    if len(cols) == 0:
        return pd.Series(dtype=float)
    if len(cols) == 1:
        return pd.Series([1.0], index=cols)
    cov = r.cov().to_numpy()
    corr = np.clip(r.corr().to_numpy(), -1, 1)
    dist = np.sqrt(np.clip(0.5 * (1 - corr), 0, None))
    np.fill_diagonal(dist, 0)
    order = list(leaves_list(linkage(squareform(dist, checks=False), method="single")))
    w = np.ones(len(cols))
    stack = [order]
    while stack:
        items = stack.pop()
        if len(items) < 2:
            continue
        half = len(items) // 2
        left, right = items[:half], items[half:]
        vl, vr = _cluster_var(cov, left), _cluster_var(cov, right)
        a = 1 - vl / (vl + vr)
        w[left] *= a
        w[right] *= 1 - a
        stack += [left, right]
    return pd.Series(w / w.sum(), index=cols).iloc[order]


def tail_dependence(returns: pd.DataFrame, q: float = 0.05) -> pd.DataFrame:
    """Ma trận λ_L thực nghiệm (đường chéo = 1)."""
    r = returns.dropna()
    u = r.rank() / (len(r) + 1)
    low = (u <= q).astype(float).to_numpy()
    lam = (low.T @ low) / len(r) / q
    np.fill_diagonal(lam, 1.0)
    return pd.DataFrame(lam, index=r.columns, columns=r.columns)


def top_pairs(lam: pd.DataFrame, n: int = 5) -> list[tuple[str, str, float]]:
    cols = list(lam.columns)
    pairs = [(cols[i], cols[j], float(lam.iat[i, j])) for i in range(len(cols))
             for j in range(i + 1, len(cols))]
    return sorted(pairs, key=lambda x: -x[2])[:n]


@dataclass
class HrpView:
    weights: pd.Series
    current: pd.Series            # tỷ trọng hiện tại trong phần đã giải ngân (nếu có)
    pairs: list = field(default_factory=list)
    sessions: int = 0
    q: float = 0.05


def format_hrp(v: HrpView) -> str:
    lines = [f"PHÂN BỔ CÂN BẰNG RỦI RO (HRP, {v.sessions} phiên — gợi ý, chưa thay trần đang dùng)",
             f"{'Mã':<6}{'HRP':>8}{'Hiện tại':>10}"]
    for s, w in v.weights.items():
        cur = v.current.get(s, float("nan"))
        lines.append(f"{s:<6}{w:>8.0%}{'—' if pd.isna(cur) else f'{cur:.0%}':>10}")
    if v.pairs:
        lines.append(f"\nCặp hay cùng giảm mạnh (phụ thuộc đuôi λ_L, ngày tệ nhất {v.q:.0%}; độc lập "
                     f"≈ {v.q:.2f}):")
        lines += [f"  {a}–{b}: {x:.2f}" for a, b, x in v.pairs]
    lines.append("\nHRP chia vốn sao cho mỗi nhánh cây (nhóm mã đi cùng nhau) gánh rủi ro tương đương. "
                 "Kiểm định: study-hrp.")
    return "\n".join(lines)
