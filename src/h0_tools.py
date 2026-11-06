"""H0/H1 persistence summaries used in the math paper.

All finite H0 bars of a Vietoris-Rips filtration are born at 0 and die at the
minimum-spanning-tree edge weights. Their tents are nested, so the k-th landscape
layer is the tent of the k-th largest death and

    xi_lambda^2 = (1/(L-1)) sum_k (d_(k)')^2 d_(k) / 2 .

`velocity_closed` evaluates this formula. `velocity_landscape` integrates the
layers numerically, which is the definition used in the physics paper.
"""
from __future__ import annotations

import numpy as np

TRAPEZOID = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def mst_prim(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum spanning tree of a dense distance matrix (zeros allowed).

    Returns the L-1 edge weights (descending) and the edges as an (L-1, 2) array.
    """
    L = D.shape[0]
    in_tree = np.zeros(L, dtype=bool)
    in_tree[0] = True
    best = D[0].astype(float).copy()
    parent = np.zeros(L, dtype=int)
    weights, edges = [], []
    for _ in range(L - 1):
        cand = np.where(in_tree, np.inf, best)
        v = int(np.argmin(cand))
        weights.append(best[v])
        edges.append((parent[v], v))
        in_tree[v] = True
        closer = D[v] < best
        best = np.where(closer, D[v], best)
        parent = np.where(closer, v, parent)
    order = np.argsort(weights)[::-1]
    return np.asarray(weights)[order], np.asarray(edges)[order]


def mst_is_nearest_neighbour(edges: np.ndarray, L: int, bc: str) -> bool:
    """True if every MST edge joins nearest neighbours of the chain (ring for PBC)."""
    sep = np.abs(edges[:, 0] - edges[:, 1])
    if bc == "pbc":
        sep = np.minimum(sep, L - sep)
    return bool(np.all(sep == 1))


def h0_nn_ties(D: np.ndarray, bc: str, rtol: float = 1e-12) -> bool:
    """True if every off-diagonal distance is >= the smallest nearest-neighbour distance.

    In PBC this is the hypothesis under which all H0 deaths equal d_1.
    """
    L = D.shape[0]
    nn = np.diagonal(D, 1)
    if bc == "pbc":
        nn = np.append(nn, D[0, L - 1])
    off = D[~np.eye(L, dtype=bool)]
    return bool(off.min() >= nn.min() * (1 - rtol))


def velocity_closed(g: np.ndarray, deaths: np.ndarray) -> np.ndarray:
    """xi_lambda from sorted H0 deaths (n_g, L-1), with second-order finite differences in g."""
    dd = np.gradient(deaths, g, axis=0, edge_order=2)
    return np.sqrt((dd ** 2 * deaths / 2.0).mean(axis=1))


def velocity_landscape(g: np.ndarray, deaths: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
    """xi_lambda from finite differences of the landscape layers integrated on a t grid."""
    if t is None:
        t = np.linspace(0.0, max(1.0, float(deaths.max())), 4001)
    out = np.empty(g.size)
    layers = np.empty((g.size, deaths.shape[1], t.size))
    for k in range(g.size):
        tents = np.maximum(0.0, np.minimum(t[None, :], deaths[k][:, None] - t[None, :]))
        layers[k] = np.sort(tents, axis=0)[::-1]
    d = np.gradient(layers, g, axis=0, edge_order=2)
    out[:] = np.sqrt(TRAPEZOID((d ** 2).sum(axis=1), t, axis=1) / deaths.shape[1])
    return out


def quadratic_peak(g: np.ndarray, y: np.ndarray) -> dict:
    """Discrete argmax refined by a three-point parabola (same rule as the physics paper)."""
    k = int(np.argmax(y))
    if k == 0 or k == y.size - 1:
        return {"x": float(g[k]), "value": float(y[k]), "edge": True}
    x = g[k - 1:k + 2]
    c = np.polyfit(x - x[1], y[k - 1:k + 2], 2)
    if c[0] >= 0:
        return {"x": float(g[k]), "value": float(y[k]), "edge": False}
    xs = -c[1] / (2 * c[0])
    return {"x": float(x[1] + xs), "value": float(np.polyval(c, xs)), "edge": False}


def h1_summary(D: np.ndarray, threshold: float = 1e-9) -> dict:
    """Vietoris-Rips H1 (Z2) of a distance matrix: most persistent bar and bar count."""
    import gudhi

    rips = gudhi.RipsComplex(distance_matrix=D)
    st = rips.create_simplex_tree(max_dimension=2)
    st.compute_persistence(homology_coeff_field=2, persistence_dim_max=False)
    bars = np.asarray(st.persistence_intervals_in_dimension(1), dtype=float).reshape(-1, 2)
    if bars.size == 0:
        return {"n_bars": 0, "birth": np.nan, "death": np.nan, "lifetime": 0.0,
                "total_persistence": 0.0}
    life = bars[:, 1] - bars[:, 0]
    keep = life > threshold
    if not keep.any():
        return {"n_bars": 0, "birth": np.nan, "death": np.nan, "lifetime": 0.0,
                "total_persistence": 0.0}
    bars, life = bars[keep], life[keep]
    k = int(np.argmax(life))
    return {"n_bars": int(keep.sum()), "birth": float(bars[k, 0]), "death": float(bars[k, 1]),
            "lifetime": float(life[k]), "total_persistence": float(life.sum())}
