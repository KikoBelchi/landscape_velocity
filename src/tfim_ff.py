"""Free-fermion ground state of Olsthoorn's TFIM, H = -J sum sx_i sx_{i+1} - h sum sz_i.

We set J = 1, so g = h/J = h. The state is always the ground state of the even-parity
sector P = prod sz = +1 (the Z2-symmetric state given by exact diagonalization at finite L).

Majoranas: a_j = (prod_{k<j} sz_k) sx_j and b_j = (prod_{k<j} sz_k) sy_j, ordered as
gamma = (a_0, b_0, a_1, b_1, ...). The covariance Gamma is real antisymmetric, with
<gamma_m gamma_n> = delta_mn + i Gamma_mn and <P> = Pf(Gamma).

Two-site Pauli expectations (i < j):
    <sz_i>         = Gamma[a_i, b_i]
    <sx_i sx_j>    = det[ Gamma[b_k, a_{l+1}] ]_{k,l=i..j-1}
    <sy_i sy_j>    = det[-Gamma[a_k, b_{l+1}] ]_{k,l=i..j-1}
    <sz_i sz_j>    = m_i m_j + Gamma[a_i, b_j] Gamma[b_i, a_j]
All other two-site Pauli expectations vanish by parity and reality. The two-site reduced
density matrix is therefore an X state, and its spectrum is known in closed form.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

LN2 = np.log(2.0)


def majorana_matrix(L: int, g: float, bc: str, J: np.ndarray | None = None) -> np.ndarray:
    """A with H = (i/4) gamma^T A gamma. In the even sector PBC means antiperiodic fermions.

    J: optional bond couplings J_j of -J_j sx_j sx_{j+1} (L-1 values for OBC, L for PBC, the
    last one closing the ring). None gives the clean chain, J_j = 1.
    """
    if bc not in ("pbc", "obc"):
        raise ValueError(bc)
    if bc == "pbc" and L % 2:
        raise ValueError("PBC requires even L")
    n_bonds = L if bc == "pbc" else L - 1
    J = np.ones(n_bonds) if J is None else np.asarray(J, dtype=float)
    if J.shape != (n_bonds,):
        raise ValueError(f"J must have {n_bonds} entries for {bc}")
    A = np.zeros((2 * L, 2 * L))

    def add(m: int, n: int, w: float) -> None:  # adds i w gamma_m gamma_n to H
        A[m, n] += 2.0 * w
        A[n, m] -= 2.0 * w

    for j in range(L):
        add(2 * j, 2 * j + 1, g)             # -g sz_j = i g a_j b_j
    for j in range(L - 1):
        add(2 * j + 1, 2 * j + 2, J[j])      # -J_j sx_j sx_{j+1} = i J_j b_j a_{j+1}
    if bc == "pbc":
        add(2 * L - 1, 0, -J[L - 1])         # antiperiodic boundary for P = +1
    return A


def _schur_blocks(A: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], np.ndarray]:
    """Real Schur form A = Z T Z^T split into 2x2 blocks (unpaired 1x1 zeros are paired)."""
    T, Z = sla.schur(A, output="real")
    n = A.shape[0]
    blocks, eps, singles = [], [], []
    i = 0
    while i < n:
        if i + 1 < n and abs(T[i + 1, i]) > 0.0:
            blocks.append((i, i + 1))
            eps.append(0.5 * (T[i, i + 1] - T[i + 1, i]))
            i += 2
        else:
            singles.append(i)
            i += 1
    if len(singles) % 2:
        raise RuntimeError("odd number of 1x1 Schur blocks")
    for p, q in zip(singles[::2], singles[1::2]):
        blocks.append((p, q))
        eps.append(0.0)
    return Z, blocks, np.asarray(eps)


def covariance(L: int, g: float, bc: str, J: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Even-parity ground-state covariance Gamma and QC numbers."""
    A = majorana_matrix(L, g, bc, J)
    Z, blocks, eps = _schur_blocks(A)
    signs = np.where(eps >= 0.0, 1.0, -1.0)
    # Pf(Gamma) = det(Z) * prod(signs) for Gamma = Z (+) s_j J Z^T, block order as found
    perm = np.array([x for b in blocks for x in b])
    det_z = np.linalg.det(Z[:, perm])
    parity = float(np.sign(det_z) * np.prod(signs))
    flipped = False
    if parity < 0:
        if bc == "pbc":
            raise RuntimeError(f"NS vacuum has odd parity (L={L}, g={g})")
        softest = int(np.argmin(np.abs(eps)))
        signs[softest] *= -1.0
        flipped = True
    Gp = np.zeros_like(A)
    for (p, q), s in zip(blocks, signs):
        Gp[p, q], Gp[q, p] = s, -s
    G = Z @ Gp @ Z.T
    G = 0.5 * (G - G.T)
    qc = {
        "softest_mode_energy": float(np.min(np.abs(eps))),
        "parity_flipped": flipped,
        "max_aa_bb": float(max(np.abs(G[0::2, 0::2]).max(), np.abs(G[1::2, 1::2]).max())),
        "orthogonality_error": float(np.abs(G @ G + np.eye(2 * L)).max()),
    }
    return G, qc


def _leading_minors(M: np.ndarray) -> tuple[np.ndarray, int]:
    """All leading principal minors of M by unpivoted LU, with slogdet fallback."""
    A = np.array(M, dtype=float, copy=True)
    n = A.shape[0]
    minors = np.empty(n)
    running = 1.0
    unstable_from = None
    for k in range(n):
        piv = A[k, k]
        if not np.isfinite(piv) or abs(piv) < 1e-12 * max(1.0, abs(running)) or running == 0.0:
            unstable_from = k
            break
        running *= piv
        minors[k] = running
        if k + 1 < n:
            A[k + 1:, k + 1:] -= np.outer(A[k + 1:, k] / piv, A[k, k + 1:])
    fallbacks = 0
    if unstable_from is not None:
        for k in range(unstable_from, n):
            sign, logdet = np.linalg.slogdet(M[:k + 1, :k + 1])
            minors[k] = sign * np.exp(logdet)
            fallbacks += 1
    return minors, fallbacks


def string_correlators(G: np.ndarray, L: int, rows: list[int] | None = None) -> dict:
    """m_i, and <sx_i sx_j>, <sy_i sy_j>, <sz_i sz_j> for i in rows (default all), j > i."""
    a = np.arange(L) * 2
    b = a + 1
    m = G[a, b].copy()
    Mxx = G[np.ix_(b[:-1], a[1:])]          # Gamma[b_k, a_{l+1}]
    Myy = -G[np.ix_(a[:-1], b[1:])]         # -Gamma[a_k, b_{l+1}]
    Gxx, Gyy = np.eye(L), np.eye(L)
    fallbacks = 0
    for i in (range(L - 1) if rows is None else rows):
        mx, fx = _leading_minors(Mxx[i:, i:])
        my, fy = _leading_minors(Myy[i:, i:])
        Gxx[i, i + 1:] = Gxx[i + 1:, i] = mx
        Gyy[i, i + 1:] = Gyy[i + 1:, i] = my
        fallbacks += fx + fy
    Gzz = np.outer(m, m) + G[np.ix_(a, b)] * G[np.ix_(b, a)]
    np.fill_diagonal(Gzz, 1.0)
    return {"m": m, "Gxx": Gxx, "Gyy": Gyy, "Gzz": Gzz, "lu_fallbacks": fallbacks}


def circulant_from_row(row: np.ndarray) -> np.ndarray:
    L = row.size
    idx = (np.arange(L)[None, :] - np.arange(L)[:, None]) % L
    return row[idx]


def xlogx(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0.0, p * np.log(p), 0.0)


def two_site_spectrum(mi, mj, Gxx, Gyy, Gzz) -> np.ndarray:
    """Eigenvalues of the X-state rho_ij, last axis of length 4."""
    r1 = np.sqrt((mi + mj) ** 2 + (Gxx - Gyy) ** 2)
    r2 = np.sqrt((mi - mj) ** 2 + (Gxx + Gyy) ** 2)
    return 0.25 * np.stack([1 + Gzz + r1, 1 + Gzz - r1, 1 - Gzz + r2, 1 - Gzz - r2], axis=-1)


def mutual_information(corr: dict) -> tuple[np.ndarray, float]:
    """MI_ij in nats (zero diagonal) and the minimum two-site eigenvalue (QC)."""
    m = corr["m"]
    mi, mj = m[:, None], m[None, :]
    spec = two_site_spectrum(mi, mj, corr["Gxx"], corr["Gyy"], corr["Gzz"])
    S2 = -xlogx(spec).sum(axis=-1)
    S1 = -(xlogx((1 + m) / 2) + xlogx((1 - m) / 2))
    MI = S1[:, None] + S1[None, :] - S2
    np.fill_diagonal(MI, 0.0)
    off = ~np.eye(m.size, dtype=bool)
    return MI, float(spec[off].min())


def ground_state_matrices(L: int, g: float, bc: str, J: np.ndarray | None = None) -> dict:
    """C_ij = <sx_i sx_j>, MI_ij (nats), single-site m_i, and QC for one (L, g, bc).

    With random couplings J the ring is not translation invariant, so only OBC is allowed.
    """
    if J is not None and bc == "pbc":
        raise ValueError("random couplings are supported with OBC only")
    G, qc = covariance(L, g, bc, J)
    if bc == "pbc":
        corr = string_correlators(G, L, rows=[0])
        trans = 0.0
        for key in ("Gxx", "Gyy", "Gzz"):
            row = corr[key][0].copy()
            trans = max(trans, float(np.abs(row[1:] - row[1:][::-1]).max()))
            corr[key] = circulant_from_row(row)
        trans = max(trans, float(np.abs(corr["m"] - corr["m"][0]).max()))
        corr["m"] = np.full(L, corr["m"][0])
        qc["translation_residual"] = trans
    else:
        corr = string_correlators(G, L)
        qc["translation_residual"] = float("nan")
    MI, min_eig = mutual_information(corr)
    qc["lu_fallbacks"] = corr["lu_fallbacks"]
    qc["min_two_site_eigenvalue"] = min_eig
    return {"C": corr["Gxx"], "MI": MI, "m": corr["m"], "Gyy": corr["Gyy"],
            "Gzz": corr["Gzz"], "qc": qc}


def distance_MI(MI: np.ndarray) -> np.ndarray:
    """Olsthoorn distance normalized by 2 ln 2: D = 1 - MI/(2 ln 2), zero diagonal."""
    D = 1.0 - MI / (2.0 * LN2)
    np.fill_diagonal(D, 0.0)
    return D


def distance_C(C: np.ndarray) -> np.ndarray:
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    return D
