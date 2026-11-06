"""Free fermions against even-sector exact diagonalization, plus H0 tool checks.

Run from the repository root:  python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import minimum_spanning_tree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import h0_tools as h0  # noqa: E402
import tfim_ff as ff  # noqa: E402

SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]])
SZ = np.diag([1.0, -1.0]).astype(complex)


def ed_even_ground_state(L, g, bc, J=None):
    states = np.arange(2 ** L)
    basis = states[np.array([bin(s).count("1") % 2 == 0 for s in states])]
    index = -np.ones(2 ** L, dtype=np.int64)
    index[basis] = np.arange(basis.size)
    bits = (basis[:, None] >> np.arange(L)[None, :]) & 1        # bit 1 <-> sz = -1
    n = basis.size
    rows, cols, vals = [np.arange(n)], [np.arange(n)], [-g * (1 - 2 * bits).sum(axis=1)]
    bonds = [(i, i + 1) for i in range(L - 1)] + ([(L - 1, 0)] if bc == "pbc" else [])
    J = np.ones(len(bonds)) if J is None else J
    for (i, j), Jb in zip(bonds, J):
        rows.append(np.arange(n))
        cols.append(index[basis ^ ((1 << i) | (1 << j))])
        vals.append(-Jb * np.ones(n))
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n))
    _, v = spla.eigsh(H, k=1, which="SA", tol=1e-14)
    psi = np.zeros(2 ** L)
    psi[basis] = v[:, 0]
    return psi.reshape([2] * L)  # axis k <-> site L-1-k


def ed_two_site_rho(T, L, i, j):
    M = np.moveaxis(T, (L - 1 - i, L - 1 - j), (0, 1)).reshape(4, -1)
    return M @ M.conj().T


def ff_two_site_rho(res, i, j):
    m = res["m"]
    rho = (np.eye(4) + m[i] * np.kron(SZ, np.eye(2)) + m[j] * np.kron(np.eye(2), SZ)
           + res["C"][i, j] * np.kron(SX, SX) + res["Gyy"][i, j] * np.kron(SY, SY)
           + res["Gzz"][i, j] * np.kron(SZ, SZ))
    return rho / 4


@pytest.mark.parametrize("bc", ["pbc", "obc"])
@pytest.mark.parametrize("L", [6, 8, 10])
@pytest.mark.parametrize("g", [0.1, 0.3, 0.7, 1.0, 1.5, 3.0, 10.0])
def test_two_site_density_matrices_match_ed(bc, L, g):
    T = ed_even_ground_state(L, g, bc)
    res = ff.ground_state_matrices(L, g, bc)
    assert res["qc"]["orthogonality_error"] < 1e-10
    worst = 0.0
    for i in range(L):
        for j in range(i + 1, L):
            worst = max(worst, np.abs(ed_two_site_rho(T, L, i, j) - ff_two_site_rho(res, i, j)).max())
    assert worst < 1e-10


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("g", [0.3, 1.0, 3.0])
def test_random_couplings_match_ed(seed, g):
    """OBC with J_j = exp(u_j), u_j uniform in [-1, 1], the ensemble of the disorder runs."""
    L = 8
    J = np.exp(np.random.default_rng(seed).uniform(-1.0, 1.0, L - 1))
    T = ed_even_ground_state(L, g, "obc", J)
    res = ff.ground_state_matrices(L, g, "obc", J)
    assert res["qc"]["orthogonality_error"] < 1e-10
    worst = max(np.abs(ed_two_site_rho(T, L, i, j) - ff_two_site_rho(res, i, j)).max()
                for i in range(L) for j in range(i + 1, L))
    assert worst < 1e-10


@pytest.mark.parametrize("bc", ["pbc", "obc"])
def test_mutual_information_matches_ed(bc):
    L, g = 8, 0.9
    T = ed_even_ground_state(L, g, bc)
    res = ff.ground_state_matrices(L, g, bc)

    def S(rho):
        w = np.linalg.eigvalsh(rho)
        w = w[w > 1e-15]
        return -(w * np.log(w)).sum()

    for i in range(L):
        for j in range(i + 1, L):
            rho = ed_two_site_rho(T, L, i, j)
            r4 = rho.reshape(2, 2, 2, 2)
            mi = S(np.einsum("abcb->ac", r4)) + S(np.einsum("abad->bd", r4)) - S(rho)
            assert abs(mi - res["MI"][i, j]) < 1e-10


def test_obc_zero_mode_stays_even_at_large_size():
    res = ff.ground_state_matrices(128, 0.5, "obc")
    qc = res["qc"]
    assert qc["orthogonality_error"] < 1e-8
    assert qc["min_two_site_eigenvalue"] > -1e-10
    assert qc["softest_mode_energy"] < 1e-15   # below machine precision, handled by Schur
    # Even (cat) state: the end-to-end correlator is the squared edge magnetization,
    # m_edge^2 = 1 - g^2 (Pfeuty 1970). An odd/even mixture would spoil it.
    assert abs(res["C"][0, -1] - 0.75) < 1e-10


def test_pbc_matrices_are_translation_invariant():
    res = ff.ground_state_matrices(32, 1.0, "pbc")
    assert res["qc"]["translation_residual"] < 1e-12
    assert abs(res["C"][0, 1] - 1.0 / (32 * np.sin(np.pi / 64))) < 1e-12  # exact at g = 1


def test_prim_matches_scipy():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    D = np.linalg.norm(X[:, None] - X[None], axis=-1)
    w, _ = h0.mst_prim(D)
    ref = minimum_spanning_tree(D).toarray()
    assert np.allclose(np.sort(w), np.sort(ref[ref > 0]))


def test_closed_form_velocity_matches_landscape_integral():
    g = np.linspace(0.5, 1.5, 2001)
    deaths = np.stack([0.3 + 0.2 * np.tanh(5 * (g - 1)), 0.25 + 0.1 * g, 0.2 + 0.05 * g ** 2], 1)
    deaths = np.sort(deaths, axis=1)[:, ::-1]
    closed = h0.velocity_closed(g, deaths)
    numeric = h0.velocity_landscape(g, deaths, t=np.linspace(0, 1, 20001))
    assert np.abs(closed - numeric)[5:-5].max() < 5e-3


def test_pbc_velocity_reduces_to_nearest_neighbour_derivative():
    g = np.linspace(0.8, 1.2, 41)
    d1 = np.array([ff.distance_C(ff.ground_state_matrices(16, x, "pbc")["C"])[0, 1] for x in g])
    deaths = np.array([h0.mst_prim(ff.distance_C(ff.ground_state_matrices(16, x, "pbc")["C"]))[0]
                       for x in g])
    assert np.allclose(deaths, d1[:, None], atol=1e-13)
    xi = h0.velocity_closed(g, deaths)
    assert np.allclose(xi, np.abs(np.gradient(d1, g, edge_order=2)) * np.sqrt(d1 / 2))
