# In this repository this module provides the plotting style and the figure functions used by
# analysis/analyze_olsthoorn_suite.py and analysis/analyze_disorder_summaries.py. Its own main()
# rebuilds the figures from a package of raw matrices that is not part of the repository.
"""Reproduce the exploratory figures of the math-paper report from the raw matrices.

Standalone port of analysis/analyze_olsthoorn_suite.py. The original reads our
production HDF5 runs. This version recomputes everything from raw_data_MI/ and raw_data_C/,
so it needs no file outside this package.

Needs numpy, scipy, matplotlib and gudhi (gudhi only for H1, see --no-h1).
Run from anywhere:
    python reproduce_plots.py --workers 8
Writes reproduced/ with the same 15 figures (PNG + PDF) as analysis/,
summary_reproduced.json, and a comparison with reference/summary_olsthoorn_v1.json.

Definitions (as in the report):
  D_MI = 1 - MI/(2 ln 2) off the diagonal, D_ii = 0      D_C = 1 - C
  H0 deaths = minimum-spanning-tree weights (Vietoris-Rips, infinite bar excluded)
  xi^2 = (1/(L-1)) sum_k d_(k) (d_(k)')^2 / 2   (exact landscape integral for bars born at 0)
  d/dg by second-order finite differences, peak refined by a three-point parabola.
  H1 = Vietoris-Rips over Z2 (GUDHI), bars shorter than 1e-9 dropped.

The package holds L <= 128. The report also shows PBC L = 256, 512 and OBC L = 256, so the
size-scaling figures here have fewer sizes and the fitted PBC slope uses L = 32, 64, 128.
Every peak and QC number for the included sizes must match the reference to round-off.
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker  # noqa: E402
from scipy.integrate import quad  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reproduced"
TRAPEZOID = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
LN2, PI = np.log(2.0), np.pi
TWO_LN2 = 2.0 * LN2
A_C = np.sqrt(0.5 - 1 / PI) / PI
A_MI = 0.10359598823077024          # exact constant, see documents/log_divergence_velocity.pdf
D01_OLSTHOORN = 0.744293
BC_SHORT = {"periodic": "pbc", "open": "obc"}
FILT_LABEL = {"MI": r"$D=1-M/2\ln2$ (MI)", "C": r"$D=1-\langle\sigma^x\sigma^x\rangle$ (C)"}
INK, MUTED, GRID = "#1f2328", "#6e7781", "#d0d7de"
H1_KEYS = ("n_bars", "birth", "death", "lifetime", "total_persistence")

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.4,
    "legend.frameon": False, "savefig.dpi": 200,
})


# ------------------------------------------------------------- H0/H1 tools (src/h0_tools.py)
def distance(name: str, M: np.ndarray) -> np.ndarray:
    D = 1.0 - M / TWO_LN2 if name == "MI" else 1.0 - M
    np.fill_diagonal(D, 0.0)
    return D


def mst_prim(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum spanning tree of a dense distance matrix: L-1 weights (descending) and edges."""
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
    sep = np.abs(edges[:, 0] - edges[:, 1])
    if bc == "pbc":
        sep = np.minimum(sep, L - sep)
    return bool(np.all(sep == 1))


def h0_nn_ties(D: np.ndarray, bc: str, rtol: float = 1e-12) -> bool:
    """True if every off-diagonal distance is >= the smallest nearest-neighbour distance."""
    L = D.shape[0]
    nn = np.diagonal(D, 1)
    if bc == "pbc":
        nn = np.append(nn, D[0, L - 1])
    off = D[~np.eye(L, dtype=bool)]
    return bool(off.min() >= nn.min() * (1 - rtol))


def velocity_closed(g: np.ndarray, deaths: np.ndarray) -> np.ndarray:
    dd = np.gradient(deaths, g, axis=0, edge_order=2)
    return np.sqrt((dd ** 2 * deaths / 2.0).mean(axis=1))


def velocity_landscape(g: np.ndarray, deaths: np.ndarray, t: np.ndarray) -> np.ndarray:
    """xi_lambda from finite differences of the landscape layers integrated on a t grid."""
    layers = np.empty((g.size, deaths.shape[1], t.size))
    for k in range(g.size):
        tents = np.maximum(0.0, np.minimum(t[None, :], deaths[k][:, None] - t[None, :]))
        layers[k] = np.sort(tents, axis=0)[::-1]
    d = np.gradient(layers, g, axis=0, edge_order=2)
    return np.sqrt(TRAPEZOID((d ** 2).sum(axis=1), t, axis=1) / deaths.shape[1])


def quadratic_peak(g: np.ndarray, y: np.ndarray) -> dict:
    k = int(np.argmax(y))
    if k == 0 or k == y.size - 1:
        return {"x": float(g[k]), "value": float(y[k]), "edge": True}
    x = g[k - 1:k + 2]
    c = np.polyfit(x - x[1], y[k - 1:k + 2], 2)
    if c[0] >= 0:
        return {"x": float(g[k]), "value": float(y[k]), "edge": False}
    xs = -c[1] / (2 * c[0])
    return {"x": float(x[1] + xs), "value": float(np.polyval(c, xs)), "edge": False}


def kink_after_max(g: np.ndarray, y: np.ndarray) -> dict:
    """V-shaped cusp of y after its maximum (Olsthoorn Fig. 12(b), OBC Delta d of MI).

    First local minimum after the global maximum, at grid resolution (no fit, the point is not
    differentiable). Returns NaN when there is none, as for Delta d of C, which decays smoothly.
    """
    km = int(np.argmax(y))
    for k in range(km + 1, y.size - 1):
        if y[k] < y[k - 1] and y[k] <= y[k + 1]:
            return {"x": float(g[k]), "value": float(y[k]), "index": k}
    return {"x": float("nan"), "value": float("nan"), "index": None}


def h1_summary(D: np.ndarray, threshold: float = 1e-9) -> dict:
    import gudhi

    st = gudhi.RipsComplex(distance_matrix=D).create_simplex_tree(max_dimension=2)
    st.compute_persistence(homology_coeff_field=2, persistence_dim_max=False)
    bars = np.asarray(st.persistence_intervals_in_dimension(1), dtype=float).reshape(-1, 2)
    life = bars[:, 1] - bars[:, 0]
    keep = life > threshold
    if not keep.any():
        return {"n_bars": 0, "birth": np.nan, "death": np.nan, "lifetime": 0.0, "total_persistence": 0.0}
    bars, life = bars[keep], life[keep]
    k = int(np.argmax(life))
    return {"n_bars": int(keep.sum()), "birth": float(bars[k, 0]), "death": float(bars[k, 1]),
            "lifetime": float(life[k]), "total_persistence": float(life.sum())}


# ------------------------------------------------------------- matrices -> derived data
def one_point(task: tuple) -> dict:
    """Everything the figures need at one (bc, L, grid, g), from the MI and C matrices."""
    paths, bc, L, do_h1 = task
    out = {}
    for name in ("MI", "C"):
        M = np.load(paths[name])
        if name == "C":
            out["M"] = float(np.sqrt(max(M.mean(), 0.0)))
        D = distance(name, M)
        deaths, edges = mst_prim(D)
        out[name] = {"deaths": deaths,
                     "nn": np.diagonal(D, 1).copy() if bc == "obc" else np.array([D[0, 1]]),
                     "mst_is_nn": mst_is_nearest_neighbour(edges, L, bc),
                     "hyp_N": h0_nn_ties(D, bc),
                     "h1": h1_summary(D) if do_h1 else None}
    return out


def read_manifests() -> dict:
    """(bc, L, grid) -> manifest rows of both observables, sorted by grid index."""
    groups = defaultdict(dict)
    for name in ("MI", "C"):
        folder = ROOT / f"raw_data_{name}"
        with (folder / "manifest.csv").open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (BC_SHORT[row["bc"]], int(row["L"]), row["grid"])
                row["abs_path"] = str(folder / row["matrix_path"])
                groups[key].setdefault(name, []).append(row)
    for key, by_name in groups.items():
        for rows in by_name.values():
            rows.sort(key=lambda r: int(r["grid_index"]))
        gm = [float(r["g"]) for r in by_name["MI"]]
        if gm != [float(r["g"]) for r in by_name["C"]]:
            raise RuntimeError(f"MI and C grids differ for {key}")
    return dict(sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])))


def compute(workers: int, do_h1: bool) -> dict:
    data = {}
    pool = Pool(workers) if workers > 1 else None
    try:
        for (bc, L, grid), rows in read_manifests().items():
            t0 = time.time()
            tasks = [({"MI": a["abs_path"], "C": b["abs_path"]}, bc, L, do_h1)
                     for a, b in zip(rows["MI"], rows["C"])]
            pts = (pool.map(one_point, tasks, chunksize=max(1, len(tasks) // (4 * workers)))
                   if pool and len(tasks) > 1 else [one_point(t) for t in tasks])
            r = {"g": np.array([float(x["g"]) for x in rows["C"]]),
                 "M": np.array([p["M"] for p in pts]),
                 "qc": {"min_two_site_eigenvalue": np.array([float(x["min_two_site_eigenvalue"]) for x in rows["C"]]),
                        "orthogonality_error": np.array([float(x["orthogonality_error"]) for x in rows["C"]]),
                        "parity_flipped": np.array([x["parity_flipped"] == "True" for x in rows["C"]])}}
            for name in ("MI", "C"):
                r[name] = {"deaths": np.array([p[name]["deaths"] for p in pts]),
                           "nn_distances": np.array([p[name]["nn"] for p in pts]),
                           "mst_is_nn": np.array([p[name]["mst_is_nn"] for p in pts]),
                           "hypothesis_N": np.array([p[name]["hyp_N"] for p in pts])}
                if do_h1:
                    r[name]["h1"] = {k: np.array([p[name]["h1"][k] for p in pts]) for k in H1_KEYS}
            data[(bc, L, grid)] = r
            print(f"{bc} L={L:<4d} {grid:9s} {len(tasks):5d} points  {time.time() - t0:6.1f} s", flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    return data


def sizes_in(data: dict, bc: str, grid: str) -> list[int]:
    return sorted(L for (b, L, gr) in data if b == bc and gr == grid)


ARGMAX_G = r"\underset{g}{\mathrm{arg\,max}}"   # g centred under the whole "arg max"
OURS_CMAP = "Oranges"   # landscape-velocity curves (this project); Olsthoorn's observables stay blue
# distances whose *_scaling right panel keeps the dashed slope-A line and its legend
# (removed for MI in the figures of the paper; add "MI" to restore it)
SLOPE_LINE = {"C"}


def size_colors(sizes, cmap="Blues"):
    """One hue, light to dark with L (L is ordinal magnitude)."""
    cmap = plt.get_cmap(cmap)
    return {L: cmap(0.35 + 0.6 * i / max(1, len(sizes) - 1)) for i, L in enumerate(sorted(sizes))}


# ------------------------------------------------------------- analytic TL curves
def _R(k, g):
    return np.maximum(np.sqrt(np.maximum(1 + g * g - 2 * g * np.cos(k), 0.0)), 1e-300)


def _tl(f, g):
    return quad(lambda k: f(k, g), 0, PI, limit=400)[0] / PI


SYM = {"m": lambda k, g: (g - np.cos(k)) / _R(k, g),
       "Gxx": lambda k, g: (1 - g * np.cos(k)) / _R(k, g),
       "Gyy": lambda k, g: (np.cos(2 * k) - g * np.cos(k)) / _R(k, g)}


def xlogx(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0.0, p * np.log(p), 0.0)


def two_site_spectrum(mi, mj, Gxx, Gyy, Gzz) -> np.ndarray:
    """Eigenvalues of the two-site X state rho_ij."""
    r1 = np.sqrt((mi + mj) ** 2 + (Gxx - Gyy) ** 2)
    r2 = np.sqrt((mi - mj) ** 2 + (Gxx + Gyy) ** 2)
    return 0.25 * np.stack([1 + Gzz + r1, 1 + Gzz - r1, 1 - Gzz + r2, 1 - Gzz - r2], axis=-1)


def _F(x):
    m, gxx, gyy = x
    spec = two_site_spectrum(m, m, gxx, gyy, m * m - gxx * gyy)
    return float(2 * -(xlogx(np.array((1 + m) / 2)) + xlogx(np.array((1 - m) / 2)))
                 + xlogx(np.asarray(spec)).sum())


def tl_velocity(g: np.ndarray) -> dict:
    """xi_infinity(g) for both distances via Lemma 3 (closed form in d_1)."""
    out = {"MI": np.full(g.size, np.nan), "C": np.full(g.size, np.nan)}
    h = 1e-6
    for i, x in enumerate(g):
        if abs(x - 1) < 2e-3:
            continue
        xs = [np.array([_tl(SYM[k], x + s) for k in ("m", "Gxx", "Gyy")]) for s in (-h, 0, h)]
        dC = [1 - v[1] for v in xs]
        dM = [1 - _F(v) / (2 * LN2) for v in xs]
        out["C"][i] = abs(dC[2] - dC[0]) / (2 * h) * np.sqrt(dC[1] / 2)
        out["MI"][i] = abs(dM[2] - dM[0]) / (2 * h) * np.sqrt(dM[1] / 2)
    return out


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def fraction_ylabel(ax, prefix: str = "") -> None:
    """Draw a 12 pt fraction with any surrounding label text at the normal size."""
    fraction = TextArea(r"$\frac{\xi}{\sqrt{L-1}}$", textprops={"fontsize": 12, "rotation": 90})
    parts = [fraction]
    if prefix:
        parts.append(TextArea(prefix, textprops={"fontsize": plt.rcParams["axes.labelsize"], "rotation": 90}))
    label = VPacker(children=parts, align="center", pad=0, sep=2)
    box = AnchoredOffsetbox(loc="center right", child=label, pad=0, frameon=False,
                            bbox_to_anchor=(-0.18, 0.5), bbox_transform=ax.transAxes, borderpad=0)
    ax.add_artist(box)


# ------------------------------------------------------------- barcodes
H0_COLOR, H1_COLOR = "#cf222e", "#0969da"   # Olsthoorn's convention, H0 red and H1 blue


def barcodes(data: dict, summary: dict) -> None:
    """Analogue of Olsthoorn Fig. 6(a-c) (PBC, L=10) and of the OBC case (L=12)."""
    for bc, L in (("pbc", 10), ("obc", 12)):
        r = data.get((bc, L, "wide"))
        if r is None:
            continue
        for name in ("MI", "C"):
            fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.8), sharey=True)
            rows = []
            for a, target, title in zip(ax, (0.1, 1.0, 10.0), (r"$g=0.1$", r"$g=1$", r"$g=10$")):
                k = int(np.argmin(np.abs(r["g"] - target)))
                d = np.sort(r[name]["deaths"][k])
                for i, x in enumerate(d):
                    a.plot([0, x], [i, i], color=H0_COLOR, lw=1.6, solid_capstyle="butt")
                a.plot([0, 1.0], [L - 1, L - 1], color=H0_COLOR, lw=1.6, alpha=0.45, ls="--")  # infinite bar
                a.axhline(L - 0.25, color=MUTED, lw=0.6)
                h1 = r[name].get("h1")
                life = 0.0
                if h1 is not None and h1["n_bars"][k] > 0:
                    b, e = h1["birth"][k], h1["death"][k]
                    life = float(e - b)
                    a.plot([b, e], [L + 0.6, L + 0.6], color=H1_COLOR, lw=2.4, solid_capstyle="butt")
                a.set(xlim=(0, 1.02), ylim=(-0.8, L + 1.6), xlabel=r"$t$ (distance)", title=title)
                a.set_yticks([(L - 2) / 2, L + 0.6], [r"$H_0$", r"$H_1$"])
                a.grid(axis="y", visible=False)
                rows.append({"g": float(r["g"][k]), "h0_deaths_min_max": [float(d.min()), float(d.max())],
                             "h1_lifetime": life})
            fig.tight_layout()
            save(fig, f"{bc}_barcodes_{name}")
            summary.setdefault("barcodes", {})[f"{bc}_{name}"] = rows


# ------------------------------------------------------------- PBC
def pbc_figures(data: dict, summary: dict) -> None:
    olst = [L for L in (10, 14, 18, 22) if ("pbc", L, "wide") in data]
    col, ours = size_colors(olst), size_colors(olst, OURS_CMAP)
    for name in ("MI", "C"):
        fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
        for L in olst:
            r = data[("pbc", L, "wide")]
            g, d = r["g"], r[name]["deaths"]
            ax[0].plot(g, d.mean(1), color=col[L], label=fr"$L={L}$")
            if "h1" in r[name]:
                ax[1].plot(g, r[name]["h1"]["lifetime"], color=col[L])
            ax[2].plot(g, velocity_closed(g, d), color=ours[L], label=fr"$L={L}$")
        for a, yl in zip(ax[:2], ("mean $H_0$ lifetime", "longest $H_1$ lifetime")):
            a.set_xscale("log")
            a.axvline(1.0, color=MUTED, ls="--", lw=0.8)
            a.set(xlabel=r"$g=h/J$", ylabel=yl)
        ax[2].set_xscale("log")
        ax[2].axvline(1.0, color=MUTED, ls="--", lw=0.8)
        ax[2].set_xlabel(r"$g=h/J$")
        fraction_ylabel(ax[2])
        ax[0].legend(loc="lower right")
        ax[2].legend(loc="upper right")      # own legend: the velocity curves are orange, not blue
        fig.tight_layout()
        save(fig, f"pbc_fig6_{name}")

    sizes = sizes_in(data, "pbc", "fine")
    col = size_colors(sizes, OURS_CMAP)
    for name in ("MI", "C"):
        rows = []
        fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
        for L in sizes:
            r = data[("pbc", L, "fine")]
            g = r["g"]
            xi = velocity_closed(g, r[name]["deaths"])
            pk = quadratic_peak(g, xi)
            row = {"L": L, "g_star": pk["x"], "height": pk["value"]}
            # H1 from the full fine grid when present (L <= 128), otherwise from the h1peak window
            # (the fine points in [0.98, 1.06], run olsthoorn_pbc_h1peak_v1) if that run is loaded
            src = r if "h1" in r[name] else data.get(("pbc", L, "h1peak"))
            if src is not None and "h1" in src[name]:
                pk_h1 = quadratic_peak(src["g"], src[name]["h1"]["lifetime"])
                if pk_h1["edge"]:                      # maximum outside the window: no reliable peak
                    row["H1_peak_at_grid_edge"] = True
                else:
                    row["g_star_H1_lifetime"], row["height_H1_lifetime"] = pk_h1["x"], pk_h1["value"]
                    row["H1_grid"] = "fine" if src is r else "h1peak"
            rows.append(row)
            ax[0].plot(g, xi, color=col[L], lw=1.1, label=fr"$L={L}$")
        ax[0].set(xlabel=r"$g=h/J$", xlim=(0.7, 1.3))
        fraction_ylabel(ax[0])
        ax[0].legend(ncol=1, loc="upper left", fontsize=6.0, handlelength=1.4, labelspacing=0.25)
        Ls = np.array([r_["L"] for r_ in rows])
        gs = np.array([r_["g_star"] for r_ in rows])
        ax[1].loglog(Ls, np.abs(gs - 1), "o-", color=col[Ls.max()], ms=4, label=r"$\xi$")
        h1 = [(r_["L"], r_["g_star_H1_lifetime"]) for r_ in rows if "g_star_H1_lifetime" in r_]
        if h1:
            a, b = np.array(h1).T
            ax[1].loglog(a, np.abs(b - 1), "s--", color=size_colors(sizes)[max(sizes)], ms=4, label=r"$H_1$ lifetime")
        ax[1].set(xlabel=r"$L$", ylabel=r"$|g_{\max,L}-1|$")
        ax[1].legend()
        hts = np.array([r_["height"] for r_ in rows])
        amp = A_MI if name == "MI" else A_C
        ax[2].plot(np.log(Ls), hts, "o", color=col[Ls.max()], ms=4, label=r"$\max_g\,\frac{\xi}{\sqrt{L-1}}$")
        if h1:   # height of Olsthoorn's summary of the middle panel, same colour and marker, no line
            a = np.array([r_["L"] for r_ in rows if "height_H1_lifetime" in r_])
            b = np.array([r_["height_H1_lifetime"] for r_ in rows if "height_H1_lifetime" in r_])
            ax[2].plot(np.log(a), b, "s", color=size_colors(sizes)[max(sizes)], ms=4,
                       label=r"$\max_g$ longest $H_1$ lifetime")
        fit = np.polyfit(np.log(Ls[Ls >= 32]), hts[Ls >= 32], 1) if (Ls >= 32).sum() >= 2 else None
        xx = np.log(np.array([Ls.min(), Ls.max()]))
        if name in SLOPE_LINE:
            ax[2].plot(xx, hts[-1] + amp * (xx - np.log(Ls[-1])), color=INK, ls="--", lw=1.0,
                       label=fr"slope $A_{{\rm {name}}}={amp:.4f}$")
        ax[2].legend(loc="upper left")
        ax[2].set(xlabel=r"$\ln L$", ylabel="peak height")
        fig.tight_layout()
        save(fig, f"pbc_scaling_{name}")
        summary["pbc"][name] = {"peaks": rows, "theory_slope": amp,
                                "fitted_slope_L>=32": None if fit is None else float(fit[0])}

    crit = sizes_in(data, "pbc", "critical")
    if crit and "h1" in data[("pbc", crit[0], "critical")]["MI"]:
        fig, ax = plt.subplots(figsize=(3.6, 3.0))
        b = [data[("pbc", L, "critical")]["MI"]["h1"] for L in crit]
        ax.plot(crit, [x["birth"][0] for x in b], "o-", color="#0969da", ms=4, label="birth")
        ax.plot(crit, [x["death"][0] for x in b], "s-", color="#8250df", ms=4, label="death")
        ax.axhline(D01_OLSTHOORN, color=MUTED, ls=":", lw=0.9)
        ax.axhline(1.0, color=MUTED, ls=":", lw=0.9)
        ax.set(xlabel=r"$L$", ylabel=r"$\epsilon\ [2\ln2]$")
        ax.legend()
        fig.tight_layout()
        save(fig, "pbc_fig7_MI")
        summary["pbc"]["fig7_MI"] = [{"L": L, "birth": float(x["birth"][0]), "death": float(x["death"][0])}
                                     for L, x in zip(crit, b)]


# ------------------------------------------------------------- OBC
def obc_figures(data: dict, summary: dict) -> None:
    olst = [L for L in (12, 16, 20) if ("obc", L, "wide") in data]
    col, ours = size_colors(olst), size_colors(olst, OURS_CMAP)
    for name in ("MI", "C"):
        fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))   # the panel of M was dropped
        has_corner = False
        for L in olst:
            r = data[("obc", L, "wide")]
            g, d = r["g"], r[name]["deaths"]
            dd = d.max(1) - d.min(1)
            ax[0].plot(g, dd, color=col[L], label=fr"$L={L}$")
            kn = kink_after_max(g, dd)
            if kn["index"] is not None:
                has_corner = True
                ax[0].plot(kn["x"], kn["value"], "o", ms=4.5, mfc="white", mec=col[L], mew=1.4, zorder=5)
            mx = quadratic_peak(g, dd)                 # the estimator of Olsthoorn's Fig. 12
            ax[0].plot(mx["x"], mx["value"], "^", ms=5, color=col[L], mec="white", mew=0.6, zorder=5)
            ax[1].plot(g, velocity_closed(g, d), color=ours[L], label=fr"$L={L}$")
        ax[0].set_xscale("log")
        ax[0].axvline(1.0, color=MUTED, ls="--", lw=0.8)
        ax[0].set(xlabel=r"$g=h/J$", ylabel=r"$\Delta d=\max\,d-\min\,d$ for $H_0$")
        ax[1].set_xscale("log")
        ax[1].axvline(1.0, color=MUTED, ls="--", lw=0.8)
        ax[1].set_xlabel(r"$g=h/J$")
        fraction_ylabel(ax[1])
        for a in ax:
            a.legend(loc="best")
        marks = [plt.Line2D([], [], ls="", marker="^", ms=5, color=MUTED, label="maximum")]
        if has_corner:                                 # Delta d of C decays smoothly, it has no corner
            marks.append(plt.Line2D([], [], ls="", marker="o", ms=4.5, mfc="white", mec=MUTED, mew=1.4,
                                    label="corner"))
        ax[0].add_artist(ax[0].legend(loc="upper right"))
        ax[0].legend(handles=marks, loc="upper left")  # above the rising branch of the curves
        fig.tight_layout()
        save(fig, f"obc_fig12_{name}")

    sizes = sizes_in(data, "obc", "fine")
    col = size_colors(sizes, OURS_CMAP)
    for name in ("MI", "C"):
        rows = []
        fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.0))
        for L in sizes:
            r = data[("obc", L, "fine")]
            g, d, nn = r["g"], r[name]["deaths"], r[name]["nn_distances"]
            xi = velocity_closed(g, d)
            # Olsthoorn's OBC estimator: the kink of Delta d after its maximum (not the maximum)
            pk, pk_dd = quadratic_peak(g, xi), kink_after_max(g, d.max(1) - d.min(1))
            mx_dd = quadratic_peak(g, d.max(1) - d.min(1))
            # bond-resolved decomposition of xi^2 at the peak (bonds = MST edges here)
            dn = np.gradient(nn, g, axis=0, edge_order=2)
            contrib = (dn ** 2 * nn / 2) / (L - 1)
            k = int(np.argmin(np.abs(g - pk["x"])))
            bulk = slice(L // 4, L - 1 - L // 4)
            row = {"L": L, "g_star": pk["x"], "height": pk["value"], "g_star_delta_d_kink": pk_dd["x"],
                   "g_star_delta_d_max": mx_dd["x"],
                   "height_delta_d_max": mx_dd["value"], "height_delta_d_kink": pk_dd["value"],
                   "bulk_half_fraction_of_xi2_at_peak": float(contrib[k, bulk].sum() / contrib[k].sum())}
            if "h1" in r[name]:
                row["max_H1_bars"] = int(r[name]["h1"]["n_bars"].max())
            rows.append(row)
            ax[0].plot(g, xi, color=col[L], lw=1.1, label=fr"$L={L}$")
        ax[0].set(xlabel=r"$g=h/J$")
        fraction_ylabel(ax[0])
        ax[0].legend(ncol=1, loc="upper left", fontsize=6.0, handlelength=1.4, labelspacing=0.25)
        Ls = np.array([r_["L"] for r_ in rows])
        # neutral axis label: the panel shows two estimators, told apart by the legend
        blues = size_colors(sizes)
        for key, mk, c, lab in (("g_star", "o-", col[Ls.max()], fr"${ARGMAX_G}\,\xi$"),
                                ("g_star_delta_d_max", "^-.", blues[sizes[len(sizes) // 2]],
                                 fr"${ARGMAX_G}\,\Delta d$"),
                                ("g_star_delta_d_kink", "s--", blues[max(sizes)],
                                 r"$g_{\mathrm{corner},L}$ of $\Delta d$")):
            ys = np.array([r_[key] for r_ in rows], dtype=float)
            if np.isfinite(ys).any():                  # Delta d of C has no corner: nothing to draw
                ax[1].plot(1 / Ls, ys, mk, color=c, ms=4, label=lab)
        ax[1].axhline(1.0, color=INK, lw=0.8, ls=":", label=r"$g_c=1$")
        ax[1].set(xlabel=r"$1/L$", ylabel=r"Estimated critical field $g_L$")
        # MI: the only free area is the gap between the xi and the Delta d maximum series
        ax[1].legend(**({"loc": "lower left", "bbox_to_anchor": (0.0, 0.26), "fontsize": 6.5}
                        if name == "MI" else {"loc": "best"}))
        ax[2].plot(np.log(Ls), [r_["height"] for r_ in rows], "o", color=col[Ls.max()], ms=4,
                   label=r"$\max_g\,\frac{\xi}{\sqrt{L-1}}$")
        # heights of Olsthoorn's summaries of the middle panel, same colours and markers, no lines
        for key, mk, c, lab in (("height_delta_d_max", "^", blues[sizes[len(sizes) // 2]],
                                 r"$\max_g\,\Delta d$"),
                                ("height_delta_d_kink", "s", blues[max(sizes)],
                                 r"$\Delta d$ at $g_{\mathrm{corner},L}$")):
            ys = np.array([r_[key] for r_ in rows], dtype=float)
            if np.isfinite(ys).any():                  # Delta d of C has no corner: nothing to draw
                ax[2].plot(np.log(Ls), ys, mk, color=c, ms=4, label=lab)
        amp = A_MI if name == "MI" else A_C
        xx = np.log(np.array([Ls.min(), Ls.max()]))
        if name in SLOPE_LINE:
            ax[2].plot(xx, rows[-1]["height"] + amp * (xx - np.log(Ls[-1])), color=INK, ls="--", lw=1.0,
                       label=fr"PBC slope $A_{{\rm {name}}}$")
        ax[2].legend(loc="upper left")
        ax[2].set(xlabel=r"$\ln L$", ylabel="peak height")
        fig.tight_layout()
        save(fig, f"obc_scaling_{name}")
        summary["obc"][name] = {"peaks": rows}


# ------------------------------------------------------------- QC
def qc(data: dict, summary: dict) -> None:
    """Same entries as the reference, except lu_fallbacks (not exported in the manifests)."""
    out = {}
    for (bc, L, grid), r in data.items():
        e = {f"mst_is_nn_{n}": bool(r[n]["mst_is_nn"].all()) for n in ("MI", "C")}
        e.update({f"hypothesis_N_{n}": bool(r[n]["hypothesis_N"].all()) for n in ("MI", "C")})
        e["min_two_site_eigenvalue"] = float(r["qc"]["min_two_site_eigenvalue"].min())
        e["max_orthogonality_error"] = float(r["qc"]["orthogonality_error"].max())
        e["parity_flips"] = int(r["qc"]["parity_flipped"].sum())
        if grid == "fine":
            g = r["g"]
            for n in ("MI", "C"):
                d = r[n]["deaths"]
                ref = quadratic_peak(g, velocity_closed(g, d))["x"]
                e[f"dg_convergence_{n}"] = {
                    str(round(0.001 * s, 3)): quadratic_peak(g[::s], velocity_closed(g[::s], d[::s]))["x"] - ref
                    for s in (2, 5, 10)}
                if L <= 32:
                    sel = slice(max(0, np.argmin(np.abs(g - ref)) - 60), np.argmin(np.abs(g - ref)) + 61)
                    num = velocity_landscape(g[sel], d[sel], t=np.linspace(0, 1, 4001))
                    clo = velocity_closed(g[sel], d[sel])
                    e[f"closed_vs_landscape_max_rel_{n}"] = float(np.max(np.abs(num - clo)[2:-2] / clo.max()))
        out[f"{bc}_L{L}_{grid}.h5"] = e     # keyed like the reference (production file names)
    summary["qc"] = out


# ------------------------------------------------------------- comparison with the reference
def _diffs(ref, new, path=""):
    """Yield (path, |ref-new|) for every number present in both, and (path, None) for mismatches."""
    if isinstance(ref, dict) and isinstance(new, dict):
        for k in ref.keys() & new.keys() - {"fitted_slope_L>=32"}:   # depends on the sizes present
            yield from _diffs(ref[k], new[k], f"{path}/{k}")
    elif isinstance(ref, list) and isinstance(new, list):
        if ref and isinstance(ref[0], dict) and "L" in ref[0]:
            by_L = {x["L"]: x for x in new}
            for x in ref:
                if x["L"] in by_L:
                    yield from _diffs(x, by_L[x["L"]], f"{path}[L={x['L']}]")
        else:
            for i, (a, b) in enumerate(zip(ref, new)):
                yield from _diffs(a, b, f"{path}[{i}]")
    elif isinstance(ref, bool) or isinstance(new, bool):
        yield path, 0.0 if ref == new else None
    elif isinstance(ref, (int, float)) and isinstance(new, (int, float)):
        if np.isnan(ref) and np.isnan(new):
            yield path, 0.0
        else:
            yield path, abs(float(ref) - float(new))


def compare(summary: dict) -> dict:
    ref_path = ROOT / "reference" / "summary_olsthoorn_v1.json"
    if not ref_path.exists():
        return {}
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    report = {}
    for section in ("pbc", "obc", "barcodes", "qc"):
        diffs = list(_diffs(ref.get(section, {}), summary.get(section, {}), section))
        bad = [p for p, d in diffs if d is None]
        num = [(d, p) for p, d in diffs if d is not None]
        worst = max(num) if num else (0.0, "")
        report[section] = {"compared": len(diffs), "boolean_mismatches": bad,
                           "max_abs_diff": worst[0], "at": worst[1]}
    for bc in ("pbc", "obc"):
        ref_L = {p["L"] for p in ref.get(bc, {}).get("MI", {}).get("peaks", [])}
        new_L = {p["L"] for p in summary.get(bc, {}).get("MI", {}).get("peaks", [])}
        report[f"{bc}_sizes_only_in_reference"] = sorted(ref_L - new_L)
    report["pbc_fitted_slope_L>=32"] = {
        n: {"reference": ref["pbc"][n]["fitted_slope_L>=32"], "reproduced": summary["pbc"][n]["fitted_slope_L>=32"],
            "note": "not expected to match, the reference also fits L = 256 and 512"}
        for n in ("MI", "C") if n in ref.get("pbc", {}) and n in summary["pbc"]}
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--no-h1", action="store_true", help="skip H1 (no gudhi needed, H1 panels stay empty)")
    ap.add_argument("--recompute", action="store_true", help="ignore reproduced/derived_data.pkl")
    a = ap.parse_args()
    do_h1 = not a.no_h1
    if do_h1:
        try:
            import gudhi  # noqa: F401
        except ImportError:
            raise SystemExit("gudhi is needed for H1 (pip install gudhi), or run with --no-h1")
    OUT.mkdir(exist_ok=True)
    cache = OUT / "derived_data.pkl"
    data = None
    if cache.exists() and not a.recompute:
        with cache.open("rb") as fh:
            stored = pickle.load(fh)
        if stored["h1"] or not do_h1:
            data = stored["data"]
            print(f"using cached {cache.name} (pass --recompute to rebuild it)")
    if data is None:
        data = compute(a.workers, do_h1)
        with cache.open("wb") as fh:
            pickle.dump({"h1": do_h1, "data": data}, fh)

    summary = {"pbc": {}, "obc": {}, "A_C": A_C, "A_MI": A_MI}
    pbc_figures(data, summary)
    obc_figures(data, summary)
    barcodes(data, summary)
    qc(data, summary)
    summary["comparison_with_reference"] = compare(summary)
    (OUT / "summary_reproduced.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    for bc in ("pbc", "obc"):
        for name in ("MI", "C"):
            for p in summary[bc][name]["peaks"]:
                print(f"{bc} {name:2s} L={p['L']:4d}  g*={p['g_star']:.6f}  max xi={p['height']:.5f}")
    print(json.dumps(summary["comparison_with_reference"], indent=1, default=float))
    print(f"figures and summary in {OUT}")


if __name__ == "__main__":
    main()
