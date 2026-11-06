"""The three disorder summaries of Section 4, Eqs. (13)-(15), on the random-bond chain.

Data: runs/disorder_obc_v1 (src/run_disorder.py), OBC, J_j = exp(u_j) with u_j uniform
in [-1, 1], R realizations per L. For each distance (MI, C) and size,
  xi_r(g)          velocity of realization r (Proposition 8, closed form in its deaths)
  xi_bar           mean of the xi_r                                   Eq. (14)
  xi_rms           root mean square of the xi_r                       Eq. (15)
  xi_lambda_bar    velocity of the mean landscape                     Eq. (13)
All velocities are divided by sqrt(L-1), the velocity per bar of the other figures.

xi_lambda_bar is computed in two exact ways that must agree:
  (i)  Proposition 22, Eq. (17), the double sum over pairs of realizations;
  (ii) the step form d_g lambda_bar_k(t) = (1/R) sum_r d_g d_{r,k} 1[d_{r,k}/2 < t < d_{r,k}],
       whose square is integrated exactly by sorting the interval ends. (ii) is linear in the
       weights of the realizations and gives the bootstrap bands.
and, for L = 16, against a direct integration of the mean landscape on a grid in t.
Derivatives in g: second-order finite differences of the sorted deaths, as in h0_tools.

Figures (exploratory, PNG + PDF), summary_disorder.json and the plotted curves (disorder_curves.npz)
go to analysis/figures_disorder/.
Run from the repository root:  python analysis\\analyze_disorder_summaries.py
Redraw the figures from disorder_curves.npz only:  add --plot-only
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
MP = HERE.parent
RUN = MP / "runs" / "disorder_obc_v1"
OUT = HERE / "figures_disorder"
FIGURES = HERE / "figures.py"   # shared style of the math figures
N_BOOT = 200
RNG = np.random.default_rng(20261103)

spec = importlib.util.spec_from_file_location("figures", FIGURES)
figs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figs)
plt = figs.plt


def load() -> dict:
    data = {}
    for p in sorted(RUN.glob("obc_L*_disorder.h5"), key=lambda q: int(q.stem.split("_L")[1].split("_")[0])):
        with h5py.File(p, "r") as f:
            data[int(f.attrs["L"])] = {"g": f["g"][()], "MI": f["data/MI/deaths"][()], "C": f["data/C/deaths"][()]}
    return data


def derivatives(g: np.ndarray, deaths: np.ndarray) -> np.ndarray:
    """d_g of the sorted deaths, deaths[r, g, k]."""
    return np.gradient(deaths, g, axis=1, edge_order=2)


def individual(deaths: np.ndarray, dd: np.ndarray) -> np.ndarray:
    """xi_r(g) / sqrt(L-1), Proposition 8."""
    return np.sqrt((dd ** 2 * deaths / 2.0).mean(axis=2))


def mean_landscape_pairs(deaths: np.ndarray, dd: np.ndarray) -> np.ndarray:
    """xi_lambda_bar / sqrt(L-1) from Eq. (17), the double sum over pairs of realizations."""
    R, n_g, n_k = deaths.shape
    out = np.zeros(n_g)
    for j in range(n_g):
        for k0 in range(0, n_k, 8):                               # blocks of bars keep R x R x 8 in memory
            d, a = deaths[:, j, k0:k0 + 8], dd[:, j, k0:k0 + 8]
            lo = np.minimum(d[:, None, :], d[None, :, :])
            hi = np.maximum(d[:, None, :], d[None, :, :])
            out[j] += (a[:, None, :] * a[None, :, :] * np.clip(lo - 0.5 * hi, 0.0, None)).sum()
    return np.sqrt(out / R ** 2 / n_k)


class StepForm:
    """Exact step form of d_g lambda_bar_k: on (d_{r,k}/2, d_{r,k}) layer k of realization r moves
    at rate d_g d_{r,k}. The interval ends do not depend on the weights of the realizations, so
    they are sorted once and every bootstrap resample is a cumulative sum."""

    def __init__(self, deaths: np.ndarray, dd: np.ndarray):
        ends = np.concatenate([deaths / 2.0, deaths], axis=0)      # (2R, g, k)
        self.order = np.argsort(ends, axis=0)
        self.gaps = np.diff(np.take_along_axis(ends, self.order, axis=0), axis=0)
        self.dd, self.n_k = dd, deaths.shape[2]

    def velocity(self, weights: np.ndarray | None = None) -> np.ndarray:
        """xi_lambda_bar / sqrt(L-1) (weights: bootstrap counts of the realizations)."""
        w = np.ones(self.dd.shape[0]) if weights is None else weights
        rates = w[:, None, None] * self.dd / w.sum()
        level = np.cumsum(np.take_along_axis(np.concatenate([rates, -rates], axis=0), self.order, axis=0), axis=0)
        return np.sqrt((level[:-1] ** 2 * self.gaps).sum(axis=(0, 2)) / self.n_k)


def mean_landscape_grid(g: np.ndarray, deaths: np.ndarray, t: np.ndarray) -> np.ndarray:
    """xi_lambda_bar / sqrt(L-1) by averaging the landscape layers on a t grid (check)."""
    R, n_g, n_k = deaths.shape
    lam = np.zeros((n_g, n_k, t.size))
    for r in range(R):
        lam += np.maximum(0.0, np.minimum(t[None, None, :], deaths[r][:, :, None] - t[None, None, :]))
    lam /= R
    d = np.gradient(lam, g, axis=0, edge_order=2)
    return np.sqrt(figs.TRAPEZOID((d ** 2).sum(axis=1), t, axis=1) / n_k)


def summaries(g: np.ndarray, deaths: np.ndarray) -> dict:
    dd = derivatives(g, deaths)
    xr = individual(deaths, dd)
    steps = StepForm(deaths, dd)
    res = {"bar": xr.mean(0), "rms": np.sqrt((xr ** 2).mean(0)), "lam": steps.velocity()}
    res["lam_pairs"] = mean_landscape_pairs(deaths, dd)
    boots = {"bar": [], "rms": [], "lam": [], "ratio": []}
    R = deaths.shape[0]
    for _ in range(N_BOOT):
        w = np.bincount(RNG.integers(0, R, R), minlength=R).astype(float)
        bar, rms = (w[:, None] * xr).sum(0) / w.sum(), np.sqrt((w[:, None] * xr ** 2).sum(0) / w.sum())
        lam = steps.velocity(w)
        for k, v in (("bar", bar), ("rms", rms), ("lam", lam), ("ratio", lam / rms)):
            boots[k].append(v)
    res["band"] = {k: np.percentile(np.array(v), [2.5, 97.5], axis=0) for k, v in boots.items()}
    res["ratio"] = res["lam"] / res["rms"]
    res["xi_r"] = xr                                               # kept for disorder_curves.npz
    res["boot"] = {k: np.array(v) for k, v in boots.items()}
    return res


# Layout: one panel of the three speeds per L, and the
# ratio of all L in a bottom panel spanning both columns. Presentation only, the curves are those
# of summaries().
SPEEDS = (("lam", r"$\xi_{\overline{\lambda}}$", "#0072B2", "-"),     # Okabe-Ito blue
          ("bar", r"$\overline{\xi}$", "#D55E00", "--"),              # vermilion
          ("rms", r"$\xi_{\mathrm{rms}}$", "#009E73", ":"))           # bluish green
RATIO_STYLE = {16: ("#999999", "-", "o"), 32: ("#707070", "--", "s"),
               64: ("#404040", "-.", "^"), 128: ("#000000", ":", "D")}
RATIO_YLIM = (0.4, 1.0)
REF_LINE = {"color": "#b5b5b5", "ls": "--", "lw": 1.0, "zorder": 0}
FIG_RC = {"font.size": 11, "axes.labelsize": 12.5, "axes.titlesize": 12, "xtick.labelsize": 10.5,
          "ytick.labelsize": 10.5, "legend.fontsize": 11, "grid.color": "#e4e7eb", "grid.linewidth": 0.6,
          "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white"}


def _markevery(n: int, i: int, n_marks: int = 10) -> np.ndarray:
    """About n_marks evenly spaced sample indices, shifted with i so markers of different L do not coincide."""
    step = (n - 1) / n_marks
    return np.unique(np.round(step * (np.arange(n_marks) + (i + 0.5) / 4)).astype(int))


def figure(name: str, sizes: list[int], g: np.ndarray, S: dict) -> None:
    with plt.rc_context(FIG_RC):
        fig = plt.figure(figsize=(10.0, 9.0), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.08)                    # room between the g labels and title (e)
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.85])
        speed_ax = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(len(sizes))]
        ratio_ax = fig.add_subplot(gs[2, :])
        xlim = (g.min(), g.max())

        # one vertical scale for the twelve speed curves (no uncertainty bands are drawn on them)
        lo = min(S[L][k].min() for L in sizes for k, *_ in SPEEDS)
        hi = max(S[L][k].max() for L in sizes for k, *_ in SPEEDS)
        pad = 0.05 * (hi - lo)
        ylim = (lo - pad, hi + pad)

        for i, (a, L) in enumerate(zip(speed_ax, sizes)):
            a.axvline(1.0, **REF_LINE)
            for key, lab, c, ls in SPEEDS:
                a.plot(g, S[L][key], color=c, ls=ls, lw=2.0, label=lab)
            a.set(xlim=xlim, ylim=ylim)
            a.set_title(f"({'abcd'[i]}) $L = {L}$")
            if i % 2 == 0:
                a.set_ylabel(r"$\frac{\xi}{\sqrt{L-1}}$", fontsize=15)
            else:
                a.tick_params(labelleft=False)
            if i < 2:
                a.tick_params(labelbottom=False)
            else:
                a.set_xlabel(r"$g$")
        fig.legend(handles=speed_ax[0].get_lines()[1:], loc="outside upper center", ncol=3,
                   handlelength=3.2, columnspacing=2.5)

        a = ratio_ax
        a.axvline(1.0, **REF_LINE)
        for i, L in enumerate(sizes):
            c, ls, m = RATIO_STYLE[L]
            a.fill_between(g, *S[L]["band"]["ratio"], color=c, alpha=0.18, lw=0)
            a.plot(g, S[L]["ratio"], color=c, ls=ls, lw=1.9, marker=m, ms=4.5,
                   markevery=_markevery(g.size, i), label=fr"$L = {L}$")
        a.set(xlim=xlim, ylim=RATIO_YLIM, xlabel=r"$g$",
              ylabel=r"$\xi_{\overline{\lambda}}\,/\,\xi_{\mathrm{rms}}$",
              title="(e) Ratio of mean-landscape speed to RMS speed")
        a.legend(loc="lower right", ncol=4, handlelength=3.2, columnspacing=1.8)
        figs.save(fig, f"obc_disorder_summaries_{name}")


# Everything summaries() produces, so the figures can be redrawn, and other summaries computed,
# without reading the runs again. Keys "<MI|C>__L<L>__<field>", all velocities divided by sqrt(L-1):
#   lam, bar, rms, ratio          (n_g,)        the plotted curves; lam_pairs is lam from Eq. (17)
#   band_<lam|bar|rms|ratio>      (2, n_g)      bootstrap percentiles 2.5 and 97.5
#   boot_<lam|bar|rms|ratio>      (N_BOOT, n_g) every bootstrap resample
#   xi_r                          (R, n_g)      velocity of each realization (Proposition 8)
CURVES = "disorder_curves.npz"
FIELDS = ("lam", "bar", "rms", "ratio")


def save_curves(path: Path, g: np.ndarray, sizes: list[int], all_S: dict) -> None:
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    meta = {"run": str(RUN.relative_to(MP)), "script": Path(__file__).name, "n_boot": N_BOOT,
            "rng_seed": 20261103, "bootstrap_percentiles": [2.5, 97.5],
            "normalization": "velocities divided by sqrt(L-1); ratio dimensionless",
            "h5_sha256": {e["file"]: e["sha256"] for e in manifest["entries"]}}
    arr = {"g": g, "sizes": np.array(sizes), "meta_json": np.array(json.dumps(meta))}
    for name, S in all_S.items():
        for L in sizes:
            s, p = S[L], f"{name}__L{L}__"
            for k in FIELDS:
                arr[p + k] = s[k]
                arr[p + "band_" + k] = s["band"][k]
                arr[p + "boot_" + k] = s["boot"][k]
            arr[p + "lam_pairs"] = s["lam_pairs"]
            arr[p + "xi_r"] = s["xi_r"]
    np.savez_compressed(path, **arr)


def load_curves(path: Path) -> tuple[np.ndarray, list[int], dict]:
    with np.load(path) as f:
        g, sizes = f["g"], [int(L) for L in f["sizes"]]
        all_S = {}
        for name in ("MI", "C"):
            all_S[name] = {L: {k: f[f"{name}__L{L}__{k}"] for k in FIELDS} for L in sizes}
            for L in sizes:
                all_S[name][L]["band"] = {k: f[f"{name}__L{L}__band_{k}"] for k in FIELDS}
    return g, sizes, all_S


def main() -> None:
    final = OUT
    figs.OUT = Path(tempfile.mkdtemp(prefix="figures_disorder_"))   # matplotlib writes outside the synced folder
    if "--plot-only" in sys.argv:                                  # redraw from the cached curves
        g, sizes, all_S = load_curves(final / CURVES)
        for name in ("MI", "C"):
            figure(name, sizes, g, all_S[name])
        for p in sorted(figs.OUT.iterdir()):
            shutil.copy2(p, final / p.name)
        print(f"figures in {final}")
        return
    data = load()
    all_S = {}
    sizes = sorted(data)
    g = data[sizes[0]]["g"]
    summary = {"run": str(RUN.relative_to(MP)), "sizes": sizes, "R": {}, "n_boot": N_BOOT}
    for name in ("MI", "C"):
        S, rows = {}, []
        for L in sizes:
            t0 = time.time()
            deaths = data[L][name]
            summary["R"][str(L)] = int(deaths.shape[0])
            s = summaries(g, deaths)
            S[L] = s
            row = {"L": L,
                   "eq17_vs_steps_max_abs": float(np.abs(s["lam_pairs"] - s["lam"]).max()),
                   "ordering_violations": int(((s["lam"] > s["bar"] * (1 + 1e-12)) |
                                               (s["bar"] > s["rms"] * (1 + 1e-12))).sum()),
                   "min_ratio": float(s["ratio"].min()), "g_min_ratio": float(g[np.argmin(s["ratio"])])}
            for key in ("lam", "bar", "rms"):
                pk = figs.quadratic_peak(g, s[key])
                row[f"peak_{key}"] = {"g": pk["x"], "height": pk["value"]}
            if L == sizes[0]:
                grid = mean_landscape_grid(g, deaths, np.linspace(0.0, 1.0, 2001))
                row["steps_vs_t_grid_max_rel"] = float(np.abs(grid - s["lam"]).max() / s["lam"].max())
            rows.append(row)
            print(name, L, f"{time.time() - t0:.0f} s", json.dumps(row), flush=True)
        figure(name, sizes, g, S)
        summary[name] = rows
        all_S[name] = S
    (figs.OUT / "summary_disorder.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    save_curves(figs.OUT / CURVES, g, sizes, all_S)
    final.mkdir(exist_ok=True)
    for p in sorted(figs.OUT.iterdir()):
        shutil.copy2(p, final / p.name)
    print(f"figures in {final}")


if __name__ == "__main__":
    main()
