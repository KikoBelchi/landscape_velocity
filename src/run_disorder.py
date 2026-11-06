"""Free-fermion disorder runs for Section 4 of the math paper (three disorder summaries).

Random-bond transverse-field Ising chain, H = -sum_j J_j sx_j sx_{j+1} - g sum_j sz_j, OBC,
with J_j = exp(u_j) and u_j uniform in [-1, 1], so that <ln J> = 0 and the infinite chain is
critical at g = 1. Realization r of size L uses np.random.default_rng([L, r]) for its L-1
couplings, independently of the other sizes. Same state, distances and H0 deaths as
run_suite.py (MI main distance, C secondary one), on a uniform grid of 201 fields.

One HDF5 per L in runs/<run_id>/, with the couplings and the H0 deaths of every
realization, plus manifest.json with SHA-256 hashes. Every realization is cached as soon as it
is done (cache_L<L>/r<r>.npz), so an interrupted run resumes where it stopped.

Example (Anaconda Prompt, from the repository root):
python src\\run_disorder.py --run-id disorder_obc_v1 --L 16 32 64 128 --R 500 --workers 12
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import h0_tools as h0  # noqa: E402
import tfim_ff as ff  # noqa: E402
from run_suite import FILTRATIONS, QC_KEYS, RUNS, sha256  # noqa: E402

GRID = np.round(np.linspace(0.5, 1.5, 201), 6)   # uniform dg = 0.005
LOG_J_WIDTH = 1.0                                 # u_j uniform in [-LOG_J_WIDTH, LOG_J_WIDTH]


def couplings(L: int, r: int) -> np.ndarray:
    return np.exp(np.random.default_rng([L, r]).uniform(-LOG_J_WIDTH, LOG_J_WIDTH, L - 1))


def one_realization(args: tuple) -> Path:
    L, r, cache = args
    path = cache / f"r{r:04d}.npz"
    if path.exists():
        return path
    J = couplings(L, r)
    out = {name: {"deaths": [], "nn": [], "mst_is_nn": [], "hyp_N": []} for name in FILTRATIONS}
    qc = {key: [] for key in QC_KEYS}
    for g in GRID:
        res = ff.ground_state_matrices(L, float(g), "obc", J)
        for key in QC_KEYS:
            qc[key].append(float(res["qc"][key]))
        for name, D in (("MI", ff.distance_MI(res["MI"])), ("C", ff.distance_C(res["C"]))):
            deaths, edges = h0.mst_prim(D)
            out[name]["deaths"].append(deaths)
            out[name]["nn"].append(np.diagonal(D, 1).copy())
            out[name]["mst_is_nn"].append(h0.mst_is_nearest_neighbour(edges, L, "obc"))
            out[name]["hyp_N"].append(h0.h0_nn_ties(D, "obc"))
    arrays = {"J": J, **{f"qc_{k}": np.array(v) for k, v in qc.items()}}
    for name in FILTRATIONS:
        arrays.update({f"{name}_{k}": np.array(v) for k, v in out[name].items()})
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return path


def run_L(L: int, R: int, out_dir: Path, workers: int) -> dict:
    cache = out_dir / f"cache_L{L}"
    cache.mkdir(exist_ok=True)
    t0 = time.time()
    tasks = [(L, r, cache) for r in range(R)]
    with Pool(workers) as pool:
        for k, _ in enumerate(pool.imap_unordered(one_realization, tasks), 1):
            if k % max(1, R // 10) == 0:
                print(f"L={L}: {k}/{R} realizations, {time.time() - t0:.0f} s", flush=True)
    parts = [np.load(cache / f"r{r:04d}.npz") for r in range(R)]
    path = out_dir / f"obc_L{L}_disorder.h5"
    with h5py.File(path, "w") as f:
        f.attrs.update({
            "model": "H = -sum_j J_j sx_j sx_{j+1} - g sum_j sz_j, random bonds, OBC",
            "disorder": f"J_j = exp(u_j), u_j uniform in [-{LOG_J_WIDTH}, {LOG_J_WIDTH}], <ln J> = 0, g_c = 1",
            "seeds": "realization r of size L: np.random.default_rng([L, r])",
            "bc": "obc", "L": L, "R": R,
            "state": "ground state of the even-parity sector P = prod sz = +1",
            "method": "free fermions (Majorana covariance, real Schur parity fixing)",
            "D_MI": "1 - MI/(2 ln 2), MI in nats (main distance)",
            "D_C": "1 - <sx_i sx_j>, raw correlator (secondary distance)",
            "h0_deaths": "minimum-spanning-tree weights, sorted descending, infinite bar removed",
            "layout": "data/<D>/<key>[realization, g, ...]",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        f["g"] = GRID
        f["realization"] = np.arange(R)
        f["J"] = np.array([p["J"] for p in parts])
        for name in FILTRATIONS:
            grp = f.create_group(f"data/{name}")
            for key, ds in (("deaths", "deaths"), ("nn", "nn_distances"), ("mst_is_nn", "mst_is_nn"),
                            ("hyp_N", "hypothesis_N")):
                grp.create_dataset(ds, data=np.array([p[f"{name}_{key}"] for p in parts]),
                                   compression="gzip", compression_opts=4)
        for key in QC_KEYS:
            f[f"qc/{key}"] = np.array([p[f"qc_{key}"] for p in parts])
    qc_all = {k: np.array([p[f"qc_{k}"] for p in parts]) for k in QC_KEYS}
    qc = {
        "max_orthogonality_error": float(qc_all["orthogonality_error"].max()),
        "min_two_site_eigenvalue": float(qc_all["min_two_site_eigenvalue"].min()),
        "parity_flips": int(qc_all["parity_flipped"].sum()),
        "lu_fallbacks": int(qc_all["lu_fallbacks"].sum()),
        "mst_is_nn_fraction": {n: float(np.mean([p[f"{n}_mst_is_nn"] for p in parts])) for n in FILTRATIONS},
    }
    entry = {"file": path.name, "sha256": sha256(path), "L": L, "R": R, "n_g": int(GRID.size),
             "seconds": round(time.time() - t0, 1), "qc": qc}
    print(json.dumps(entry), flush=True)
    return entry


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--L", type=int, nargs="+", required=True)
    p.add_argument("--R", type=int, default=500)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()

    out_dir = RUNS / a.run_id
    manifest_path = out_dir / "manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "run_id": a.run_id, "created_utc": datetime.now(timezone.utc).isoformat(), "entries": []}
    manifest["code_sha256"] = {f.name: sha256(f) for f in sorted(HERE.glob("*.py"))}
    manifest["environment"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                               "h5py": h5py.__version__, "platform": platform.platform()}
    manifest.setdefault("invocations", []).append(" ".join(sys.argv))
    for L in a.L:
        entry = run_L(L, a.R, out_dir, a.workers)
        manifest["entries"] = [e for e in manifest["entries"] if e["file"] != entry["file"]] + [entry]
        manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
