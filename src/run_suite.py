"""Free-fermion production runs for the math paper (Olsthoorn benchmark, PBC and OBC).

One HDF5 per (bc, L, grid) in runs/<run_id>/, plus manifest.json with SHA-256
hashes of every output and of the source files. The landscape velocity is computed in the
analysis step from the stored H0 deaths. Only a preview of its maximum goes to the manifest.

Example (Anaconda Prompt, from the repository root):
python src\\run_suite.py --run-id olsthoorn_pbc_v1 --bc pbc --L 10 14 18 22 32 64 128 256 512 --grids wide fine --workers 10
H1 near its peak for the large chains (L = 128 is a check against olsthoorn_pbc_v1):
python src\\run_suite.py --run-id olsthoorn_pbc_h1peak_v1 --bc pbc --L 128 256 512 --grids h1peak --h1-max-L 512 --workers 10
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
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

RUNS = HERE.parent / "runs"
GRIDS = {
    "wide": lambda: np.logspace(-1.0, 1.0, 401),               # Olsthoorn's axis, 10^-1..10^1
    "fine": lambda: np.round(np.linspace(0.5, 1.5, 1001), 6),  # uniform dg = 0.001
    "critical": lambda: np.array([1.0]),                       # Fig. 7 (H1 birth/death at g=1)
}
# H1 for L > 128 only around the peak of the H1 lifetime: the 81 points of "fine" in [0.98, 1.06]
GRIDS["h1peak"] = lambda: (lambda x: x[(x >= 0.98 - 1e-9) & (x <= 1.06 + 1e-9)])(GRIDS["fine"]())
FILTRATIONS = ("MI", "C")  # MI is the main distance, C the secondary one
QC_KEYS = ("softest_mode_energy", "parity_flipped", "max_aa_bb", "orthogonality_error",
           "translation_residual", "lu_fallbacks", "min_two_site_eigenvalue")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def one_point(args: tuple) -> dict:
    L, g, bc, do_h1, keep_matrices = args
    res = ff.ground_state_matrices(L, float(g), bc)
    out = {"m": res["m"], "M": float(np.sqrt(max(res["C"].mean(), 0.0))), "qc": res["qc"]}
    for name, D in (("MI", ff.distance_MI(res["MI"])), ("C", ff.distance_C(res["C"]))):
        deaths, edges = h0.mst_prim(D)
        nn = np.diagonal(D, 1).copy()
        out[name] = {
            "deaths": deaths,
            "nn": nn if bc == "obc" else np.array([D[0, 1]]),
            "mst_is_nn": h0.mst_is_nearest_neighbour(edges, L, bc),
            "hyp_N": h0.h0_nn_ties(D, bc),
            "h1": h0.h1_summary(D) if do_h1 else None,
            "row": D[0] if bc == "pbc" else None,
            "matrix": D if keep_matrices else None,
        }
    return out


def run_one(bc: str, L: int, grid: str, out_dir: Path, workers: int, h1_max_L: int,
            matrices_max_L: int) -> dict:
    g = GRIDS[grid]()
    do_h1 = L <= h1_max_L
    keep = bc == "obc" and L <= matrices_max_L
    t0 = time.time()
    tasks = [(L, x, bc, do_h1, keep) for x in g]
    if workers > 1 and g.size > 1:
        with Pool(workers) as pool:
            points = pool.map(one_point, tasks, chunksize=max(1, g.size // (4 * workers)))
    else:
        points = [one_point(t) for t in tasks]
    path = out_dir / f"{bc}_L{L}_{grid}.h5"
    with h5py.File(path, "w") as f:
        f.attrs.update({
            "model": "H = -J sum sx_i sx_{i+1} - h sum sz_i (Olsthoorn PRB 107, 115174)",
            "J": 1.0, "g_definition": "g = h/J", "bc": bc, "L": L, "grid": grid,
            "state": "ground state of the even-parity sector P = prod sz = +1",
            "method": "free fermions (Majorana covariance, real Schur parity fixing)",
            "D_MI": "1 - MI/(2 ln 2), MI in nats (main distance)",
            "D_C": "1 - <sx_i sx_j>, raw correlator = Cstag of the AFM (secondary distance)",
            "h0_deaths": "minimum-spanning-tree weights, sorted descending, infinite bar removed",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        f["g"] = g
        f["data/m"] = np.array([p["m"] for p in points])
        f["data/M_sqrt_mean_C"] = np.array([p["M"] for p in points])
        for name in FILTRATIONS:
            grp = f.create_group(f"data/{name}")
            grp["deaths"] = np.array([p[name]["deaths"] for p in points])
            grp["nn_distances"] = np.array([p[name]["nn"] for p in points])
            grp["mst_is_nn"] = np.array([p[name]["mst_is_nn"] for p in points])
            grp["hypothesis_N"] = np.array([p[name]["hyp_N"] for p in points])
            if bc == "pbc":
                grp["distance_row0"] = np.array([p[name]["row"] for p in points])
            if keep:
                grp.create_dataset("distance_matrix", data=np.array([p[name]["matrix"] for p in points]),
                                   compression="gzip", compression_opts=4)
            if do_h1:
                h1g = f.create_group(f"h1/{name}")
                for key in ("n_bars", "birth", "death", "lifetime", "total_persistence"):
                    h1g[key] = np.array([p[name]["h1"][key] for p in points])
        for key in QC_KEYS:
            f[f"qc/{key}"] = np.array([p["qc"][key] for p in points], dtype=float)
    preview = {}
    if g.size >= 5:
        with h5py.File(path, "r") as f:
            for name in FILTRATIONS:
                xi = h0.velocity_closed(g, f[f"data/{name}/deaths"][()])
                preview[name] = h0.quadratic_peak(g, xi)
    qc = {
        "mst_is_nn_all": {n: bool(all(p[n]["mst_is_nn"] for p in points)) for n in FILTRATIONS},
        "hypothesis_N_all": {n: bool(all(p[n]["hyp_N"] for p in points)) for n in FILTRATIONS},
        "max_orthogonality_error": max(p["qc"]["orthogonality_error"] for p in points),
        "min_two_site_eigenvalue": min(p["qc"]["min_two_site_eigenvalue"] for p in points),
        "parity_flips": int(sum(p["qc"]["parity_flipped"] for p in points)),
        "lu_fallbacks": int(sum(p["qc"]["lu_fallbacks"] for p in points)),
    }
    entry = {"file": path.name, "sha256": sha256(path), "bc": bc, "L": L, "grid": grid,
             "n_g": int(g.size), "h1": do_h1, "matrices": keep,
             "seconds": round(time.time() - t0, 1), "qc": qc, "velocity_peak_preview": preview}
    print(json.dumps({k: entry[k] for k in ("file", "seconds", "qc", "velocity_peak_preview")}),
          flush=True)
    return entry


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True)
    p.add_argument("--bc", choices=("pbc", "obc"), required=True)
    p.add_argument("--L", type=int, nargs="+", required=True)
    p.add_argument("--grids", nargs="+", choices=tuple(GRIDS), default=["wide", "fine"])
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--h1-max-L", type=int, default=128)
    p.add_argument("--matrices-max-L", type=int, default=64)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    out_dir = RUNS / a.run_id
    manifest_path = out_dir / "manifest.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "run_id": a.run_id, "created_utc": datetime.now(timezone.utc).isoformat(), "entries": []}
    done = {e["file"] for e in manifest["entries"]}
    manifest["code_sha256"] = {f.name: sha256(f) for f in sorted(HERE.glob("*.py"))}
    manifest["environment"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                               "h5py": h5py.__version__, "platform": platform.platform()}
    manifest.setdefault("invocations", []).append(" ".join(sys.argv))
    for L in a.L:
        for grid in a.grids:
            name = f"{a.bc}_L{L}_{grid}.h5"
            if name in done and not a.overwrite:
                print(f"SKIP {name} (already in manifest)", flush=True)
                continue
            entry = run_one(a.bc, L, grid, out_dir, a.workers, a.h1_max_L, a.matrices_max_L)
            manifest["entries"] = [e for e in manifest["entries"] if e["file"] != name] + [entry]
            manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
