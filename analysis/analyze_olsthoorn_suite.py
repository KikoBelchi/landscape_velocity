"""The 15 figures of the Olsthoorn benchmark (PBC and OBC), drawn by analysis/figures.py.

The plotting code is imported unchanged from figures.py. The data come from the production runs runs/olsthoorn_{pbc,obc}_v1, so
all sizes are included (PBC up to L = 512, OBC up to L = 256).
The H1 lifetime of the periodic chain for L = 256 and 512 comes from runs/olsthoorn_pbc_h1peak_v1,
which is checked against the full grid at L = 128.
Writes PNG + PDF and summary_olsthoorn.json into analysis/figures_olsthoorn/ and compares every
peak with analysis/summary_olsthoorn_v1.json.

Run from the repository root:  python analysis\\analyze_olsthoorn_suite.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
MP = HERE.parent
# olsthoorn_pbc_h1peak_v1: H1 on the fine points in [0.98, 1.06] for L = 128, 256, 512 (grid "h1peak").
# Its L = 128 file only checks the window against the full fine grid of olsthoorn_pbc_v1.
RUNS = [("pbc", MP / "runs" / "olsthoorn_pbc_v1"), ("obc", MP / "runs" / "olsthoorn_obc_v1"),
        ("pbc", MP / "runs" / "olsthoorn_pbc_h1peak_v1")]
FIGURES = HERE / "figures.py"

spec = importlib.util.spec_from_file_location("figures", FIGURES)
figs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figs)
figs.OUT = HERE / "figures_olsthoorn"


def load_runs() -> dict:
    """(bc, L, grid) -> the same structure that figures.compute() builds from matrices."""
    data = {}
    for bc, run in RUNS:
        for p in sorted(run.glob(f"{bc}_L*_*.h5")):
            L, grid = int(p.stem.split("_L")[1].split("_")[0]), p.stem.split("_")[-1]
            with h5py.File(p, "r") as f:
                r = {"g": f["g"][()], "M": f["data/M_sqrt_mean_C"][()],
                     "qc": {k: f[f"qc/{k}"][()] for k in ("min_two_site_eigenvalue", "orthogonality_error",
                                                          "parity_flipped")}}
                for name in ("MI", "C"):
                    r[name] = {k: f[f"data/{name}/{k}"][()]
                               for k in ("deaths", "nn_distances", "mst_is_nn", "hypothesis_N")}
                    if f"h1/{name}" in f:
                        r[name]["h1"] = {k: f[f"h1/{name}/{k}"][()] for k in figs.H1_KEYS}
            data[(bc, L, grid)] = r
    return data


def publish(src: Path, dst: Path, tries: int = 20) -> None:
    """Copy the figures into the output folder, retrying while the sync client holds a file."""
    import shutil
    import time
    dst.mkdir(exist_ok=True)
    for p in sorted(src.iterdir()):
        for t in range(tries):
            try:
                shutil.copy2(p, dst / p.name)
                break
            except OSError:
                time.sleep(0.5 * (t + 1))
        else:
            raise OSError(f"could not write {dst / p.name} (file locked?)")


def h1peak_check(data: dict) -> dict:
    """The h1peak window against the production runs at the same fields.

    For every L in both runs, the largest difference of the H0 deaths (same state, must be 0) and,
    where the production file has H1, of the H1 lifetime and of its quadratic peak."""
    out = {}
    for (bc, L, grid), w in sorted(data.items()):
        full = data.get((bc, L, "fine"))
        if grid != "h1peak" or full is None:
            continue
        idx = np.searchsorted(full["g"], w["g"])
        assert np.array_equal(full["g"][idx], w["g"]), "h1peak grid is not a subset of fine"
        row = {}
        for name in ("MI", "C"):
            row[f"{name}_max_abs_diff_deaths"] = float(np.abs(full[name]["deaths"][idx] - w[name]["deaths"]).max())
            if "h1" in full[name]:
                a, b = full[name]["h1"]["lifetime"], w[name]["h1"]["lifetime"]
                row[f"{name}_max_abs_diff_H1_lifetime"] = float(np.abs(a[idx] - b).max())
                pa, pb = figs.quadratic_peak(full["g"], a), figs.quadratic_peak(w["g"], b)
                row[f"{name}_H1_peak_full_vs_window"] = [pa["x"], pb["x"], pa["value"], pb["value"]]
            pk = figs.quadratic_peak(w["g"], w[name]["h1"]["lifetime"])
            row[f"{name}_H1_peak_window"] = {"x": pk["x"], "value": pk["value"], "edge": pk["edge"]}
        out[f"{bc}_L{L}"] = row
    return out


def main() -> None:
    import tempfile
    final = figs.OUT
    figs.OUT = Path(tempfile.mkdtemp(prefix="figures_olsthoorn_"))   # matplotlib writes outside the synced folder
    data = load_runs()
    summary = {"pbc": {}, "obc": {}, "A_C": figs.A_C, "A_MI": figs.A_MI}
    figs.pbc_figures(data, summary)
    figs.obc_figures(data, summary)
    figs.barcodes(data, summary)
    figs.qc(data, summary)
    ref = json.loads((HERE / "summary_olsthoorn_v1.json").read_text(encoding="utf-8"))
    report = {}
    for section in ("pbc", "obc", "barcodes"):
        d = [x for _, x in figs._diffs(ref[section], summary[section], section) if x is not None]
        report[section] = {"compared": len(d), "max_abs_diff": max(d, default=0.0)}
    report["fitted_slope_L>=32"] = {n: [ref["pbc"][n]["fitted_slope_L>=32"], summary["pbc"][n]["fitted_slope_L>=32"]]
                                    for n in ("MI", "C")}
    summary["comparison_with_analysis_summary"] = report
    summary["h1peak_check"] = h1peak_check(data)
    (figs.OUT / "summary_olsthoorn.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    publish(figs.OUT, final)
    print(json.dumps(report, indent=1, default=float))
    print(f"figures in {final}")


if __name__ == "__main__":
    main()
