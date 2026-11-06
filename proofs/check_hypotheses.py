"""Hypothesis N (nearest-neighbour distance is the minimum) and P (positive two-site spectrum)
for PBC on Olsthoorn's field range. Run from the repository root:
python proofs\check_hypotheses.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import h0_tools as h0  # noqa: E402
import tfim_ff as ff  # noqa: E402

grid = np.unique(np.concatenate([np.logspace(-1, 1, 201), np.linspace(0.9, 1.1, 41)]))
report = {}
for L in (10, 14, 18, 22, 32, 64, 128, 256):
    bad = {"C": [], "MI": []}
    min_eig = np.inf
    for g in grid:
        res = ff.ground_state_matrices(L, float(g), "pbc")
        min_eig = min(min_eig, res["qc"]["min_two_site_eigenvalue"])
        for name, D in (("C", ff.distance_C(res["C"])), ("MI", ff.distance_MI(res["MI"]))):
            if not h0.h0_nn_ties(D, "pbc"):
                bad[name].append(float(g))
    report[L] = {"n_g": int(grid.size), "N_fails_C": bad["C"], "N_fails_MI": bad["MI"],
                 "min_two_site_eigenvalue": float(min_eig)}
    print(L, report[L], flush=True)
(HERE / "check_hypotheses_output.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
