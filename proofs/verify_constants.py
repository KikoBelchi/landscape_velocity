"""Numerical checks of every formula and constant stated in log_divergence_velocity.tex.

Run from the repository root:  python proofs\\verify_constants.py
Writes proofs/verify_constants_output.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
import tfim_ff as ff  # noqa: E402

PI, LN2, EULER = np.pi, np.log(2.0), 0.5772156649015329
out = {}


def R(k, g):
    return np.maximum(np.sqrt(np.maximum(1 + g * g - 2 * g * np.cos(k), 0.0)), 1e-300)  # symbols vanish at k=0 when g=1


# Symbols: G_r = (1/pi) int_0^pi [cos k(r-1) - g cos kr] / R dk, m = -G_0.
def tl(f, g):
    return quad(lambda k: f(k, g), 0, PI, limit=400, points=[1e-9])[0] / PI


SYM = {
    "m": lambda k, g: (g - np.cos(k)) / R(k, g),
    "Gxx": lambda k, g: (1 - g * np.cos(k)) / R(k, g),
    "Gyy": lambda k, g: (np.cos(2 * k) - g * np.cos(k)) / R(k, g),
}
DSYM = {  # exact g-derivatives of the symbols
    "m": lambda k, g: np.sin(k) ** 2 / R(k, g) ** 3,
    "Gxx": lambda k, g: -g * np.sin(k) ** 2 / R(k, g) ** 3,
    "Gyy": lambda k, g: np.sin(k) ** 2 * (g - 2 * np.cos(k)) / R(k, g) ** 3,
}


def finite(f, g, L):
    k = PI * (2 * np.arange(L // 2) + 1) / L
    return 2.0 / L * f(k, g).sum()


# 1) finite-L Riemann sums reproduce the free-fermion state exactly (PBC, even sector)
worst = 0.0
for L in (8, 16, 64, 128):
    for g in (0.3, 1.0, 1.7):
        res = ff.ground_state_matrices(L, g, "pbc")
        vals = {"m": res["m"][0], "Gxx": res["C"][0, 1], "Gyy": res["Gyy"][0, 1]}
        for key, val in vals.items():
            worst = max(worst, abs(finite(SYM[key], g, L) - val))
        gzz = vals["m"] ** 2 - vals["Gxx"] * vals["Gyy"]
        worst = max(worst, abs(gzz - res["Gzz"][0, 1]))
out["finite_L_sums_vs_free_fermions_max_abs"] = worst

# 2) derivative symbols against finite differences of the symbols
worst = 0.0
for key in SYM:
    for g in (0.4, 0.95, 1.3):
        for k in (0.1, 1.0, 2.5):
            h = 1e-6
            fd = (SYM[key](k, g + h) - SYM[key](k, g - h)) / (2 * h)
            worst = max(worst, abs(fd - DSYM[key](k, g)))
out["derivative_symbols_max_abs"] = worst

# 3) critical values
out["critical_values"] = {
    "m": tl(SYM["m"], 1.0), "2/pi": 2 / PI,
    "Gxx": tl(SYM["Gxx"], 1.0),
    "Gyy": tl(SYM["Gyy"], 1.0), "-2/(3pi)": -2 / (3 * PI),
}


# 4) I(g) = int_0^pi sin^2 k / R^3 = ln(1/|g-1|) + 3 ln 2 - 2 + o(1)
def I(g):
    return quad(lambda u: u * u * np.sqrt(1 - u * u / 4) / ((g - 1) ** 2 + g * u * u) ** 1.5,
                0, 2, limit=400, points=[abs(g - 1)])[0]


out["I_minus_log"] = {f"{d:g}": I(1 + s * d) - np.log(1 / d) for d in (1e-2, 1e-3, 1e-4, 1e-5)
                      for s in (1,)}
out["I_minus_log_below"] = {f"{d:g}": I(1 - d) - np.log(1 / d) for d in (1e-2, 1e-3, 1e-4, 1e-5)}
out["3ln2-2"] = 3 * LN2 - 2
# substitution check: I in k variable equals I in u variable
out["I_substitution_check"] = abs(quad(lambda k: np.sin(k) ** 2 / R(k, 1.3) ** 3, 0, PI)[0] - I(1.3))

# 5) finite size at g = 1: d_C'(1;L) = (1/pi)[ln L + gamma + ln(8/pi) - 1] + o(1)
c_fs = (EULER + np.log(8 / PI) - 1)
out["finite_size_constant"] = c_fs
out["dC_prime_at_1_minus_prediction"] = {
    str(L): finite(lambda k, g: g * np.sin(k) ** 2 / R(k, g) ** 3, 1.0, L) - (np.log(L) + c_fs) / PI
    for L in (16, 64, 256, 1024, 2 ** 14, 2 ** 18)}
out["dC_at_1_exact_vs_sum"] = max(
    abs((1 - 1 / (L * np.sin(PI / (2 * L)))) - (1 - finite(SYM["Gxx"], 1.0, L))) for L in (10, 64, 512))

# 6) C filtration: amplitude A_C = (1/pi) sqrt((1 - 2/pi)/2) and the full TL expansion
A_C = np.sqrt((1 - 2 / PI) / 2) / PI
out["A_C"] = A_C
xiC = {}
for d in (1e-2, 1e-3, 1e-4, 1e-5):
    g = 1 - d
    dC = 1 - tl(SYM["Gxx"], g)
    xi = g * I(g) / PI * np.sqrt(dC / 2)
    xiC[f"{d:g}"] = xi - A_C * (np.log(1 / d) + 3 * LN2 - 2)
out["xi_C_TL_minus_expansion"] = xiC


# 7) MI filtration: d_MI = 1 - F(m, Gxx, Gyy, Gzz)/(2 ln 2) for the nearest-neighbour pair
def F(x):
    m, gxx, gyy, gzz = x
    spec = ff.two_site_spectrum(m, m, gxx, gyy, gzz)
    S2 = -ff.xlogx(np.asarray(spec)).sum()
    S1 = -(ff.xlogx(np.array((1 + m) / 2)) + ff.xlogx(np.array((1 - m) / 2)))
    return float(2 * S1 - S2)


def x_of(g):
    m, gxx, gyy = (tl(SYM[k], g) for k in ("m", "Gxx", "Gyy"))
    return np.array([m, gxx, gyy, m * m - gxx * gyy])


x1 = x_of(1.0)
spec1 = ff.two_site_spectrum(x1[0], x1[0], x1[1], x1[2], x1[3])
grad = np.array([(F(x1 + e) - F(x1 - e)) / 2e-7 for e in np.eye(4) * 1e-7])
v = np.array([1.0, -1.0, -1.0, 2 * x1[0] + x1[1] + x1[2]])
dMI1 = 1 - F(x1) / (2 * LN2)
A_MI = abs(grad @ v) * np.sqrt(dMI1 / 2) / (2 * PI * LN2)
out["MI"] = {"x_at_1": x1.tolist(), "two_site_spectrum_at_1": np.asarray(spec1).tolist(),
             "grad_F": grad.tolist(), "v": v.tolist(), "grad_dot_v": float(grad @ v),
             "d_MI_at_1": dMI1, "Olsthoorn_D01": 0.744293, "A_MI": A_MI}


def xi_MI_TL(g, h=1e-7):
    xp, xm = x_of(g + h), x_of(g - h)
    dprime = -(F(xp) - F(xm)) / (2 * h) / (2 * LN2)
    return abs(dprime) * np.sqrt((1 - F(x_of(g)) / (2 * LN2)) / 2)


out["xi_MI_TL_over_log"] = {f"{d:g}": xi_MI_TL(1 - d) / np.log(1 / d) for d in (1e-2, 1e-3, 1e-4, 1e-5)}
out["xi_MI_TL_minus_A_log"] = {f"{d:g}": xi_MI_TL(1 - d) - A_MI * np.log(1 / d)
                               for d in (1e-2, 1e-3, 1e-4, 1e-5)}

path = HERE / "verify_constants_output.json"
path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print(json.dumps(out, indent=1, default=float))
