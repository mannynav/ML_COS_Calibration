
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""
Classical Calibration of Merton, Heston, and Bates Models
=========================================================

Standard (non-ML) calibration of three option-pricing models to a market
implied-volatility surface, using the derivative-free and quasi-Newton
optimizers covered in Ali Hirsa, "Computational Methods in Finance"
(Chapman & Hall/CRC), Part II: Calibration and Estimation.

Optimizers implemented / wrapped:
    - Nelder-Mead      (downhill simplex; derivative-free)     [SciPy]
    - Powell           (direction-set method; derivative-free) [SciPy]
    - DFP              (Davidon-Fletcher-Powell quasi-Newton)  [from scratch]
    - L-BFGS-B         (bound-constrained quasi-Newton)        [SciPy, reference]

DFP is implemented from scratch because SciPy ships BFGS (the successor to DFP)
but not DFP itself. The DFP inverse-Hessian update is:

    H_{k+1} = H_k + (s s^T)/(s^T y) - (H_k y y^T H_k)/(y^T H_k y)

    where  s = x_{k+1} - x_k   and   y = grad_{k+1} - grad_k

Gradients are approximated by forward finite differences. A backtracking Armijo
line search is used, and box bounds are enforced by projection so the optimizer
cannot leave the physical parameter region.

Objective: mean squared error between the model and market implied-vol surfaces,
computed by pricing with the validated COS method and inverting each price to
Black-Scholes implied volatility. Each model's pricer/helpers are imported from
its ML-calibration module, so the objective is identical to the ML pipeline's,
making the comparison apples-to-apples.

Models are selected at runtime; the optimizers are entirely model-agnostic and
depend only on each model's `surface_ivs_from_params`, bounds, and parameter names.

Run:  python classical_calibration.py            # all three models
      python classical_calibration.py heston      # single model
Deps: numpy scipy matplotlib
       (+ ml_calibration.py, heston_ml_calibration.py, bates_ml_calibration.py)
"""

import sys
import time
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Import the three model modules. Each exposes the same interface:
#   surface_ivs_from_params(params, S0) -> (35,) implied-vol surface
#   bs_call_np, P_LO, P_HI, PNAMES, GRID_K, GRID_T, N_GRID, r
import ml_calibration          as merton_mod
import heston_ml_calibration   as heston_mod
import bates_ml_calibration    as bates_mod


# ─────────────────────────────────────────────────────────────────────────────
# 0.  MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
#   name : (module, cold-start x0)
# Cold-start guesses are deliberately mid-range and identical in spirit across
# models, so no optimizer is handed an unfair warm start.
MODELS = {
    "Merton": dict(
        mod=merton_mod,
        #      sigma  lambda  mu_J   sigma_J
        x0=np.array([0.20, 0.50, -0.10, 0.15]),
    ),
    "Heston": dict(
        mod=heston_mod,
        #       v0   kappa  theta  sigma_v  rho
        x0=np.array([0.04, 2.00, 0.04, 0.40, -0.60]),
    ),
    "Bates": dict(
        mod=bates_mod,
        #       v0  kappa theta sigma_v rho    lam  mu_J  sigma_J
        x0=np.array([0.04, 2.00, 0.04, 0.40, -0.60, 0.30, -0.10, 0.12]),
    ),
}

GRID_K = merton_mod.GRID_K
GRID_T = merton_mod.GRID_T


# ─────────────────────────────────────────────────────────────────────────────
# 1.  OBJECTIVE (model-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
def make_objective(mod, market_ivs, S0):
    """
    Returns f(params) -> mean squared IV error, and a dict tracking the number
    of objective evaluations (a key comparison metric between methods).
    """
    stats = {"n_eval": 0}
    P_LO, P_HI = mod.P_LO, mod.P_HI

    def objective(params):
        stats["n_eval"] += 1
        p = np.clip(params, P_LO, P_HI)                 # stay physical
        ivs = mod.surface_ivs_from_params(p, S0)
        if np.any(ivs <= 0.0):                          # failed IV inversion
            return 1.0e3
        return float(np.mean((ivs - market_ivs) ** 2))

    return objective, stats


def rmse_vol_points(mod, params, market_ivs, S0):
    ivs = mod.surface_ivs_from_params(np.clip(params, mod.P_LO, mod.P_HI), S0)
    return float(np.sqrt(np.mean((ivs - market_ivs) ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DFP (Davidon-Fletcher-Powell) — implemented from scratch
# ─────────────────────────────────────────────────────────────────────────────
def finite_diff_grad(f, x, h=1e-5):
    """Forward finite-difference gradient."""
    n = len(x)
    g = np.zeros(n)
    fx = f(x)
    for i in range(n):
        xp = x.copy()
        xp[i] += h
        g[i] = (f(xp) - fx) / h
    return g, fx


def project(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def dfp_minimize(f, x0, lo, hi, max_iter=200, tol=1e-8,
                 c1=1e-4, ls_shrink=0.5, ls_max=40):
    """
    Davidon-Fletcher-Powell quasi-Newton minimizer with:
      - forward finite-difference gradients
      - backtracking Armijo line search
      - box constraints enforced by projection

    Returns (x_best, f_best, n_iter). Exact on quadratics in n steps (in theory).
    """
    x = project(np.asarray(x0, dtype=float), lo, hi)
    n = len(x)
    H = np.eye(n)                        # inverse-Hessian approximation
    g, fx = finite_diff_grad(f, x)

    for it in range(max_iter):
        d = -H @ g
        if not np.all(np.isfinite(d)):
            H = np.eye(n)
            d = -g

        gd = g @ d
        if gd >= 0:                      # not descent — reset to steepest descent
            H = np.eye(n)
            d = -g
            gd = g @ d

        # Backtracking Armijo line search, respecting bounds by projection
        alpha = 1.0
        x_new, f_new = x, fx
        improved = False
        for _ in range(ls_max):
            x_try = project(x + alpha * d, lo, hi)
            f_try = f(x_try)
            if f_try <= fx + c1 * alpha * gd:
                x_new, f_new = x_try, f_try
                improved = True
                break
            alpha *= ls_shrink
        if not improved:
            break

        g_new, _ = finite_diff_grad(f, x_new)
        s = (x_new - x).reshape(-1, 1)
        y = (g_new - g).reshape(-1, 1)

        step = np.linalg.norm(x_new - x)
        if step < tol or abs(fx - f_new) < tol:
            return x_new, f_new, it + 1

        sy = float((s.T @ y).item())
        if sy > 1e-12:                              # curvature condition
            Hy = H @ y
            yHy = float((y.T @ Hy).item())
            if yHy > 1e-12:
                H = H + (s @ s.T) / sy - (Hy @ Hy.T) / yHy

        x, fx, g = x_new, f_new, g_new

    return x, fx, max_iter


# ─────────────────────────────────────────────────────────────────────────────
# 3.  UNIFIED CALIBRATION DRIVER
# ─────────────────────────────────────────────────────────────────────────────
def calibrate(method, mod, market_ivs, S0, x0, max_iter=200):
    obj, stats = make_objective(mod, market_ivs, S0)
    bounds = list(zip(mod.P_LO, mod.P_HI))
    P_LO, P_HI = mod.P_LO, mod.P_HI

    t0 = time.time()
    if method == "Nelder-Mead":
        res = minimize(obj, x0, method="Nelder-Mead",
                       options={"maxiter": max_iter * 50, "xatol": 1e-6,
                                "fatol": 1e-10, "adaptive": True})
        x_best = np.clip(res.x, P_LO, P_HI)
    elif method == "Powell":
        res = minimize(obj, x0, method="Powell", bounds=bounds,
                       options={"maxiter": max_iter * 50, "xtol": 1e-6, "ftol": 1e-10})
        x_best = np.clip(res.x, P_LO, P_HI)
    elif method == "DFP":
        x_best, _, _ = dfp_minimize(obj, x0, P_LO, P_HI, max_iter=max_iter)
    elif method == "L-BFGS-B":
        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-9})
        x_best = np.clip(res.x, P_LO, P_HI)
    else:
        raise ValueError(f"Unknown method: {method}")

    elapsed = time.time() - t0
    rmse = rmse_vol_points(mod, x_best, market_ivs, S0)
    return dict(method=method, params=x_best, rmse=rmse,
                time_s=elapsed, n_eval=stats["n_eval"])


def calibrate_model(model_name, market_ivs, S0, max_iter=200,
                    methods=("Nelder-Mead", "Powell", "DFP", "L-BFGS-B")):
    spec = MODELS[model_name]
    mod, x0 = spec["mod"], spec["x0"]
    n_par = len(x0)
    print(f"\nCalibrating {model_name} ({n_par} params) to SPY surface...")
    print(f"  Cold-start x0 = {dict(zip(mod.PNAMES, np.round(x0, 4)))}")

    # Powell's direction-set search scales poorly in dimension; for the 8-param
    # Bates model it needs thousands of extra evaluations for no better result,
    # so it is skipped above 5 parameters (report notes this).
    run_methods = list(methods)
    if n_par > 5 and "Powell" in run_methods:
        run_methods.remove("Powell")
        print(f"  (Powell skipped for {n_par}-param model — too costly in high dimension)")

    results = []
    for m in run_methods:
        print(f"    {m} ...", flush=True)
        results.append(calibrate(m, mod, market_ivs, S0, x0, max_iter))
    return mod, results


# ─────────────────────────────────────────────────────────────────────────────
# 4.  REPORTING
# ─────────────────────────────────────────────────────────────────────────────
def print_report(model_name, mod, results):
    print("\n" + "═" * 74)
    print(f"  CLASSICAL CALIBRATION — {model_name} Model")
    print("═" * 74)

    print(f"\n{'Method':>12} {'RMSE (vol pts)':>15} {'Time (s)':>10} {'# f-evals':>10}")
    print("  " + "-" * 60)
    for res in results:
        print(f"{res['method']:>12} {res['rmse']*100:>14.3f} "
              f"{res['time_s']:>10.2f} {res['n_eval']:>10}")

    print("\n── Calibrated Parameters ─────────────────────────────────────────")
    print(f"{'Method':>12}" + "".join(f"{n:>9}" for n in mod.PNAMES))
    print("  " + "-" * (12 + 9 * len(mod.PNAMES)))
    for res in results:
        print(f"{res['method']:>12}" + "".join(f"{v:>9.4f}" for v in res["params"]))


def plot_model_comparison(model_name, mod, results, market_ivs, S0):
    market_2d = market_ivs.reshape(len(GRID_K), len(GRID_T))
    colors = {"Nelder-Mead": "#1f77b4", "Powell": "#ff7f0e",
              "DFP": "#2ca02c", "L-BFGS-B": "#d62728"}
    markers = {"Nelder-Mead": "s", "Powell": "^", "DFP": "D", "L-BFGS-B": "v"}

    fig, axes = plt.subplots(1, len(GRID_T), figsize=(4 * len(GRID_T), 4), sharey=True)
    for j, (T, ax) in enumerate(zip(GRID_T, axes)):
        ax.plot(GRID_K, market_2d[:, j], "o-", color="black", lw=2.5,
                label="Market", markersize=7)
        for res in results:
            ivs2d = mod.surface_ivs_from_params(res["params"], S0).reshape(len(GRID_K), len(GRID_T))
            ax.plot(GRID_K, ivs2d[:, j], marker=markers[res["method"]],
                    linestyle="--", color=colors[res["method"]], lw=1.3,
                    markersize=4, label=res["method"], alpha=0.8)
        ax.set_title(f"T = {T:.2f}")
        ax.set_xlabel("Log-moneyness")
        if j == 0:
            ax.set_ylabel("Implied Vol")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.suptitle(f"Classical Calibration — {model_name}: Method Comparison vs Market", y=1.02)
    plt.tight_layout()
    fname = f"classical_calibration_{model_name.lower()}.png"
    plt.savefig(fname, dpi=140, bbox_inches="tight")
    plt.close()
    return fname


def print_cross_model_summary(all_results):
    """Best RMSE per model across all methods — the headline comparison."""
    print("\n" + "═" * 74)
    print("  CROSS-MODEL SUMMARY — best classical fit per model")
    print("═" * 74)
    print(f"\n{'Model':>10} {'Best method':>14} {'RMSE (vol pts)':>16} {'# params':>10}")
    print("  " + "-" * 56)
    for model_name, (mod, results) in all_results.items():
        best = min(results, key=lambda r: r["rmse"])
        print(f"{model_name:>10} {best['method']:>14} "
              f"{best['rmse']*100:>15.3f} {len(mod.PNAMES):>10}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Same SPY surface used throughout the project
    market_ivs = np.array([
        0.2629, 0.1799, 0.1330, 0.1733, 0.1634,
        0.1789, 0.1629, 0.1564, 0.1702, 0.1639,
        0.1431, 0.1475, 0.1538, 0.1610, 0.1564,
        0.1260, 0.1335, 0.1417, 0.1484, 0.1455,
        0.1232, 0.1240, 0.1273, 0.1350, 0.1339,
        0.1305, 0.1222, 0.1176, 0.1230, 0.1245,
        0.1485, 0.1372, 0.1242, 0.1136, 0.1205,
    ], dtype=np.float32)
    S0 = 711.58

    # Which models to run: all three by default, or one from the command line
    if len(sys.argv) > 1:
        arg = sys.argv[1].capitalize()
        to_run = [arg] if arg in MODELS else list(MODELS)
    else:
        to_run = list(MODELS)   # ["Merton", "Heston", "Bates"]

    all_results = {}
    for model_name in to_run:
        mod, results = calibrate_model(model_name, market_ivs, S0)
        print_report(model_name, mod, results)
        fname = plot_model_comparison(model_name, mod, results, market_ivs, S0)
        print(f"  saved {fname}")
        all_results[model_name] = (mod, results)

    if len(all_results) > 1:
        print_cross_model_summary(all_results)