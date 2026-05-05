# Created: April 23, 2026

"""
Three-Model Calibration Comparison: Merton vs Heston vs Bates
==============================================================

Loads the three saved inversion networks and calibrates each to the same
market IV surface. Produces:

    - Parameter table for each model
    - RMSE comparison (network alone, network + L-BFGS-B polish)
    - Wall-clock timing comparison
    - Smile-slice plot showing all three model fits vs market
    - Residual heatmap showing where each model fails

Run AFTER training all three networks:
    python merton_ml_calibration.py    →  merton_inversion_net.pt
    python heston_ml_calibration.py    →  heston_inversion_net.pt
    python bates_ml_calibration.py     →  bates_inversion_net.pt
    python compare_models.py

Deps: torch numpy scipy matplotlib
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Import the three model modules — must be importable from cwd.
# These files must be in the same folder as compare_models.py:
#   ml_calibration.py            (the Merton ML script)
#   heston_ml_calibration.py
#   bates_ml_calibration.py
import ml_calibration         as merton_mod
import heston_ml_calibration  as heston_mod
import bates_ml_calibration   as bates_mod


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use the same SPY surface across all three calibrations
MARKET_IVS = np.array([
    0.2629, 0.1799, 0.1330, 0.1733, 0.1634,   # k=-0.20
    0.1789, 0.1629, 0.1564, 0.1702, 0.1639,   # k=-0.12
    0.1431, 0.1475, 0.1538, 0.1610, 0.1564,   # k=-0.06
    0.1260, 0.1335, 0.1417, 0.1484, 0.1455,   # k= 0.00
    0.1232, 0.1240, 0.1273, 0.1350, 0.1339,   # k= 0.06
    0.1305, 0.1222, 0.1176, 0.1230, 0.1245,   # k= 0.12
    0.1485, 0.1372, 0.1242, 0.1136, 0.1205,   # k= 0.20
], dtype=np.float32)
S0_MARKET = 711.58

GRID_K = merton_mod.GRID_K
GRID_T = merton_mod.GRID_T
N_GRID = merton_mod.N_GRID
r      = merton_mod.r


# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD CHECKPOINTS
# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path, ModelClass):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Missing {checkpoint_path}. Train it first."
        )
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=DEVICE)
    model = ModelClass().to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["mu_x"], ckpt["std_x"]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PER-MODEL CALIBRATION (returns timing + fit metrics)
# ─────────────────────────────────────────────────────────────────────────────
def calibrate_one(name, model, mu_x, std_x, mod, market_ivs, S0_market):
    """
    mod : the imported model module (merton_mod, heston_mod, or bates_mod)
          — provides bs_call_np, surface_ivs_from_params, P_LO, P_HI, PNAMES
    """
    # ── IV → normalized prices ──────────────────────────────────────────────
    prices_over_s = np.empty(N_GRID, dtype=np.float32)
    idx = 0
    for T in GRID_T:
        F = S0_market * np.exp(r * T)
        for lm in GRID_K:
            K = F * np.exp(lm)
            prices_over_s[idx] = mod.bs_call_np(S0_market, K, T, market_ivs[idx]) / S0_market
            idx += 1

    x_norm = (torch.tensor(prices_over_s).unsqueeze(0) - mu_x) / std_x

    # ── Network forward pass ────────────────────────────────────────────────
    t0 = time.time()
    with torch.no_grad():
        theta_net = model(x_norm.to(DEVICE)).cpu().numpy()[0]
    t_net = time.time() - t0

    fitted_net = mod.surface_ivs_from_params(theta_net, S0_market)
    rmse_net   = float(np.sqrt(np.mean((fitted_net - market_ivs) ** 2)))

    # ── L-BFGS-B polish ─────────────────────────────────────────────────────
    def obj(p):
        p = np.clip(p, mod.P_LO, mod.P_HI)
        ivs = mod.surface_ivs_from_params(p, S0_market)
        return float(np.mean((ivs - market_ivs) ** 2))

    t0 = time.time()
    res = minimize(obj, theta_net, method="L-BFGS-B",
                   bounds=list(zip(mod.P_LO, mod.P_HI)),
                   options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9})
    t_polish = time.time() - t0

    theta_polish = res.x
    rmse_polish  = float(np.sqrt(res.fun))
    fitted_polish = mod.surface_ivs_from_params(theta_polish, S0_market)

    return dict(
        name           = name,
        n_params       = len(theta_net),
        pnames         = mod.PNAMES,
        theta_net      = theta_net,
        theta_polish   = theta_polish,
        rmse_net       = rmse_net,
        rmse_polish    = rmse_polish,
        t_net_ms       = t_net * 1000,
        t_polish_ms    = t_polish * 1000,
        fitted_net     = fitted_net,
        fitted_polish  = fitted_polish,
        polish_iters   = res.nit,
        polish_success = res.success,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PRINT SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(results):
    print("\n" + "═" * 78)
    print("  THREE-MODEL CALIBRATION COMPARISON")
    print("═" * 78)
    print(f"  Surface:  SPY  S0={S0_MARKET}  {len(GRID_K)} log-moneyness × {len(GRID_T)} maturities")
    print(f"  Grid_K = {GRID_K.tolist()}")
    print(f"  Grid_T = {GRID_T.tolist()}")

    # ── Calibrated parameters ───────────────────────────────────────────────
    for r_ in results:
        print(f"\n── {r_['name']}  ({r_['n_params']} parameters) ─────────────────────────────")
        for n, vn, vp in zip(r_["pnames"], r_["theta_net"], r_["theta_polish"]):
            print(f"    {n:>10}    network={vn:+.5f}    polish={vp:+.5f}")

    # ── Headline comparison table ───────────────────────────────────────────
    print("\n── Headline Results ─────────────────────────────────────────────")
    print(f"  {'Model':>8} {'#par':>5} "
          f"{'NET RMSE':>10} {'POL RMSE':>10} "
          f"{'NET (ms)':>10} {'POL (ms)':>10} "
          f"{'iters':>6}")
    print("  " + "-"*72)
    for r_ in results:
        print(f"  {r_['name']:>8} {r_['n_params']:>5} "
              f"{r_['rmse_net']:>10.5f} {r_['rmse_polish']:>10.5f} "
              f"{r_['t_net_ms']:>10.2f} {r_['t_polish_ms']:>10.0f} "
              f"{r_['polish_iters']:>6}")

    # In vol points (basis points × 100)
    print("\n── Same RMSE in vol points (× 100) ──────────────────────────────")
    print(f"  {'Model':>8} {'NET':>10} {'POLISH':>10}")
    print("  " + "-"*30)
    for r_ in results:
        print(f"  {r_['name']:>8} {r_['rmse_net']*100:>9.2f}  {r_['rmse_polish']*100:>9.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PLOTS
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "Merton": "#d62728",   # red
    "Heston": "#1f77b4",   # blue
    "Bates":  "#2ca02c",   # green
}
MARKERS = {"Merton": "s", "Heston": "^", "Bates": "D"}


def plot_smile_comparison(results, market_ivs):
    """Side-by-side smile slices for each maturity, all three models overlaid."""
    market_2d = market_ivs.reshape(len(GRID_K), len(GRID_T))

    fig, axes = plt.subplots(1, len(GRID_T), figsize=(4 * len(GRID_T), 4),
                             sharey=True)
    for j, (T, ax) in enumerate(zip(GRID_T, axes)):
        ax.plot(GRID_K, market_2d[:, j], "o-", color="black",
                lw=2.5, label="Market", markersize=7)
        for r_ in results:
            ivs2d = r_["fitted_polish"].reshape(len(GRID_K), len(GRID_T))
            ax.plot(GRID_K, ivs2d[:, j],
                    marker=MARKERS[r_["name"]], linestyle="--",
                    color=COLORS[r_["name"]], lw=1.6, markersize=5,
                    label=r_["name"], alpha=0.85)
        ax.set_title(f"T = {T:.2f}")
        ax.set_xlabel("Log-moneyness  k = log(K/F)")
        if j == 0:
            ax.set_ylabel("Implied Vol")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle("Volatility Smile Fits — Market vs Models (post-polish)",
                 y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig("comparison_smiles.png", dpi=140, bbox_inches="tight")
    plt.show()


def plot_residual_heatmaps(results, market_ivs):
    """For each model, plot the (k, T) heatmap of (model - market) IV in vol points."""
    market_2d = market_ivs.reshape(len(GRID_K), len(GRID_T))

    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
    if len(results) == 1:
        axes = [axes]

    # Common colour scale across all heatmaps
    all_resids = np.concatenate([
        (r_["fitted_polish"].reshape(len(GRID_K), len(GRID_T)) - market_2d).ravel()
        for r_ in results
    ])
    vmax = max(0.005, np.abs(all_resids).max())   # at least ±0.5 vol points
    vmin = -vmax

    for ax, r_ in zip(axes, results):
        resid = r_["fitted_polish"].reshape(len(GRID_K), len(GRID_T)) - market_2d
        im = ax.imshow(resid * 100,                 # show in vol-points (× 100)
                       cmap="RdBu_r", origin="lower",
                       vmin=vmin*100, vmax=vmax*100, aspect="auto",
                       extent=[0, len(GRID_T), 0, len(GRID_K)])
        ax.set_xticks(np.arange(len(GRID_T)) + 0.5)
        ax.set_xticklabels([f"{t:.2f}" for t in GRID_T], fontsize=8)
        ax.set_yticks(np.arange(len(GRID_K)) + 0.5)
        ax.set_yticklabels([f"{k:+.2f}" for k in GRID_K], fontsize=8)
        ax.set_title(f"{r_['name']}\nRMSE = {r_['rmse_polish']*100:.2f} vol pts",
                     fontsize=11)
        ax.set_xlabel("Maturity T"); ax.set_ylabel("Log-moneyness k")
        plt.colorbar(im, ax=ax, fraction=0.046, label="Model − Market (vol pts)")

    plt.suptitle("Residual Heatmaps — Where Each Model Fails",
                 y=1.04, fontsize=13)
    plt.tight_layout()
    plt.savefig("comparison_residuals.png", dpi=140, bbox_inches="tight")
    plt.show()


def plot_rmse_speed(results):
    """Bar chart: RMSE and timing for each model."""
    names      = [r_["name"] for r_ in results]
    rmse_net   = [r_["rmse_net"]    * 100 for r_ in results]
    rmse_pol   = [r_["rmse_polish"] * 100 for r_ in results]
    t_net      = [r_["t_net_ms"]    for r_ in results]
    t_pol      = [r_["t_polish_ms"] for r_ in results]
    cols       = [COLORS[n] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # RMSE
    x = np.arange(len(names)); w = 0.35
    axes[0].bar(x - w/2, rmse_net, w, label="Network only", color=cols, alpha=0.55)
    axes[0].bar(x + w/2, rmse_pol, w, label="Network + polish", color=cols)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names)
    axes[0].set_ylabel("IV RMSE (vol points)")
    axes[0].set_title("Calibration Quality")
    axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

    # Timing (log scale because polish dominates)
    axes[1].bar(x - w/2, t_net, w, label="Network only", color=cols, alpha=0.55)
    axes[1].bar(x + w/2, t_pol, w, label="Network + polish", color=cols)
    axes[1].set_xticks(x); axes[1].set_xticklabels(names)
    axes[1].set_ylabel("Wall-clock time (ms, log scale)")
    axes[1].set_yscale("log")
    axes[1].set_title("Calibration Speed")
    axes[1].legend(); axes[1].grid(axis="y", which="both", alpha=0.3)

    plt.suptitle("Speed vs Quality Trade-off Across Three Models", y=1.02)
    plt.tight_layout()
    plt.savefig("comparison_rmse_speed.png", dpi=140, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading saved models...")
    merton_model, m_mu, m_std = load_model("merton_inversion_net.pt",
                                            merton_mod.MertonInversionNet)
    print("  Merton loaded.")
    heston_model, h_mu, h_std = load_model("heston_inversion_net.pt",
                                            heston_mod.HestonInversionNet)
    print("  Heston loaded.")
    bates_model,  b_mu, b_std = load_model("bates_inversion_net.pt",
                                            bates_mod.BatesInversionNet)
    print("  Bates loaded.")

    print("\nCalibrating all three to the same SPY surface...")
    results = [
        calibrate_one("Merton", merton_model, m_mu, m_std, merton_mod,
                      MARKET_IVS, S0_MARKET),
        calibrate_one("Heston", heston_model, h_mu, h_std, heston_mod,
                      MARKET_IVS, S0_MARKET),
        calibrate_one("Bates",  bates_model,  b_mu, b_std, bates_mod,
                      MARKET_IVS, S0_MARKET),
    ]

    print_summary(results)

    print("\nGenerating plots...")
    plot_smile_comparison(results, MARKET_IVS)
    plot_residual_heatmaps(results, MARKET_IVS)
    plot_rmse_speed(results)

    print("\nDone. Saved:")
    print("  comparison_smiles.png")
    print("  comparison_residuals.png")
    print("  comparison_rmse_speed.png")
