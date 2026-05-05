# Created: February 22, 2026

"""
Full Pipeline: Market Data → Cleaned Surface → Merton Calibration
==================================================================

Steps
-----
  1. Fetch  : pull option chain from yfinance (SPY by default)
              falls back to synthetic Merton surface if offline
  2. Filter : remove bad quotes (zero volume, crossed spreads, deep OTM)
  3. Invert : mid-price  →  implied vol via Brent root-find on BS
  4. Arbitrage check : calendar spread + butterfly convexity
  5. Interpolate : scattered market quotes → fixed log-moneyness grid
  6. Calibrate   : MLP direct inversion  +  L-BFGS-B polish
  7. Verify      : COS pricer residual, surface fit plot

Run locally with live data:
    pip install yfinance scipy numpy matplotlib
    python pipeline.py

Note: the COS pricer functions are inlined directly in this file —
no other scripts are required. cos_pricer_validation.py is a separate
standalone script used earlier to verify correctness of the pricer.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats    import norm
from scipy.optimize import brentq, minimize
from scipy.interpolate import RectBivariateSpline
import warnings
warnings.filterwarnings("ignore")

# ── reuse the validated COS pricer ───────────────────────────────────────────
# Inline key functions so this file is self-contained
S0_GLOBAL = None   # set at runtime from spot price
r_GLOBAL  = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# 0.  COS PRICER  (copied from validated script)
# ─────────────────────────────────────────────────────────────────────────────
def merton_cf(u, T, sigma, lam, mu_j, sigma_j, r, S0):
    drift_corr = lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1.0)
    phi_jump   = np.exp(1j * u * mu_j - 0.5 * (u * sigma_j)**2)
    Psi = (
        1j * u * (r - 0.5 * sigma**2 - drift_corr)
        - 0.5 * (u * sigma)**2
        + lam * (phi_jump - 1.0)
    )
    return np.exp(T * Psi)

def cos_truncation(T, sigma, lam, mu_j, sigma_j, r, L=12):
    drift_corr = lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1.0)
    c1 = (r - 0.5 * sigma**2 - drift_corr) * T
    c2 = (sigma**2 + lam * (mu_j**2 + sigma_j**2)) * T
    c4 = lam * (mu_j**4 + 6 * mu_j**2 * sigma_j**2 + 3 * sigma_j**4) * T
    H  = L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    return c1 - H, c1 + H

def call_cos_coefficients(a, b, k):
    bw = b - a
    kp = k * np.pi / bw
    upper = np.exp(b) * np.cos(k * np.pi)
    lower = np.cos(-kp * a) + kp * np.sin(-kp * a)
    chi = np.where(k == 0, np.exp(b) - 1.0,
                   (1.0 / (1.0 + kp**2)) * (upper - lower))
    with np.errstate(invalid="ignore", divide="ignore"):
        psi = np.where(k == 0, b,
                       (np.sin(kp * bw) - np.sin(-kp * a)) / kp)
    return (2.0 / bw) * (chi - psi)

def price_call_cos(K, T, sigma, lam, mu_j, sigma_j, S0, r, n=128):
    a, b = cos_truncation(T, sigma, lam, mu_j, sigma_j, r)
    k    = np.arange(n, dtype=float)
    u    = k * np.pi / (b - a)
    x    = np.log(S0 / K)
    phi  = merton_cf(u, T, sigma, lam, mu_j, sigma_j, r, S0)
    V    = call_cos_coefficients(a, b, k)
    series    = np.real(phi * np.exp(1j * k * np.pi * (x - a) / (b - a)))
    series[0] *= 0.5
    return max(K * np.exp(-r * T) * np.dot(series, V), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA FETCH
# ─────────────────────────────────────────────────────────────────────────────
def fetch_live(ticker="SPY"):
    """
    Pull option chain from yfinance, selecting expiries that give good
    maturity coverage across GRID_T = [1m, 3m, 6m, 12m, 18m].

    For each target maturity, the closest available expiry is selected,
    so the surface spans the full range the calibration needs.
    """
    import yfinance as yf
    from datetime import datetime

    t     = yf.Ticker(ticker)
    spot  = t.fast_info["last_price"]
    today = datetime.today()

    # Compute T (years) for every available expiry
    all_expiries = []
    for exp_str in t.options:
        exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
        T      = (exp_dt - today).days / 365.0
        if T > 0.01:   # skip anything expiring within ~4 days
            all_expiries.append((T, exp_str))

    if not all_expiries:
        raise ValueError("No usable expiries found")

    all_T = np.array([x[0] for x in all_expiries])

    # Target maturities — match GRID_T roughly
    targets = [1/12, 2/12, 3/12, 6/12, 9/12, 12/12, 18/12]

    # For each target pick the closest available expiry (no duplicates)
    chosen = {}
    for tgt in targets:
        idx = int(np.argmin(np.abs(all_T - tgt)))
        exp_str  = all_expiries[idx][1]
        T_actual = all_expiries[idx][0]
        if exp_str not in chosen:
            chosen[exp_str] = T_actual

    print(f"\nSelected expiries for {ticker}  (spot={spot:.2f}):")
    rows = []
    for exp_str, T in sorted(chosen.items(), key=lambda x: x[1]):
        chain    = t.option_chain(exp_str).calls
        n_before = len(rows)
        for _, row in chain.iterrows():
            if row.bid > 0 and row.ask > 0:
                vol = row.volume
                oi  = row.openInterest
                rows.append(dict(
                    T      = T,
                    K      = float(row.strike),
                    bid    = float(row.bid),
                    ask    = float(row.ask),
                    mid    = 0.5 * (float(row.bid) + float(row.ask)),
                    volume = int(vol) if (vol is not None and vol == vol) else 0,
                    oi     = int(oi)  if (oi  is not None and oi  == oi)  else 0,
                ))
        print(f"  {exp_str}  T={T:.3f}  quotes={len(rows)-n_before}")

    print(f"Total: {len(rows)} raw quotes")
    return rows, spot


def synthetic_surface(true_params=(0.18, 0.40, -0.06, 0.12),
                      S0=500.0, r=0.05, noise=0.003):
    """
    Generate a synthetic option surface that mimics what yfinance returns.
    Uses the validated COS pricer with known Merton parameters + Gaussian noise.
    """
    sigma, lam, mu_j, sigma_j = true_params
    strikes    = np.array([0.75, 0.80, 0.85, 0.90, 0.95,
                            1.00, 1.05, 1.10, 1.15, 1.20, 1.25]) * S0
    maturities = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 12/12, 18/12])

    rows = []
    for T in maturities:
        for K in strikes:
            price = price_call_cos(K, T, sigma, lam, mu_j, sigma_j, S0, r)
            spread = max(price * 0.01, 0.05)
            mid    = price * (1 + np.random.normal(0, noise))
            rows.append(dict(
                T=T, K=K,
                bid=mid - spread/2,
                ask=mid + spread/2,
                mid=mid,
                volume=int(np.random.exponential(500)),
                oi=int(np.random.exponential(2000)),
            ))

    print(f"Synthetic surface: {len(rows)} quotes  S0={S0}  true_params={true_params}")
    return rows, S0


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FILTER: remove bad quotes
# ─────────────────────────────────────────────────────────────────────────────
def filter_quotes(rows, S0, r,
                  moneyness_lo=0.70, moneyness_hi=1.35,
                  min_volume=0, min_mid=0.05):
    """
    Remove:
      - Crossed or zero spreads
      - Deep OTM / deep ITM (outside moneyness band)
      - Below intrinsic value (model-free no-arbitrage)
      - Very low mid prices (dominated by bid-ask noise)
      - Zero volume (if filtering by volume)
    """
    clean = []
    n_removed = {"spread": 0, "moneyness": 0, "intrinsic": 0,
                 "min_price": 0, "volume": 0}

    for q in rows:
        K, T, mid = q["K"], q["T"], q["mid"]
        F = S0 * np.exp(r * T)                  # forward price
        m = K / F                               # moneyness K/F

        if q["bid"] <= 0 or q["ask"] <= q["bid"]:
            n_removed["spread"] += 1;  continue
        if not (moneyness_lo <= m <= moneyness_hi):
            n_removed["moneyness"] += 1;  continue
        intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
        if mid < intrinsic - 0.01:
            n_removed["intrinsic"] += 1;  continue
        if mid < min_mid:
            n_removed["min_price"] += 1;  continue
        if q["volume"] < min_volume:
            n_removed["volume"] += 1;  continue
        clean.append(q)

    print(f"\nFilter: {len(rows)} → {len(clean)} quotes kept")
    for k, v in n_removed.items():
        if v: print(f"  removed {v:>4} ({k})")
    return clean


# ─────────────────────────────────────────────────────────────────────────────
# 3.  IMPLIED VOL INVERSION  (Brent root-find on Black-Scholes)
# ─────────────────────────────────────────────────────────────────────────────
def bs_call(S0, K, T, r, sigma):
    """Black-Scholes call price."""
    if sigma <= 0 or T <= 0:
        return max(S0 - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(mid, S0, K, T, r, tol=1e-6, lo=1e-4, hi=5.0):
    """
    Invert BS formula to get implied vol via Brent's method.
    Returns np.nan if inversion fails (intrinsic only, illiquid, etc.)
    """
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)
    if mid <= intrinsic + tol:
        return np.nan
    try:
        f_lo = bs_call(S0, K, T, r, lo) - mid
        f_hi = bs_call(S0, K, T, r, hi) - mid
        if f_lo * f_hi > 0:
            return np.nan
        return brentq(lambda s: bs_call(S0, K, T, r, s) - mid,
                      lo, hi, xtol=tol, maxiter=200)
    except Exception:
        return np.nan


def invert_surface(rows, S0, r):
    """Add implied_vol field to each quote. Drop failures."""
    result = []
    n_fail = 0
    for q in rows:
        iv = implied_vol(q["mid"], S0, q["K"], q["T"], r)
        if np.isnan(iv):
            n_fail += 1
            continue
        result.append({**q, "iv": iv})
    print(f"\nIV inversion: {len(result)} succeeded, {n_fail} failed (NaN dropped)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ARBITRAGE CHECKS
# ─────────────────────────────────────────────────────────────────────────────
def check_calendar_spread(rows):
    """
    Calendar spread: total variance w(k,T) = iv^2 * T must be
    non-decreasing in T for fixed log-moneyness k = log(K/F).
    Reports violations; does NOT remove them (user decides).
    """
    from collections import defaultdict
    # Bin by approximate moneyness (round to nearest 0.02)
    buckets = defaultdict(list)
    for q in rows:
        k_bin = round(np.log(q["K"] / (S0_GLOBAL * np.exp(r_GLOBAL * q["T"]))) / 0.02) * 0.02
        buckets[k_bin].append((q["T"], q["iv"]))

    violations = 0
    for k_bin, pts in buckets.items():
        pts.sort()
        for i in range(len(pts) - 1):
            T1, iv1 = pts[i]
            T2, iv2 = pts[i+1]
            w1, w2  = iv1**2 * T1, iv2**2 * T2
            if w2 < w1 - 1e-4:
                violations += 1
    if violations:
        print(f"  ⚠  Calendar spread: {violations} violation(s) found")
    else:
        print("  ✓  Calendar spread: no violations")
    return violations


def check_butterfly(rows):
    """
    Butterfly: for fixed T, the call price must be convex in K.
    Check: C(K-h) - 2*C(K) + C(K+h) >= 0 for consecutive strikes.
    """
    from collections import defaultdict
    by_T = defaultdict(list)
    for q in rows:
        by_T[round(q["T"], 4)].append((q["K"], q["mid"]))

    violations = 0
    for T, pts in by_T.items():
        pts.sort()
        for i in range(1, len(pts) - 1):
            Km, Cm = pts[i-1]
            K0, C0 = pts[i]
            Kp, Cp = pts[i+1]
            h1, h2 = K0 - Km, Kp - K0
            if abs(h1) < 1e-6 or abs(h2) < 1e-6: continue
            # Finite difference approximation of d^2C/dK^2
            butterfly = (Cp / h2 - C0 * (1/h1 + 1/h2) + Cm / h1)
            if butterfly < -0.005:
                violations += 1
    if violations:
        print(f"  ⚠  Butterfly: {violations} violation(s) found")
    else:
        print("  ✓  Butterfly: no violations")
    return violations


# ─────────────────────────────────────────────────────────────────────────────
# 5.  INTERPOLATE ONTO FIXED LOG-MONEYNESS GRID
# ─────────────────────────────────────────────────────────────────────────────
# Fixed grid: 7 log-moneyness × 5 maturities = 35 nodes
GRID_K = np.array([-0.20, -0.12, -0.06, 0.00, 0.06, 0.12, 0.20])  # log(K/F)
GRID_T = np.array([1/12, 3/12, 6/12, 12/12, 18/12])                # years
N_GRID = len(GRID_K) * len(GRID_T)                                   # 35


def interpolate_surface(rows, S0, r):
    """
    Interpolate scattered (log-moneyness, T, IV) data onto GRID_K × GRID_T.

    Strategy:
      1. Convert all quotes to log-moneyness coordinates
      2. Print diagnostics so range mismatches are visible
      3. Try SmoothBivariateSpline (best quality)
      4. Fall back to LinearNDInterpolator + NearestNDInterpolator for
         any grid points that fall outside the convex hull of the data

    Returns float32 array of shape (N_GRID,) = (35,).
    """
    from scipy.interpolate import (SmoothBivariateSpline,
                                   LinearNDInterpolator,
                                   NearestNDInterpolator)

    # ── Convert to log-moneyness ─────────────────────────────────────────────
    pts = []
    for q in rows:
        if np.isnan(q.get("iv", np.nan)):
            continue
        F  = S0 * np.exp(r * q["T"])
        lm = np.log(q["K"] / F)
        pts.append((lm, q["T"], q["iv"]))

    if len(pts) == 0:
        raise ValueError("No valid quotes after IV inversion — cannot interpolate.")

    pts = np.array(pts)
    lm_ = pts[:, 0]
    T_  = pts[:, 1]
    iv_ = pts[:, 2]

    # ── Diagnostics ──────────────────────────────────────────────────────────
    print(f"\n  Data range:  log-moneyness [{lm_.min():.3f}, {lm_.max():.3f}]"
          f"   T [{T_.min():.3f}, {T_.max():.3f}]")
    print(f"  Grid range:  log-moneyness [{GRID_K.min():.3f}, {GRID_K.max():.3f}]"
          f"   T [{GRID_T.min():.3f}, {GRID_T.max():.3f}]")
    print(f"  Points available: {len(pts)}")

    # ── Warn if grid extends well outside data ───────────────────────────────
    if GRID_T.min() < T_.min() - 0.05:
        print(f"  ⚠  Grid T_min={GRID_T.min():.3f} < data T_min={T_.min():.3f} "
              f"— short maturities will be extrapolated")
    if GRID_T.max() > T_.max() + 0.05:
        print(f"  ⚠  Grid T_max={GRID_T.max():.3f} > data T_max={T_.max():.3f} "
              f"— long maturities will be extrapolated")

    # ── Build grid ───────────────────────────────────────────────────────────
    KK, TT = np.meshgrid(GRID_K, GRID_T, indexing="ij")   # both (7, 5)
    pts2d  = np.column_stack([KK.ravel(), TT.ravel()])     # (35, 2)

    # Try smooth bivariate spline first
    grid_ivs = None
    try:
        spline   = SmoothBivariateSpline(lm_, T_, iv_,
                                         kx=3, ky=3,
                                         s=len(lm_) * 0.0002)
        grid_ivs = spline(GRID_K, GRID_T)   # (7, 5)

        # Reject spline if it produces nonsense (NaN or out of range)
        if np.any(np.isnan(grid_ivs)) or grid_ivs.min() < 0.01 or grid_ivs.max() > 3.0:
            raise ValueError("Spline produced out-of-range values")

        print("  Interpolation: smooth bivariate spline ✓")

    except Exception as e:
        print(f"  Spline failed ({e}) — using linear + nearest-neighbour")

        data2d   = np.column_stack([lm_, T_])
        linear   = LinearNDInterpolator(data2d, iv_)
        nearest  = NearestNDInterpolator(data2d, iv_)

        vals          = linear(pts2d)            # NaN outside convex hull
        nan_mask      = np.isnan(vals)
        vals[nan_mask] = nearest(pts2d[nan_mask])  # fill extrapolation with nearest

        grid_ivs = vals.reshape(len(GRID_K), len(GRID_T))

    # ── Clip to sensible IV range ────────────────────────────────────────────
    grid_ivs = np.clip(grid_ivs, 0.02, 2.0)

    # ── Print surface ────────────────────────────────────────────────────────
    print(f"\nInterpolated surface  ({len(GRID_K)} log-moneyness × {len(GRID_T)} maturities):")
    print("           " + "".join(f"  T={t:.2f}" for t in GRID_T))
    for i, k in enumerate(GRID_K):
        row = f"  k={k:+.2f}  " + "  ".join(f"{grid_ivs[i,j]:.4f}"
                                              for j in range(len(GRID_T)))
        print(row)

    return grid_ivs.astype(np.float32).ravel()   # (35,)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CALIBRATION
#     6a. Analytic inversion via COS: grid-search warm start
#     6b. L-BFGS-B local polish
# ─────────────────────────────────────────────────────────────────────────────
P_LO = np.array([0.05, 0.00, -0.30, 0.02])
P_HI = np.array([0.60, 1.50,  0.10, 0.40])
PNAMES = ["sigma", "lambda", "mu_J", "sigma_J"]


def surface_from_params(params, S0, r):
    """
    Compute model-implied IVs on the GRID_K × GRID_T grid.
    Returns (N_GRID,) float array.
    """
    sigma, lam, mu_j, sigma_j = params
    ivs = []
    for T in GRID_T:
        F = S0 * np.exp(r * T)
        for lm in GRID_K:
            K     = F * np.exp(lm)
            price = price_call_cos(K, T, sigma, lam, mu_j, sigma_j, S0, r)
            iv    = implied_vol(price, S0, K, T, r)
            ivs.append(iv if not np.isnan(iv) else 0.0)
    return np.array(ivs, dtype=np.float32)


def objective(params, market_ivs, S0, r):
    """MSE between model IVs and market IVs on the grid."""
    params = np.clip(params, P_LO, P_HI)
    model_ivs = surface_from_params(params, S0, r)
    return float(np.mean((model_ivs - market_ivs) ** 2))


def grid_search_warmstart(market_ivs, S0, r, n_grid=6):
    """
    Coarse grid search over parameter space using current P_LO / P_HI bounds.
    Finds a good starting point for L-BFGS-B, avoiding bad local minima.
    """
    print("\nGrid search warm start...")
    best_loss = np.inf
    best_p    = None

    # Use actual bounds — evenly spaced across each parameter range
    sigma_vals   = np.linspace(P_LO[0], P_HI[0], n_grid)
    lam_vals     = np.linspace(max(P_LO[1], 0.10), P_HI[1], n_grid)
    mu_j_vals    = np.linspace(P_LO[2], P_HI[2], 4)
    sigma_j_vals = np.linspace(P_LO[3], P_HI[3], 4)

    total = len(sigma_vals)*len(lam_vals)*len(mu_j_vals)*len(sigma_j_vals)
    print(f"  Searching {total} grid points  "
          f"(sigma×{len(sigma_vals)}, lam×{len(lam_vals)}, "
          f"mu_J×{len(mu_j_vals)}, sigma_J×{len(sigma_j_vals)})")

    for s in sigma_vals:
        for l in lam_vals:
            for mj in mu_j_vals:
                for sj in sigma_j_vals:
                    p    = np.array([s, l, mj, sj])
                    loss = objective(p, market_ivs, S0, r)
                    if loss < best_loss:
                        best_loss = loss
                        best_p    = p.copy()

    print(f"  Best grid loss = {best_loss:.6f}  "
          f"params = {dict(zip(PNAMES, best_p.round(4)))}")
    return best_p


def lbfgsb_polish(p0, market_ivs, S0, r):
    """
    L-BFGS-B local optimization starting from p0.
    Uses current P_LO / P_HI bounds.
    """
    print("\nL-BFGS-B polish...")
    bounds = list(zip(P_LO, P_HI))
    res = minimize(
        objective, p0, args=(market_ivs, S0, r),
        method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9},
    )
    print(f"  Converged: {res.success}   loss = {res.fun:.8f}   iters = {res.nit}")
    if not res.success:
        print(f"  Reason: {res.message}")
    for name, val, lo, hi in zip(PNAMES, res.x, P_LO, P_HI):
        at_bound = " ← at bound" if abs(val-lo)<1e-6 or abs(val-hi)<1e-6 else ""
        print(f"    {name:>10} = {val:.5f}  [{lo}, {hi}]{at_bound}")
    return res.x, res.fun


def calibrate(market_ivs, S0, r):
    """Full calibration: grid search → L-BFGS-B."""
    p0     = grid_search_warmstart(market_ivs, S0, r)
    p_cal, loss = lbfgsb_polish(p0, market_ivs, S0, r)
    return p_cal, loss


# ─────────────────────────────────────────────────────────────────────────────
# 7.  VERIFY AND PLOT
# ─────────────────────────────────────────────────────────────────────────────
def verify_and_plot(p_cal, market_ivs, S0, r, true_params=None):
    model_ivs = surface_from_params(p_cal, S0, r)
    rmse      = np.sqrt(np.mean((model_ivs - market_ivs)**2))

    print("\n── Calibrated Parameters ────────────────────────────────────────")
    for name, val in zip(PNAMES, p_cal):
        truth = f"  (true: {true_params[PNAMES.index(name)]:.4f})" if true_params else ""
        print(f"  {name:>10} = {val:.5f}{truth}")
    print(f"\n  Surface RMSE (IV) = {rmse:.5f}  ({rmse*100:.2f} vol points)")

    # Surface fit heatmap
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    market_2d = market_ivs.reshape(len(GRID_K), len(GRID_T))
    model_2d  = model_ivs.reshape(len(GRID_K), len(GRID_T))
    resid_2d  = model_2d - market_2d

    T_labels  = [f"{t:.2f}" for t in GRID_T]
    K_labels  = [f"{k:+.2f}" for k in GRID_K]
    extent    = [0, len(GRID_T), 0, len(GRID_K)]

    for ax, data, title in zip(
        axes,
        [market_2d, model_2d, resid_2d],
        ["Market IV", "Model IV (Merton)", "Residual (Model - Market)"]
    ):
        im = ax.imshow(data, aspect="auto", origin="lower",
                       cmap="RdYlGn_r" if "Resid" not in title else "coolwarm",
                       extent=extent)
        ax.set_xticks(np.arange(len(GRID_T)) + 0.5)
        ax.set_xticklabels(T_labels, fontsize=8)
        ax.set_yticks(np.arange(len(GRID_K)) + 0.5)
        ax.set_yticklabels(K_labels, fontsize=8)
        ax.set_xlabel("Maturity T"); ax.set_ylabel("Log-moneyness k")
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f"Merton Calibration — Surface Fit  (RMSE={rmse:.4f})", y=1.02)
    plt.tight_layout()
    plt.savefig("surface_fit.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved surface_fit.png")

    # Smile slices
    fig, axes = plt.subplots(1, len(GRID_T), figsize=(16, 4), sharey=True)
    for j, (T, ax) in enumerate(zip(GRID_T, axes)):
        ax.plot(GRID_K, market_2d[:, j], "o-",  label="Market", lw=2)
        ax.plot(GRID_K, model_2d[:,  j], "s--", label="Merton", lw=2)
        ax.set_title(f"T={T:.2f}", fontsize=10)
        ax.set_xlabel("Log-moneyness")
        if j == 0: ax.set_ylabel("Implied Vol")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle("Volatility Smile Slices", y=1.02)
    plt.tight_layout()
    plt.savefig("smile_slices.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved smile_slices.png")

    return rmse


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TRUE_PARAMS = (0.18, 0.40, -0.06, 0.12)   # known only for synthetic test

    # ── Step 1: Fetch data ───────────────────────────────────────────────────
    try:
        rows, S0 = fetch_live("SPY")
        true_params_for_plot = None    # unknown for real data
    except Exception as e:
        print(f"yfinance unavailable ({e})  →  using synthetic surface")
        rows, S0 = synthetic_surface(TRUE_PARAMS)
        true_params_for_plot = list(TRUE_PARAMS)

    S0_GLOBAL = S0
    r = r_GLOBAL = 0.05

    # ── Step 2: Filter ───────────────────────────────────────────────────────
    rows = filter_quotes(rows, S0, r)

    # ── Step 3: Implied vol inversion ────────────────────────────────────────
    rows = invert_surface(rows, S0, r)

    # ── Step 4: Arbitrage checks ─────────────────────────────────────────────
    print("\nArbitrage checks:")
    check_calendar_spread(rows)
    check_butterfly(rows)

    # ── Step 5: Interpolate to fixed grid ────────────────────────────────────
    market_ivs = interpolate_surface(rows, S0, r)   # (35,)

    # ── Step 6: Calibrate ────────────────────────────────────────────────────
    p_cal, final_loss = calibrate(market_ivs, S0, r)

    # ── Step 7: Verify ───────────────────────────────────────────────────────
    true_list = [TRUE_PARAMS[i] for i in range(4)] if true_params_for_plot else None
    verify_and_plot(p_cal, market_ivs, S0, r,
                    true_params=true_list)
