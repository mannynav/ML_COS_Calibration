# Created: March 15, 2026

"""
ML Calibration of Merton Jump Diffusion via CF-Informed Neural Network
========================================================================

Pipeline:
  1.  Generate synthetic (theta, surface) pairs using the validated COS pricer
  2.  Train MLP for direct inversion: surface -> theta
       - Supervised loss: predicted theta vs true theta
       - Physics loss   : torch_COS(predicted theta) vs input surface
  3.  Evaluate on synthetic test set
  4.  Calibrate to the real SPY market_ivs vector (compared against L-BFGS-B)

Surface representation:
  Call prices / S0 on a fixed log-moneyness × maturity grid (35-dim).
  Scale-invariant; same format for training and deployment.

Run:  python ml_calibration.py
Deps: torch numpy scipy matplotlib  (pip install torch ...)
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from scipy.stats    import norm
from scipy.optimize import brentq, minimize
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────────────────
# 0.  CONFIG
# ────────────────────────────────────────────────────────────────────────────
torch.manual_seed(0); np.random.seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S0_TRAIN = 100.0     # all training surfaces use spot=100; deployment rescales
r        = 0.05
N_COS    = 256       # COS terms (numpy data gen) — needed for high lambda
N_TRAIN_COS = 128    # COS terms in differentiable torch pricer

# Fixed grid — must match pipeline.py
GRID_K = np.array([-0.20, -0.12, -0.06, 0.00, 0.06, 0.12, 0.20])  # log(K/F)
GRID_T = np.array([1/12, 3/12, 6/12, 12/12, 18/12])               # years
N_GRID = len(GRID_K) * len(GRID_T)                                  # 35

# Parameter bounds — realistic Merton range for index options.
# Lambda is deliberately kept < 2 because higher intensities + large jumps
# push the COS method into a numerically unstable regime requiring very high N.
# Real-world Merton calibrations to SPY rarely produce lambda > 2.
P_LO   = np.array([0.05, 0.05, -0.30, 0.05], dtype=np.float32)
P_HI   = np.array([0.40, 2.00,  0.05, 0.25], dtype=np.float32)
PNAMES = ["sigma", "lambda", "mu_J", "sigma_J"]


# ────────────────────────────────────────────────────────────────────────────
# 1.  NUMPY COS PRICER  (data generation — fast, validated)
# ────────────────────────────────────────────────────────────────────────────
def merton_cf_np(u, T, sigma, lam, mu_j, sigma_j):
    drift_corr = lam * (np.exp(mu_j + 0.5*sigma_j**2) - 1.0)
    phi_jump   = np.exp(1j*u*mu_j - 0.5*(u*sigma_j)**2)
    Psi = (1j*u*(r - 0.5*sigma**2 - drift_corr)
           - 0.5*(u*sigma)**2
           + lam*(phi_jump - 1.0))
    return np.exp(T * Psi)

def cos_truncation_np(T, sigma, lam, mu_j, sigma_j, L=10):
    drift_corr = lam * (np.exp(mu_j + 0.5*sigma_j**2) - 1.0)
    c1 = (r - 0.5*sigma**2 - drift_corr)*T
    c2 = (sigma**2 + lam*(mu_j**2 + sigma_j**2))*T
    c4 = lam*(mu_j**4 + 6*mu_j**2*sigma_j**2 + 3*sigma_j**4)*T
    H  = L*np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    return c1-H, c1+H

def call_cos_coefficients_np(a, b, k):
    bw = b - a
    kp = k * np.pi / bw
    upper = np.exp(b) * np.cos(k*np.pi)
    lower = np.cos(-kp*a) + kp*np.sin(-kp*a)
    chi = np.where(k==0, np.exp(b)-1.0,
                   (1/(1+kp**2)) * (upper - lower))
    with np.errstate(invalid="ignore", divide="ignore"):
        psi = np.where(k==0, b,
                       (np.sin(kp*bw) - np.sin(-kp*a)) / kp)
    return (2/bw) * (chi - psi)

def price_call_cos_np(K, T, sigma, lam, mu_j, sigma_j, S0=S0_TRAIN, n=N_COS):
    a, b = cos_truncation_np(T, sigma, lam, mu_j, sigma_j)
    k    = np.arange(n, dtype=float)
    u    = k * np.pi / (b - a)
    x    = np.log(S0 / K)
    phi  = merton_cf_np(u, T, sigma, lam, mu_j, sigma_j)
    V    = call_cos_coefficients_np(a, b, k)
    series    = np.real(phi * np.exp(1j*k*np.pi*(x-a)/(b-a)))
    series[0] *= 0.5
    return max(K*np.exp(-r*T)*np.dot(series, V), 0.0)


# ────────────────────────────────────────────────────────────────────────────
# 2.  TORCH COS PRICER  (differentiable, used in CF physics loss)
# ────────────────────────────────────────────────────────────────────────────
PI = np.pi

def merton_cf_torch(u, T, sigma, lam, mu_j, sigma_j):
    """Batched: u (F,), T scalar, params (B,1) → (B,F) complex."""
    drift_corr = lam * (torch.exp(mu_j + 0.5*sigma_j**2) - 1.0)
    phi_jump   = torch.exp(1j*u*mu_j - 0.5*(u*sigma_j)**2)
    Psi = (1j*u*(r - 0.5*sigma**2 - drift_corr)
           - 0.5*(u*sigma)**2
           + lam*(phi_jump - 1.0))
    return torch.exp(T * Psi)

def cos_truncation_torch(T, sigma, lam, mu_j, sigma_j, L=10):
    drift_corr = lam * (torch.exp(mu_j + 0.5*sigma_j**2) - 1.0)
    c1 = (r - 0.5*sigma**2 - drift_corr)*T
    c2 = (sigma**2 + lam*(mu_j**2 + sigma_j**2))*T
    c4 = lam*(mu_j**4 + 6*mu_j**2*sigma_j**2 + 3*sigma_j**4)*T
    H  = L*torch.sqrt(torch.abs(c2) + torch.sqrt(torch.abs(c4)))
    return c1-H, c1+H

def price_surface_torch(theta, S0=S0_TRAIN, n=N_TRAIN_COS):
    """
    Differentiable COS surface pricer.

    theta : (B, 4) float32 [sigma, lam, mu_j, sigma_j]
    return: (B, N_GRID=35) float32 — call prices / S0 on GRID_K x GRID_T

    Implementation notes for autograd safety:
      - The k=0 term is computed *separately* from k>=1 to avoid the
        well-known torch.where gradient trap where the unused branch
        still contributes NaN/inf gradients via the chain rule.
      - All intermediate tensors are clamped to physically valid ranges
        before any division.
    """
    th  = theta.double()
    sigma   = th[:, 0:1].clamp(min=1e-4)
    lam     = th[:, 1:2].clamp(min=0.0)
    mu_j    = th[:, 2:3]
    sigma_j = th[:, 3:4].clamp(min=1e-4)

    # k=0 term and k>=1 terms handled separately
    k_pos = torch.arange(1, n, dtype=torch.float64, device=theta.device)  # (F-1,)

    all_prices = []
    for T_val in GRID_T:
        T  = torch.tensor(T_val,           dtype=torch.float64, device=theta.device)
        eT = torch.tensor(np.exp(-r*T_val),dtype=torch.float64, device=theta.device)

        a, b = cos_truncation_torch(T, sigma, lam, mu_j, sigma_j)        # (B,1)
        bw   = (b - a).clamp(min=1e-3)

        # ── k=0 contributions (handled separately) ──────────────────────
        # phi(0) = 1 always for a valid CF
        # V_0    = (2/bw) * (chi_0 - psi_0)  with  chi_0 = e^b - 1,  psi_0 = b
        chi_0 = torch.exp(b) - 1.0                                          # (B,1)
        psi_0 = b                                                            # (B,1)
        V_0   = (2.0 / bw) * (chi_0 - psi_0)                                 # (B,1)

        # ── k>=1 contributions ──────────────────────────────────────────
        u_pos  = k_pos * PI / bw                                            # (B,F-1)
        phi_p  = merton_cf_torch(u_pos, T, sigma, lam, mu_j, sigma_j)       # (B,F-1)

        kp     = k_pos * PI / bw                                            # (B,F-1)
        upper  = torch.exp(b) * torch.cos(k_pos * PI)                       # (B,F-1)
        lower  = torch.cos(-kp*a) + kp * torch.sin(-kp*a)                   # (B,F-1)
        chi_p  = (1.0 / (1.0 + kp**2)) * (upper - lower)                    # (B,F-1)
        psi_p  = (torch.sin(kp*bw) - torch.sin(-kp*a)) / kp                 # safe — kp>0
        V_p    = (2.0 / bw) * (chi_p - psi_p)                               # (B,F-1)

        F_T = S0 * np.exp(r * T_val)
        for lm_val in GRID_K:
            K = F_T * np.exp(lm_val)
            x = np.log(S0 / K)

            # k=0 contribution (halved per midpoint rule)
            # Re[phi(0) * exp(0)] = 1
            term_0 = 0.5 * 1.0 * V_0.squeeze(-1)                            # (B,)

            # k>=1 contributions
            series_p = torch.real(phi_p * torch.exp(1j*k_pos*PI*(x-a)/bw))  # (B,F-1)
            term_p   = (series_p * V_p).sum(dim=-1)                         # (B,)

            price = K * eT * (term_0 + term_p)                              # (B,)
            price = torch.clamp(price, min=0.0).float() / S0
            all_prices.append(price)

    return torch.stack(all_prices, dim=1)   # (B, 35)


# ────────────────────────────────────────────────────────────────────────────
# 3.  HELPERS — IV ↔ price conversion (BS, scipy)
# ────────────────────────────────────────────────────────────────────────────
def bs_call_np(S0, K, T, sigma):
    if sigma <= 0 or T <= 0:
        return max(S0 - K*np.exp(-r*T), 0.0)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def implied_vol_np(price, S0, K, T):
    intrinsic = max(S0 - K*np.exp(-r*T), 0.0)
    if price <= intrinsic + 1e-8: return np.nan
    try:
        return brentq(lambda s: bs_call_np(S0, K, T, s) - price,
                      1e-4, 5.0, xtol=1e-7)
    except ValueError:
        return np.nan


# ────────────────────────────────────────────────────────────────────────────
# 4.  DATASET GENERATION
# ────────────────────────────────────────────────────────────────────────────
def generate_surface_np(theta, S0=S0_TRAIN):
    """Surface = call prices / S0  on GRID_K × GRID_T   →   (35,) float32."""
    sigma, lam, mu_j, sigma_j = theta
    out = np.empty(N_GRID, dtype=np.float32)
    idx = 0
    for T in GRID_T:
        F = S0 * np.exp(r*T)
        for lm in GRID_K:
            K = F * np.exp(lm)
            out[idx] = price_call_cos_np(K, T, sigma, lam, mu_j, sigma_j, S0) / S0
            idx += 1
    return out

def generate_dataset(n, verbose=True, max_price_over_s=1.0):
    """
    Sample params from prior, price surfaces, return (X, Y) arrays.

    A price/S0 above 1.0 is impossible for a call (call price <= S0 always),
    so any sample exceeding this is a numerical artefact and is resampled
    until a valid surface is produced.  This keeps high-lambda/large-jump samples
    in the dataset without their corrupted prices poisoning training.
    """
    if verbose: print(f"Generating {n} training samples...")
    Y = np.empty((n, 4), dtype=np.float32)
    X = np.empty((n, N_GRID), dtype=np.float32)
    t0 = time.time()
    n_resampled = 0
    for i in range(n):
        attempts = 0
        while True:
            theta = np.random.uniform(P_LO, P_HI, 4).astype(np.float32)
            surf  = generate_surface_np(theta)
            if surf.max() < max_price_over_s and np.all(np.isfinite(surf)):
                break
            attempts += 1
            n_resampled += 1
            if attempts > 50:
                # extremely rare — just take it
                break
        Y[i] = theta
        X[i] = surf
        if verbose and (i+1) % max(1, n//10) == 0:
            print(f"  {i+1}/{n}  ({time.time()-t0:.1f}s)  resampled={n_resampled}")
    if verbose:
        print(f"  Total resamples: {n_resampled}")
    return X, Y


# ────────────────────────────────────────────────────────────────────────────
# 5.  NETWORK
# ────────────────────────────────────────────────────────────────────────────
class MertonInversionNet(nn.Module):
    """
    Direct-inversion MLP: surface (35-dim) → Merton parameters (4-dim).
    Output passed through sigmoid then scaled to [P_LO, P_HI] so predictions
    always lie inside the physical parameter range.
    """
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_GRID,  hidden), nn.SiLU(),
            nn.Linear(hidden,  hidden), nn.SiLU(),
            nn.Linear(hidden,  hidden), nn.SiLU(),
            nn.Linear(hidden,  hidden//2), nn.SiLU(),
            nn.Linear(hidden//2, 4),    nn.Sigmoid(),
        )
        self.register_buffer("lo", torch.tensor(P_LO))
        self.register_buffer("hi", torch.tensor(P_HI))

    def forward(self, x):
        u = self.net(x)              # (B, 4) in [0, 1]
        return self.lo + u * (self.hi - self.lo)


# ────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING
# ────────────────────────────────────────────────────────────────────────────
def train(n_train=20_000, n_val=2_000, epochs=120, batch_size=256,
          lam_cf=10.0, cf_warmup=20, lr=1e-3):
    """
    Loss = L_param  +  cf_weight(epoch) * L_cf

      L_param   : range-normalized MSE between predicted theta and true theta
                  (so lambda doesn't dominate sigma/mu_J/sigma_J in scale)
      L_cf      : MSE between torch_COS(predicted theta) and input surface
                  — this enforces characteristic-function consistency

      cf_weight(epoch) ramps linearly from 0 to lam_cf over the first
      `cf_warmup` epochs.  This stops the physics loss from blowing up
      early when the network's predictions are random.
    """
    # ── Data ────────────────────────────────────────────────────────────────
    X_np, Y_np = generate_dataset(n_train + n_val)
    X_all      = torch.tensor(X_np)
    Y_all      = torch.tensor(Y_np)

    # Standardize inputs from training stats only
    mu_x  = X_all[:n_train].mean(0)
    std_x = X_all[:n_train].std(0).clamp(min=1e-8)
    X_norm = (X_all - mu_x) / std_x

    X_tr, Y_tr  = X_norm[:n_train].to(DEVICE), Y_all[:n_train].to(DEVICE)
    X_va, Y_va  = X_norm[n_train:].to(DEVICE), Y_all[n_train:].to(DEVICE)
    X_raw_tr    = X_all[:n_train].to(DEVICE)        # raw prices for CF loss

    # ── Model / optimizer ───────────────────────────────────────────────────
    model = MertonInversionNet().to(DEVICE)
    opt   = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # ── Loop ────────────────────────────────────────────────────────────────
    print(f"\nTraining on {DEVICE}  |  n_train={n_train}  |  epochs={epochs}  |  lam_cf={lam_cf}")
    print(f"{'Ep':>4} {'Tot':>10} {'Lparam':>10} {'Lcf':>10} {'Val':>10}")
    print("-" * 50)
    n_batches = max(1, n_train // batch_size)
    history = []

    for ep in range(epochs):
        model.train()
        idx = torch.randperm(n_train, device=DEVICE)
        ep_lp = ep_lcf = 0.0

        for b in range(n_batches):
            bi      = idx[b*batch_size : (b+1)*batch_size]
            xb      = X_tr[bi]
            yb      = Y_tr[bi]
            xb_raw  = X_raw_tr[bi]

            theta_hat   = model(xb)                       # (B, 4)

            # Normalize parameter MSE by parameter range so lambda doesn't
            # dominate (lambda in [0,5], others in [0, 0.5])
            scale       = (model.hi - model.lo).to(theta_hat.device)
            L_param     = nn.functional.mse_loss(theta_hat / scale,
                                                  yb / scale)

            prices_hat  = price_surface_torch(theta_hat)  # (B, 35) — physics loss
            L_cf        = nn.functional.mse_loss(prices_hat, xb_raw)

            cf_weight = lam_cf * min(1.0, ep / max(1, cf_warmup))
            loss = L_param + cf_weight * L_cf
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_lp  += L_param.item()
            ep_lcf += L_cf.item()
        sched.step()

        avg_lp  = ep_lp  / n_batches
        avg_lcf = ep_lcf / n_batches
        model.eval()
        with torch.no_grad():
            scale    = (model.hi - model.lo).to(X_va.device)
            val_loss = nn.functional.mse_loss(model(X_va) / scale,
                                              Y_va / scale).item()
        cf_weight = lam_cf * min(1.0, ep / max(1, cf_warmup))
        history.append((avg_lp + cf_weight*avg_lcf, val_loss, avg_lp, avg_lcf))

        if (ep+1) % 10 == 0 or ep == 0:
            print(f"{ep+1:>4} {history[-1][0]:>10.6f} {avg_lp:>10.6f} "
                  f"{avg_lcf:>10.6f} {val_loss:>10.6f}  (cfw={cf_weight:.3f})")

    return model, history, mu_x, std_x


# ────────────────────────────────────────────────────────────────────────────
# 7.  EVALUATION HELPERS
# ────────────────────────────────────────────────────────────────────────────
def evaluate_synthetic(model, mu_x, std_x, n_test=1000):
    """Per-parameter R² and RMSE on a held-out synthetic test set."""
    Xv, Yv = generate_dataset(n_test, verbose=False)
    Xn     = (torch.tensor(Xv) - mu_x) / std_x
    model.eval()
    with torch.no_grad():
        Pv = model(Xn.to(DEVICE)).cpu().numpy()

    print("\n── Synthetic Test Set ─────────────────────────────────────")
    print(f"{'Param':>10} {'RMSE':>10} {'R²':>8}")
    for i, name in enumerate(PNAMES):
        rmse = float(np.sqrt(np.mean((Yv[:,i] - Pv[:,i])**2)))
        r2   = float(np.corrcoef(Yv[:,i], Pv[:,i])[0,1]**2)
        print(f"{name:>10} {rmse:>10.5f} {r2:>8.3f}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i, ax in enumerate(axes):
        ax.scatter(Yv[:,i], Pv[:,i], s=6, alpha=0.3)
        ax.plot([P_LO[i], P_HI[i]], [P_LO[i], P_HI[i]], "r--", lw=1)
        ax.set_xlabel("True"); ax.set_ylabel("Predicted")
        ax.set_title(PNAMES[i])
    plt.suptitle("ML Calibration — Synthetic Test")
    plt.tight_layout()
    plt.savefig("ml_synthetic.png", dpi=130)
    plt.show()


def calibrate_market(model, mu_x, std_x, market_ivs, S0_market):
    """
    Calibrate to a real market IV vector.

    market_ivs : (35,) float32 IVs on GRID_K × GRID_T
    S0_market  : current spot price

    1. Convert IVs to scaled call prices (price/S0)
    2. Normalize and run through network
    3. Optional L-BFGS-B polish from network warm start
    """
    # ── Convert IVs to normalized prices ────────────────────────────────────
    prices_over_s = np.empty(N_GRID, dtype=np.float32)
    idx = 0
    for T in GRID_T:
        F = S0_market * np.exp(r*T)
        for lm in GRID_K:
            K = F * np.exp(lm)
            iv = market_ivs[idx]
            prices_over_s[idx] = bs_call_np(S0_market, K, T, iv) / S0_market
            idx += 1

    # ── Network forward pass ────────────────────────────────────────────────
    x_norm = (torch.tensor(prices_over_s).unsqueeze(0) - mu_x) / std_x
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        theta_net = model(x_norm.to(DEVICE)).cpu().numpy()[0]
    t_net = time.time() - t0

    # ── Surface fit from network output ─────────────────────────────────────
    fitted_ivs = surface_ivs_from_params(theta_net, S0_market)
    rmse_net   = float(np.sqrt(np.mean((fitted_ivs - market_ivs)**2)))

    print("\n── Network Calibration ─────────────────────────────────────")
    print(f"  Time = {t_net*1000:.2f} ms")
    for n, v in zip(PNAMES, theta_net):
        print(f"    {n:>10} = {v:.5f}")
    print(f"  IV RMSE = {rmse_net:.5f}  ({rmse_net*100:.2f} vol points)")

    # ── L-BFGS-B polish from network warm start ─────────────────────────────
    def obj(p):
        p = np.clip(p, P_LO, P_HI)
        ivs = surface_ivs_from_params(p, S0_market)
        return float(np.mean((ivs - market_ivs)**2))

    t0  = time.time()
    res = minimize(obj, theta_net, method="L-BFGS-B",
                   bounds=list(zip(P_LO, P_HI)),
                   options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9})
    t_polish = time.time() - t0
    theta_polish = res.x
    rmse_polish  = float(np.sqrt(res.fun))

    print(f"\n── L-BFGS-B Polish (warm start = network) ────────────────")
    print(f"  Time = {t_polish*1000:.1f} ms   converged={res.success}   iters={res.nit}")
    for n, v in zip(PNAMES, theta_polish):
        print(f"    {n:>10} = {v:.5f}")
    print(f"  IV RMSE = {rmse_polish:.5f}  ({rmse_polish*100:.2f} vol points)")

    return dict(theta_net=theta_net, theta_polish=theta_polish,
                rmse_net=rmse_net, rmse_polish=rmse_polish,
                fitted_ivs_net=fitted_ivs,
                fitted_ivs_polish=surface_ivs_from_params(theta_polish, S0_market))


def surface_ivs_from_params(theta, S0):
    """Compute IV surface (35,) from model parameters via COS + BS inversion."""
    sigma, lam, mu_j, sigma_j = theta
    ivs = np.empty(N_GRID, dtype=np.float32)
    idx = 0
    for T in GRID_T:
        F = S0 * np.exp(r*T)
        for lm in GRID_K:
            K     = F * np.exp(lm)
            price = price_call_cos_np(K, T, sigma, lam, mu_j, sigma_j, S0)
            iv    = implied_vol_np(price, S0, K, T)
            ivs[idx] = iv if not np.isnan(iv) else 0.0
            idx += 1
    return ivs


def plot_smile_fit(market_ivs, theta_net, theta_polish, S0):
    ivs_net    = surface_ivs_from_params(theta_net,    S0).reshape(len(GRID_K), len(GRID_T))
    ivs_polish = surface_ivs_from_params(theta_polish, S0).reshape(len(GRID_K), len(GRID_T))
    market_2d  = market_ivs.reshape(len(GRID_K), len(GRID_T))

    fig, axes = plt.subplots(1, len(GRID_T), figsize=(16, 4), sharey=True)
    for j, (T, ax) in enumerate(zip(GRID_T, axes)):
        ax.plot(GRID_K, market_2d[:, j], "o-",  label="Market",   lw=2)
        ax.plot(GRID_K, ivs_net[:, j],   "s--", label="Network",  lw=1.5)
        ax.plot(GRID_K, ivs_polish[:,j], "^:",  label="+Polish",  lw=1.5)
        ax.set_title(f"T={T:.2f}"); ax.set_xlabel("Log-moneyness")
        if j == 0: ax.set_ylabel("IV")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.suptitle("ML Calibration vs Market — Smile Slices")
    plt.tight_layout()
    plt.savefig("ml_smile_fit.png", dpi=130)
    plt.show()


# ────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Step 1: train ───────────────────────────────────────────────────────
    model, history, mu_x, std_x = train(
        n_train   = 20_000,
        n_val     = 2_000,
        epochs    = 120,
        batch_size= 256,
        lam_cf    = 10.0,    # weight on physics (CF) loss, tuned for ~0.05 price scale
        cf_warmup = 20,      # ramp up CF loss over first 20 epochs
    )

    # Save weights and normalization stats
    torch.save({
        "state_dict": model.state_dict(),
        "mu_x":  mu_x,
        "std_x": std_x,
        "P_LO":  P_LO,
        "P_HI":  P_HI,
        "GRID_K": GRID_K,
        "GRID_T": GRID_T,
    }, "merton_inversion_net.pt")
    print("\nSaved merton_inversion_net.pt")

    # ── Step 2: synthetic eval ──────────────────────────────────────────────
    evaluate_synthetic(model, mu_x, std_x)

    # ── Step 3: calibrate to real SPY surface ──────────────────────────────
    # Paste your market_ivs and S0 from the pipeline run here:
    market_ivs = np.array([
        0.2629, 0.1799, 0.1330, 0.1733, 0.1634,   # k=-0.20
        0.1789, 0.1629, 0.1564, 0.1702, 0.1639,   # k=-0.12
        0.1431, 0.1475, 0.1538, 0.1610, 0.1564,   # k=-0.06
        0.1260, 0.1335, 0.1417, 0.1484, 0.1455,   # k= 0.00
        0.1232, 0.1240, 0.1273, 0.1350, 0.1339,   # k= 0.06
        0.1305, 0.1222, 0.1176, 0.1230, 0.1245,   # k= 0.12
        0.1485, 0.1372, 0.1242, 0.1136, 0.1205,   # k= 0.20
    ], dtype=np.float32)
    S0_market = 711.58

    result = calibrate_market(model, mu_x, std_x, market_ivs, S0_market)
    plot_smile_fit(market_ivs, result["theta_net"], result["theta_polish"], S0_market)
