# Created: March 27, 2026

"""
Heston ML Calibration via CF-Informed Neural Network
=====================================================

Heston SDEs (risk-neutral):
    dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW^S_t
    dv_t = kappa * (theta - v_t) * dt + sigma_v * sqrt(v_t) * dW^v_t
    d<W^S, W^v>_t = rho * dt

5 parameters: (v0, kappa, theta, sigma_v, rho)

Pipeline:
  1.  Generate (theta, surface) pairs using the validated COS pricer
  2.  Train MLP for direct inversion: surface -> theta
        - Supervised loss : range-normalized MSE on parameters
        - Physics loss    : torch_COS(predicted theta) vs input surface
        - CF warmup       : physics term ramps up over first 20 epochs
  3.  Evaluate on synthetic test set
  4.  Calibrate to real SPY market_ivs vector (network + L-BFGS-B polish)

Surface: call prices / S0 on a fixed log-moneyness x maturity grid (35-dim).
Scale-invariant — same network works for any spot price.

Run:  python heston_ml_calibration.py
Deps: torch numpy scipy matplotlib
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from scipy.stats    import norm
from scipy.optimize import brentq, minimize
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(0); np.random.seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S0_TRAIN    = 100.0
r           = 0.05
N_COS       = 256
N_TRAIN_COS = 128

GRID_K = np.array([-0.20, -0.12, -0.06, 0.00, 0.06, 0.12, 0.20])
GRID_T = np.array([1/12, 3/12, 6/12, 12/12, 18/12])
N_GRID = len(GRID_K) * len(GRID_T)

# ── Heston parameter bounds ─────────────────────────────────────────────────
# (v0, kappa, theta, sigma_v, rho)  — index options regime
#
# v0      : initial variance (typical SPY: 0.01 - 0.10  ⇒ sqrt → 10–32% vol)
# kappa   : mean reversion speed (1 - 5 typical)
# theta   : long-run variance (similar range to v0)
# sigma_v : vol-of-vol (0.1 - 1.0 typical)
# rho     : leverage correlation (very negative for index — typical -0.9 to -0.3)
P_LO   = np.array([0.01, 0.50, 0.01, 0.10, -0.95], dtype=np.float32)
P_HI   = np.array([0.10, 5.00, 0.10, 1.00, -0.10], dtype=np.float32)
PNAMES = ["v0", "kappa", "theta", "sigma_v", "rho"]
N_PARAM = len(PNAMES)

# Feller condition: 2*kappa*theta > sigma_v^2 ensures v stays positive.
# Not enforced during sampling (calibration to SPY routinely violates
# the condition) but flagged during evaluation.


# ─────────────────────────────────────────────────────────────────────────────
# 1.  NUMPY HESTON COS PRICER  (data generation)
# ─────────────────────────────────────────────────────────────────────────────
def heston_cf_np(u, T, v0, kappa, theta, sigma_v, rho):
    """Heston CF, "little trap" form (numerically stable)."""
    iu = 1j * u
    xi = kappa - iu * sigma_v * rho
    d  = np.sqrt(xi**2 + sigma_v**2 * (iu + u**2))
    g2 = (xi - d) / (xi + d)
    edt = np.exp(-d*T)
    D = (xi - d) / (sigma_v**2) * (1.0 - edt) / (1.0 - g2*edt)
    C = (kappa*theta/sigma_v**2) * (
        (xi - d)*T - 2.0*np.log((1.0 - g2*edt) / (1.0 - g2))
    )
    return np.exp(iu*r*T + C + D*v0)


def cos_truncation_np(T, v0, kappa, theta, sigma_v, rho, L=10):
    """Conservative truncation using closed-form cumulants of log(S_T/S_0)."""
    if kappa*T < 1e-6:
        v_avg = v0
    else:
        v_avg = theta + (v0 - theta)*(1.0 - np.exp(-kappa*T))/(kappa*T)
    c1 = (r - 0.5*v_avg)*T
    c2 = v_avg*T + 0.5*sigma_v**2 * T**2 * v_avg / kappa
    H  = L*np.sqrt(abs(c2))
    return c1 - H, c1 + H


def call_cos_coefficients_np(a, b, k):
    """V_k for call payoff (model-independent)."""
    bw = b - a
    kp = k * np.pi / bw
    upper = np.exp(b) * np.cos(k*np.pi)
    lower = np.cos(-kp*a) + kp*np.sin(-kp*a)
    chi = np.where(k==0, np.exp(b)-1.0,
                   (1/(1+kp**2))*(upper - lower))
    with np.errstate(invalid="ignore", divide="ignore"):
        psi = np.where(k==0, b, (np.sin(kp*bw) - np.sin(-kp*a))/kp)
    return (2/bw)*(chi - psi)


def price_call_cos_np(K, T, v0, kappa, theta, sigma_v, rho, S0=S0_TRAIN, n=N_COS):
    a, b = cos_truncation_np(T, v0, kappa, theta, sigma_v, rho)
    k    = np.arange(n, dtype=float)
    u    = k * np.pi / (b - a)
    x    = np.log(S0 / K)
    phi  = heston_cf_np(u, T, v0, kappa, theta, sigma_v, rho)
    V    = call_cos_coefficients_np(a, b, k)
    series    = np.real(phi * np.exp(1j*k*np.pi*(x-a)/(b-a)))
    series[0] *= 0.5
    return max(K*np.exp(-r*T)*np.dot(series, V), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TORCH HESTON COS PRICER  (differentiable, used in CF physics loss)
# ─────────────────────────────────────────────────────────────────────────────
PI = np.pi


def heston_cf_torch(u, T, v0, kappa, theta, sigma_v, rho):
    """
    Batched, differentiable Heston CF in 'little trap' form.

    Args
    ----
    u       : (F,)   real frequencies         [torch float64]
    T       : scalar maturity                  [torch float64]
    v0..rho : (B, 1) batched, broadcast over u

    Returns
    -------
    (B, F) complex128
    """
    iu  = 1j * u                                              # (F,) complex
    xi  = kappa - iu * sigma_v * rho                          # (B,F)
    d   = torch.sqrt(xi**2 + sigma_v**2 * (iu + u**2))        # (B,F)
    g2  = (xi - d) / (xi + d)                                 # (B,F)
    edt = torch.exp(-d * T)                                   # (B,F)

    D = (xi - d) / (sigma_v**2) * (1.0 - edt) / (1.0 - g2*edt)
    C = (kappa*theta/sigma_v**2) * (
            (xi - d)*T - 2.0*torch.log((1.0 - g2*edt) / (1.0 - g2))
        )
    return torch.exp(iu*r*T + C + D*v0)


def cos_truncation_torch(T, v0, kappa, theta, sigma_v, rho, L=10):
    # if kappa*T very small, fall back to v0 — but kappa is never sampled near 0
    v_avg = theta + (v0 - theta) * (1.0 - torch.exp(-kappa*T)) / (kappa*T)
    c1 = (r - 0.5*v_avg) * T
    c2 = v_avg*T + 0.5*sigma_v**2 * T**2 * v_avg / kappa
    H  = L * torch.sqrt(torch.abs(c2))
    return c1 - H, c1 + H


def price_surface_torch(theta_vec, S0=S0_TRAIN, n=N_TRAIN_COS):
    """
    Differentiable Heston COS surface pricer.

    theta_vec : (B, 5) float32  [v0, kappa, theta, sigma_v, rho]
    return    : (B, N_GRID=35) float32  call prices / S0 on GRID_K x GRID_T

    The k=0 term is computed separately to avoid the autograd 'where' trap.
    """
    th = theta_vec.double()
    v0      = th[:, 0:1].clamp(min=1e-4)
    kappa   = th[:, 1:2].clamp(min=1e-3)
    theta_p = th[:, 2:3].clamp(min=1e-4)
    sigma_v = th[:, 3:4].clamp(min=1e-3)
    rho     = th[:, 4:5].clamp(min=-0.999, max=0.999)

    k_pos = torch.arange(1, n, dtype=torch.float64, device=theta_vec.device)

    all_prices = []
    for T_val in GRID_T:
        T  = torch.tensor(T_val,            dtype=torch.float64, device=theta_vec.device)
        eT = torch.tensor(np.exp(-r*T_val), dtype=torch.float64, device=theta_vec.device)

        a, b = cos_truncation_torch(T, v0, kappa, theta_p, sigma_v, rho)   # (B,1)
        bw   = (b - a).clamp(min=1e-3)

        # ── k=0 term (handled separately) ───────────────────────────────
        # phi(0) for Heston includes the i*u*r*T factor: exp(0) = 1
        # V_0 has the closed form: (2/bw) * ((e^b - 1) - b)
        chi_0 = torch.exp(b) - 1.0
        psi_0 = b
        V_0   = (2.0 / bw) * (chi_0 - psi_0)             # (B,1)

        # ── k>=1 terms ──────────────────────────────────────────────────
        u_pos = k_pos * PI / bw                                              # (B,F-1)
        phi_p = heston_cf_torch(u_pos, T, v0, kappa, theta_p, sigma_v, rho)  # (B,F-1)

        kp    = k_pos * PI / bw
        upper = torch.exp(b) * torch.cos(k_pos * PI)
        lower = torch.cos(-kp*a) + kp*torch.sin(-kp*a)
        chi_p = (1.0/(1.0 + kp**2)) * (upper - lower)
        psi_p = (torch.sin(kp*bw) - torch.sin(-kp*a)) / kp                   # safe
        V_p   = (2.0/bw) * (chi_p - psi_p)                                    # (B,F-1)

        F_T = S0 * np.exp(r * T_val)
        for lm_val in GRID_K:
            K = F_T * np.exp(lm_val)
            x = np.log(S0 / K)

            term_0 = 0.5 * 1.0 * V_0.squeeze(-1)                              # (B,)
            series_p = torch.real(phi_p * torch.exp(1j*k_pos*PI*(x-a)/bw))   # (B,F-1)
            term_p   = (series_p * V_p).sum(dim=-1)                           # (B,)

            price = K * eT * (term_0 + term_p)
            price = torch.clamp(price, min=0.0).float() / S0
            all_prices.append(price)

    return torch.stack(all_prices, dim=1)   # (B, 35)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  IV ↔ price helpers  (Black-Scholes inversion)
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_surface_np(theta_vec, S0=S0_TRAIN):
    """Surface = call prices / S0 on GRID_K × GRID_T   →   (35,) float32."""
    v0, kappa, theta_p, sigma_v, rho = theta_vec
    out = np.empty(N_GRID, dtype=np.float32)
    idx = 0
    for T in GRID_T:
        F = S0 * np.exp(r*T)
        for lm in GRID_K:
            K = F * np.exp(lm)
            out[idx] = price_call_cos_np(K, T, v0, kappa, theta_p, sigma_v, rho, S0)/S0
            idx += 1
    return out


def generate_dataset(n, verbose=True, max_price_over_s=1.0):
    """
    Sample params from prior, price surfaces, return (X, Y).
    Resamples any draw that produces a numerically corrupt surface.
    """
    if verbose: print(f"Generating {n} Heston training samples...")
    Y = np.empty((n, N_PARAM), dtype=np.float32)
    X = np.empty((n, N_GRID),   dtype=np.float32)
    t0 = time.time()
    n_resampled = 0
    for i in range(n):
        for attempts in range(50):
            theta = np.random.uniform(P_LO, P_HI, N_PARAM).astype(np.float32)
            surf  = generate_surface_np(theta)
            if surf.max() < max_price_over_s and np.all(np.isfinite(surf)):
                break
            n_resampled += 1
        Y[i] = theta
        X[i] = surf
        if verbose and (i+1) % max(1, n//10) == 0:
            print(f"  {i+1}/{n}  ({time.time()-t0:.1f}s)  resampled={n_resampled}")
    if verbose:
        print(f"  Total resamples: {n_resampled}")
    return X, Y


# ─────────────────────────────────────────────────────────────────────────────
# 5.  NETWORK
# ─────────────────────────────────────────────────────────────────────────────
class HestonInversionNet(nn.Module):
    """
    Direct-inversion MLP: surface (35) → Heston parameters (5).
    Sigmoid output scaled to [P_LO, P_HI] so predictions stay in physical range.
    """
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_GRID, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden//2), nn.SiLU(),
            nn.Linear(hidden//2, N_PARAM), nn.Sigmoid(),
        )
        self.register_buffer("lo", torch.tensor(P_LO))
        self.register_buffer("hi", torch.tensor(P_HI))

    def forward(self, x):
        u = self.net(x)
        return self.lo + u * (self.hi - self.lo)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train(n_train=20_000, n_val=2_000, epochs=120, batch_size=256,
          lam_cf=10.0, cf_warmup=20, lr=1e-3):
    """
    Loss = L_param  +  cf_weight(epoch) * L_cf

      L_param   : range-normalized MSE on parameters
      L_cf      : MSE between torch_COS(predicted theta) and input surface
      cf_weight : ramps linearly from 0 to lam_cf over first cf_warmup epochs
    """
    X_np, Y_np = generate_dataset(n_train + n_val)
    X_all = torch.tensor(X_np)
    Y_all = torch.tensor(Y_np)

    mu_x  = X_all[:n_train].mean(0)
    std_x = X_all[:n_train].std(0).clamp(min=1e-8)
    X_norm = (X_all - mu_x) / std_x

    X_tr, Y_tr = X_norm[:n_train].to(DEVICE), Y_all[:n_train].to(DEVICE)
    X_va, Y_va = X_norm[n_train:].to(DEVICE), Y_all[n_train:].to(DEVICE)
    X_raw_tr   = X_all[:n_train].to(DEVICE)

    model = HestonInversionNet().to(DEVICE)
    opt   = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    print(f"\nTraining on {DEVICE}  |  n_train={n_train}  |  epochs={epochs}  |  lam_cf={lam_cf}")
    print(f"{'Ep':>4} {'Tot':>10} {'Lparam':>10} {'Lcf':>10} {'Val':>10}")
    print("-" * 56)
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

            theta_hat = model(xb)
            scale     = (model.hi - model.lo).to(theta_hat.device)
            L_param   = nn.functional.mse_loss(theta_hat/scale, yb/scale)

            prices_hat = price_surface_torch(theta_hat)
            L_cf       = nn.functional.mse_loss(prices_hat, xb_raw)

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
            val_loss = nn.functional.mse_loss(model(X_va)/scale, Y_va/scale).item()
        cf_weight = lam_cf * min(1.0, ep / max(1, cf_warmup))
        history.append((avg_lp + cf_weight*avg_lcf, val_loss, avg_lp, avg_lcf))

        if (ep+1) % 10 == 0 or ep == 0:
            print(f"{ep+1:>4} {history[-1][0]:>10.6f} {avg_lp:>10.6f} "
                  f"{avg_lcf:>10.6f} {val_loss:>10.6f}  (cfw={cf_weight:.2f})")

    return model, history, mu_x, std_x


# ─────────────────────────────────────────────────────────────────────────────
# 7.  EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_synthetic(model, mu_x, std_x, n_test=1000):
    """Per-parameter R² and RMSE on a held-out synthetic test set."""
    Xv, Yv = generate_dataset(n_test, verbose=False)
    Xn = (torch.tensor(Xv) - mu_x) / std_x
    model.eval()
    with torch.no_grad():
        Pv = model(Xn.to(DEVICE)).cpu().numpy()

    print("\n── Synthetic Test Set ─────────────────────────────────────")
    print(f"{'Param':>10} {'RMSE':>10} {'R²':>8}")
    for i, name in enumerate(PNAMES):
        rmse = float(np.sqrt(np.mean((Yv[:,i] - Pv[:,i])**2)))
        r2   = float(np.corrcoef(Yv[:,i], Pv[:,i])[0,1]**2)
        print(f"{name:>10} {rmse:>10.5f} {r2:>8.3f}")

    fig, axes = plt.subplots(1, N_PARAM, figsize=(4*N_PARAM, 4))
    for i, ax in enumerate(axes):
        ax.scatter(Yv[:,i], Pv[:,i], s=6, alpha=0.3)
        ax.plot([P_LO[i], P_HI[i]], [P_LO[i], P_HI[i]], "r--", lw=1)
        ax.set_xlabel("True"); ax.set_ylabel("Predicted")
        ax.set_title(PNAMES[i])
    plt.suptitle("Heston ML Calibration — Synthetic Test")
    plt.tight_layout()
    plt.savefig("heston_ml_synthetic.png", dpi=130)
    plt.show()


def surface_ivs_from_params(theta_vec, S0):
    """IV surface (35,) from Heston params via COS + BS inversion."""
    v0, kappa, theta_p, sigma_v, rho = theta_vec
    ivs = np.empty(N_GRID, dtype=np.float32)
    idx = 0
    for T in GRID_T:
        F = S0 * np.exp(r*T)
        for lm in GRID_K:
            K     = F * np.exp(lm)
            price = price_call_cos_np(K, T, v0, kappa, theta_p, sigma_v, rho, S0)
            iv    = implied_vol_np(price, S0, K, T)
            ivs[idx] = iv if not np.isnan(iv) else 0.0
            idx += 1
    return ivs


def calibrate_market(model, mu_x, std_x, market_ivs, S0_market):
    """Network calibration + L-BFGS-B polish."""
    # IVs → normalized prices
    prices_over_s = np.empty(N_GRID, dtype=np.float32)
    idx = 0
    for T in GRID_T:
        F = S0_market * np.exp(r*T)
        for lm in GRID_K:
            K = F * np.exp(lm)
            prices_over_s[idx] = bs_call_np(S0_market, K, T, market_ivs[idx]) / S0_market
            idx += 1

    x_norm = (torch.tensor(prices_over_s).unsqueeze(0) - mu_x) / std_x
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        theta_net = model(x_norm.to(DEVICE)).cpu().numpy()[0]
    t_net = time.time() - t0

    fitted_ivs = surface_ivs_from_params(theta_net, S0_market)
    rmse_net   = float(np.sqrt(np.mean((fitted_ivs - market_ivs)**2)))

    print("\n── Network Calibration ─────────────────────────────────────")
    print(f"  Time = {t_net*1000:.2f} ms")
    for n, v in zip(PNAMES, theta_net):
        print(f"    {n:>10} = {v:.5f}")
    feller = 2*theta_net[1]*theta_net[2] - theta_net[3]**2
    print(f"  Feller condition  2*kappa*theta - sigma_v^2 = {feller:+.4f}  "
          f"({'satisfied' if feller>0 else 'violated — common for SPY'})")
    print(f"  IV RMSE = {rmse_net:.5f}  ({rmse_net*100:.2f} vol points)")

    # L-BFGS-B polish from network warm start
    def obj(p):
        p = np.clip(p, P_LO, P_HI)
        ivs = surface_ivs_from_params(p, S0_market)
        return float(np.mean((ivs - market_ivs)**2))

    t0 = time.time()
    res = minimize(obj, theta_net, method="L-BFGS-B",
                   bounds=list(zip(P_LO, P_HI)),
                   options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9})
    t_polish = time.time() - t0
    theta_polish = res.x
    rmse_polish  = float(np.sqrt(res.fun))

    print(f"\n── L-BFGS-B Polish (warm start = network) ────────────────")
    print(f"  Time = {t_polish*1000:.0f} ms   converged={res.success}   iters={res.nit}")
    for n, v in zip(PNAMES, theta_polish):
        print(f"    {n:>10} = {v:.5f}")
    feller_p = 2*theta_polish[1]*theta_polish[2] - theta_polish[3]**2
    print(f"  Feller condition  2*kappa*theta - sigma_v^2 = {feller_p:+.4f}  "
          f"({'satisfied' if feller_p>0 else 'violated'})")
    print(f"  IV RMSE = {rmse_polish:.5f}  ({rmse_polish*100:.2f} vol points)")

    return dict(theta_net=theta_net, theta_polish=theta_polish,
                rmse_net=rmse_net, rmse_polish=rmse_polish)


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
    plt.suptitle("Heston ML Calibration vs Market — Smile Slices")
    plt.tight_layout()
    plt.savefig("heston_ml_smile_fit.png", dpi=130)
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Train
    model, history, mu_x, std_x = train(
        n_train   = 20_000,
        n_val     = 2_000,
        epochs    = 120,
        batch_size= 256,
        lam_cf    = 10.0,
        cf_warmup = 20,
    )

    torch.save({
        "state_dict": model.state_dict(),
        "mu_x":  mu_x,
        "std_x": std_x,
        "P_LO":  P_LO,
        "P_HI":  P_HI,
        "GRID_K": GRID_K,
        "GRID_T": GRID_T,
        "PNAMES": PNAMES,
    }, "heston_inversion_net.pt")
    print("\nSaved heston_inversion_net.pt")

    # Synthetic eval
    evaluate_synthetic(model, mu_x, std_x)

    # Calibrate to real SPY surface (paste your latest from pipeline.py)
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
