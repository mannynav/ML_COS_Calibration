# Created: March 3, 2026

"""
CF-Informed Neural Network Calibration — Merton Jump Diffusion
==============================================================

Architecture:
  - Forward model : Merton CF + COS method (numpy, fast data generation)
  - Network       : MLP direct inversion  (surface → parameters)
  - Physics loss  : differentiable torch COS pricer enforces
                    COS(theta_predicted) ≈ market prices
  - Total loss    : L_param (supervised) + lambda * L_cf (physics)

Parameters: sigma, lambda, mu_J, sigma_J  (4-dim output)
Surface   : 5 strikes × 4 maturities = 20-dim input

Run:  python merton_cf_calibration.py
Deps: numpy torch matplotlib  (pip install torch matplotlib numpy)
"""

"""
Note: This is the prototype of the Merton COS Pricer - ml_calibation.py

"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 0.  GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

r   = 0.05       # risk-free rate
S0  = 100.0      # spot price
N   = 64         # COS terms for data generation
N_T = 32         # COS terms inside the torch pricer (fewer = faster grad)

STRIKES    = np.array([80., 90., 100., 110., 120.])
MATURITIES = np.array([0.25, 0.5, 1.0, 2.0])
N_OPT      = len(STRIKES) * len(MATURITIES)   # 20

# Merton parameter bounds: [sigma, lambda, mu_J, sigma_J]
P_LO = np.array([0.05,  0.00, -0.20, 0.05], dtype=np.float32)
P_HI = np.array([0.40,  1.00,  0.05, 0.30], dtype=np.float32)
NAMES = ["sigma", "lambda", "mu_J", "sigma_J"]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MERTON CHARACTERISTIC FUNCTION  (numpy — used for fast data generation)
# ─────────────────────────────────────────────────────────────────────────────
def merton_cf_np(u, T, sigma, lam, mu_j, sigma_j):
    """
    CF of log(S_T / S_0) under risk-neutral Merton Jump Diffusion.
    Lévy-Khintchine form:
        phi(u, T) = exp( T * Psi(u) )
    where the Lévy exponent Psi decomposes as:
        Psi(u) = diffusion_term + jump_term
    with martingale drift correction built in.
    """
    drift_corr = lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1.0)   # keeps E[S_T]=S0*e^{rT}
    phi_jump   = np.exp(1j * u * mu_j - 0.5 * u**2 * sigma_j**2)  # CF of a single jump
    Psi = (
        1j * u * (r - 0.5 * sigma**2 - drift_corr)   # drift
        - 0.5 * u**2 * sigma**2                        # diffusion
        + lam * (phi_jump - 1.0)                       # Poisson-weighted jump contribution
    )
    return np.exp(T * Psi)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  COS METHOD  (numpy)
# ─────────────────────────────────────────────────────────────────────────────
def cos_truncation_np(T, sigma, lam, mu_j, sigma_j, L=12):
    """
    Cumulant-based integration interval [a, b].
    Uses c1 (mean), c2 (variance), c4 (kurtosis contribution) of log-return.
    Ref: Fang & Oosterlee (2008), equation (52).
    """
    drift_corr = lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1.0)
    c1 = (r - 0.5 * sigma**2 - drift_corr) * T
    c2 = (sigma**2 + lam * (mu_j**2 + sigma_j**2)) * T
    c4 = lam * (mu_j**4 + 6 * mu_j**2 * sigma_j**2 + 3 * sigma_j**4) * T
    H  = L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    return c1 - H, c1 + H


def cos_payoff_coefficients_np(a, b, k):
    """
    Cosine coefficients V_k of the European call payoff max(e^x - 1, 0)
    on [a, b], integrated over [0, b] (the in-the-money region).
    Analytic: V_k = (2 / (b-a)) * (chi_k - psi_k)
    """
    kp  = k * np.pi / (b - a)
    chi = np.where(
        k == 0,
        np.exp(b) - 1.0,
        (1.0 / (1.0 + kp**2)) * (
            np.exp(b) * (np.cos(kp * (b - a)) + kp * np.sin(kp * (b - a))) - 1.0
        ),
    )
    psi = np.where(k == 0, b, np.sin(kp * (b - a)) / kp)
    return (2.0 / (b - a)) * (chi - psi)


def price_call_cos_np(K, T, sigma, lam, mu_j, sigma_j):
    """Price a single European call via COS."""
    a, b = cos_truncation_np(T, sigma, lam, mu_j, sigma_j)
    k    = np.arange(N, dtype=float)
    u    = k * np.pi / (b - a)
    x    = np.log(S0 / K)

    phi    = merton_cf_np(u, T, sigma, lam, mu_j, sigma_j)
    V      = cos_payoff_coefficients_np(a, b, k)
    series = np.real(phi * np.exp(1j * k * np.pi * (x - a) / (b - a)))
    series[0] *= 0.5

    return max(K * np.exp(-r * T) * np.dot(series, V), 0.0)


def price_surface_np(sigma, lam, mu_j, sigma_j):
    """Price full 20-instrument surface. Returns float32 array (N_OPT,)."""
    prices = []
    for T in MATURITIES:
        for K in STRIKES:
            prices.append(price_call_cos_np(K, T, sigma, lam, mu_j, sigma_j))
    return np.array(prices, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  DIFFERENTIABLE COS PRICER  (torch — used in physics loss)
# ─────────────────────────────────────────────────────────────────────────────
#
# This is the physics-informed piece:  given predicted parameters theta_hat,
# COS prices are computed inside the computational graph so gradients flow
# back through the CF formula into the network weights.
#
# Key: the CF is the "physics" — the network must predict parameters
# whose CF, when inverted via COS, reproduces the observed market prices.
# ─────────────────────────────────────────────────────────────────────────────

def merton_cf_torch(u, T, sigma, lam, mu_j, sigma_j):
    """
    Batched, differentiable Merton CF.
    u      : (F,)    real frequencies  [torch float64]
    T      : scalar  maturity          [torch float64]
    params : (B, 1)  batched, cast to complex
    Returns: (B, F)  complex
    """
    drift_corr = lam * (torch.exp(mu_j + 0.5 * sigma_j**2) - 1.0)
    phi_jump   = torch.exp(1j * u * mu_j - 0.5 * u**2 * sigma_j**2)
    Psi = (
        1j * u * (r - 0.5 * sigma**2 - drift_corr)
        - 0.5 * u**2 * sigma**2
        + lam * (phi_jump - 1.0)
    )
    return torch.exp(T * Psi)   # (B, F)


def cos_truncation_torch(T, sigma, lam, mu_j, sigma_j, L=12):
    """Differentiable truncation — returns (B, 1) tensors a, b."""
    drift_corr = lam * (torch.exp(mu_j + 0.5 * sigma_j**2) - 1.0)
    c1 = (r - 0.5 * sigma**2 - drift_corr) * T
    c2 = (sigma**2 + lam * (mu_j**2 + sigma_j**2)) * T
    c4 = lam * (mu_j**4 + 6 * mu_j**2 * sigma_j**2 + 3 * sigma_j**4) * T
    H  = L * torch.sqrt(torch.abs(c2) + torch.sqrt(torch.abs(c4)))
    return c1 - H, c1 + H


PI = torch.tensor(np.pi, dtype=torch.float64)


def price_surface_torch(theta, n_cos=N_T):
    """
    Differentiable COS surface pricer.
    theta  : (B, 4)  float32  [sigma, lam, mu_j, sigma_j]
    returns: (B, N_OPT) float32
    """
    B = theta.shape[0]
    # promote to float64 for numerical accuracy in complex ops
    th = theta.double()
    sigma   = th[:, 0:1]   # (B, 1)
    lam     = th[:, 1:2]
    mu_j    = th[:, 2:3]
    sigma_j = th[:, 3:4]

    all_prices = []

    for T_val in MATURITIES:
        T  = torch.tensor(T_val, dtype=torch.float64, device=theta.device)
        eT = torch.tensor(np.exp(-r * T_val), dtype=torch.float64, device=theta.device)

        a, b = cos_truncation_torch(T, sigma, lam, mu_j, sigma_j)   # (B, 1)
        bw   = b - a                                                   # (B, 1)

        k    = torch.arange(n_cos, dtype=torch.float64, device=theta.device)   # (F,)
        u    = k * PI / bw          # (B, F)  frequencies

        phi  = merton_cf_torch(u, T, sigma, lam, mu_j, sigma_j)       # (B, F) complex

        # Payoff cosine coefficients V_k  (call on [0, b])
        kp   = k * PI / bw          # (B, F)
        chi  = torch.where(
            k == 0,
            torch.exp(b) - 1.0,
            (1.0 / (1.0 + kp**2)) * (
                torch.exp(b) * (torch.cos(kp * bw) + kp * torch.sin(kp * bw)) - 1.0
            ),
        )
        psi  = torch.where(k == 0, b, torch.sin(kp * bw) / kp)
        V    = (2.0 / bw) * (chi - psi)                               # (B, F) real

        for K_val in STRIKES:
            x      = torch.tensor(np.log(S0 / K_val), dtype=torch.float64, device=theta.device)
            series = torch.real(phi * torch.exp(1j * k * PI * (x - a) / bw))  # (B, F)
            series = series.clone()
            series[:, 0] = series[:, 0] * 0.5

            price  = K_val * eT * (series * V).sum(dim=-1)            # (B,)
            price  = torch.clamp(price, min=0.0).float()
            all_prices.append(price)

    return torch.stack(all_prices, dim=1)   # (B, N_OPT)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  DATASET GENERATION
# ─────────────────────────────────────────────────────────────────────────────
def generate_dataset(n, verbose=True):
    """
    Sample random parameter vectors, price full surfaces via COS.
    Returns surfaces (n, N_OPT) and params (n, 4) as float32 arrays.
    """
    if verbose:
        print(f"Generating {n} training samples via COS pricer...")
    params   = np.random.uniform(P_LO, P_HI, (n, 4)).astype(np.float32)
    surfaces = np.array([price_surface_np(*p) for p in params], dtype=np.float32)
    return surfaces, params


# ─────────────────────────────────────────────────────────────────────────────
# 5.  NETWORK — MLP DIRECT INVERSION
# ─────────────────────────────────────────────────────────────────────────────
class MertonInversionNet(nn.Module):
    """
    Input  : normalized option price surface  (B, N_OPT=20)
    Output : Merton parameters                (B, 4)  in physical units
    """
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_OPT, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 4), nn.Sigmoid(),   # output in [0, 1]
        )
        lo = torch.tensor(P_LO, dtype=torch.float32)
        hi = torch.tensor(P_HI, dtype=torch.float32)
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)

    def forward(self, x):
        raw = self.net(x)                          # (B, 4) in [0, 1]
        return raw * (self.hi - self.lo) + self.lo  # (B, 4) in physical units


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train(
    n_train   = 8_000,
    n_val     = 1_000,
    epochs    = 80,
    batch_size= 256,
    lam_cf    = 0.5,     # weight on CF consistency loss
    lr        = 1e-3,
):
    """
    Two-term loss:
      L = L_param  +  lam_cf * L_cf
      L_param : MSE between predicted and true parameters (supervised)
      L_cf    : MSE between COS(predicted params) and market prices
                — this is the characteristic function physics constraint
    """
    X_np, Y_np = generate_dataset(n_train + n_val)

    X     = torch.tensor(X_np)
    Y     = torch.tensor(Y_np)

    # Standardize inputs using training statistics
    mu_x  = X[:n_train].mean(0)
    std_x = X[:n_train].std(0).clamp(min=1e-8)
    X_norm = (X - mu_x) / std_x

    X_tr, Y_tr   = X_norm[:n_train].to(DEVICE), Y[:n_train].to(DEVICE)
    X_va, Y_va   = X_norm[n_train:].to(DEVICE), Y[n_train:].to(DEVICE)
    X_raw_tr     = X[:n_train].to(DEVICE)         # raw prices for CF loss

    model = MertonInversionNet().to(DEVICE)
    opt   = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_losses, val_losses = [], []
    n_batches = n_train // batch_size

    print(f"\nTraining on {DEVICE}  |  epochs={epochs}  |  lam_cf={lam_cf}")
    print(f"{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'L_param':>10}  {'L_cf':>10}")
    print("-" * 55)

    for ep in range(epochs):
        model.train()
        idx    = torch.randperm(n_train, device=DEVICE)
        ep_lp, ep_lcf = 0.0, 0.0

        for b in range(n_batches):
            bi      = idx[b * batch_size : (b + 1) * batch_size]
            xb      = X_tr[bi]
            yb      = Y_tr[bi]
            xb_raw  = X_raw_tr[bi]

            theta_hat = model(xb)                        # (B, 4) predicted params

            # ── Loss 1: Supervised parameter MSE ─────────────────────────
            L_param = nn.functional.mse_loss(theta_hat, yb)

            # ── Loss 2: CF Consistency ────────────────────────────────────
            # Compute COS prices from predicted params (differentiable)
            # and compare to the market prices that were fed in.
            # This enforces: network output must be consistent with the CF.
            prices_hat = price_surface_torch(theta_hat)  # (B, N_OPT)
            L_cf       = nn.functional.mse_loss(prices_hat, xb_raw)

            loss = L_param + lam_cf * L_cf

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_lp  += L_param.item()
            ep_lcf += L_cf.item()

        sched.step()

        avg_lp  = ep_lp  / n_batches
        avg_lcf = ep_lcf / n_batches

        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(model(X_va), Y_va).item()

        train_losses.append(avg_lp + lam_cf * avg_lcf)
        val_losses.append(val_loss)

        if (ep + 1) % 10 == 0:
            print(f"{ep+1:>6}  {train_losses[-1]:>10.5f}  {val_loss:>10.6f}"
                  f"  {avg_lp:>10.5f}  {avg_lcf:>10.5f}")

    return model, train_losses, val_losses, mu_x, std_x


# ─────────────────────────────────────────────────────────────────────────────
# 7.  EVALUATION & PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, mu_x, std_x, n_test=500):
    model.eval()
    X_np, Y_np = generate_dataset(n_test, verbose=False)
    X_norm     = (torch.tensor(X_np) - mu_x) / std_x

    with torch.no_grad():
        pred = model(X_norm.to(DEVICE)).cpu().numpy()

    # ── Scatter plots: true vs predicted ─────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.scatter(Y_np[:, i], pred[:, i], alpha=0.25, s=8, color="steelblue")
        lo, hi = P_LO[i], P_HI[i]
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect")
        r2 = float(np.corrcoef(Y_np[:, i], pred[:, i])[0, 1] ** 2)
        rmse = float(np.sqrt(np.mean((Y_np[:, i] - pred[:, i]) ** 2)))
        ax.set_title(f"{NAMES[i]}   R²={r2:.3f}  RMSE={rmse:.4f}", fontsize=11)
        ax.set_xlabel("True"); ax.set_ylabel("Predicted")
    plt.suptitle("Merton JDM — CF-Informed Direct Inversion", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("calibration_scatter.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved calibration_scatter.png")
    return pred, Y_np


def plot_training_curve(train_losses, val_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Validation")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training Curve — CF-Informed Merton Calibration")
    plt.legend(); plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    plt.show()
    print("Saved training_curve.png")


def demo_single_calibration(model, mu_x, std_x):
    """Calibrate one test surface and print parameter comparison."""
    model.eval()
    # Ground truth
    true_params = np.array([0.20, 0.35, -0.08, 0.12], dtype=np.float32)
    surface     = price_surface_np(*true_params)

    # Add small noise (simulate bid-ask spread)
    noisy_surface = surface * (1.0 + np.random.normal(0, 0.005, surface.shape))

    x_in = (torch.tensor(noisy_surface).unsqueeze(0) - mu_x) / std_x
    with torch.no_grad():
        cal = model(x_in.to(DEVICE)).cpu().numpy()[0]

    print("\n── Single Surface Calibration Demo ──────────────────────────────")
    print(f"{'Param':>10}  {'True':>8}  {'Predicted':>10}  {'|Error|%':>9}")
    print("-" * 45)
    for name, t, p in zip(NAMES, true_params, cal):
        err_pct = abs(p - t) / abs(t) * 100
        print(f"{name:>10}  {t:8.4f}  {p:10.4f}  {err_pct:8.1f}%")

    # Plot surface fit
    surface_pred = price_surface_np(*cal)
    x_axis       = np.arange(N_OPT)
    plt.figure(figsize=(10, 4))
    plt.plot(x_axis, surface,      "o-",  label="True surface",      lw=2)
    plt.plot(x_axis, noisy_surface,"s--", label="Noisy market",      lw=1, alpha=0.6)
    plt.plot(x_axis, surface_pred, "^-",  label="Predicted surface", lw=2)
    plt.xlabel("Option index  (maturity × strike)")
    plt.ylabel("Call price")
    plt.title("Surface Fit: True vs Network-Calibrated Parameters")
    plt.legend(); plt.tight_layout()
    plt.savefig("surface_fit.png", dpi=150)
    plt.show()
    print("Saved surface_fit.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8.  ABLATION — compare with vs without CF loss
# ─────────────────────────────────────────────────────────────────────────────
def ablation(n_train=4000, epochs=40):
    """
    Train two models: one with CF constraint, one without.
    Show that the CF loss improves calibration accuracy.
    """
    print("\n── Ablation: with vs without CF loss ────────────────────────────")
    X_np, Y_np = generate_dataset(n_train + 500)

    results = {}
    for lam in [0.0, 0.5]:
        X     = torch.tensor(X_np)
        Y     = torch.tensor(Y_np)
        mu_x  = X[:n_train].mean(0)
        std_x = X[:n_train].std(0).clamp(min=1e-8)
        X_norm = (X - mu_x) / std_x

        X_tr, Y_tr   = X_norm[:n_train].to(DEVICE), Y[:n_train].to(DEVICE)
        X_va, Y_va   = X_norm[n_train:].to(DEVICE), Y[n_train:].to(DEVICE)
        X_raw_tr     = X[:n_train].to(DEVICE)

        model = MertonInversionNet().to(DEVICE)
        opt   = Adam(model.parameters(), lr=1e-3)

        for ep in range(epochs):
            idx = torch.randperm(n_train, device=DEVICE)
            for b in range(n_train // 256):
                bi     = idx[b*256:(b+1)*256]
                th_hat = model(X_tr[bi])
                Lp     = nn.functional.mse_loss(th_hat, Y_tr[bi])
                Lc     = nn.functional.mse_loss(price_surface_torch(th_hat), X_raw_tr[bi])
                loss   = Lp + lam * Lc
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            pred = model(X_va).cpu().numpy()
        rmse = np.sqrt(np.mean((pred - Y_np[n_train:]) ** 2, axis=0))
        label = f"lam_cf={lam}"
        results[label] = rmse
        print(f"\n{label}")
        for n, r2 in zip(NAMES, rmse):
            print(f"  {n:>10}  RMSE={r2:.5f}")

    # Bar chart
    x = np.arange(len(NAMES))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, results["lam_cf=0.0"], w, label="No CF loss")
    ax.bar(x + w/2, results["lam_cf=0.5"], w, label="With CF loss")
    ax.set_xticks(x); ax.set_xticklabels(NAMES)
    ax.set_ylabel("RMSE"); ax.set_title("Ablation: Effect of CF Physics Loss")
    ax.legend(); plt.tight_layout()
    plt.savefig("ablation.png", dpi=150)
    plt.show()
    print("\nSaved ablation.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Train
    model, tr_losses, va_losses, mu_x, std_x = train(
        n_train=8_000, n_val=1_000, epochs=80, batch_size=256, lam_cf=0.5
    )

    # Plots
    plot_training_curve(tr_losses, va_losses)
    evaluate(model, mu_x, std_x)
    demo_single_calibration(model, mu_x, std_x)

    # Optional: ablation study (takes ~2× training time)
    # ablation()
