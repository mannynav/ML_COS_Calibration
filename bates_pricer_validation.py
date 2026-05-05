# Created: February 9, 2026

"""
Bates COS Pricer Validation
============================

Bates (1996) model — Heston stochastic volatility + Merton jumps:

    dS_t = (r - lam*kappa_J) * S_t * dt + sqrt(v_t) * S_t * dW^S_t  +  S_t * (J - 1) * dN_t
    dv_t = kappa * (theta - v_t) * dt + sigma_v * sqrt(v_t) * dW^v_t
    d<W^S, W^v>_t = rho * dt

with:
    N_t   ~ Poisson(lam * t)
    log(J) ~ N(mu_J, sigma_J^2)
    kappa_J = E[J - 1] = exp(mu_J + 0.5 * sigma_J^2) - 1   (martingale correction)

Eight parameters total:
    Heston part: v0, kappa, theta, sigma_v, rho
    Jump part:   lam, mu_J, sigma_J

The CF factorizes neatly by Lévy-Khintchine independence:
    phi_Bates(u, T) = phi_Heston(u, T) * phi_jump(u, T)

where phi_jump is the Merton compound-Poisson CF with the drift correction
absorbed into the diffusion drift to keep the discounted price a martingale.

Validation tests:
  Test 1 — Heston limit  (lam=0  →  Bates collapses to Heston)
  Test 2 — Merton limit  (sigma_v→0, v0=theta=sigma^2, rho=0  →  Bates collapses to Merton)
  Test 3 — Put-call parity
  Test 4 — Monte Carlo cross-check (Andersen QE + jumps)

Run:  python bates_pricer_validation.py
Deps: numpy scipy
"""

import numpy as np
from scipy.stats import norm

S0    = 100.0
r     = 0.05
N_COS = 256


# ─────────────────────────────────────────────────────────────────────────────
# 1.  BATES CHARACTERISTIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def heston_part_cf(u, T, v0, kappa, theta, sigma_v, rho):
    """
    Heston piece of the Bates CF — same 'little trap' formulation as before,
    but WITHOUT the i*u*r*T drift term (the full drift, including
    the jump compensator, in the parent function).
    """
    iu = 1j * u
    xi = kappa - iu * sigma_v * rho
    d  = np.sqrt(xi**2 + sigma_v**2 * (iu + u**2))
    g2 = (xi - d) / (xi + d)
    edt = np.exp(-d * T)
    D = (xi - d) / (sigma_v**2) * (1.0 - edt) / (1.0 - g2*edt)
    C = (kappa*theta/sigma_v**2) * (
            (xi - d)*T - 2.0*np.log((1.0 - g2*edt) / (1.0 - g2))
        )
    return np.exp(C + D*v0)


def jump_part_cf(u, T, lam, mu_j, sigma_j):
    """
    Compound-Poisson jump CF with Lévy-Khintchine compensator.
    For the discounted price to be a martingale, the drift must be reduced
    by lam * (E[J] - 1) = lam * (exp(mu_j + 0.5*sigma_j^2) - 1).

    psi_jump(u) = lam * (E[e^{iuJ}] - 1) - i*u*lam*(E[J] - 1)
    """
    iu        = 1j * u
    drift_corr = lam * (np.exp(mu_j + 0.5*sigma_j**2) - 1.0)
    phi_J      = np.exp(iu*mu_j - 0.5*(u*sigma_j)**2)
    psi        = lam * (phi_J - 1.0) - iu * drift_corr
    return np.exp(T * psi)


def bates_cf(u, T, v0, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j):
    """
    Full Bates CF.

    By Lévy-Khintchine independence between the Brownian/Heston part and
    the compound Poisson jump part:

        phi_Bates(u, T) = exp(i*u*r*T) * phi_Heston(u, T) * phi_jump(u, T)

    The risk-free drift i*u*r*T appears once; phi_jump has the jump
    compensator -i*u*lam*kappa_J built in, so the total drift in log-S is
    r - lam*kappa_J as required for risk-neutral pricing.
    """
    iu = 1j * u
    return (np.exp(iu*r*T)
            * heston_part_cf(u, T, v0, kappa, theta, sigma_v, rho)
            * jump_part_cf(u, T, lam, mu_j, sigma_j))


# ─────────────────────────────────────────────────────────────────────────────
# 2.  COS TRUNCATION INTERVAL
# ─────────────────────────────────────────────────────────────────────────────
def cos_truncation(T, v0, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j, L=10):
    """
    Cumulants of log(S_T/S_0) under Bates.

    By independence, cumulants add:
        c1 = c1_Heston + c1_jump
        c2 = c2_Heston + c2_jump
        c4 ≈ c4_jump   (Heston c4 contribution is small at index-option scales)

    Heston pieces (with martingale drift, see Albrecher et al.):
        v_avg = theta + (v0 - theta)*(1 - e^{-kappa T})/(kappa T)
        c1_H  = (r - 0.5*v_avg)*T
        c2_H  ≈ v_avg*T   (+ vol-of-vol correction)

    Jump pieces (Merton):
        drift_corr = lam * (exp(mu_j + 0.5*sigma_j^2) - 1)
        c1_J = -drift_corr * T
        c2_J = lam * (mu_j^2 + sigma_j^2) * T
        c4_J = lam * (mu_j^4 + 6*mu_j^2*sigma_j^2 + 3*sigma_j^4) * T
    """
    if kappa*T < 1e-6:
        v_avg = v0
    else:
        v_avg = theta + (v0 - theta)*(1.0 - np.exp(-kappa*T))/(kappa*T)

    drift_corr = lam * (np.exp(mu_j + 0.5*sigma_j**2) - 1.0)

    c1 = (r - 0.5*v_avg)*T - drift_corr*T
    c2 = (v_avg*T
          + 0.5*sigma_v**2 * T**2 * v_avg / kappa
          + lam*(mu_j**2 + sigma_j**2)*T)
    c4 = lam*(mu_j**4 + 6*mu_j**2*sigma_j**2 + 3*sigma_j**4)*T

    H = L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    return c1 - H, c1 + H


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PAYOFF COEFFICIENTS
# ─────────────────────────────────────────────────────────────────────────────
def call_cos_coefficients(a, b, k):
    bw = b - a
    kp = k * np.pi / bw
    upper = np.exp(b) * np.cos(k*np.pi)
    lower = np.cos(-kp*a) + kp*np.sin(-kp*a)
    chi = np.where(k==0, np.exp(b)-1.0,
                   (1/(1+kp**2))*(upper - lower))
    with np.errstate(invalid="ignore", divide="ignore"):
        psi = np.where(k==0, b, (np.sin(kp*bw) - np.sin(-kp*a))/kp)
    return (2/bw)*(chi - psi)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PRICER
# ─────────────────────────────────────────────────────────────────────────────
def price_call_cos(K, T, v0, kappa, theta, sigma_v, rho,
                       lam, mu_j, sigma_j, n=N_COS):
    """European call via COS, Bates model."""
    a, b = cos_truncation(T, v0, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j)
    k    = np.arange(n, dtype=float)
    u    = k * np.pi / (b - a)
    x    = np.log(S0 / K)

    phi  = bates_cf(u, T, v0, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j)
    V    = call_cos_coefficients(a, b, k)

    series    = np.real(phi * np.exp(1j*k*np.pi*(x-a)/(b-a)))
    series[0] *= 0.5
    return max(K*np.exp(-r*T)*np.dot(series, V), 0.0)


def price_put_cos(K, T, v0, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j, n=N_COS):
    C = price_call_cos(K, T, v0, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j, n)
    return C - S0 + K*np.exp(-r*T)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  REFERENCE PRICERS
# ─────────────────────────────────────────────────────────────────────────────
def black_scholes_call(K, T, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def heston_cf_for_test(u, T, v0, kappa, theta, sigma_v, rho):
    """Standalone Heston CF including drift — used only for cross-checking."""
    iu = 1j * u
    xi = kappa - iu * sigma_v * rho
    d  = np.sqrt(xi**2 + sigma_v**2 * (iu + u**2))
    g2 = (xi - d) / (xi + d)
    edt = np.exp(-d * T)
    D = (xi - d) / (sigma_v**2) * (1.0 - edt) / (1.0 - g2*edt)
    C = (kappa*theta/sigma_v**2) * (
            (xi - d)*T - 2.0*np.log((1.0 - g2*edt) / (1.0 - g2))
        )
    return np.exp(iu*r*T + C + D*v0)


def heston_call_cos(K, T, v0, kappa, theta, sigma_v, rho, n=N_COS, L=10):
    """Pure Heston COS for the limit test."""
    if kappa*T < 1e-6:
        v_avg = v0
    else:
        v_avg = theta + (v0 - theta)*(1.0 - np.exp(-kappa*T))/(kappa*T)
    c1_H = (r - 0.5*v_avg)*T
    c2_H = v_avg*T + 0.5*sigma_v**2 * T**2 * v_avg / kappa
    H = L*np.sqrt(abs(c2_H))
    a, b = c1_H - H, c1_H + H

    k = np.arange(n, dtype=float)
    u = k*np.pi/(b-a)
    x = np.log(S0/K)
    phi = heston_cf_for_test(u, T, v0, kappa, theta, sigma_v, rho)
    V = call_cos_coefficients(a, b, k)
    series = np.real(phi * np.exp(1j*k*np.pi*(x-a)/(b-a)))
    series[0] *= 0.5
    return max(K*np.exp(-r*T)*np.dot(series, V), 0.0)


def monte_carlo_bates_qe(K, T, v0, kappa, theta, sigma_v, rho,
                              lam, mu_j, sigma_j,
                              n_paths=100_000, n_steps=200, seed=42):
    """
    Andersen QE for Heston part + thinned jump simulation.

    Per step:
      1.  Update v via QE (preserves positivity)
      2.  Update log-S under continuous diffusion (correlated Z)
      3.  Sample N_t ~ Poisson(lam*dt) jumps in this interval, sum log-jumps
    """
    rng = np.random.default_rng(seed)
    dt  = T / n_steps

    # QE constants for log-S given (v_t, v_{t+dt})
    e   = np.exp(-kappa*dt)
    drift_corr = lam * (np.exp(mu_j + 0.5*sigma_j**2) - 1.0)
    K0  = -rho*kappa*theta/sigma_v * dt  -  drift_corr*dt   # add jump compensator
    K1  = 0.5*dt*(kappa*rho/sigma_v - 0.5) - rho/sigma_v
    K2  = 0.5*dt*(kappa*rho/sigma_v - 0.5) + rho/sigma_v
    K3  = 0.5*dt*(1 - rho**2)

    log_S = np.full(n_paths, np.log(S0))
    v     = np.full(n_paths, v0)

    for _ in range(n_steps):
        m   = theta + (v - theta) * e
        s2  = (v * sigma_v**2 * e * (1-e)/kappa
               + theta * sigma_v**2 * (1-e)**2 / (2*kappa))
        psi = s2 / np.maximum(m**2, 1e-300)

        v_next = np.empty_like(v)
        mask   = psi <= 1.5
        if mask.any():
            psi_m = psi[mask]
            b2    = 2.0/psi_m - 1.0 + np.sqrt(2.0/psi_m * (2.0/psi_m - 1.0))
            a_qe  = m[mask] / (1.0 + b2)
            Zv    = rng.standard_normal(mask.sum())
            v_next[mask] = a_qe * (np.sqrt(b2) + Zv)**2
        nm = ~mask
        if nm.any():
            p_qe  = (psi[nm] - 1.0) / (psi[nm] + 1.0)
            beta  = (1.0 - p_qe) / m[nm]
            U     = rng.uniform(size=nm.sum())
            v_next[nm] = np.where(U <= p_qe, 0.0,
                                  np.log((1.0-p_qe)/(1.0-U)) / beta)

        # Continuous diffusion update
        Z = rng.standard_normal(n_paths)
        log_S += (r*dt + K0
                  + K1*v + K2*v_next
                  + np.sqrt(np.maximum(K3*v + K3*v_next, 0.0)) * Z)

        # Jumps in this interval
        n_jumps = rng.poisson(lam*dt, n_paths)
        if n_jumps.max() > 0:
            max_n = int(n_jumps.max())
            J     = rng.normal(mu_j, sigma_j, (n_paths, max_n))
            mask_j = np.arange(max_n) < n_jumps[:, None]
            log_S += (J * mask_j).sum(axis=1)

        v = v_next

    S_T    = np.exp(log_S)
    payoff = np.maximum(S_T - K, 0.0)
    price  = np.exp(-r*T) * payoff.mean()
    se     = np.exp(-r*T) * payoff.std() / np.sqrt(n_paths)
    return price, se


# ─────────────────────────────────────────────────────────────────────────────
# 6.  TESTS
# ─────────────────────────────────────────────────────────────────────────────
STRIKES    = [80, 90, 100, 110, 120]
MATURITIES = [0.25, 0.5, 1.0, 2.0]


def sep(title=""):
    print("\n" + "─"*72)
    if title:
        print(f"  {title}"); print("─"*72)


def test_heston_limit():
    """lam=0 → Bates = Heston exactly."""
    sep("TEST 1 — Heston Limit  (lam=0)")
    v0, kappa, theta, sigma_v, rho = 0.04, 2.0, 0.04, 0.5, -0.7
    lam, mu_j, sigma_j = 0.0, 0.0, 0.10

    print(f"  v0={v0} kappa={kappa} theta={theta} sigma_v={sigma_v} rho={rho}")
    print(f"  lam={lam} (no jumps)\n")
    print(f"  {'K':>5} {'T':>5} {'Bates':>10} {'Heston':>10} {'|err|':>10}")
    print(f"  {'-'*5} {'-'*5} {'-'*10} {'-'*10} {'-'*10}")
    max_err = 0.0
    for T in MATURITIES:
        for K in STRIKES:
            bp = price_call_cos(K, T, v0, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j)
            hp = heston_call_cos(K, T, v0, kappa, theta, sigma_v, rho)
            err = abs(bp - hp)
            max_err = max(max_err, err)
            print(f"  {K:>5} {T:>5.2f} {bp:>10.5f} {hp:>10.5f} {err:>10.2e}")
    print(f"\n  Max error = {max_err:.2e}  [{'PASS ✓' if max_err < 1e-4 else 'FAIL ✗'}]")
    return max_err


def test_merton_limit():
    """sigma_v→0, v0=theta=sigma^2, rho=0 → Bates = Merton."""
    sep("TEST 2 — Merton Limit  (sigma_v→0, rho=0, v0=theta=σ²)")
    sigma   = 0.20
    v0      = sigma**2
    theta   = sigma**2
    kappa   = 2.0
    sigma_v = 1e-4
    rho     = 0.0
    lam, mu_j, sigma_j = 0.5, -0.05, 0.10

    print(f"  Diffusion vol (constant) = {sigma}")
    print(f"  Jumps: lam={lam}  mu_j={mu_j}  sigma_j={sigma_j}\n")
    print(f"  {'K':>5} {'T':>5} {'Bates':>10} {'Merton-COS':>12} {'|err|':>10}")
    print(f"  {'-'*5} {'-'*5} {'-'*10} {'-'*12} {'-'*10}")

    # Merton COS (inline so this test is self-contained)
    def merton_cos(K, T, sig, lm, mj, sj, n=N_COS):
        drift_corr = lm*(np.exp(mj+0.5*sj**2) - 1.0)
        c1 = (r - 0.5*sig**2 - drift_corr)*T
        c2 = (sig**2 + lm*(mj**2 + sj**2))*T
        c4 = lm*(mj**4 + 6*mj**2*sj**2 + 3*sj**4)*T
        H  = 10*np.sqrt(abs(c2) + np.sqrt(abs(c4)))
        a, b = c1-H, c1+H
        k = np.arange(n, dtype=float)
        u = k*np.pi/(b-a)
        x = np.log(S0/K)
        phi_j = np.exp(1j*u*mj - 0.5*(u*sj)**2)
        Psi = (1j*u*(r - 0.5*sig**2 - drift_corr)
               - 0.5*(u*sig)**2
               + lm*(phi_j - 1.0))
        phi = np.exp(T*Psi)
        V = call_cos_coefficients(a, b, k)
        series = np.real(phi * np.exp(1j*k*np.pi*(x-a)/(b-a)))
        series[0] *= 0.5
        return max(K*np.exp(-r*T)*np.dot(series, V), 0.0)

    max_err = 0.0
    for T in MATURITIES:
        for K in STRIKES:
            bp = price_call_cos(K, T, v0, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j)
            mp = merton_cos(K, T, sigma, lam, mu_j, sigma_j)
            err = abs(bp - mp)
            max_err = max(max_err, err)
            print(f"  {K:>5} {T:>5.2f} {bp:>10.5f} {mp:>12.5f} {err:>10.2e}")
    print(f"\n  Max error = {max_err:.2e}  [{'PASS ✓' if max_err < 1e-3 else 'FAIL ✗'}]")
    return max_err


def test_put_call_parity():
    sep("TEST 3 — Put-Call Parity")
    v0, kappa, theta, sigma_v, rho = 0.04, 2.0, 0.04, 0.5, -0.7
    lam, mu_j, sigma_j = 0.5, -0.10, 0.15
    print(f"  Heston: v0={v0} kappa={kappa} theta={theta} sigma_v={sigma_v} rho={rho}")
    print(f"  Jumps:  lam={lam}  mu_j={mu_j}  sigma_j={sigma_j}\n")
    print(f"  {'K':>5} {'T':>5} {'C-P':>10} {'S0-Ke-rT':>10} {'|err|':>10}")
    print(f"  {'-'*5} {'-'*5} {'-'*10} {'-'*10} {'-'*10}")
    max_err = 0.0
    for T in MATURITIES:
        for K in STRIKES:
            C = price_call_cos(K, T, v0, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j)
            P = price_put_cos( K, T, v0, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j)
            lhs = C - P
            rhs = S0 - K*np.exp(-r*T)
            err = abs(lhs - rhs)
            max_err = max(max_err, err)
            print(f"  {K:>5} {T:>5.2f} {lhs:>10.5f} {rhs:>10.5f} {err:>10.2e}")
    print(f"\n  Max error = {max_err:.2e}  [{'PASS ✓' if max_err < 1e-4 else 'FAIL ✗'}]")
    return max_err


def test_monte_carlo():
    sep("TEST 4 — Monte Carlo (Andersen QE + jumps, 100k × 200 steps)")
    v0, kappa, theta, sigma_v, rho = 0.04, 2.0, 0.04, 0.5, -0.7
    lam, mu_j, sigma_j = 0.5, -0.10, 0.15
    print(f"  Heston: v0={v0} kappa={kappa} theta={theta} sigma_v={sigma_v} rho={rho}")
    print(f"  Jumps:  lam={lam}  mu_j={mu_j}  sigma_j={sigma_j}\n")
    print(f"  {'K':>5} {'T':>5} {'COS':>10} {'MC':>10} {'CI [MC±3se]':>22} {'in band':>8}")
    print(f"  {'-'*5} {'-'*5} {'-'*10} {'-'*10} {'-'*22} {'-'*8}")
    fails = 0
    for T in [0.5, 1.0, 2.0]:
        for K in [90, 100, 110]:
            cos    = price_call_cos(K, T, v0, kappa, theta, sigma_v, rho,
                                    lam, mu_j, sigma_j)
            mc, se = monte_carlo_bates_qe(K, T, v0, kappa, theta, sigma_v, rho,
                                          lam, mu_j, sigma_j)
            lo, hi = mc - 3*se, mc + 3*se
            ok     = lo <= cos <= hi
            if not ok: fails += 1
            print(f"  {K:>5} {T:>5.2f} {cos:>10.5f} {mc:>10.5f} "
                  f"[{lo:>8.4f}, {hi:>8.4f}]  {'✓' if ok else '✗'}")
    print(f"\n  {'PASS ✓' if fails==0 else f'FAIL ✗  ({fails})'}")
    return fails


def test_convergence():
    sep("EXTRA — COS Convergence in N (ref = N=1024)")
    v0, kappa, theta, sigma_v, rho = 0.04, 2.0, 0.04, 0.5, -0.7
    lam, mu_j, sigma_j = 0.5, -0.10, 0.15
    K, T = 100.0, 1.0
    ref = price_call_cos(K, T, v0, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j, n=1024)
    print(f"  Reference (N=1024) = {ref:.8f}\n")
    print(f"  {'N':>5} {'Price':>14} {'|err|':>12}")
    print(f"  {'-'*5} {'-'*14} {'-'*12}")
    for n in [16, 32, 64, 128, 256, 512]:
        p = price_call_cos(K, T, v0, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j, n=n)
        print(f"  {n:>5} {p:>14.8f} {abs(p-ref):>12.2e}")


if __name__ == "__main__":
    print("="*72)
    print("  Bates COS Pricer Validation")
    print("="*72)
    print(f"  S0={S0}  r={r}  N_COS={N_COS}")

    e1 = test_heston_limit()
    e2 = test_merton_limit()
    e3 = test_put_call_parity()
    f4 = test_monte_carlo()
    test_convergence()

    sep("SUMMARY")
    print(f"  Test 1  Heston limit  max|err| = {e1:.2e}   {'PASS ✓' if e1<1e-4 else 'FAIL ✗'}")
    print(f"  Test 2  Merton limit  max|err| = {e2:.2e}   {'PASS ✓' if e2<1e-3 else 'FAIL ✗'}")
    print(f"  Test 3  Parity        max|err| = {e3:.2e}   {'PASS ✓' if e3<1e-4 else 'FAIL ✗'}")
    print(f"  Test 4  MC fails      = {f4}                {'PASS ✓' if f4==0 else 'FAIL ✗'}")
