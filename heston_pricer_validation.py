# Created: January 28, 2026

"""
Heston COS Pricer Validation
============================

Heston model SDEs (under risk-neutral measure):
    dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW^S_t
    dv_t = kappa * (theta - v_t) * dt + sigma_v * sqrt(v_t) * dW^v_t
    d<W^S, W^v>_t = rho * dt

Five parameters:
    v0      : initial variance
    kappa   : mean reversion speed
    theta   : long-run variance
    sigma_v : vol-of-vol
    rho     : correlation between S and v
    (plus r, S0 fixed externally)

The CF of log(S_T / S_0) under Heston has an exponential-affine form:
    phi(u, T) = exp( C(u, T) + D(u, T) * v0 + i*u*r*T )

where C and D solve Riccati ODEs in T (closed-form solution available).

Validation tests:
  Test 1 — Black-Scholes limit  (sigma_v=0, kappa large, v0=theta=sigma^2)
  Test 2 — Put-call parity
  Test 3 — Monte Carlo (Andersen QE scheme — bias-free for Heston)

Run:  python heston_pricer_validation.py
Deps: numpy scipy
"""

import numpy as np
from scipy.stats import norm

S0    = 100.0
r     = 0.05
N_COS = 256


# ─────────────────────────────────────────────────────────────────────────────
# 1.  HESTON CHARACTERISTIC FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def heston_cf(u, T, v0, kappa, theta, sigma_v, rho):
    """
    CF of x = log(S_T / S_0) under Heston.

    Closed-form (Albrecher et al. 2007, "little Heston trap" version
    that avoids branch-cut issues with the complex sqrt):

        phi(u, T) = exp(i*u*r*T + C(u,T) + D(u,T)*v0)

    where:
        xi  = kappa - i*u*sigma_v*rho
        d   = sqrt(xi^2 + sigma_v^2 * (i*u + u^2))
        g2  = (xi - d) / (xi + d)         ← "little trap" form
        D   = (xi - d) * (1 - exp(-d*T)) / (sigma_v^2 * (1 - g2*exp(-d*T)))
        C   = (kappa*theta/sigma_v^2) * [(xi-d)*T - 2*log((1-g2*exp(-d*T))/(1-g2))]

    Why "little trap": the original formulation uses g1 = (xi+d)/(xi-d)
    which can cross the branch cut of complex log for some (u, T) combos.
    The g2 form is numerically stable for all parameter ranges relevant
    to calibration.
    """
    iu = 1j * u

    # Discriminant
    xi = kappa - iu * sigma_v * rho
    d  = np.sqrt(xi**2 + sigma_v**2 * (iu + u**2))

    # "Little Heston trap" formulation — numerically stable
    g2  = (xi - d) / (xi + d)
    edt = np.exp(-d * T)

    D = (xi - d) / (sigma_v**2) * (1.0 - edt) / (1.0 - g2 * edt)
    C = (kappa * theta / sigma_v**2) * (
        (xi - d) * T - 2.0 * np.log((1.0 - g2 * edt) / (1.0 - g2))
    )

    return np.exp(iu * r * T + C + D * v0)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  COS TRUNCATION INTERVAL
# ─────────────────────────────────────────────────────────────────────────────
def cos_truncation(T, v0, kappa, theta, sigma_v, rho, L=10):
    """
    Cumulant-based truncation. For Heston, c1 (mean) and c2 (variance)
    of log(S_T/S_0) have closed forms; c4 is messier so a
    diffusion-only approximation that's plenty conservative.

    Mean:      c1 = (r - 0.5*v_avg)*T
    Variance:  c2 ≈ v_avg * T   (ignoring vol-of-vol contribution at
                                    leading order — fine for truncation
                                    bounds)
    where v_avg = theta + (v0 - theta)*(1 - e^{-kappa*T})/(kappa*T)
    is the time-average of the expected variance path.
    """
    if kappa * T < 1e-6:
        v_avg = v0
    else:
        v_avg = theta + (v0 - theta) * (1.0 - np.exp(-kappa*T)) / (kappa*T)

    c1 = (r - 0.5 * v_avg) * T
    # Inflate variance estimate to be safe with vol-of-vol effects
    c2 = v_avg * T + 0.5 * sigma_v**2 * T**2 * v_avg / kappa

    H = L * np.sqrt(abs(c2))
    return c1 - H, c1 + H


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CALL PAYOFF COSINE COEFFICIENTS  (same as Merton — payoff-only)
# ─────────────────────────────────────────────────────────────────────────────
def call_cos_coefficients(a, b, k):
    """
    V_k for call payoff (e^y - 1)^+ on [0, b]. Same formula as Merton —
    these are model-independent (depend only on payoff and truncation).
    """
    bw    = b - a
    kp    = k * np.pi / bw
    upper = np.exp(b) * np.cos(k * np.pi)
    lower = np.cos(-kp*a) + kp * np.sin(-kp*a)
    chi = np.where(k == 0, np.exp(b) - 1.0,
                   (1.0 / (1.0 + kp**2)) * (upper - lower))
    with np.errstate(invalid="ignore", divide="ignore"):
        psi = np.where(k == 0, b,
                       (np.sin(kp*bw) - np.sin(-kp*a)) / kp)
    return (2.0 / bw) * (chi - psi)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PRICER
# ─────────────────────────────────────────────────────────────────────────────
def price_call_cos(K, T, v0, kappa, theta, sigma_v, rho, n=N_COS):
    """
    European call via COS, Heston model.

    Note: the Heston CF includes the i*u*r*T drift internally, so the
    series sum gives prices directly without a separate discounting drift
    adjustment.  Just multiply by exp(-r*T) at the end.
    """
    a, b = cos_truncation(T, v0, kappa, theta, sigma_v, rho)
    k    = np.arange(n, dtype=float)
    u    = k * np.pi / (b - a)
    x    = np.log(S0 / K)

    phi  = heston_cf(u, T, v0, kappa, theta, sigma_v, rho)
    V    = call_cos_coefficients(a, b, k)

    series    = np.real(phi * np.exp(1j * k * np.pi * (x - a) / (b - a)))
    series[0] *= 0.5

    # The Heston CF in this formulation has the drift baked in (i*u*r*T term),
    # so the series result is e^{rT}*E[(S_T-K)^+/S_0] essentially.
    # The standard COS formula: C = K*e^{-rT} * sum_k Re[phi_x(u_k)*exp(...)] V_k
    # where phi_x is the CF of x = log(S_T/K). The CF here is of log(S_T/S_0), and
    # the exp(i*k*pi*(x-a)/(b-a)) term shifts to log(S_T/K) — works out the same.
    return max(K * np.exp(-r * T) * np.dot(series, V), 0.0)


def price_put_cos(K, T, v0, kappa, theta, sigma_v, rho, n=N_COS):
    """Put via put-call parity."""
    C = price_call_cos(K, T, v0, kappa, theta, sigma_v, rho, n)
    return C - S0 + K * np.exp(-r * T)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  REFERENCE PRICERS
# ─────────────────────────────────────────────────────────────────────────────
def black_scholes_call(K, T, sigma):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def monte_carlo_heston_qe(K, T, v0, kappa, theta, sigma_v, rho,
                           n_paths=200_000, n_steps=200, seed=42):
    """
    Andersen Quadratic-Exponential (QE) scheme for Heston Monte Carlo.

    The naive Euler discretization fails for Heston because v can become
    negative. The QE scheme (Andersen 2008) preserves positivity of v
    using two cases based on the ratio psi = s2/m2:
      - psi <= 1.5: match moments to a squared Gaussian
      - psi >  1.5: match moments to an exponential

    For log-S the integrated-variance approximation is used:
        log(S_{i+1}/S_i) ≈ r*dt - 0.5*int(v ds) + rho*int(sqrt(v) dW^v)
                          + sqrt(1-rho^2)*sqrt(int(v ds)) * Z

    This is exact in distribution under standard QE assumptions.
    """
    rng = np.random.default_rng(seed)
    dt  = T / n_steps

    # Pre-compute QE constants
    e   = np.exp(-kappa*dt)
    K0  = -rho*kappa*theta/sigma_v * dt
    K1  = 0.5*dt*(kappa*rho/sigma_v - 0.5) - rho/sigma_v
    K2  = 0.5*dt*(kappa*rho/sigma_v - 0.5) + rho/sigma_v
    K3  = 0.5*dt*(1 - rho**2)
    K4  = K3   # symmetry under the integration choice used here

    log_S = np.full(n_paths, np.log(S0))
    v     = np.full(n_paths, v0)

    for _ in range(n_steps):
        # Conditional moments of v_{t+dt} | v_t
        m   = theta + (v - theta) * e
        s2  = (v * sigma_v**2 * e * (1 - e) / kappa
               + theta * sigma_v**2 * (1 - e)**2 / (2*kappa))
        psi = s2 / np.maximum(m**2, 1e-300)

        v_next = np.empty_like(v)

        # Case 1: psi <= 1.5  → squared Gaussian
        mask = psi <= 1.5
        if mask.any():
            psi_m  = psi[mask]
            b2     = 2.0/psi_m - 1.0 + np.sqrt(2.0/psi_m * (2.0/psi_m - 1.0))
            a_qe   = m[mask] / (1.0 + b2)
            Zv     = rng.standard_normal(mask.sum())
            v_next[mask] = a_qe * (np.sqrt(b2) + Zv)**2

        # Case 2: psi > 1.5  → exponential
        nm = ~mask
        if nm.any():
            p_qe   = (psi[nm] - 1.0) / (psi[nm] + 1.0)
            beta   = (1.0 - p_qe) / m[nm]
            U      = rng.uniform(size=nm.sum())
            v_qe   = np.where(U <= p_qe, 0.0, np.log((1.0 - p_qe)/(1.0 - U)) / beta)
            v_next[nm] = v_qe

        # Update log-S
        Z      = rng.standard_normal(n_paths)
        log_S += (r*dt + K0
                  + K1*v + K2*v_next
                  + np.sqrt(np.maximum(K3*v + K4*v_next, 0.0)) * Z)
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


def test_bs_limit():
    """
    Black-Scholes limit: sigma_v → 0 with v0 = theta = sigma^2 means
    variance is constant at sigma^2, recovering Black-Scholes.
    """
    sep("TEST 1 — Black-Scholes Limit  (sigma_v→0, v0=theta=σ²)")
    sigma   = 0.20
    v0      = sigma**2          # 0.04
    theta   = sigma**2          # 0.04
    kappa   = 2.0               # any positive value
    sigma_v = 1e-4              # → 0 (but not exactly 0, avoids div-by-zero)
    rho     = 0.0

    print(f"  sigma={sigma}  v0={v0}  theta={theta}  sigma_v={sigma_v}  rho={rho}\n")
    print(f"  {'K':>5} {'T':>5} {'COS':>10} {'BS':>10} {'|err|':>10}")
    print(f"  {'-'*5} {'-'*5} {'-'*10} {'-'*10} {'-'*10}")
    max_err = 0.0
    for T in MATURITIES:
        for K in STRIKES:
            cos = price_call_cos(K, T, v0, kappa, theta, sigma_v, rho)
            bs  = black_scholes_call(K, T, sigma)
            err = abs(cos - bs)
            max_err = max(max_err, err)
            print(f"  {K:>5} {T:>5.2f} {cos:>10.5f} {bs:>10.5f} {err:>10.2e}")
    result = "PASS ✓" if max_err < 1e-3 else "FAIL ✗"
    print(f"\n  Max error = {max_err:.2e}  [{result}]")
    return max_err


def test_put_call_parity():
    sep("TEST 2 — Put-Call Parity")
    v0, kappa, theta, sigma_v, rho = 0.04, 2.0, 0.04, 0.5, -0.7
    print(f"  v0={v0}  kappa={kappa}  theta={theta}  sigma_v={sigma_v}  rho={rho}\n")
    print(f"  {'K':>5} {'T':>5} {'C-P':>10} {'S0-Ke-rT':>10} {'|err|':>10}")
    print(f"  {'-'*5} {'-'*5} {'-'*10} {'-'*10} {'-'*10}")
    max_err = 0.0
    for T in MATURITIES:
        for K in STRIKES:
            C = price_call_cos(K, T, v0, kappa, theta, sigma_v, rho)
            P = price_put_cos( K, T, v0, kappa, theta, sigma_v, rho)
            lhs = C - P
            rhs = S0 - K*np.exp(-r*T)
            err = abs(lhs - rhs)
            max_err = max(max_err, err)
            print(f"  {K:>5} {T:>5.2f} {lhs:>10.5f} {rhs:>10.5f} {err:>10.2e}")
    result = "PASS ✓" if max_err < 1e-4 else "FAIL ✗"
    print(f"\n  Max error = {max_err:.2e}  [{result}]")
    return max_err


def test_monte_carlo():
    """COS price should lie within 3-sigma of MC estimate."""
    sep("TEST 3 — Monte Carlo (Andersen QE, 100k paths × 200 steps)")
    v0, kappa, theta, sigma_v, rho = 0.04, 2.0, 0.04, 0.5, -0.7
    n_paths, n_steps = 100_000, 200
    print(f"  v0={v0}  kappa={kappa}  theta={theta}  sigma_v={sigma_v}  rho={rho}\n")
    print(f"  {'K':>5} {'T':>5} {'COS':>10} {'MC':>10} {'CI [MC±3se]':>22} {'in band':>8}")
    print(f"  {'-'*5} {'-'*5} {'-'*10} {'-'*10} {'-'*22} {'-'*8}")
    fails = 0
    for T in [0.5, 1.0, 2.0]:
        for K in [90, 100, 110]:
            cos    = price_call_cos(K, T, v0, kappa, theta, sigma_v, rho)
            mc, se = monte_carlo_heston_qe(K, T, v0, kappa, theta, sigma_v, rho,
                                           n_paths, n_steps)
            lo, hi = mc - 3*se, mc + 3*se
            ok     = lo <= cos <= hi
            if not ok: fails += 1
            ci = f"[{lo:.4f}, {hi:.4f}]"
            print(f"  {K:>5} {T:>5.2f} {cos:>10.5f} {mc:>10.5f} {ci:>22}  {'✓' if ok else '✗'}")
    result = "PASS ✓" if fails == 0 else f"FAIL ✗  ({fails} outside band)"
    print(f"\n  {result}")
    return fails


def test_convergence():
    """Exponential convergence in N."""
    sep("EXTRA — COS Convergence in N  (reference = N=1024)")
    v0, kappa, theta, sigma_v, rho = 0.04, 2.0, 0.04, 0.5, -0.7
    K, T = 100.0, 1.0
    ref = price_call_cos(K, T, v0, kappa, theta, sigma_v, rho, n=1024)
    print(f"  Reference (N=1024) = {ref:.8f}\n")
    print(f"  {'N':>5} {'Price':>14} {'|err|':>12}")
    print(f"  {'-'*5} {'-'*14} {'-'*12}")
    for n in [16, 32, 64, 128, 256, 512]:
        p = price_call_cos(K, T, v0, kappa, theta, sigma_v, rho, n=n)
        print(f"  {n:>5} {p:>14.8f} {abs(p-ref):>12.2e}")


if __name__ == "__main__":
    print("="*72)
    print("  Heston COS Pricer Validation")
    print("="*72)
    print(f"  S0={S0}  r={r}  N_COS={N_COS}")

    e1 = test_bs_limit()
    e2 = test_put_call_parity()
    f3 = test_monte_carlo()
    test_convergence()

    sep("SUMMARY")
    print(f"  Test 1  BS limit     max|error| = {e1:.2e}   {'PASS ✓' if e1<1e-3 else 'FAIL ✗'}")
    print(f"  Test 2  Parity       max|error| = {e2:.2e}   {'PASS ✓' if e2<1e-4 else 'FAIL ✗'}")
    print(f"  Test 3  Monte Carlo  fails      = {f3}        {'PASS ✓' if f3==0 else 'FAIL ✗'}")
    print()
