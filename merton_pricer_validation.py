# Created: January 14, 2026

"""
Merton COS pricer validation
============================

"""
import numpy as np
from scipy.stats import norm

S0 = 100.0
r  = 0.05
N_COS = 128

def merton_cf(u, T, sigma, lam, mu_j, sigma_j):
    drift_corr = lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1.0)
    phi_jump   = np.exp(1j * u * mu_j - 0.5 * (u * sigma_j)**2)
    Psi = (
        1j * u * (r - 0.5 * sigma**2 - drift_corr)
        - 0.5 * (u * sigma)**2
        + lam * (phi_jump - 1.0)
    )
    return np.exp(T * Psi)

def cos_truncation(T, sigma, lam, mu_j, sigma_j, L=12):
    """
    Integration interval [a, b] via cumulants of log(S_T/S_0).
        c1 = mean, c2 = variance, c4 = 4th cumulant contribution from jumps
        H  = L * sqrt(|c2| + sqrt(|c4|))
    Ref: Fang & Oosterlee (2008) eq. (52).
    """
    drift_corr = lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1.0)
    c1 = (r - 0.5 * sigma**2 - drift_corr) * T
    c2 = (sigma**2 + lam * (mu_j**2 + sigma_j**2)) * T
    c4 = lam * (mu_j**4 + 6 * mu_j**2 * sigma_j**2 + 3 * sigma_j**4) * T
    H  = L * np.sqrt(abs(c2) + np.sqrt(abs(c4)))
    return c1 - H, c1 + H

def call_cos_coefficients(a, b, k):
    """
    Cosine coefficients V_k for call payoff g(y) = (e^y - 1)^+
    integrated over [c, d] = [0, b]  (assuming a < 0).

    From Fang & Oosterlee (2008) eq. (20)-(21):

        chi_k = (1/(1+kp^2)) * [ e^d*(cos(kp*(d-a)) + kp*sin(kp*(d-a)))
                                - e^c*(cos(kp*(c-a)) + kp*sin(kp*(c-a))) ]

        psi_k = [sin(kp*(d-a)) - sin(kp*(c-a))] / kp    k > 0
              = d - c = b                                  k = 0

    With c=0, d=b:
        kp*(d-a) = k*pi  →  cos(k*pi)=(-1)^k, sin(k*pi)=0
        kp*(c-a) = -kp*a →  must be evaluated explicitly  ← BUG WAS HERE
    """
    bw = b - a
    kp = k * np.pi / bw

    # Upper bound (y=b): kp*(b-a) = k*pi
    upper = np.exp(b) * np.cos(k * np.pi)         # e^b * (-1)^k

    # Lower bound (y=0): kp*(0-a) = -kp*a
    lower = np.cos(-kp * a) + kp * np.sin(-kp * a)

    chi = np.where(
        k == 0,
        np.exp(b) - 1.0,
        (1.0 / (1.0 + kp**2)) * (upper - lower),
    )

    psi = np.where(
        k == 0,
        b,
        (np.sin(kp * bw) - np.sin(-kp * a)) / kp,   # = sin(kp*a)/kp
    )

    return (2.0 / bw) * (chi - psi)

def price_call_cos(K, T, sigma, lam, mu_j, sigma_j, n=N_COS):
    """
    COS formula (Fang & Oosterlee 2008, eq. 9):
        C = K * e^{-rT} * sum_k' Re[phi(kpi/(b-a)) * e^{ikpi(x-a)/(b-a)}] * V_k
    where x = log(S0/K) converts CF of log(S_T/S_0) to log(S_T/K).
    """
    a, b = cos_truncation(T, sigma, lam, mu_j, sigma_j)
    k    = np.arange(n, dtype=float)
    u    = k * np.pi / (b - a)
    x    = np.log(S0 / K)
    phi  = merton_cf(u, T, sigma, lam, mu_j, sigma_j)
    V    = call_cos_coefficients(a, b, k)
    series    = np.real(phi * np.exp(1j * k * np.pi * (x - a) / (b - a)))
    series[0] *= 0.5
    return max(K * np.exp(-r * T) * np.dot(series, V), 0.0)

def price_put_cos(K, T, sigma, lam, mu_j, sigma_j, n=N_COS):
    C = price_call_cos(K, T, sigma, lam, mu_j, sigma_j, n)
    return C - S0 + K * np.exp(-r * T)

def black_scholes_call(K, T, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def monte_carlo_merton(K, T, sigma, lam, mu_j, sigma_j, n_paths=200_000, seed=42):
    """
    Exact simulation (no time-step error).
    log(S_T/S_0) = drift*T + sigma*sqrt(T)*Z + sum_{i=1}^{N_T} J_i
    N_T ~ Poisson(lam*T),  J_i ~ N(mu_j, sigma_j^2)
    """
    rng        = np.random.default_rng(seed)
    drift_corr = lam * (np.exp(mu_j + 0.5 * sigma_j**2) - 1.0)
    log_ret    = ((r - 0.5 * sigma**2 - drift_corr) * T
                  + sigma * np.sqrt(T) * rng.standard_normal(n_paths))
    n_jumps = rng.poisson(lam * T, n_paths)
    max_n   = max(n_jumps.max(), 1)
    all_J   = rng.normal(mu_j, sigma_j, (n_paths, max_n))
    mask    = np.arange(max_n) < n_jumps[:, None]
    log_ret += (all_J * mask).sum(axis=1)
    S_T    = S0 * np.exp(log_ret)
    payoff = np.maximum(S_T - K, 0.0)
    price  = np.exp(-r * T) * payoff.mean()
    se     = np.exp(-r * T) * payoff.std() / np.sqrt(n_paths)
    return price, se

STRIKES    = [80, 90, 100, 110, 120]
MATURITIES = [0.25, 0.5, 1.0, 2.0]

def sep(title=""):
    print("\n" + "─"*68)
    if title:
        print(f"  {title}"); print("─"*68)

def test_bs_limit():
    sep("TEST 1 — Black-Scholes Limit  (lambda=0, no jumps)")
    sigma, lam, mu_j, sigma_j = 0.20, 0.0, 0.0, 0.10
    print(f"  sigma={sigma}   r={r}   S0={S0}\n")
    print(f"  {'K':>5} {'T':>5} {'COS':>10} {'BS':>10} {'|Error|':>10}")
    print(f"  {'-'*5} {'-'*5} {'-'*10} {'-'*10} {'-'*10}")
    max_err = 0.0
    for T in MATURITIES:
        for K in STRIKES:
            cos = price_call_cos(K, T, sigma, lam, mu_j, sigma_j)
            bs  = black_scholes_call(K, T, sigma)
            err = abs(cos - bs)
            max_err = max(max_err, err)
            print(f"  {K:>5} {T:>5.2f} {cos:>10.5f} {bs:>10.5f} {err:>10.2e}")
    result = "PASS ✓" if max_err < 1e-4 else "FAIL ✗"
    print(f"\n  Max error = {max_err:.2e}   [{result}]")
    return max_err

def test_put_call_parity():
    sep("TEST 2 — Put-Call Parity")
    sigma, lam, mu_j, sigma_j = 0.20, 0.50, -0.05, 0.10
    print(f"  sigma={sigma}  lam={lam}  mu_j={mu_j}  sigma_j={sigma_j}\n")
    print(f"  {'K':>5} {'T':>5} {'C-P':>10} {'S0-Ke-rT':>10} {'|Error|':>10}")
    print(f"  {'-'*5} {'-'*5} {'-'*10} {'-'*10} {'-'*10}")
    max_err = 0.0
    for T in MATURITIES:
        for K in STRIKES:
            C   = price_call_cos(K, T, sigma, lam, mu_j, sigma_j)
            P   = price_put_cos( K, T, sigma, lam, mu_j, sigma_j)
            lhs = C - P
            rhs = S0 - K * np.exp(-r * T)
            err = abs(lhs - rhs)
            max_err = max(max_err, err)
            print(f"  {K:>5} {T:>5.2f} {lhs:>10.5f} {rhs:>10.5f} {err:>10.2e}")
    result = "PASS ✓" if max_err < 1e-4 else "FAIL ✗"
    print(f"\n  Max error = {max_err:.2e}   [{result}]")
    return max_err

def test_monte_carlo():
    sep("TEST 3 — Monte Carlo Cross-Check  (200k paths)")
    sigma, lam, mu_j, sigma_j = 0.20, 0.50, -0.05, 0.10
    n_paths = 200_000
    print(f"  sigma={sigma}  lam={lam}  mu_j={mu_j}  sigma_j={sigma_j}")
    print(f"  MC paths = {n_paths:,}\n")
    print(f"  {'K':>5} {'T':>5} {'COS':>10} {'MC':>10} {'CI [MC±3se]':>22} {'in band':>8}")
    print(f"  {'-'*5} {'-'*5} {'-'*10} {'-'*10} {'-'*22} {'-'*8}")
    fails = 0
    for T in [0.5, 1.0, 2.0]:
        for K in [90, 100, 110]:
            cos    = price_call_cos(K, T, sigma, lam, mu_j, sigma_j)
            mc, se = monte_carlo_merton(K, T, sigma, lam, mu_j, sigma_j, n_paths)
            lo, hi = mc - 3*se, mc + 3*se
            ok     = lo <= cos <= hi
            if not ok: fails += 1
            ci = f"[{lo:.4f}, {hi:.4f}]"
            print(f"  {K:>5} {T:>5.2f} {cos:>10.5f} {mc:>10.5f} {ci:>22}  {'✓' if ok else '✗'}")
    result = "PASS ✓" if fails == 0 else f"FAIL ✗  ({fails} outside band)"
    print(f"\n  {result}")
    return fails

def test_convergence():
    sep("EXTRA — Convergence in N  (reference = N=512)")
    sigma, lam, mu_j, sigma_j = 0.20, 0.50, -0.05, 0.10
    K, T = 100.0, 1.0
    ref  = price_call_cos(K, T, sigma, lam, mu_j, sigma_j, n=512)
    print(f"  K={K}  T={T}  Reference (N=512) = {ref:.8f}\n")
    print(f"  {'N':>5} {'Price':>14} {'|Error|':>12}")
    print(f"  {'-'*5} {'-'*14} {'-'*12}")
    for n in [8, 16, 32, 64, 128, 256]:
        p   = price_call_cos(K, T, sigma, lam, mu_j, sigma_j, n=n)
        print(f"  {n:>5} {p:>14.8f} {abs(p-ref):>12.2e}")

if __name__ == "__main__":
    print("="*68)
    print("  COS Pricer Validation — Merton JDM  (corrected)")
    print("="*68)
    print(f"  S0={S0}  r={r}  N_COS={N_COS}")
    e1 = test_bs_limit()
    e2 = test_put_call_parity()
    f3 = test_monte_carlo()
    test_convergence()
    sep("SUMMARY")
    print(f"  Test 1  BS limit     max|error| = {e1:.2e}   {'PASS ✓' if e1<1e-4 else 'FAIL ✗'}")
    print(f"  Test 2  Parity       max|error| = {e2:.2e}   {'PASS ✓' if e2<1e-4 else 'FAIL ✗'}")
    print(f"  Test 3  Monte Carlo  fails      = {f3}        {'PASS ✓' if f3==0 else 'FAIL ✗'}")
    print()
