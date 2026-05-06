# CF-Informed Neural Calibration of Stochastic Volatility Models

End-to-end pipeline for calibrating jump-diffusion and stochastic-volatility option-pricing models to live S&P 500 (SPY) market data, using the COS Fourier method for pricing and physics-informed neural networks for parameter inversion.

The framework progresses through three models of increasing complexity — **Merton**, **Heston**, and **Bates** — and demonstrates that calibration time can be reduced from seconds to milliseconds while quantifying the structural limits of the affine model class on real index option surfaces.

---

## Headline Results

Calibration to a live SPY surface (S₀ = 711.58, 7 strikes × 5 maturities):

| Model | # parameters | Network calibration | RMSE (vol pts) |
|---|---|---|---|
| Merton | 4 | ~22 ms | 1.82 |
| Heston | 5 | ~6 ms | 1.71 |
| Bates  | 8 | ~1 ms | **1.42** |

The neural network calibration runs **roughly 1000× faster** than cold-start L-BFGS-B (which takes 30–100 seconds), with comparable or better fit quality. Each model successively reduces RMSE, and each one hits the same parameter bound (wanting faster skew decay than the model class can produce) — the empirical signature that motivates rough-volatility extensions in the literature.

---

## Why CF + COS

For affine models like Heston, the price density has no closed form but the **characteristic function** does. The Fang & Oosterlee (2008) COS method recovers option prices as a finite cosine series of CF evaluations, with exponential convergence in the series length.

Three properties matter for this project:

1. **Speed** — full surface pricing in milliseconds
2. **Modularity** — adding jumps to Heston is a one-line CF multiplication (Lévy-Khintchine independence), where it would require switching from PDE to PIDE solvers in the time-domain approach
3. **Differentiability** — the COS series can be implemented in PyTorch so gradients flow through the pricer, enabling physics-informed losses that wouldn't be possible with PDE-based pricing

---

## Models

### Merton Jump Diffusion (1976)
```
dS_t = (r - λκ_J) S_t dt + σ S_t dW_t + S_t (J - 1) dN_t,    log(J) ~ N(μ_J, σ_J²)
```
Four parameters: `(σ, λ, μ_J, σ_J)`. CF available in closed form via Lévy-Khintchine.

### Heston Stochastic Volatility (1993)
```
dS_t = r S_t dt + √v_t S_t dW^S_t
dv_t = κ(θ - v_t) dt + σ_v √v_t dW^v_t,    d⟨W^S, W^v⟩_t = ρ dt
```
Five parameters: `(v_0, κ, θ, σ_v, ρ)`. CF in exponential-affine form; implemented using the Albrecher et al. (2007) "little Heston trap" formulation to avoid branch-cut instabilities.

### Bates (1996)
Heston + Merton: stochastic variance with compound Poisson jumps overlaid on the price. Eight parameters total. The CF factorizes neatly by Lévy-Khintchine independence:

```
phi_Bates(u, T) = exp(i u r T) · phi_Heston(u, T) · phi_jump(u, T)
```

---

## Pipeline

```
yfinance SPY chain
       │
       ▼
   Filter / clean         (drop crossed spreads, deep OTM, sub-intrinsic quotes)
       │
       ▼
   Implied vol inversion  (Brent's method on Black-Scholes)
       │
       ▼
   Arbitrage check        (calendar spread, butterfly convexity)
       │
       ▼
   Surface interpolation  (bivariate spline onto 7 × 5 log-moneyness × T grid)
       │
       ▼
   Calibration            (L-BFGS-B  OR  neural network direct inversion)
       │
       ▼
   Verify / plot          (smile slices, residual heatmaps)
```

---

## ML Calibration Architecture

Each ML calibration script trains a direct-inversion MLP with two losses:

```
L = L_param  +  λ_cf · L_cf

  L_param : range-normalized MSE between predicted and true parameters
  L_cf    : MSE between the input surface and the COS-priced surface
            generated from the network's predicted parameters
```

The CF consistency loss requires a **differentiable PyTorch implementation** of the COS pricer so gradients flow back through the Fourier series into the network weights. This forces the network's outputs to be parameter values that actually reproduce the input surface, not just statistically similar to training labels.

After training (~5–30 minutes on CPU per model), inference is a single forward pass: ~1 ms per surface. An optional L-BFGS-B polish step uses the network output as a warm start and refines locally — converging in 10–40 iterations rather than the 200+ a cold start typically requires.

---

## File Structure

```
.
├── README.md
│
├── cos_pricer_validation.py      # Merton COS pricer + validation tests
├── heston_pricer_validation.py   # Heston COS pricer + validation tests
├── bates_pricer_validation.py    # Bates  COS pricer + validation tests
│
├── pipeline.py                   # Live market data → cleaned surface → cold-start calibration (Merton)
│
├── ml_calibration.py             # Merton ML calibration (direct inversion + CF physics loss)
├── heston_ml_calibration.py      # Heston ML calibration
├── bates_ml_calibration.py       # Bates  ML calibration
│
├── compare_models.py             # Three-model comparison: parameters, RMSE, timing, plots
│
└── merton_cf_calibration.py      # Early prototype that became ml_calibration.py (kept for history)
```

---

## Validation

Each pricer is validated against three benchmarks:

| Test | What it checks |
|---|---|
| **Black-Scholes limit** | Setting jump intensity / vol-of-vol to zero recovers the analytic BS price exactly |
| **Put-call parity** | `C - P = S₀ - K·exp(-rT)` must hold to machine precision regardless of model |
| **Monte Carlo** | Andersen QE scheme for Heston/Bates; exact simulation for Merton |

All three pricers pass with errors at machine precision for closed-form tests and within 3-sigma Monte Carlo bands for path simulations.

The Bates pricer is additionally validated to reduce to Merton in the limit `(σ_v → 0, ρ = 0, v_0 = θ = σ²)` and to Heston in the limit `λ = 0` — confirming the CF factorization is correct.

---

## ML Synthetic Test Performance

R² of predicted parameters on held-out synthetic test sets (1000 surfaces):

| Parameter | Heston | Bates |
|---|---|---|
| v_0 | 0.999 | 0.999 |
| kappa | 0.994 | 0.990 |
| theta | 0.996 | 0.995 |
| sigma_v | 0.998 | 0.996 |
| rho | 0.995 | 0.991 |
| lam (Bates) | – | 0.942 |
| mu_J (Bates) | – | 0.983 |
| sigma_J (Bates) | – | 0.955 |

The lower R² for Bates jump parameters is expected — at the index-option scale, jumps and stochastic-vol skew partially overlap, creating identifiability ambiguity. The CF physics loss ensures the network's *fitted surface* is still accurate even when individual parameters are not uniquely recoverable.

---

## Requirements

```
numpy
scipy
matplotlib
yfinance
torch
```

```bash
pip install numpy scipy matplotlib yfinance torch

# Run the pricer validation suites
python cos_pricer_validation.py
python heston_pricer_validation.py
python bates_pricer_validation.py

# Calibrate Merton to live SPY data via cold-start L-BFGS-B
python pipeline.py

# Train each ML calibrator (saves a .pt checkpoint, then runs SPY calibration)
python ml_calibration.py
python heston_ml_calibration.py
python bates_ml_calibration.py

# Side-by-side comparison (requires all three .pt checkpoints)
python compare_models.py
```
---

## Related Models and Research Directions

This project focuses on stochastic-volatility and jump-diffusion models that admit characteristic-function pricing via the COS method. A few related directions worth noting:

- **SABR (Hagan et al. 2002)** — the dominant per-maturity smile-interpolation model on rates and equity desks. SABR doesn't fit the CF/COS framework: its closed-form asymptotic formula gives implied vol directly, making calibration trivial via Levenberg-Marquardt (~5 ms per smile). It excels at interpolating individual smiles but lacks term-structure dynamics, so the typical desk practice is to use SABR for smile interpolation and a richer model (Heston, Bates, or rough Heston) for exotic pricing and risk.

- **Rough Heston (El Euch & Rosenbaum 2018)** — affine models hit a structural limit on SPY's short-dated skew, visible in this project as `kappa` pinned at its ceiling for both Heston and Bates calibrations. Rough Heston (Hurst index `H ≈ 0.1`) produces skew decay of `T^(H−0.5)`, matching the empirically observed `T^(−0.4)` decay on SPY. Its CF requires a fractional Riccati solver, making the ML-calibration speedup even more impactful.

- **Neural SDEs** — instead of calibrating a parametric model, train a network to *be* the diffusion directly from prices. Removes the model-selection problem at the cost of interpretability.

- **Mixture density / Bayesian networks** — output a distribution over parameters rather than point estimates, making identifiability ambiguity explicit (relevant for Bates' `lam`/`sigma_v` overlap).

---

## References

- Fang, F., & Oosterlee, C. W. (2008). *A novel pricing method for European options based on Fourier-cosine series expansions.* SIAM Journal on Scientific Computing.
- Albrecher, H., Mayer, P., Schoutens, W., & Tistaert, J. (2007). *The little Heston trap.* Wilmott Magazine.
- Andersen, L. (2008). *Simple and efficient simulation of the Heston stochastic volatility model.* Journal of Computational Finance.
- Hernandez, A. (2016). *Model calibration with neural networks.* SSRN.
- Bayer, C., & Stemper, B. (2018). *Deep calibration of rough stochastic volatility models.* arXiv:1810.03399.
- Liu, S., Borovykh, A., Grzelak, L. A., & Oosterlee, C. W. (2019). *A neural network-based framework for financial model calibration.* Journal of Mathematical Industry.
