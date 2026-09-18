# Optimization

What the construction layer does, why each formulation is written the way it is,
and what it still does not do.

## Three separable pieces

**Risk model** — how the covariance is estimated. `app/riskmodel.py`.
**Mandate** — what the portfolio is allowed to be. `app/convex.py`.
**Objective** — what it optimizes for. `app/strategies.py`.

They are separate because they belong to different people. A researcher supplies
an objective. The desk owns the mandate, and every objective inherits it without
knowing it exists. The risk model is a research choice that both depend on and
neither should hard-code.

## Risk models

| Model | Estimates | Where it fits |
| --- | --- | --- |
| `sample` | every pairwise covariance | the baseline; unusable once assets approach observations |
| `ledoit_wolf` | shrinkage toward a scaled identity | a good default at a few dozen assets |
| `statistical_factor` | `Σ = BBᵀ + D` from principal components | what makes a large universe tractable |

A 500-name sample covariance has 125,250 free parameters. Two years of daily data
gives about 500 observations per name, so most of that matrix is noise — and its
smallest eigenvalues, the directions a variance minimizer loads into hardest, are
the least trustworthy part of it. The factor model replaces it with a handful of
factors plus specific variance: estimable, invertible, and a quadratic form that
costs `k` inner products instead of an `n × n` multiply.

**The number of factors is not tuned.** It is the count of eigenvalues above the
Marchenko-Pastur upper edge, `(1 + √(n/T))²` — the largest eigenvalue a
correlation matrix of pure noise with the same shape would produce. Components
below it are indistinguishable from randomness. On data built with three factors,
the rule finds three.

Every result reports the model's condition number, its factor count, the share of
correlation structure explained, and the portfolio's variance split into common
and specific parts.

**This is not a vendor fundamental risk model.** Barra, Axioma and MSCI build
factors from fundamental and industry data with decades of point-in-time history.
These factors come from the return covariance itself. They need no licensed data
and they have no economic names.

## Convex formulations

Every problem is written in disciplined convex form and solved by a conic solver
(CLARABEL, falling back to OSQP and SCS). That buys three things a local search
cannot give: a certificate that the answer is the global optimum, dual variables
saying what each binding constraint costs, and a definite "infeasible" when the
mandate cannot be met.

| Objective | Formulation |
| --- | --- |
| Minimum variance | `min wᵀΣw` |
| Mean-variance | `min ½γwᵀΣw − μᵀw` |
| Maximum Sharpe | Schaible transform: optimize unnormalized `y` with `(μ−rf)ᵀy = 1`, then rescale |
| Maximum diversification | the same ratio transform against `σᵀy = 1` |
| Risk parity | `min ½wᵀΣw − Σ bᵢ log wᵢ` |
| Volatility target | `max μᵀw` subject to `‖Lᵀw‖ ≤ target` |
| Active return at a risk budget | `max μᵀ(w−b)` subject to `‖Lᵀ(w−b)‖ ≤ limit` |

Two of these are worth dwelling on.

**Risk parity** looks non-convex written as "equalize the risk contributions", and
the previous implementation minimized the dispersion of contributions with a
multi-start local search. The log-barrier form is convex, and its first-order
condition `Σw = b/w` *is* the equal-contribution condition. The difference is not
cosmetic: at 40 assets the local search left a spread of 3.4 percentage points
across contributions, while the convex form reaches 3.5e-6. It was degrading
silently as the universe grew. Spinu (2013); Maillard, Roncalli and Teïletche
(2010).

**Maximum Sharpe** is a ratio and not concave, but pinning the excess return to
one and rescaling turns it into a quadratic minimization. That is only valid
while every constraint is homogeneous in the unnormalized variable — a turnover
budget and a tracking-error limit are absolute, and are not. Those cases fall
back to scanning the frontier, which is a sequence of convex problems, and the
result says which route it took.

### A numerical detail that mattered

The ratio transforms pin a linear functional to one, so the optimizing variable
inherits the units of the input: quote returns in percent rather than decimals
and it shrinks a hundredfold while the solver's absolute tolerances stay put. The
answer should not depend on units, and it did, by about 1.7e-3. Normalizing the
problem scale before solving fixed it — the conformance harness caught this, on
its scale-invariance check.

## Mandates

Position bounds, group limits (sector, country, sleeve), a turnover budget, a
tracking-error ceiling, and transaction costs in the objective.

Costs are `spread × |Δw| + coefficient × |Δw|^exponent`, convex for any exponent
above one. The default 1.5 comes from the square-root law in price terms: filling
a larger share of a day's volume moves the price against you, so cost per unit
traded rises with size. It sits in the objective rather than being subtracted
afterwards, so the cost of reaching a portfolio is weighed against the risk it
saves inside one problem.

**Infeasibility is diagnosed, not merely reported.** A cap that cannot fill the
book, a floor that exceeds the portfolio, group caps that cannot cover it, and a
turnover budget below the trading a mandate forces — each returns the arithmetic:

```
No portfolio satisfies this mandate: reaching a portfolio inside a 10% cap from
the current one requires at least 1.00 of turnover, against a budget of 0.25.
```

**Cardinality is a heuristic and says so.** Holding at most `k` names is a
combinatorial constraint; written exactly it is a mixed-integer program, and this
system carries no MIP solver. The two-pass heuristic — solve, keep the largest
positions, re-solve on that subset — gives a sparse portfolio and is not
guaranteed optimal. It reports that in its own diagnostics.

## Scale

Minimum variance, 750 days of history, on this machine:

| Assets | Dense (Ledoit-Wolf) | Factor model |
| ---: | ---: | ---: |
| 50 | 9 ms | 7 ms |
| 200 | 36 ms | 8 ms |
| 500 | 177 ms | 15 ms |

The local search, for comparison, took 540 ms at 200 assets from a single start,
and is run from five.

## What is still missing

- **Point-in-time institutional data.** Yahoo adjusted closes are revised, survivorship-biased, and have no delisting returns. Fixing this needs a CRSP or Compustat licence; it is not an engineering gap and pretending otherwise would be worse than stating it.
- **Fundamental factor models.** The statistical model has no economic interpretation, so there is no "value exposure" to report.
- **Exact cardinality**, for want of a MIP solver.
- **Intraday liquidity.** The impact model has no ADV input, so it prices the shape of the cost curve without its level.
- **Long-short and leverage.** Everything here is long-only and fully invested.
