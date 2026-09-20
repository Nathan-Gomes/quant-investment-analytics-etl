# Adding a methodology

The path a construction rule takes from a rough idea in a notebook to something
that can run unattended. It's the workflow the rest of the app is built around,
and the reason the code is shaped the way it is.

## The contract

A methodology is a function from a `Context` to a weight vector, registered with
metadata. That is the whole interface.

```python
from app.strategies import Context, Strategy, register
from app.optimize import minimum_variance

def inverse_variance(context: Context):
    deviation = (context.covariance()[0].diagonal()) ** 0.5
    weights = (1 / deviation) ** 2
    return weights / weights.sum()

register(Strategy(
    name="inverse_variance",
    label="Inverse variance",
    description="Weights proportional to the reciprocal of each holding's variance.",
    solve=inverse_variance,
    author="r.chen",
))
```

Nothing else changes. The API advertises it at `/api/strategies`, the interface
builds its menu from that list, the walk-forward loop can run it, and the
conformance harness picks it up. `test_a_methodology_can_be_added_without_touching_the_engine`
holds that promise in place.

### What a `Context` gives you, and what it deliberately does not

| Available | Why |
| --- | --- |
| `returns` | the trailing estimation window, most recent row last |
| `constraints` | the mandate: long-only, fully invested, cap per holding |
| `risk_model(kind)` | sample, Ledoit-Wolf or statistical factor, built once and shared |
| `mandate()` | position bounds, group limits and a turnover budget, assembled from the request |
| `covariance()` | the risk model as a plain matrix, for methodologies that want one |
| `expected_returns()` | shrunk means, for the rules that need them |
| `previous_weights` | the portfolio being moved from, for cost-aware rules |
| `risk_free_rate`, `transaction_cost_bps` | the economics |
| `parameters` | whatever the methodology declares it needs |

There is no access to prices after the window, to the identity of the assets, to
the benchmark, or to the rest of the backtest. A methodology that needs one of
those is asking for something the live system would not have at that moment, and
that is the conversation to have before the code is written rather than after
the backtest looks good.

### The mandate isn't set here

`mandate()` supplies the limits everything already runs under: position bounds,
sector and country limits, a turnover budget. They apply to whatever gets built.

What it deliberately does **not** give you is a tracking-error ceiling or a cost
term, because those change what is being optimized rather than constraining it.
A methodology that wants them asks explicitly, with `ctx.mandate(tracking_error_limit=...)`.
This distinction came out of a real defect: parameters intended for one
methodology were being read by every other, so "minimum variance" was quietly
solving a benchmark-relative problem. `test_the_mandate_does_not_inherit_objectives_from_the_parameter_bag`
holds the line.

### Convex where it can be

New objectives should be written as convex programs and handed to the solver in
`app/convex.py` rather than to a local search. You get a certified optimum, duals
that price each constraint, and a real infeasibility answer. See
[`OPTIMIZATION.md`](OPTIMIZATION.md); the risk-parity rewrite there is the
argument in one example.

If your objective genuinely is not convex, say so in the strategy's description
and expect the conformance harness to be strict about determinism.

## The gate

```sh
python -m app.conformance                       # everything registered
python -m app.conformance --strategy risk_parity --verbose
python -m app.conformance --json                # for CI
```

Fifteen checks, run against identical data for every methodology so that a new
arrival is held to what already shipped. Non-zero exit on failure, and it runs
in CI beside the unit tests.

| Check | The failure it is looking for |
| --- | --- |
| produces weights, finite, fully invested, long only | the obvious ones, which still happen |
| honours a weight cap | a constraint satisfied on the author's data by luck |
| deterministic | an unseeded draw, making a backtest unreproducible and an incident uninvestigable |
| independent of asset order | an index assumed rather than carried — the classic production bug |
| scale invariant | a hidden dependence on the units returns arrive in |
| survives degenerate data | perfectly correlated holdings, a halted name with zero variance, one extreme print, a short window |
| weight stability | advisory: how far the answer moves when the window is resampled |
| runs within budget | a solve that is fine once and hopeless inside a walk-forward loop |
| numerical tolerance | declared per strategy: a local search reproduces to machine precision, an interior-point solver to its convergence tolerance, and the difference is a property of the method rather than something to wave through |
| scales to 60 assets | a method that only works on the eight names it was written for |

Two of these earn their keep immediately. The asset-order check fails any rule
that assumes column zero is a particular holding, which is invisible in a
notebook where the order never changes. The runtime check failed
`maximum_sharpe` on its first run at 1,070 ms against a 750 ms budget; the fix
was to stop using multi-start restarts on the convex subproblems inside the
frontier sweep, since a positive-definite quadratic over a convex set has one
optimum and restarts buy nothing there. That is a five-fold speed-up with no
change to any result.

The suite in `tests/application/test_strategies.py` defines methodologies that are broken in
each of these ways and asserts the harness fails them on the check that names
the fault. A gate nobody has tried to fool is decoration.

## The runtime gate

`solve_with_diagnostics` re-checks the weights every single time, in production,
before anything acts on them: finite, summing to one, non-negative, inside the
cap. The harness proves a methodology behaved on the data it was given; this
proves it behaved on the data it actually got. The specific failure it exists to
catch is a solver that quietly returns its last iterate when it fails to
converge — the result looks like a portfolio and is not one.

## What the app reports back to the researcher

- **Shrinkage intensity actually applied**, so the amount of structure imposed on
  the covariance is visible rather than assumed.
- **Weight next to risk contribution**, because ten holdings can be one bet.
- **Weight stability**: the 5th-95th percentile range of each weight across
  resampled windows. On the shipped universe, maximum Sharpe's widest holding
  spans roughly 50 percentage points and risk parity's spans 2. That single
  comparison explains why practitioners reach for risk-based methods more often
  than the textbook frontier suggests they should.
- **Turnover and the cost it incurred**, per rebalance and per year.
- **Estimated against realized volatility**, which is usually an uncomfortable
  ratio and should be.
- **The in-sample frontier with realized outcomes plotted against it**: the gap
  is the cost of not knowing the future.

## A worked example: pricing the trade into the objective

`minimum_variance_net_of_costs` was added through the contract, with no change to
the engine. It adds the round-trip cost of each move to the objective, so the
portfolio only changes when the variance saved is worth paying for.

On the bundled universe at 10 bps, against plain minimum variance:

| | Turnover a year | Costs paid | Sharpe |
| --- | ---: | ---: | ---: |
| Minimum variance | 1.40 | $1,698 | 0.79 |
| Net of trading | 0.43 | $563 | 0.76 |

Turnover falls by two thirds and the risk-adjusted return is slightly lower in
this sample. The penalty parameter was **not** tuned: choosing it by maximizing
Sharpe on the same window is precisely the overfitting the rest of this app
exists to expose, and the honest report of a middling result is worth more than
a tuned one. If it is to be tuned, it should be chosen on a period that is then
never used to evaluate it.

## The finding to expect

Walk-forward minimum variance earns a lower Sharpe ratio than equal weighting on
the bundled universe, and trades about three times as much. This is DeMiguel,
Garlappi and Uppal (2009), reproduced out of sample. It is not a defect in the
optimizer. It is what estimation error does to one: with a handful of assets and
a few years of daily data, the differences between candidate portfolios are
frequently smaller than the error in the inputs that generated them.

This system is built to surface that rather than to flatter the method.
