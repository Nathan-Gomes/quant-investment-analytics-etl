# Strata

An interactive front end for the research pipeline. Enter any set of tickers and
weights, pick a window, and get the realized backtest plus the block-bootstrap
scenario range — the same method the pipeline applies to its four fixed
portfolios, generalized to whatever universe you ask for.

## Run it with live prices

```sh
cd quant-investment-analytics-etl-main
python3 -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-app.txt
python -m app.server
```

Open <http://127.0.0.1:8000>.

The older `PORTFOLIO_LAB_*` environment variables still work, so a deployment
configured before the rename keeps running untouched. **Price data** at the top of the study-window
controls is already set to Yahoo Finance, so type any symbol Yahoo knows —
`AAPL`, `NVDA`, `BNS.TO`, `VTI`, `BTC-USD` — and it fetches the history on the
first run.

Symbols follow Yahoo's conventions: `.TO` for Toronto, `.V` for the TSX Venture,
`.L` for London, and no suffix for US listings. If a symbol comes back empty the
app says so and names it rather than guessing a substitute.

The first run for a new ticker takes a few seconds. After that it is served from
`app/cache/`, which holds each ticker's full history plus a provenance record,
and refreshes when it is more than twelve hours old. Delete the folder to force
a clean pull.

With no network, switch **Price data** to the bundled dataset, or start the
server so that is the only option:

```sh
STRATA_SOURCE=bundled python -m app.server
```

There is also a single-file build that needs no server at all:

```sh
python tools/build_demo.py     # writes dist/strata-demo.html
```

That file carries the nine frozen securities and runs the whole engine in the
browser. It is for showing the tool to someone without asking them to install
Python; for any ticker outside the frozen set, run the served app.

## Speed, and what the app tells you about it

Every response carries `meta.timings_ms`, a breakdown by stage, and each stage is
reported to the browser as it finishes over `POST /api/analyze/stream` —
newline-delimited JSON, because the request is a POST and `EventSource` cannot
make one. The interface shows the stage, a progress bar and a running clock, and
falls back to the plain endpoint if the stream is unavailable.

The scenario stage reuses float64 arrays to reduce memory traffic. Dense
minimum-variance problems reuse compiled solver problems when their constraint
structure matches. Factor models retain their factor formulation. The compiled
cache is bounded to 24 entries, and parameter assignment and solving are locked
so concurrent requests cannot overwrite one another's inputs.

Identical studies reuse a bounded result cache, after loading and fingerprinting
the prices. Its key includes the resolved dates, request, prices, security
metadata, data source and source-code digest. `STRATA_RESULT_CACHE` sets its size
(default 24; zero disables it); `/api/health` reports `cached_results`.
`meta.result_cache_hit` identifies reuse. Cached `meta.timings_ms` describes the
original calculation, not the current request's latency. Downloads can still
take time even when the analysis result is cached.

Actual speed depends on the window, strategies, paths and host. Cold starts and
Yahoo throttling remain possible; no paid hosting changes are required by this
update. A persistent price-cache disk requires a separate hosting decision.

### Sampling precision

The default remains 5,000 paths. Live-server results include a 95% confidence
interval for each estimated 5th, 50th and 95th percentile, using binomial order
statistics rather than assuming normally distributed terminal wealth. These
intervals measure Monte Carlo sampling error conditional on the bootstrap model,
not the model's accuracy or uncertainty about future markets. Their width varies
by portfolio and inputs; there is no universal percentage error at 5,000 paths.
The offline JavaScript demo retains its existing percentile outputs without
these additional intervals.

## What the interface shows

| Plate | What it answers |
| --- | --- |
| One continuous picture | What the money did, then where the scenarios spread |
| What the window actually did | Return, volatility, Sharpe, Sortino, drawdown, costs paid, gap to the benchmark |
| Depth of the holes | Value against its own running peak, for every portfolio at once |
| Where N years could land | Median, 5th and 95th percentile, chance of ending below the start, typical and severe drawdowns, head-to-head odds against the benchmark |
| Inside *portfolio* | Target versus drifted weights, each holding's contribution, sector mix, correlation between holdings |
| Assumptions | Every choice the engine made, in plain sentences |

Clicking a portfolio in the legend or the metrics table makes it the focus: the
scenario fan, the callouts and the composition plate all follow it.

## How a request is answered

1. **Prices.** yfinance adjusted closes, cached per ticker under `app/cache/`
   with a provenance record, or the frozen CSV. A download failure is reported;
   it is never replaced with simulated data.
2. **Alignment.** Only dates on which *every* holding traded are kept. Tickers
   from different exchanges do not share a holiday calendar, and dropping a
   mismatched day is safer than carrying a stale price forward, which would
   invent a zero-return session. The count of dropped days is shown.
3. **Weights.** As entered (normalized, in either percent or fraction form),
   equal, or inverse volatility calibrated on a leading window that is then
   excluded from every reported result.
4. **Backtest.** Previous-close weights times current returns; drift between
   rebalances; `pre-cost NAV x sum of absolute weight changes x bps / 10,000`
   charged at each rebalance; the reported series is net of those costs.
5. **Scenarios.** A block bootstrap of that portfolio's own completed net
   returns. Every portfolio receives the identical block positions, so a
   comparison between them is paired rather than a lucky draw.


## Portfolio construction

Beyond fixed weights, the app will solve for them and re-solve at every
rebalance. Pick an objective under **Weighting**; the holdings you list become
the universe and the weights you typed are ignored.

| Objective | Needs a return forecast | What it solves for |
| --- | --- | --- |
| Minimum variance | no | the lowest-variance mix |
| Risk parity | no | weights where every holding supplies the same share of risk |
| Maximum diversification | no | the largest gap between holdings' own risk and the portfolio's |
| Maximum Sharpe | yes | the tangency portfolio on the estimated frontier |

Constraints are long-only and fully invested, with an optional cap per holding.
A cap that cannot be spread across the universe is rejected with the arithmetic
explained rather than quietly relaxed.

### Three decisions worth defending

**Covariance is shrunk, not sampled.** With 25 assets and a year of daily data,
a sample covariance matrix fits 325 parameters on 252 observations. Its smallest
eigenvalues are biased toward zero, and those are exactly the directions a
variance minimizer loads into, so the optimizer's confidence is largest where
the estimate is worst. The default is Ledoit-Wolf shrinkage toward a scaled
identity, with the intensity estimated from the data and reported in the
interface so the amount of structure imposed is visible. Setting the estimator
to `sample` reproduces the unshrunk behaviour for comparison.

**Expected returns are shrunk harder, and mostly avoided.** Estimating a mean to
the precision an optimizer needs takes decades. Three of the four objectives ask
for no return forecast at all; maximum Sharpe does, and its inputs are pulled
toward the cross-sectional mean by a James-Stein factor before the solve.

**Everything is walk-forward.** At each rebalance the optimizer sees only the
trailing estimation window, solves, and the result is held forward. The
efficient frontier drawn in the results is the deliberate exception: it is
fitted to the whole window and labelled as hindsight, so the realized dots
plotted beneath it show the distance between what a perfect forecast would have
allowed and what the rules actually delivered.

### Adding another methodology

Construction rules live in a registry, not in the engine. Writing one function
and registering it puts it in the API, the interface menu and the walk-forward
loop; `python -m app.conformance` then holds it to the same fifteen checks as
everything already shipped. `docs/ADDING_A_METHODOLOGY.md` covers the contract,
the checks and why each one exists.

### What the results plate reports

Shrinkage intensity, effective number of bets, diversification ratio, estimated
versus realized volatility, turnover per year and the cost it incurred, each
holding's share of money next to its share of risk, and the full weight history
so the churn of a rule is visible rather than summarized.

A finding worth expecting: on the bundled universe, walk-forward minimum
variance earns a lower Sharpe ratio than equal weighting and trades roughly
three times as much. That is the DeMiguel, Garlappi and Uppal (2009) result,
reproduced here out of sample. It is not a defect in the optimizer; it is what
estimation error does to one, and the app is built to show it rather than to
flatter the method.

## Adding a methodology

Construction rules live in a registry rather than in the engine. See
`docs/ADDING_A_METHODOLOGY.md` for the contract, the conformance battery every
methodology passes, and a worked example that prices trading costs into the
objective.

## Reproducibility

`app/rng.py` and `static/engine.js` implement the same mulberry32 generator, so
a given seed selects the same blocks in Python and in the browser. Two tests
hold that in place: `test_prng_matches_javascript` checks the generator bit for
bit, and `test_browser_engine_agrees_with_python` runs both engines over the
same request and compares the results.

The Yahoo path is covered by tests that substitute a stand-in provider, so the
download, caching, profile lookup and failure messages are exercised without a
network and without depending on the provider being up.

`tests/application/test_optimize.py` checks the solvers against closed forms where one
exists — the two-asset minimum-variance weight, equal risk contributions under
risk parity, the exact Euler decomposition of volatility, and the equality of
weight share and risk share at a minimum-variance optimum — and the shrinkage
estimator against scikit-learn's implementation of the same paper. Its most
important test rewrites the final month of prices and asserts that every earlier
weight and NAV is unchanged, because a walk-forward rule that can see forward is
not walk-forward.

`test_app_reproduces_the_published_pipeline_results` checks the app against the
numbers in `output/portfolio_summary.csv`. If a change to the app moves Growth's
annualized return away from 27.795%, that test fails. The app and the research
are meant to stay the same calculation.

## API

`POST /api/analyze`

```json
{
  "portfolios": [
    {"name": "My five", "weights": {"RY.TO": 25, "ENB.TO": 20, "CNR.TO": 20, "FTS.TO": 15, "SHOP.TO": 20}, "scheme": "custom"},
    {"name": "Equal", "weights": {"RY.TO": 1, "ENB.TO": 1, "CNR.TO": 1, "FTS.TO": 1, "SHOP.TO": 1}, "scheme": "equal"}
  ],
  "benchmark": "XIC.TO",
  "start": "2016-09-01", "end": "2026-09-01",
  "initial_capital": 100000, "transaction_cost_bps": 10, "risk_free_rate": 0.03,
  "rebalance": "monthly", "horizon_years": 10, "paths": 5000,
  "block_days": 20, "seed": 42, "scenario_basis": "equal", "source": "auto"
}
```

Up to six portfolios and 25 tickers per request. Schemes: `custom`, `equal`,
`inverse_volatility`. Schedules: `monthly`, `quarterly`, `annual`, `none`.
Bases: `equal` (same starting balance for every portfolio, so terminal values
compare construction) or `continuation` (each portfolio carries on from its own
final value, which answers a wealth question instead). Errors come back as HTTP
400 with a sentence explaining what to change.

Every number the interface draws is in the response, and **Export results**
downloads it, so a reviewer can check a chart against the payload.

## Limits worth stating to anyone you show this to

- **Survivorship and selection bias.** Tickers are chosen today, knowing which
  ones survived and did well. That flatters any backtest and nothing here
  corrects for it. It is the single largest reason a good-looking backtest
  overstates what was achievable.
- **The fan is not a forecast.** It resamples one window's return distribution,
  including its luck and its regime. Sampling blocks keeps volatility clustering
  but assumes the future is drawn from the same distribution as the past. The
  median line at successive dates is not a path anyone could have held.
- **No FX.** Mixing `AAPL` with `.TO` names measures each in its own currency.
  The app warns; it does not convert.
- **Not modelled.** Taxes, inflation, cash dividends, delistings, market impact
  beyond the fixed spread, bid-ask spreads that widen in stress, or any change
  in the companies themselves.
- **Adjusted prices** approximate a dividend-reinvested total return and can be
  revised by the provider. This is not point-in-time institutional data.
- **A constant risk-free rate** across a decade is an assumption, and Sharpe is
  sensitive to it.

This is research tooling for studying historical data under stated assumptions.
It is not investment advice, and no output is a recommendation to buy or sell
anything.
