# Review guide: Quantitative Investment Analytics ETL Pipeline

## The question

How did four deliberately different Canadian equity allocations behave against XIC after comparable transaction costs, and what range of five-year outcomes follows when their completed historical return patterns are resampled?

## What this project demonstrates

This is a quantitative research and decision-support pipeline. It turns raw price and holdings data into validated portfolio accounting, historical return and drawdown comparisons, and equal-capital five-year scenario ranges. It does not claim to know which portfolio will win or predict a specific future price.

## A 10-minute walkthrough

1. Read the summary and assumptions in the [README](../README.md).
2. Inspect [config.json](../config.json) for the starting capital, costs, rebalancing, historical window, five-year horizon, 5,000 paths, and random seed.
3. Read [analytics.py](../src/analytics.py) for portfolio return, NAV, cost, and drawdown calculations.
4. Read [forward.py](../src/forward.py) for the five-year block-bootstrap scenario engine.
5. Read [scenario_risk.py](../src/scenario_risk.py) for downside, paired Growth-versus-Balanced, and sampling-sensitivity measures.
6. Open `output/report.html` or `output/notebook.html` for the evidence a non-technical reviewer can scan.
7. Review `tests/research/test_pipeline.py` for calculations, data validation, chronology, and reproducibility checks.

## Decision flow

```text
Frozen prices + declared holdings
        -> input validation
        -> monthly portfolio accounting and costs
        -> historical return, volatility, and drawdown comparison
        -> equal CAD 100,000 five-year scenario paths
        -> downside and sensitivity review
        -> HTML report, CSV exports, and SQLite research mart
```

## Why the five-year scenarios are not machine-learning forecasts

The engine samples consecutive 20-session blocks from completed net-return history and compounds them for five years. Keeping days together preserves short-run clustering such as turbulent and calm stretches. The same sampled market sequence is applied across portfolios, so Growth and Balanced can be compared in like-for-like scenarios. The output is a transparent range of historically conditional outcomes, not a single price target or a claim of prediction.

## Timing and bias controls

Historical returns use prior-close positions. Low-volatility weights are calibrated from the first 252 sessions only and then frozen. Rebalances affect only later days, while scenario paths begin after the final completed historical backtest date. Future scenario outcomes never feed back into a historical allocation decision.

The project still discloses selection and survivorship risk: the stock universe and allocation rules were selected retrospectively, and adjusted-price data is not an institutional point-in-time archive.

## What to discuss

- Growth earned the highest return in the chosen sample, but it also had the deepest drawdown and wider adverse scenario paths.
- Balanced had the strongest observed risk-adjusted result, which may matter more to a mandate with a drawdown budget.
- Equal starting capital makes the future scenario comparison fair; a separate historical-continuation view answers a different question about wealth already accumulated.
- The scenario outputs help compare risk tolerances and set questions for further research. They do not select a universally best allocation.
