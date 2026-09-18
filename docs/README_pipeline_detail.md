# Pipeline reference

Detailed notes on the research pipeline: portfolio definitions, financial
conventions, the supplemental volatility experiment, the SQL mart and the
validation rules. The project README summarizes these; this is the long form.

## Portfolios

| Portfolio | Construction |
| --- | --- |
| Growth | 65% technology, with bank and railway exposure |
| Income | Banks, energy and utilities; not a dividend cash-flow model |
| Balanced | Diversified equity sectors; no bond allocation |
| Low volatility | Inverse-volatility weights calibrated from the initial 252 sessions |
| Benchmark | XIC.TO, a broad Canadian equity ETF |

Editable `data/holdings.csv` contains target weights for the first three portfolios. Low-volatility weights are calculated using only the calibration history, frozen, and used at subsequent rebalances. It is an inverse-volatility heuristic, not a minimum-variance optimizer. The initial calibration is excluded from all reported portfolio performance.

## Financial conventions

- Prices are CAD adjusted closes, approximating dividend-reinvested total returns. Adjusted units are research accounting units, not actual broker shares. No external deposits or withdrawals are modeled.
- Daily return: current adjusted close / previous adjusted close - 1.
- Portfolio gross return: sum of previous-close weight times current asset return.
- Rebalance at the close of the first observed session of a new month. Allocations drift between rebalances. Costs equal pre-cost NAV times sum of absolute weight changes times basis points / 10,000. Both buy and sell notional count. Initial funding costs are excluded consistently.
- Net return: post-cost NAV / previous NAV - 1. Sector weights use current post-rebalance positions.
- Annualized return: compounded wealth raised to 252 / observed return days, minus 1.
- Volatility: sample daily standard deviation times square root of 252.
- Sharpe: mean daily return minus equivalent daily risk-free rate, divided by sample standard deviation, times square root of 252. It is undefined for zero volatility.
- Drawdown: NAV / running peak NAV - 1, including starting capital in the peak. Maximum drawdown is its minimum.
- Excess annualized return is the difference between portfolio and benchmark CAGR, not regression alpha.

## Supplemental volatility experiment

Linear regression, Ridge, Lasso and ElasticNet predict next-20-session realized annualized volatility, not future asset prices. This supplemental experiment is not used to create the five-year scenarios or select a portfolio. Features include trailing returns, trailing volatility, benchmark return and benchmark volume relative to its rolling mean. Features use information through the prediction close only.

The first 80% of available observations is the development period and the final 20% is untouched holdout. A 20-session purge prevents forward labels crossing a boundary. Four-fold `TimeSeriesSplit` with the same gap drives `GridSearchCV`. `StandardScaler` is fitted inside each fold through an sklearn Pipeline. Hyperparameters and the displayed model are selected on CV MSE, never on test performance. Predictions are clipped at zero. A persistence forecast of trailing 20-day volatility is evaluated on the identical test rows. CSVs retain predictions, RMSE, R-squared, coefficients, selected parameters and split dates.

Forecasts do not alter holdings. Overlapping targets make errors dependent; this project does not claim statistical significance, predictive profitability, or that any model must beat persistence. Negative test R-squared is reported honestly.

## SQL and validation

The mart contains security analytics, position facts, daily NAV, sector exposures, research holdings, target weights, model outputs and run history. `sql/analysis_queries.sql` demonstrates joins, CTEs, aggregations, `LAG`, `ROW_NUMBER` and `DENSE_RANK`. Monthly returns exclude the first partial month when no preceding month-end exists.

Validation rejects nonpositive/nonfinite prices, missing prices, conflicting duplicates, unknown holdings and weights that fail to sum to one. Exact duplicates are removed and counted. Position values and weights reconcile to NAV. Tests exercise hand-calculated compounding, initial losses, fees, lagged weights, future-data invariance, forward label construction and failed database publication. CI tests and executes a synthetic smoke run without depending on a market API.

## Limitations

The chosen present-day universe introduces survivorship and selection bias. Adjusted history can be revised by the provider. This is not point-in-time institutional data. Shared absent dates across every security cannot be detected without an exchange calendar; per-security date gaps are rejected. No FX, taxes, delistings, market impact beyond fixed basis points, or actual dividend payments are modeled. A constant risk-free rate is an assumption. Sector classifications are manually supplied. All portfolios are long-only equities and have materially different exposures. Historical performance does not establish future suitability.

## References

- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
- [scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [yfinance source and documentation](https://github.com/ranaroussi/yfinance)

Market-data access and redistribution remain subject to the provider's terms. This repository is a local research demonstration, not a trading service.
