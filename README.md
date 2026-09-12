# Quantitative Investment Analytics ETL Pipeline

## Five-year conditional scenarios

The historical chart continues from its last NAV into pointwise median scenarios. A second chart rebases all portfolios to 100 at the projection date; separate panels show 25th-75th and 5th-95th percentile ranges on identical scales. Colors match the original historical notebook. XIC is included in both comparisons.

The engine samples the same complete 20-session historical blocks across all portfolios, preserving contemporaneous dependence. It compounds 1,260 modeled returns over exactly five calendar years, including starting NAV as the initial drawdown peak. Dates are model coordinates rather than an exchange calendar. Sampled net returns retain historical costs; future holdings and trading costs are not recomputed. CSV tables and SQLite retain the bands and summary. A fixed seed makes the run reproducible.

These scenarios repeat the sample's return distribution, including its strong growth. They are not independently estimated expected returns, validated five-year forecasts, or guarantees. Parameter uncertainty, unobserved regimes, taxes and inflation are omitted. Medians at successive dates are not one investable path.

Look-ahead controls apply to backtest timing and regression validation: previous-close weights, pre-investment inverse-volatility calibration, chronological folds, purged forward labels and training-only scaling. Today's security universe and manually selected policies still create selection and survivorship bias; the research does not claim an entirely unbiased backtest. Scenario information never enters prior historical decisions.

[Live case study](https://www.nathan-gomes.com/Project-Investment-Analytics.dc.html) · [Interactive report](https://www.nathan-gomes.com/investment-analytics/output/report.html) · [Executed notebook](https://www.nathan-gomes.com/investment-analytics/output/notebook.html)

A reproducible Python, pandas and SQL research pipeline that compares four Canadian equity allocations against XIC, models transaction costs, and evaluates next-period volatility forecasts without leaking future data into training.

## What it found

This sample begins with CAD 100,000 on 3 January 2019 and ends on 31 August 2026. Values use adjusted prices and modeled monthly-rebalance costs.

| Portfolio | Annualized return | Volatility | Sharpe | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: |
| Growth | 27.80% | 28.60% | 0.90 | -48.72% |
| Balanced | 19.12% | 16.29% | 0.97 | -28.73% |
| Low volatility | 17.57% | 15.29% | 0.94 | -29.71% |
| Income | 16.13% | 16.06% | 0.83 | -33.06% |
| XIC benchmark | 16.22% | 16.37% | 0.82 | -37.21% |

Growth had the highest raw return, while Balanced had the strongest observed risk-adjusted return. Ridge regression reduced held-out volatility forecast error against trailing-volatility persistence for Growth, Balanced and Low volatility, but not Income. All selected models had negative test R-squared, so the project reports a mixed forecasting result rather than claiming that the model can reliably predict markets.

## Project structure

```text
data/        Frozen prices, holdings, and security metadata
src/         Extract, validation, analytics, modeling, reporting, and SQLite loading
sql/         Views using joins, CTEs, aggregations, and window functions
tests/       Calculation, data-quality, chronology, and publication tests
notebooks/   Executed research notebook with saved visualizations
output/      Reproducible report, run manifest, and compact result exports
```

## Run

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline --source cached
python -m pytest -q
python scripts/build_notebook.py
```

Open `output/report.html` directly in a browser. `output/notebook.html` is the executed research notebook; `notebooks/portfolio_exploration.ipynb` opens in JupyterLab. All report charts work offline. CSV exports work with Tableau or Power BI, and `output/investment_analytics.db` is the SQLite research mart.

To refresh Yahoo Finance data use `python -m src.pipeline --source download`. To run an explicitly labeled simulated example use `--source synthetic`. Download failures never silently substitute simulated returns. The default cached run checks the frozen CSV checksum. Configure dates, starting capital, transaction costs, risk-free rate and validation settings in `config.json`, or pass `--config path.json`.

## Architecture

CSV/Yahoo prices + holdings + security master -> strict validation -> pandas security analytics -> monthly portfolio simulation -> purged regression validation -> CSV/HTML report -> atomic SQLite publication.

Modules are deliberately separate: `extract.py`, `validation.py`, `analytics.py`, `models.py`, `report.py`, `load.py`, and `pipeline.py`. The pipeline prints eight stage updates, logs failures, and records successful run provenance and input hashes. Repeated successful runs retain run history in SQLite. Database replacement is atomic; report/CSV artifacts are not a transactional snapshot, so after a failed run use the last successful database and rerun before relying on exported files.

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

## Forecast experiment

Linear regression, Ridge, Lasso and ElasticNet predict next-20-session realized annualized volatility, not future asset prices. Features include trailing returns, trailing volatility, benchmark return and benchmark volume relative to its rolling mean. Features use information through the prediction close only.

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
