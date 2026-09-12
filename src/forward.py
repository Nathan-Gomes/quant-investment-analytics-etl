"""Forward-looking portfolio scenarios built from completed historical returns."""

import numpy as np
import pandas as pd


def forward_scenarios(daily, config):
    """Create deterministic block-bootstrap projections without using future data.

    The simulation starts immediately after the final backtest date. Consecutive
    blocks of realized net returns preserve short-run market behaviour while the
    percentile bands communicate uncertainty rather than a point prediction.
    """
    paths = int(config["simulation_paths"])
    days = int(config["simulation_years"] * config["trading_days"])
    block = int(config["bootstrap_block_days"])
    if paths < 100 or days < 1 or block < 1:
        raise ValueError("Invalid forward-scenario configuration")

    actual = daily.sort_values(["date", "portfolio_id"]).copy()
    returns = actual.pivot(index="date", columns="portfolio_id", values="net_return").iloc[1:]
    initial_capital = float(config["initial_capital"])
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("Initial scenario capital must be positive and finite")
    starts = pd.Series(initial_capital, index=returns.columns)
    if len(returns) < block:
        raise ValueError("Not enough completed return history for the selected bootstrap block")

    # Map 252 modeled sessions per year to an exact calendar horizon, not an exchange calendar.
    dates = pd.date_range(returns.index.max(), returns.index.max() + pd.DateOffset(years=int(config["simulation_years"])), periods=days + 1)
    rng = np.random.default_rng(config["seed"])
    blocks_needed = int(np.ceil(days / block))
    block_starts = rng.integers(0, len(returns) - block + 1, size=(paths, blocks_needed))
    sampled_index = (block_starts[:, :, None] + np.arange(block)).reshape(paths, -1)[:, :days]
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    bands, summaries = [], []
    for portfolio_id in returns.columns:
        series = returns[portfolio_id].to_numpy(dtype=float)
        sampled_returns = series[sampled_index]
        values = starts[portfolio_id] * np.cumprod(1 + sampled_returns, axis=1)
        values = np.column_stack([np.full(paths, starts[portfolio_id]), values])
        percentile_values = np.quantile(values, quantiles, axis=0)
        bands.append(pd.DataFrame({
            "date": dates,
            "portfolio_id": portfolio_id,
            "p05": percentile_values[0],
            "p25": percentile_values[1],
            "median": percentile_values[2],
            "p75": percentile_values[3],
            "p95": percentile_values[4],
        }))
        drawdowns = values / np.maximum.accumulate(values, axis=1) - 1
        terminal = values[:, -1]
        summaries.append({
            "portfolio_id": portfolio_id,
            "as_of_date": returns.index.max(),
            "start_value": starts[portfolio_id],
            "terminal_p05": np.quantile(terminal, 0.05),
            "terminal_median": np.quantile(terminal, 0.50),
            "terminal_p95": np.quantile(terminal, 0.95),
            "probability_terminal_loss": np.mean(terminal < starts[portfolio_id]),
            "median_max_drawdown": np.quantile(drawdowns.min(axis=1), 0.50),
            "scenario_paths": paths,
            "horizon_years": config["simulation_years"],
            "bootstrap_block_days": block,
        })
    return pd.concat(bands, ignore_index=True), pd.DataFrame(summaries)
