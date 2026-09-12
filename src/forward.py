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

    actual = daily[daily.portfolio_id != "Benchmark"].copy()
    returns = actual.pivot(index="date", columns="portfolio_id", values="net_return").iloc[1:]
    starts = actual.groupby("portfolio_id").nav.last()
    if len(returns) < block:
        raise ValueError("Not enough completed return history for the selected bootstrap block")

    dates = pd.bdate_range(returns.index.max() + pd.offsets.BDay(), periods=days)
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    bands, summaries = [], []
    for offset, portfolio_id in enumerate(returns.columns):
        rng = np.random.default_rng(config["seed"] + offset)
        series = returns[portfolio_id].to_numpy(dtype=float)
        blocks_needed = int(np.ceil(days / block))
        block_starts = rng.integers(0, len(series), size=(paths, blocks_needed))
        offsets = np.arange(block)
        sampled_index = (block_starts[:, :, None] + offsets).reshape(paths, -1) % len(series)
        sampled_returns = series[sampled_index[:, :days]]
        values = starts[portfolio_id] * np.cumprod(1 + sampled_returns, axis=1)
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
