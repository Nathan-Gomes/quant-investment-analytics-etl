"""Backtest and forward-scenario engine for user-defined portfolios.

This is the research pipeline's method applied to an arbitrary universe. The
conventions in ``src/analytics.py`` and ``src/forward.py`` are preserved
exactly:

* a portfolio's daily gross return is the previous close's weights times each
  asset's current return, so no holding is ever set using the return it earns;
* weights drift between rebalances, and a rebalance charges
  ``pre-cost NAV x sum of absolute weight changes x bps / 10,000``;
* the reported return series is net of those costs;
* forward scenarios are a block bootstrap of the portfolio's own completed net
  returns, never a price forecast, and every portfolio is sampled on the same
  block positions so a comparison is paired rather than a lucky draw.

What is generalized: the universe, the weights, the window, the rebalance
schedule and the horizon all come from the request instead of ``config.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

TRADING_DAYS = 252
SCHEDULES = ("none", "monthly", "quarterly", "annual")
SCHEMES = ("custom", "equal", "inverse_volatility", "optimized")


@dataclass
class Settings:
    """Everything the engine needs that is not price data."""

    initial_capital: float = 100_000.0
    transaction_cost_bps: float = 10.0
    risk_free_rate: float = 0.03
    rebalance: str = "monthly"
    benchmark: str | None = None
    calibration_days: int = 252
    horizon_years: float = 5.0
    paths: int = 5000
    block_days: int = 20
    seed: int = 42
    scenario_basis: str = "equal"  # "equal" capital, or "continuation" of final NAV
    grid_points: int = 240

    def validate(self) -> None:
        if not np.isfinite(self.initial_capital) or self.initial_capital <= 0:
            raise ValueError("Starting amount must be a positive number.")
        if self.transaction_cost_bps < 0 or self.transaction_cost_bps > 500:
            raise ValueError("Transaction cost must be between 0 and 500 basis points.")
        if self.risk_free_rate <= -1 or self.risk_free_rate > 1:
            raise ValueError("Risk-free rate must be between -100% and 100%.")
        if self.rebalance not in SCHEDULES:
            raise ValueError(f"Rebalance schedule must be one of {', '.join(SCHEDULES)}.")
        if not 0.5 <= self.horizon_years <= 30:
            raise ValueError("Scenario horizon must be between 0.5 and 30 years.")
        if not 200 <= self.paths <= 20000:
            raise ValueError("Scenario count must be between 200 and 20,000.")
        if self.block_days < 1:
            raise ValueError("Bootstrap block must be at least one session.")
        if self.scenario_basis not in ("equal", "continuation"):
            raise ValueError("Scenario basis must be 'equal' or 'continuation'.")


@dataclass
class Portfolio:
    """A named allocation over some subset of the loaded universe.

    ``scheme`` decides where the weights come from. ``custom`` uses what was
    entered; ``equal`` and ``inverse_volatility`` are rules; ``optimized`` hands
    the choice to ``app/optimize.py``, re-solved at every rebalance from a
    trailing window, with the settings below.
    """

    name: str
    weights: dict[str, float] = field(default_factory=dict)
    scheme: str = "custom"
    objective: str = "minimum_variance"
    estimation_days: int = 252
    max_weight: float = 1.0
    estimator: str = "ledoit_wolf"
    volatility_target: float | None = None
    parameters: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.scheme not in SCHEMES:
            raise ValueError(f"{self.name}: unknown weighting scheme '{self.scheme}'.")
        if self.scheme == "optimized":
            from .strategies import REGISTRY
            if self.objective not in REGISTRY:
                raise ValueError(
                    f"{self.name}: unknown methodology '{self.objective}'. "
                    f"Registered: {', '.join(sorted(REGISTRY))}."
                )
            if not 60 <= self.estimation_days <= 2520:
                raise ValueError(f"{self.name}: the estimation window must be between 60 and 2,520 sessions.")
            from .riskmodel import MODELS
            if self.estimator not in MODELS:
                raise ValueError(
                    f"{self.name}: risk model must be one of {', '.join(MODELS)}.")

    def resolve(self, columns: Sequence[str], calibration_returns: pd.DataFrame) -> pd.Series:
        if self.scheme == "equal":
            chosen = list(self.weights) or list(columns)
            raw = pd.Series(1.0 / len(chosen), index=chosen)
        elif self.scheme == "inverse_volatility":
            chosen = list(self.weights) or list(columns)
            deviation = calibration_returns[chosen].std(ddof=1)
            if not np.isfinite(deviation).all() or (deviation <= 0).any():
                raise ValueError(
                    f"{self.name}: inverse-volatility weights need a non-zero, finite "
                    "volatility for every holding in the calibration window."
                )
            inverse = 1.0 / deviation
            raw = inverse / inverse.sum()
        else:
            raw = pd.Series(self.weights, dtype=float)
            if raw.empty:
                raise ValueError(f"{self.name}: needs at least one holding.")
            if not np.isfinite(raw).all() or (raw < 0).any():
                raise ValueError(f"{self.name}: weights must be finite and not negative.")
            total = raw.sum()
            if total <= 0:
                raise ValueError(f"{self.name}: weights must add up to more than zero.")
            raw = raw / total  # normalized, and the rescaling is reported back
        unknown = set(raw.index) - set(columns)
        if unknown:
            raise ValueError(f"{self.name}: no price data loaded for {', '.join(sorted(unknown))}.")
        return raw.reindex(columns, fill_value=0.0)


# --------------------------------------------------------------------------- #
# Price handling
# --------------------------------------------------------------------------- #

def to_wide(prices: pd.DataFrame) -> pd.DataFrame:
    """Long price records to a date-by-ticker frame on shared trading days.

    Tickers from different exchanges do not share a holiday calendar, so the
    frame keeps only the dates on which every requested ticker traded. Dropping
    a mismatched day is preferred to carrying a stale price forward, which would
    invent a zero-return session. The count of dropped days is reported.
    """
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["adjusted_price"] = pd.to_numeric(frame["adjusted_price"], errors="raise")
    if frame.duplicated(["date", "ticker"]).any():
        conflicting = frame[frame.duplicated(["date", "ticker"], keep=False)]
        if conflicting.groupby(["date", "ticker"]).adjusted_price.nunique().gt(1).any():
            raise ValueError("The price source returned conflicting values for the same ticker and date.")
        frame = frame.drop_duplicates(["date", "ticker"])
    wide = frame.pivot(index="date", columns="ticker", values="adjusted_price").sort_index()
    if not np.isfinite(wide.to_numpy(dtype=float, na_value=np.nan)[~wide.isna().to_numpy()]).all():
        raise ValueError("The price source returned a non-finite price.")
    if (wide.fillna(1) <= 0).any().any():
        raise ValueError("Adjusted prices must be positive.")
    return wide


def align(wide: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep the dates every ticker traded, and describe what was removed."""
    complete = wide.dropna(how="any")
    per_ticker = wide.notna().sum().to_dict()
    quality = {
        "calendar_days_in_window": int(len(wide)),
        "shared_trading_days": int(len(complete)),
        "days_dropped_for_missing_prices": int(len(wide) - len(complete)),
        "observations_by_ticker": {k: int(v) for k, v in per_ticker.items()},
        "first_shared_date": complete.index.min().date().isoformat() if len(complete) else None,
        "last_shared_date": complete.index.max().date().isoformat() if len(complete) else None,
    }
    return complete, quality


def rebalance_mask(index: pd.DatetimeIndex, schedule: str) -> np.ndarray:
    """True on the first observed session of each new month, quarter or year."""
    if schedule == "none":
        return np.zeros(len(index), dtype=bool)
    if schedule == "monthly":
        key = index.year * 12 + index.month
    elif schedule == "quarterly":
        key = index.year * 4 + index.quarter
    else:
        key = index.year.to_numpy()
    key = np.asarray(key)
    changed = np.empty(len(index), dtype=bool)
    changed[0] = False
    changed[1:] = key[1:] != key[:-1]
    return changed


# --------------------------------------------------------------------------- #
# Backtest
# --------------------------------------------------------------------------- #

def metrics(returns: Iterable[float], risk_free_rate: float = 0.03) -> dict:
    """Identical definitions to ``src/analytics.py`` so results stay comparable."""
    series = pd.Series(list(returns), dtype=float).dropna()
    if series.empty:
        raise ValueError("No return observations in the selected window.")
    wealth = (1 + series).cumprod()
    drawdown = wealth / wealth.cummax().clip(lower=1) - 1
    deviation = series.std(ddof=1)
    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    downside = series[series < rf_daily]
    downside_deviation = downside.std(ddof=1) if len(downside) > 1 else np.nan
    return {
        "cumulative_return": float(wealth.iloc[-1] - 1),
        "annualized_return": float(wealth.iloc[-1] ** (TRADING_DAYS / len(series)) - 1),
        "volatility": float(deviation * np.sqrt(TRADING_DAYS)),
        "sharpe": float((series.mean() - rf_daily) / deviation * np.sqrt(TRADING_DAYS)) if deviation > 0 else float("nan"),
        "sortino": float((series.mean() - rf_daily) / downside_deviation * np.sqrt(TRADING_DAYS))
        if downside_deviation and np.isfinite(downside_deviation) and downside_deviation > 0 else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "best_day": float(series.max()),
        "worst_day": float(series.min()),
        "positive_days": float((series > 0).mean()),
        "observations": int(len(series)),
    }


def run_backtest(wide: pd.DataFrame, target: pd.Series, settings: Settings, schedule: str | None = None,
                 target_provider=None) -> dict:
    """Walk the window one session at a time, exactly as the pipeline does.

    ``target_provider`` turns a fixed allocation into a rule. It is called with
    the position of the session whose close is being traded and must return the
    weights to hold from that close onward, so a walk-forward optimizer can
    re-estimate at each rebalance. It is only ever given data through that
    close; the engine never shows it a future return.
    """
    schedule = schedule or settings.rebalance
    columns = list(wide.columns)
    prices = wide.to_numpy(dtype=float)
    returns = np.zeros_like(prices)
    returns[1:] = prices[1:] / prices[:-1] - 1
    if target_provider is not None:
        target = pd.Series(target_provider(0), index=columns)
    weights = target.to_numpy(dtype=float).copy()
    target_array = weights.copy()
    flags = rebalance_mask(wide.index, schedule)
    target_history = [{"date": wide.index[0], "weights": target_array.copy()}]

    days = len(wide)
    nav = np.empty(days)
    net = np.zeros(days)
    gross = np.zeros(days)
    turnover = np.zeros(days)
    cost = np.zeros(days)
    contribution = np.zeros(len(columns))
    exposure_sum = np.zeros(len(columns))
    nav[0] = settings.initial_capital

    for i in range(1, days):
        r = returns[i]
        step = float(weights @ r)
        before_cost = nav[i - 1] * (1 + step)
        contribution += weights * r
        exposure_sum += weights
        drifted = weights * (1 + r) / (1 + step) if (1 + step) != 0 else weights
        if flags[i]:
            if target_provider is not None:
                # Re-estimated from data through this close, never beyond it.
                target_array = np.asarray(target_provider(i), dtype=float)
                target_history.append({"date": wide.index[i], "weights": target_array.copy()})
            traded = float(np.abs(target_array - drifted).sum())
            charge = before_cost * traded * settings.transaction_cost_bps / 10_000
        else:
            traded, charge = 0.0, 0.0
        nav[i] = before_cost - charge
        if nav[i] <= 0:
            raise ValueError("The portfolio was wiped out inside the window; check the inputs.")
        gross[i], turnover[i], cost[i] = step, traded, charge
        net[i] = nav[i] / nav[i - 1] - 1
        weights = target_array.copy() if flags[i] else drifted

    exposure_sum += weights  # include the final session's holdings
    summary = metrics(net[1:], settings.risk_free_rate)
    peak = np.maximum.accumulate(nav)
    summary.update(
        final_value=float(nav[-1]),
        total_cost=float(cost.sum()),
        total_turnover=float(turnover.sum()),
        rebalances=int(flags.sum()),
        years=float((days - 1) / TRADING_DAYS),
    )
    return {
        "dates": wide.index,
        "target_history": target_history,
        "nav": nav,
        "net_returns": net,
        "gross_returns": gross,
        "drawdown": nav / peak - 1,
        "cost": cost,
        "turnover": turnover,
        "summary": summary,
        "final_weights": pd.Series(weights, index=columns),
        "average_weights": pd.Series(exposure_sum / days, index=columns),
        "contribution": pd.Series(contribution, index=columns),
    }


def rolling_returns(nav: np.ndarray, window_days: int) -> dict:
    """Annualized return over every overlapping window of the given length."""
    if len(nav) <= window_days:
        return {}
    ratio = nav[window_days:] / nav[:-window_days]
    annual = ratio ** (TRADING_DAYS / window_days) - 1
    return {
        "worst": float(annual.min()),
        "median": float(np.median(annual)),
        "best": float(annual.max()),
        "share_negative": float((annual < 0).mean()),
        "windows": int(len(annual)),
    }


# --------------------------------------------------------------------------- #
# Forward scenarios
# --------------------------------------------------------------------------- #

def block_positions(sample_length: int, days: int, block: int, paths: int, seed: int) -> np.ndarray:
    """Shared block starts: every portfolio is sampled on the same positions."""
    if sample_length < block:
        raise ValueError("The selected window has fewer completed sessions than the bootstrap block.")
    rng_high = sample_length - block + 1
    from .rng import Mulberry32

    generator = Mulberry32(seed)
    blocks_needed = int(np.ceil(days / block))
    starts = generator.integers(rng_high, paths * blocks_needed).reshape(paths, blocks_needed)
    index = (starts[:, :, None] + np.arange(block)).reshape(paths, -1)[:, :days]
    return index


def scenario_statistics(
    net_returns: np.ndarray,
    index: np.ndarray,
    start_value: float,
    floor_fraction: float = 0.8,
    grid_points: int = 240,
) -> dict:
    """Compound sampled returns and reduce to bands, terminals and drawdowns."""
    paths, days = index.shape
    # Column 0 is the starting value, so the starting balance is the first
    # drawdown peak, exactly as in src/forward.py.
    grid = np.unique(np.linspace(0, days, min(grid_points, days + 1)).astype(int))
    terminal = np.empty(paths)
    worst_drawdown = np.empty(paths)
    trough = np.empty(paths)
    banded = np.empty((paths, len(grid)))

    # This stage is bound by memory traffic rather than arithmetic, so the work
    # is done in place: one gathered array is compounded, reused as the running
    # peak, and reused again as the drawdown, instead of allocating a 5,000 by
    # 1,260 temporary for each step. Same numbers, in float64 throughout.
    growth = 1.0 + net_returns
    chunk = max(1, int(8_000_000 / max(days, 1)))
    sampled = grid[grid > 0] - 1
    for begin in range(0, paths, chunk):
        stop = min(begin + chunk, paths)
        values = growth[index[begin:stop]]
        np.cumprod(values, axis=1, out=values)
        values *= start_value
        peak = np.maximum.accumulate(values, axis=1)
        np.maximum(peak, start_value, out=peak)
        np.divide(values, peak, out=peak)          # peak now holds 1 + drawdown
        terminal[begin:stop] = values[:, -1]
        worst_drawdown[begin:stop] = peak.min(axis=1) - 1
        trough[begin:stop] = np.minimum(start_value, values.min(axis=1))
        if grid[0] == 0:
            banded[begin:stop, 0] = start_value
            banded[begin:stop, 1:] = values[:, sampled]
        else:
            banded[begin:stop] = values[:, sampled]

    sorted_terminal = np.sort(terminal)
    quantiles = np.quantile(banded, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
    tail = terminal <= np.quantile(terminal, 0.05)
    return {
        "grid": grid,
        "bands": {
            "p05": quantiles[0], "p25": quantiles[1], "median": quantiles[2],
            "p75": quantiles[3], "p95": quantiles[4],
        },
        "terminal": terminal,
        "summary": {
            "start_value": float(start_value),
            "terminal_p05": float(np.quantile(terminal, 0.05)),
            "terminal_p25": float(np.quantile(terminal, 0.25)),
            "terminal_median": float(np.median(terminal)),
            "terminal_p75": float(np.quantile(terminal, 0.75)),
            "terminal_p95": float(np.quantile(terminal, 0.95)),
            "worst5_mean": float(terminal[tail].mean()),
            "probability_terminal_loss": float((terminal < start_value).mean()),
            "probability_ever_below_floor": float((trough < floor_fraction * start_value).mean()),
            "median_max_drawdown": float(np.median(worst_drawdown)),
            "severe_max_drawdown_p05": float(np.quantile(worst_drawdown, 0.05)),
            "paths": int(paths),
            # How much of each reported figure is the sample rather than the
            # model. A percentile estimated from a finite number of paths has a
            # confidence interval, and quoting one without it invites the reader
            # to take the last digits seriously.
            "sampling_error": {
                "terminal_p05": quantile_interval(sorted_terminal, 0.05),
                "terminal_median": quantile_interval(sorted_terminal, 0.50),
                "terminal_p95": quantile_interval(sorted_terminal, 0.95),
            },
        },
    }


def quantile_interval(sorted_values: np.ndarray, q: float, confidence: float = 0.95) -> dict:
    """A distribution-free confidence interval for a sample quantile.

    The number of observations below a quantile is binomial, so an interval can
    be read straight off the order statistics without assuming a shape for the
    distribution — which matters here, because the terminal values are skewed by
    construction and a normal approximation would understate the upper tail.
    """
    from scipy.stats import binom

    if not 0 < q < 1 or not 0 < confidence < 1:
        raise ValueError("Quantile and confidence must be strictly between zero and one.")
    n = len(sorted_values)
    if n < 30:
        return {"low": None, "high": None, "width": None}
    alpha = 1 - confidence
    low = int(binom.ppf(alpha / 2, n, q)) - 1
    high = int(binom.ppf(1 - alpha / 2, n, q))
    if low < 0 or high >= n:
        return {"low": None, "high": None, "width": None}
    estimate = float(np.quantile(sorted_values, q))
    return {
        "confidence": confidence,
        "low": float(sorted_values[low]),
        "high": float(sorted_values[high]),
        # As a fraction of the estimate, which is how a reader should judge it.
        "width": float((sorted_values[high] - sorted_values[low]) / estimate) if estimate else None,
    }


def paired_comparison(left: np.ndarray, right: np.ndarray, left_name: str, right_name: str) -> dict:
    """Compare two portfolios on the same simulated sequences, path by path."""
    difference = np.asarray(left) - np.asarray(right)
    ratio = np.asarray(left) / np.asarray(right)
    return {
        "portfolio": left_name,
        "versus": right_name,
        "probability_ahead": float((difference > 0).mean()),
        "difference_p05": float(np.quantile(difference, 0.05)),
        "difference_median": float(np.median(difference)),
        "difference_p95": float(np.quantile(difference, 0.95)),
        "ratio_median": float(np.median(ratio)),
    }


def scenario_dates(last_date: pd.Timestamp, days: int, horizon_years: float) -> pd.DatetimeIndex:
    """Model coordinates: 252 sessions a year mapped onto the calendar horizon."""
    end = last_date + pd.DateOffset(days=int(round(horizon_years * 365.25)))
    return pd.date_range(last_date, end, periods=days + 1)
