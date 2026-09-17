"""Turns a request into the payload the interface draws.

One request describes a universe, one or more allocations and a set of
assumptions. This module loads prices, runs each allocation through the
backtest, runs every allocation through one shared set of bootstrap blocks, and
returns plain JSON-ready structures. It also collects the caveats that a
reviewer should see next to the numbers rather than buried in a footnote.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import engine
from .engine import Portfolio, Settings

ENGINE_VERSION = "1.0.0"
MAX_CURVE_POINTS = 900


def _thin(length: int, limit: int = MAX_CURVE_POINTS) -> np.ndarray:
    if length <= limit:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, limit).astype(int))


def _round(values, digits: int = 4) -> list:
    array = np.asarray(values, dtype=float)
    array = np.where(np.isfinite(array), array, np.nan)
    return [None if not np.isfinite(v) else round(float(v), digits) for v in array]


def _clean(value):
    """JSON has no NaN. Report an undefined statistic as null instead."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def analyze(
    prices: pd.DataFrame,
    portfolios: list[Portfolio],
    settings: Settings,
    metadata: pd.DataFrame | None = None,
    start: str | None = None,
    end: str | None = None,
    source_note: str = "",
) -> dict:
    settings.validate()
    if not portfolios:
        raise ValueError("Add at least one portfolio.")

    warnings: list[str] = []
    wide_all = engine.to_wide(prices)
    if start:
        wide_all = wide_all[wide_all.index >= pd.Timestamp(start)]
    if end:
        wide_all = wide_all[wide_all.index <= pd.Timestamp(end)]
    wide, quality = engine.align(wide_all)
    dropped = quality["days_dropped_for_missing_prices"]
    if dropped:
        warnings.append(
            f"{dropped} session{'s' if dropped != 1 else ''} were dropped because at least one "
            "holding had no price that day. Tickers on different exchanges do not share a holiday "
            "calendar, and a dropped day is safer than carrying a stale price forward."
        )
    if len(wide) < 60:
        raise ValueError(
            "Fewer than 60 shared trading days are available. Widen the date range or remove the "
            "ticker with the shortest history."
        )

    needs_calibration = any(p.scheme == "inverse_volatility" for p in portfolios)
    warmup = min(settings.calibration_days, len(wide) // 3) if needs_calibration else 0
    if needs_calibration and warmup < settings.calibration_days:
        warnings.append(
            f"Inverse-volatility weights were calibrated on {warmup} sessions rather than "
            f"{settings.calibration_days}, because the window is short. Those calibration sessions "
            "are excluded from every reported result."
        )
    returns_all = wide.pct_change(fill_method=None)
    calibration = returns_all.iloc[1:warmup + 1] if warmup else returns_all.iloc[1:]
    window = wide.iloc[warmup:]
    if len(window) < 40:
        raise ValueError("Too little history remains after the calibration window. Widen the date range.")

    meta = _metadata_lookup(metadata, wide.columns)
    currencies = {meta[t]["currency"] for t in wide.columns if meta[t].get("currency")}
    if len(currencies) > 1:
        warnings.append(
            "The universe mixes " + " and ".join(sorted(currencies)) + ". Returns are measured in "
            "each security's own currency and no FX conversion is applied, so a portfolio return "
            "here is not what a single-currency investor would have experienced."
        )

    benchmark_name = None
    if settings.benchmark:
        if settings.benchmark not in wide.columns:
            raise ValueError(f"No price data loaded for the benchmark {settings.benchmark}.")
        benchmark_name = f"{settings.benchmark} benchmark"
        portfolios = list(portfolios) + [
            Portfolio(name=benchmark_name, weights={settings.benchmark: 1.0}, scheme="custom")
        ]

    results = []
    for portfolio in portfolios:
        requested_total = sum(portfolio.weights.values()) if portfolio.scheme == "custom" else 1.0
        target = portfolio.resolve(list(wide.columns), calibration)
        # Weights arrive either as fractions (0.35) or as percentages (35). Both
        # are normalized; only a total that is neither is worth flagging.
        if portfolio.scheme == "custom" and min(abs(requested_total - 1), abs(requested_total - 100)) > 0.005:
            warnings.append(
                f"{portfolio.name}: the weights you entered do not add up to a whole portfolio, "
                "so they were rescaled to 100% while keeping their proportions."
            )
        schedule = "none" if portfolio.name == benchmark_name else settings.rebalance
        run = engine.run_backtest(window, target, settings, schedule=schedule)
        results.append((portfolio, target, run))

    # One set of block positions, shared by every portfolio: the differences
    # between them come from construction, not from a luckier sample.
    horizon_days = int(round(settings.horizon_years * engine.TRADING_DAYS))
    sample_length = len(window) - 1
    block = min(settings.block_days, sample_length)
    if block < settings.block_days:
        warnings.append(
            f"The bootstrap block was shortened to {block} sessions because the window holds only "
            f"{sample_length} completed returns."
        )
    index = engine.block_positions(sample_length, horizon_days, block, settings.paths, settings.seed)
    forward_dates = engine.scenario_dates(window.index[-1], horizon_days, settings.horizon_years)

    scenarios = {}
    for portfolio, _target, run in results:
        start_value = (
            float(run["summary"]["final_value"]) if settings.scenario_basis == "continuation"
            else float(settings.initial_capital)
        )
        scenarios[portfolio.name] = engine.scenario_statistics(
            run["net_returns"][1:], index, start_value, grid_points=settings.grid_points
        )

    terminal_pool = np.concatenate([s["terminal"] for s in scenarios.values()])
    hist_range = (float(np.quantile(terminal_pool, 0.005)), float(np.quantile(terminal_pool, 0.98)))
    edges = np.linspace(hist_range[0], max(hist_range[1], hist_range[0] * 1.01), 41)

    benchmark_terminal = scenarios[benchmark_name]["terminal"] if benchmark_name else None
    benchmark_annual = None
    if benchmark_name:
        benchmark_annual = next(
            run["summary"]["annualized_return"] for p, _t, run in results if p.name == benchmark_name
        )

    payload_portfolios = []
    for order, (portfolio, target, run) in enumerate(results):
        dates = run["dates"]
        keep = _thin(len(dates))
        scenario = scenarios[portfolio.name]
        held = target[target > 0]
        counts, _ = np.histogram(scenario["terminal"], bins=edges)
        sectors = (
            pd.Series({t: meta[t]["sector"] for t in target.index})
            .to_frame("sector").assign(weight=run["final_weights"].values)
            .groupby("sector").weight.sum().sort_values(ascending=False)
        )
        summary = dict(run["summary"])
        summary["excess_annualized_return"] = (
            summary["annualized_return"] - benchmark_annual if benchmark_annual is not None else None
        )
        summary["tracking_error"] = _tracking_error(run, results, benchmark_name)
        payload_portfolios.append({
            "name": portfolio.name,
            "is_benchmark": portfolio.name == benchmark_name,
            "scheme": portfolio.scheme,
            "order": order,
            "weights": {k: round(float(v), 6) for k, v in held.sort_values(ascending=False).items()},
            "summary": summary,
            "rolling": {
                "one_year": engine.rolling_returns(run["nav"], engine.TRADING_DAYS),
                "three_year": engine.rolling_returns(run["nav"], engine.TRADING_DAYS * 3),
            },
            "curve": {
                "nav": _round(run["nav"][keep], 2),
                "drawdown": _round(run["drawdown"][keep], 5),
            },
            "contribution": [
                {
                    "ticker": ticker,
                    "contribution": round(float(run["contribution"][ticker]), 5),
                    "average_weight": round(float(run["average_weights"][ticker]), 5),
                    "final_weight": round(float(run["final_weights"][ticker]), 5),
                }
                for ticker in held.index
            ],
            "sectors": [{"sector": s, "weight": round(float(w), 5)} for s, w in sectors.items()],
            "scenario": {
                "bands": {k: _round(v, 2) for k, v in scenario["bands"].items()},
                "summary": scenario["summary"],
                "histogram": {"counts": [int(c) for c in counts]},
            },
            "versus_benchmark": (
                engine.paired_comparison(
                    scenario["terminal"], benchmark_terminal, portfolio.name, benchmark_name
                )
                if benchmark_terminal is not None and portfolio.name != benchmark_name else None
            ),
        })

    first = results[0][2]
    keep = _thin(len(first["dates"]))
    asset_rows, asset_returns = _asset_table(window, meta, settings)
    correlation = asset_returns.corr()

    return _clean({
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "engine_version": ENGINE_VERSION,
            "source": source_note,
            "window_start": window.index[0].date().isoformat(),
            "window_end": window.index[-1].date().isoformat(),
            "trading_days": int(len(window)),
            "years": round(float((len(window) - 1) / engine.TRADING_DAYS), 2),
            "calibration_days": int(warmup),
            "benchmark": benchmark_name,
            "settings": {
                "initial_capital": settings.initial_capital,
                "transaction_cost_bps": settings.transaction_cost_bps,
                "risk_free_rate": settings.risk_free_rate,
                "rebalance": settings.rebalance,
                "horizon_years": settings.horizon_years,
                "paths": settings.paths,
                "block_days": block,
                "seed": settings.seed,
                "scenario_basis": settings.scenario_basis,
            },
        },
        "quality": quality,
        "warnings": warnings,
        "history": {"dates": [d.date().isoformat() for d in first["dates"][keep]]},
        "forward": {
            "dates": [forward_dates[i].date().isoformat() for i in scenarios[results[0][0].name]["grid"]],
            "histogram_edges": _round(edges, 2),
        },
        "assets": asset_rows,
        "correlation": {
            "tickers": list(correlation.columns),
            "matrix": [[round(float(v), 3) for v in row] for row in correlation.to_numpy()],
        },
        "portfolios": payload_portfolios,
    })


def _tracking_error(run, results, benchmark_name) -> float | None:
    if not benchmark_name:
        return None
    benchmark = next((r for p, _t, r in results if p.name == benchmark_name), None)
    if benchmark is None or run is benchmark:
        return None
    difference = run["net_returns"][1:] - benchmark["net_returns"][1:]
    return float(np.std(difference, ddof=1) * np.sqrt(engine.TRADING_DAYS))


def _metadata_lookup(metadata: pd.DataFrame | None, tickers) -> dict:
    lookup = {t: {"name": t, "sector": "Unclassified", "currency": None} for t in tickers}
    if metadata is not None and len(metadata):
        for row in metadata.itertuples():
            if row.ticker in lookup:
                lookup[row.ticker] = {
                    "name": getattr(row, "name", row.ticker) or row.ticker,
                    "sector": getattr(row, "sector", "Unclassified") or "Unclassified",
                    "currency": getattr(row, "currency", None),
                }
    return lookup


def _asset_table(window: pd.DataFrame, meta: dict, settings: Settings):
    returns = window.pct_change(fill_method=None).iloc[1:]
    rows = []
    for ticker in window.columns:
        stats = engine.metrics(returns[ticker], settings.risk_free_rate)
        rows.append({
            "ticker": ticker,
            "name": meta[ticker]["name"],
            "sector": meta[ticker]["sector"],
            "currency": meta[ticker]["currency"],
            "annualized_return": stats["annualized_return"],
            "volatility": stats["volatility"],
            "sharpe": stats["sharpe"],
            "max_drawdown": stats["max_drawdown"],
            "first_price": float(window[ticker].iloc[0]),
            "last_price": float(window[ticker].iloc[-1]),
        })
    return rows, returns
