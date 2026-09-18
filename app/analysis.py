"""Turns a request into the payload the interface draws.

One request describes a universe, one or more allocations and a set of
assumptions. This module loads prices, runs each allocation through the
backtest, runs every allocation through one shared set of bootstrap blocks, and
returns plain JSON-ready structures. It also collects the caveats that a
reviewer should see next to the numbers rather than buried in a footnote.
"""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter

import numpy as np
import pandas as pd

from . import engine, optimize, provenance, riskmodel, strategies
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
    request: dict | None = None,
    progress=None,
) -> dict:
    settings.validate()
    if not portfolios:
        raise ValueError("Add at least one portfolio.")

    started = perf_counter()
    timings: dict[str, float] = {}

    def phase(name: str, detail: str = "") -> None:
        """Record how long each stage took and, if anyone is listening, say so.

        The timings are returned with the results, so a slow run can be diagnosed
        from the payload instead of guessed at.
        """
        elapsed = (perf_counter() - started) * 1000
        timings[name] = round(elapsed - sum(timings.values()), 1)
        if progress:
            progress({"phase": name, "detail": detail, "elapsed_ms": round(elapsed, 1)})

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

    for portfolio in portfolios:
        portfolio.validate()
    phase("align", f"{len(wide)} shared sessions across {len(wide.columns)} securities")
    needs_calibration = any(p.scheme == "inverse_volatility" for p in portfolios)
    estimation_need = max([p.estimation_days for p in portfolios if p.scheme == "optimized"], default=0)
    # Any portfolio that has to be estimated before it can be held needs history
    # in front of the study window. Every portfolio then starts on the same day,
    # so the comparison stays like for like.
    requested_warmup = max(settings.calibration_days if needs_calibration else 0, estimation_need)
    warmup = min(requested_warmup, len(wide) // 3) if requested_warmup else 0
    if requested_warmup and warmup < requested_warmup:
        warnings.append(
            f"Weights were estimated on {warmup} sessions rather than the {requested_warmup} requested, "
            "because the window is short. Those sessions are used only to set the opening weights and are "
            "excluded from every reported result."
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

    columns = list(wide.columns)
    returns_matrix = returns_all.to_numpy(dtype=float)
    optimizer_state: dict[str, dict] = {}

    def make_provider(portfolio: Portfolio):
        """Weights re-solved at each rebalance from the trailing window only.

        ``position`` is the offset inside the study window of the session being
        traded. The slice ends at that session's close and never reaches past
        it, which is what makes the result walk-forward rather than fitted.
        """
        chosen = [c for c in columns if c in portfolio.weights]
        if len(chosen) < 2:
            raise ValueError(f"{portfolio.name}: optimization needs at least two holdings.")
        positions = [columns.index(c) for c in chosen]
        constraints = optimize.Constraints(max_weight=portfolio.max_weight)
        constraints.validate(len(chosen))
        state = {"previous": None, "last": None, "history": [], "universe": chosen}
        optimizer_state[portfolio.name] = state

        strategy = strategies.get(portfolio.objective)
        state["strategy"] = strategy
        state["constraints"] = constraints
        state["positions"] = positions

        def provider(position: int) -> np.ndarray:
            end = warmup + position + 1
            start = max(1, end - portfolio.estimation_days)
            sample = returns_matrix[start:end][:, positions]
            context = strategies.Context(
                returns=sample,
                constraints=constraints,
                benchmark=(np.asarray(portfolio.parameters["benchmark_weights"], dtype=float)
                           if portfolio.parameters.get("benchmark_weights") else None),
                risk_free_rate=settings.risk_free_rate,
                previous_weights=state["previous"],
                estimator=portfolio.estimator,
                volatility_target=portfolio.volatility_target,
                transaction_cost_bps=settings.transaction_cost_bps,
                effort="fast",
                parameters=dict(portfolio.parameters),
            )
            state["last_context"] = context
            solved = strategies.solve_with_diagnostics(strategy, context)
            state["last_risk_model"] = context.risk_model()
            state["last_solution"] = context._cache.get("solution")
            state["previous"] = solved["weights"]
            state["last"] = solved
            state["history"].append({"date": wide.index[end - 1], "weights": solved["weights"]})
            full = np.zeros(len(columns))
            full[positions] = solved["weights"]
            return full

        return provider

    results = []
    for number, portfolio in enumerate(portfolios, start=1):
        if progress:
            progress({"phase": "backtest", "elapsed_ms": round((perf_counter() - started) * 1000, 1),
                      "detail": f"{portfolio.name} ({number} of {len(portfolios)})",
                      "step": number, "steps": len(portfolios)})
        requested_total = sum(portfolio.weights.values()) if portfolio.scheme == "custom" else 1.0
        provider = make_provider(portfolio) if portfolio.scheme == "optimized" else None
        target = (pd.Series(0.0, index=columns) if provider
                  else portfolio.resolve(columns, calibration))
        # Weights arrive either as fractions (0.35) or as percentages (35). Both
        # are normalized; only a total that is neither is worth flagging.
        if portfolio.scheme == "custom" and min(abs(requested_total - 1), abs(requested_total - 100)) > 0.005:
            warnings.append(
                f"{portfolio.name}: the weights you entered do not add up to a whole portfolio, "
                "so they were rescaled to 100% while keeping their proportions."
            )
        schedule = "none" if portfolio.name == benchmark_name else settings.rebalance
        if provider and schedule == "none":
            raise ValueError(
                f"{portfolio.name}: an optimized portfolio needs a rebalance schedule, because that is "
                "when it re-estimates. Choose monthly, quarterly or annual."
            )
        run = engine.run_backtest(window, target, settings, schedule=schedule, target_provider=provider)
        if provider:
            target = pd.Series(run["target_history"][0]["weights"], index=columns)
        results.append((portfolio, target, run))
    phase("backtest", f"{len(results)} portfolios walked through {len(window)} sessions")

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
    for number, (portfolio, _target, run) in enumerate(results, start=1):
        if progress:
            progress({"phase": "scenarios", "elapsed_ms": round((perf_counter() - started) * 1000, 1),
                      "detail": f"{portfolio.name} ({number} of {len(results)})",
                      "step": number, "steps": len(results)})
        start_value = (
            float(run["summary"]["final_value"]) if settings.scenario_basis == "continuation"
            else float(settings.initial_capital)
        )
        scenarios[portfolio.name] = engine.scenario_statistics(
            run["net_returns"][1:], index, start_value, grid_points=settings.grid_points
        )

    phase("scenarios", f"{settings.paths:,} paths for each of {len(results)} portfolios")
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
            "optimization": _optimizer_report(portfolio, optimizer_state.get(portfolio.name), run, meta),
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
    if progress:
        progress({"phase": "diagnostics", "elapsed_ms": round((perf_counter() - started) * 1000, 1),
                  "detail": "risk decomposition and the efficient frontier"})
    frontier = _frontier_report(window, portfolios, results, settings, benchmark_name, asset_rows)
    phase("diagnostics", "frontier, correlations and weight stability")

    record = provenance.manifest(
        request if request is not None else _reconstruct_request(portfolios, settings, start, end),
        wide_all.reset_index().melt(id_vars="date", var_name="ticker", value_name="adjusted_price").dropna(),
        ENGINE_VERSION, source_note,
    )
    return _clean({
        "manifest": record,
        "meta": {
            "run_id": record["run_id"],
            "generated_at": record["generated_at"],
            "engine_version": ENGINE_VERSION,
            "source": source_note,
            "window_start": window.index[0].date().isoformat(),
            "window_end": window.index[-1].date().isoformat(),
            "trading_days": int(len(window)),
            "years": round(float((len(window) - 1) / engine.TRADING_DAYS), 2),
            "calibration_days": int(warmup),
            "timings_ms": {**timings, "total": round((perf_counter() - started) * 1000, 1)},
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
        "frontier": frontier,
        "correlation": {
            "tickers": list(correlation.columns),
            "matrix": [[round(float(v), 3) for v in row] for row in correlation.to_numpy()],
        },
        "portfolios": payload_portfolios,
    })


def _optimizer_report(portfolio: Portfolio, state: dict | None, run: dict, meta: dict) -> dict | None:
    """What the optimizer was given, what it did, how much it churned, and how
    much of the answer is the sample rather than the signal."""
    if not state or not state.get("last"):
        return None
    solved = state["last"]
    universe = state["universe"]
    decomposition = solved["risk"]
    history = state["history"]
    rebalances = max(len(history) - 1, 0)
    turnover = float(run["turnover"].sum())
    stability = _weight_stability(state)
    risk = state.get("last_risk_model")
    convex_solution = state.get("last_solution")
    return {
        "objective": portfolio.objective,
        "risk_model": (riskmodel.report(risk, solved["weights"]) if risk else None),
        "solver": ({
            "name": convex_solution.solver,
            "status": convex_solution.status,
            "seconds": convex_solution.solve_seconds,
            "binding_constraints": convex_solution.binding,
            "shadow_prices": {k: v for k, v in convex_solution.duals.items() if v > 1e-7},
            "reformulation": convex_solution.diagnostics.get("reformulation"),
            "note": convex_solution.diagnostics.get("note"),
        } if convex_solution else None),
        "label": strategies.get(portfolio.objective).label,
        "estimator": portfolio.estimator,
        "estimation_days": int(portfolio.estimation_days),
        "max_weight": float(portfolio.max_weight),
        "reoptimizations": int(len(history)),
        "turnover_per_rebalance": turnover / rebalances if rebalances else 0.0,
        "annual_turnover": turnover / max(run["summary"]["years"], 1e-9),
        # Estimated at the final rebalance, so these describe the portfolio as held today.
        "shrinkage_intensity": float(solved["shrinkage_intensity"]),
        "mean_shrinkage_intensity": float(solved["mean_shrinkage_intensity"]),
        "expected_volatility": float(solved["expected_volatility"]),
        "expected_return": float(solved["expected_return"]),
        "diversification_ratio": float(decomposition["diversification_ratio"]),
        "effective_bets": float(decomposition["effective_bets"]),
        "holdings": [
            {
                "ticker": ticker,
                "weight": float(solved["weights"][i]),
                "risk_share": float(decomposition["share"][i]),
                "marginal_risk": float(decomposition["marginal"][i]),
                "weight_p05": stability["percentiles"]["p05"][i] if stability else None,
                "weight_p95": stability["percentiles"]["p95"][i] if stability else None,
            }
            for i, ticker in enumerate(universe)
        ],
        "stability": stability,
        "weight_history": {
            "dates": [row["date"].date().isoformat() for row in history],
            "tickers": universe,
            "weights": [[round(float(v), 6) for v in row["weights"]] for row in history],
        },
    }


def _weight_stability(state: dict) -> dict | None:
    """Re-solve on bootstrapped samples of the final estimation window.

    The weights the optimizer reports are one draw from a sampling distribution.
    This measures the width of that distribution, which is the honest way to say
    how much of the difference between two candidate portfolios is real. It uses
    the same routine the conformance harness applies to every methodology.
    """
    context = state.get("last_context")
    strategy = state.get("strategy")
    if context is None or strategy is None:
        return None
    try:
        from .conformance import weight_stability

        measured = weight_stability(strategy, context.returns, context.constraints, draws=20)
    except Exception:  # noqa: BLE001 - diagnostics must never sink an analysis
        return None
    if not measured.get("percentiles"):
        return None
    return measured


def _frontier_report(window, portfolios, results, settings, benchmark_name, asset_rows) -> dict | None:
    """The frontier over the whole window, with what each portfolio realized.

    This frontier is fitted to the same data it is drawn against, so every point
    on it is a decision made with hindsight. It is here precisely so the
    walk-forward portfolios can be plotted against it: the gap between the curve
    and where a portfolio actually landed is the cost of not knowing the future,
    and it is the most honest thing this app can show about optimization.
    """
    optimized = [p for p in portfolios if p.scheme == "optimized"]
    universe = list(optimized[0].weights) if optimized else [
        row["ticker"] for row in asset_rows if row["ticker"] != (settings.benchmark or "")
    ]
    universe = [t for t in window.columns if t in universe]
    if len(universe) < 2:
        return None
    cap = min([p.max_weight for p in optimized], default=1.0)
    returns = window[universe].pct_change(fill_method=None).iloc[1:].to_numpy(dtype=float)
    covariance, intensity = optimize.ledoit_wolf_covariance(returns)
    means = returns.mean(axis=0) * engine.TRADING_DAYS
    constraints = optimize.Constraints(max_weight=cap)
    try:
        curve = optimize.efficient_frontier(covariance, means, constraints, points=24)
    except ValueError:
        return None
    return {
        "universe": universe,
        "max_weight": cap,
        "shrinkage_intensity": float(intensity),
        "in_sample": True,
        "volatilities": [round(v, 5) for v in curve["volatilities"]],
        "returns": [round(r, 5) for r in curve["returns"]],
        "assets": [
            {"ticker": ticker,
             "volatility": float(np.sqrt(covariance[i, i])),
             "expected_return": float(means[i])}
            for i, ticker in enumerate(universe)
        ],
        "realized": [
            {"name": portfolio.name,
             "is_benchmark": portfolio.name == benchmark_name,
             "volatility": run["summary"]["volatility"],
             "annualized_return": run["summary"]["annualized_return"]}
            for portfolio, _target, run in results
        ],
    }


def _reconstruct_request(portfolios, settings, start, end) -> dict:
    """The request as the engine understood it, for callers that did not pass one."""
    return {
        "portfolios": [
            {"name": p.name, "weights": {k: float(v) for k, v in sorted(p.weights.items())},
             "scheme": p.scheme, "objective": p.objective, "estimation_days": p.estimation_days,
             "max_weight": p.max_weight, "estimator": p.estimator,
             "parameters": dict(sorted(p.parameters.items()))}
            for p in portfolios
        ],
        "start": start, "end": end,
        "settings": {
            "initial_capital": settings.initial_capital,
            "transaction_cost_bps": settings.transaction_cost_bps,
            "risk_free_rate": settings.risk_free_rate,
            "rebalance": settings.rebalance,
            "benchmark": settings.benchmark,
            "horizon_years": settings.horizon_years,
            "paths": settings.paths,
            "block_days": settings.block_days,
            "seed": settings.seed,
            "scenario_basis": settings.scenario_basis,
        },
    }


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
