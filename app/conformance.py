"""The gate a methodology passes before it is allowed to run against money.

A construction rule that works in a researcher's notebook can still be unfit to
run unattended. The failures that matter are rarely wrong maths; they are an
asset order assumed rather than checked, a constraint satisfied on the sample
that was tried, a solver that returns whatever it had when it ran out of
iterations, a result that changes between two identical calls, or a run time
that is fine on eight assets and hopeless on five hundred.

Each check below exists because that class of failure is silent: the code
returns a plausible weight vector and the damage only appears in the P&L. The
harness runs the same battery against every registered strategy, so a new
methodology is held to what already shipped rather than to its author's
judgement.

Run it from the command line::

    python -m app.conformance                  # every registered strategy
    python -m app.conformance --strategy risk_parity --verbose

Exit status is non-zero when any required check fails, so it belongs in CI next
to the unit tests rather than in a notebook someone remembers to open.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from .optimize import Constraints, portfolio_volatility
from .strategies import REGISTRY, Context, Strategy, get

TOLERANCE = 1e-6
RUNTIME_BUDGET_MS = 750.0


@dataclass
class Check:
    name: str
    passed: bool
    required: bool
    detail: str
    measurement: float | None = None


@dataclass
class Report:
    strategy: str
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.required)

    def add(self, name, passed, detail, required=True, measurement=None) -> None:
        self.checks.append(Check(name, bool(passed), required, detail, measurement))

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "passed": self.passed,
            "failures": [c.name for c in self.checks if c.required and not c.passed],
            "warnings": [c.name for c in self.checks if not c.required and not c.passed],
            "checks": [
                {"name": c.name, "passed": c.passed, "required": c.required,
                 "detail": c.detail, "measurement": c.measurement}
                for c in self.checks
            ],
        }


def sample_returns(seed: int = 17, observations: int = 320, assets: int = 6) -> np.ndarray:
    """A market with a common factor and unequal idiosyncratic risk.

    Fixed seed on purpose: every strategy is judged against the same data, and a
    failure is reproducible from the report alone.
    """
    rng = np.random.default_rng(seed)
    loadings = rng.uniform(0.4, 1.4, (assets, 2))
    common = rng.normal(0, 0.009, (observations, 2))
    scale = np.linspace(0.004, 0.018, assets)
    return common @ loadings.T + rng.normal(0, 1, (observations, assets)) * scale


def _context(returns: np.ndarray, constraints: Constraints, **kwargs) -> Context:
    parameters = {"volatility_target": 0.12, "turnover_lambda": 1.0}
    parameters.update(kwargs.pop("parameters", {}))
    return Context(returns=returns, constraints=constraints, parameters=parameters, **kwargs)


def evaluate(strategy: Strategy, returns: np.ndarray | None = None) -> Report:
    """Run the battery and describe what happened, check by check."""
    returns = sample_returns() if returns is None else returns
    assets = returns.shape[1]
    constraints = Constraints()
    report = Report(strategy=strategy.name)

    # 1. It returns a portfolio at all.
    try:
        weights = strategy(_context(returns, constraints))
    except Exception as error:  # noqa: BLE001
        report.add("produces weights", False, f"raised {type(error).__name__}: {error}")
        return report
    report.add("produces weights", True, f"returned {assets} weights")

    finite = bool(np.isfinite(weights).all())
    report.add("weights are finite", finite,
               "no NaN or infinity" if finite else "returned a non-finite weight")
    if not finite:
        return report

    total = float(weights.sum())
    report.add("fully invested", abs(total - 1) < TOLERANCE,
               f"weights sum to {total:.10f}", measurement=total)
    report.add("long only", bool((weights >= -TOLERANCE).all()),
               f"smallest weight {weights.min():+.6f}", measurement=float(weights.min()))

    # 2. It respects a constraint it was not tested against by its author.
    capped = Constraints(max_weight=0.3)
    try:
        capped_weights = strategy(_context(returns, capped))
        respected = bool(capped_weights.max() <= 0.3 + TOLERANCE and abs(capped_weights.sum() - 1) < TOLERANCE)
        report.add("honours a weight cap", respected,
                   f"largest weight under a 30% cap was {capped_weights.max():.4f}",
                   measurement=float(capped_weights.max()))
    except Exception as error:  # noqa: BLE001
        report.add("honours a weight cap", False, f"raised {type(error).__name__}: {error}")

    # 3. Two identical calls give one answer. Anything else makes a backtest
    #    unreproducible and a production incident impossible to investigate.
    repeat = strategy(_context(returns, constraints))
    drift = float(np.abs(repeat - weights).max())
    report.add("deterministic", drift < 1e-10,
               f"largest difference across two identical calls: {drift:.2e}", measurement=drift)

    # 4. Reordering the assets must reorder the weights and nothing else. This
    #    is where an index assumed rather than carried shows up.
    order = np.random.default_rng(0).permutation(assets)
    permuted = strategy(_context(returns[:, order], constraints))
    gap = float(np.abs(permuted - weights[order]).max())
    report.add("independent of asset order", gap < 1e-6,
               f"largest difference after permuting the universe: {gap:.2e}", measurement=gap)

    # 5. A risk-based rule should not care what units the returns are quoted in.
    if strategy.scale_invariant:
        scaled = strategy(_context(returns * 2.0, constraints))
        shift = float(np.abs(scaled - weights).max())
        report.add("scale invariant", shift < 1e-5,
                   f"doubling every return moved weights by {shift:.2e}", measurement=shift)

    # 6. Degenerate inputs are the ones that arrive on a bad data day.
    for label, bad in _degenerate_cases(returns):
        try:
            result = strategy(_context(bad, constraints))
            survived = bool(np.isfinite(result).all() and abs(result.sum() - 1) < 1e-4)
            detail = "returned a valid portfolio" if survived else "returned an invalid portfolio"
        except Exception as error:  # noqa: BLE001
            survived, detail = False, f"raised {type(error).__name__}"
        report.add(f"survives {label}", survived, detail)

    # 7. How far the answer moves when the sample is resampled. Not pass or fail:
    #    it is the size of the estimation error the weights are standing on, and
    #    a number worth carrying into any discussion of the result.
    instability = weight_stability(strategy, returns, constraints)
    report.add("weight stability", instability["mean_absolute_move"] < 0.5,
               f"resampling the window moves weights by {instability['mean_absolute_move']:.1%} "
               f"on average; the widest holding spans {instability['widest_range']:.1%}",
               required=False, measurement=instability["mean_absolute_move"])

    # 8. Walk-forward means solving this hundreds of times per backtest.
    elapsed = _median_runtime(strategy, returns, constraints)
    report.add("runs within budget", elapsed < RUNTIME_BUDGET_MS,
               f"median solve {elapsed:.0f} ms against a {RUNTIME_BUDGET_MS:.0f} ms budget",
               measurement=elapsed)

    # 9. Scaling: a universe of 60 is a normal ask and must not fall over.
    wide = sample_returns(seed=5, observations=400, assets=60)
    try:
        started = time.perf_counter()
        large = strategy(_context(wide, Constraints(max_weight=0.1)))
        large_ms = (time.perf_counter() - started) * 1000
        ok = bool(np.isfinite(large).all() and abs(large.sum() - 1) < 1e-4 and large.max() <= 0.1 + TOLERANCE)
        report.add("scales to 60 assets", ok, f"solved in {large_ms:.0f} ms", measurement=large_ms)
    except Exception as error:  # noqa: BLE001
        report.add("scales to 60 assets", False, f"raised {type(error).__name__}: {error}")

    return report


def _degenerate_cases(returns: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """The data that actually breaks optimizers in production."""
    duplicated = returns.copy()
    duplicated[:, 1] = duplicated[:, 0]          # two identical assets: singular covariance
    constant = returns.copy()
    constant[:, 2] = 0.0                          # a halted or stale name: zero variance
    extreme = returns.copy()
    extreme[10, 3] = 0.6                          # a bad print or a real crash
    short = returns[:35]                          # barely enough history
    return [
        ("perfectly correlated holdings", duplicated),
        ("a zero-variance holding", constant),
        ("a single extreme return", extreme),
        ("a short estimation window", short),
    ]


def weight_stability(strategy: Strategy, returns: np.ndarray, constraints: Constraints,
                     draws: int = 24, seed: int = 9) -> dict:
    """Re-solve on bootstrapped samples to size the estimation error in the weights.

    The optimum is a point estimate computed from a sample. Resampling that
    sample and re-solving shows how much of the difference between candidate
    portfolios is signal and how much is the window that happened to be used.
    """
    rng = np.random.default_rng(seed)
    observations = returns.shape[0]
    solutions = []
    for _ in range(draws):
        rows = rng.integers(0, observations, observations)
        try:
            solutions.append(strategy(_context(returns[rows], constraints)))
        except Exception:  # noqa: BLE001 - a failure here is reported by other checks
            continue
    if len(solutions) < 3:
        return {"mean_absolute_move": float("nan"), "widest_range": float("nan"),
                "draws": len(solutions), "percentiles": {}}
    stack = np.vstack(solutions)
    base = strategy(_context(returns, constraints))
    spread = np.quantile(stack, 0.95, axis=0) - np.quantile(stack, 0.05, axis=0)
    return {
        "mean_absolute_move": float(np.abs(stack - base).mean()),
        "widest_range": float(spread.max()),
        "draws": len(solutions),
        "percentiles": {
            "p05": [float(v) for v in np.quantile(stack, 0.05, axis=0)],
            "median": [float(v) for v in np.median(stack, axis=0)],
            "p95": [float(v) for v in np.quantile(stack, 0.95, axis=0)],
        },
    }


def _median_runtime(strategy: Strategy, returns: np.ndarray, constraints: Constraints,
                    repeats: int = 5) -> float:
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        strategy(_context(returns, constraints))
        timings.append((time.perf_counter() - started) * 1000)
    return float(np.median(timings))


def run_all(names: list[str] | None = None) -> list[Report]:
    chosen = [get(name) for name in names] if names else list(REGISTRY.values())
    return [evaluate(strategy) for strategy in sorted(chosen, key=lambda s: s.name)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", action="append", help="check one strategy; repeatable")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--verbose", action="store_true", help="show every check, not just failures")
    args = parser.parse_args()

    reports = run_all(args.strategy)
    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
        return 0 if all(r.passed for r in reports) else 1

    for report in reports:
        mark = "PASS" if report.passed else "FAIL"
        print(f"\n{mark}  {report.strategy}")
        for check in report.checks:
            if check.passed and not args.verbose:
                continue
            flag = "  ok  " if check.passed else ("  --  " if not check.required else "  XX  ")
            print(f"{flag}{check.name}: {check.detail}")
    failed = [r.strategy for r in reports if not r.passed]
    print(f"\n{len(reports) - len(failed)}/{len(reports)} strategies conform"
          + (f"; failing: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
