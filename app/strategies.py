"""The contract between a researcher's methodology and this system.

A construction rule arrives as a description of what to solve. Turning it into
something that can run against money means pinning down four things: what inputs
it may see, what it must return, what it is allowed to assume, and how anyone
can tell whether a change to it broke something. That is what this module is.

A strategy is a function from a :class:`Context` to a weight vector, registered
with metadata. It receives only a trailing window of returns, so it cannot see
past the day it is being asked about; it returns weights over the universe it
was given, and nothing else. Everything downstream — the walk-forward loop, the
API, the interface, the conformance harness — works from the registry, so adding
a methodology means writing one function and registering it, not editing the
engine.

``app/conformance.py`` runs every registered strategy through the same battery
of checks. A methodology that has not passed it has not shipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .optimize import (
    Constraints,
    inverse_volatility_weights,
    ledoit_wolf_covariance,
    maximum_diversification,
    maximum_sharpe,
    minimum_variance,
    portfolio_volatility,
    risk_parity,
    sample_covariance,
    shrink_means,
    target_volatility,
)


@dataclass
class Context:
    """Everything a strategy is allowed to see, and nothing more.

    ``returns`` is the trailing estimation window, most recent row last, already
    sliced by the caller so that it ends on the session being traded. A strategy
    that wants information from outside this object is asking for information
    the live system would not have.
    """

    returns: np.ndarray
    constraints: Constraints
    risk_free_rate: float = 0.03
    previous_weights: np.ndarray | None = None
    estimator: str = "ledoit_wolf"
    volatility_target: float | None = None
    transaction_cost_bps: float = 10.0
    effort: str = "full"
    parameters: dict = field(default_factory=dict)

    @property
    def assets(self) -> int:
        return int(self.returns.shape[1])

    def covariance(self) -> tuple[np.ndarray, float]:
        """The risk model, chosen once so every strategy uses the same one."""
        if self.estimator == "sample":
            return sample_covariance(self.returns), 0.0
        return ledoit_wolf_covariance(self.returns)

    def expected_returns(self, shrink: bool = True) -> tuple[np.ndarray, float]:
        return shrink_means(self.returns, None if shrink else 0.0)

    def solver_options(self, convex: bool = False) -> dict:
        """How hard to search, given what is being solved.

        Restarts protect against a local search settling short of the optimum,
        which can only happen when the objective is not convex. Asking for them
        on a convex problem buys nothing and costs a multiple of the run time,
        which matters when the solve repeats at every rebalance of every
        backtest.
        """
        if self.previous_weights is not None:
            return {"starts": [np.asarray(self.previous_weights, dtype=float),
                               np.full(self.assets, 1 / self.assets)]}
        if convex:
            return {"attempts": 1}
        return {"attempts": 2 if self.effort != "full" else 5}


@dataclass
class Strategy:
    """A registered methodology and the facts a caller needs to use it."""

    name: str
    label: str
    description: str
    solve: Callable[[Context], np.ndarray]
    needs_expected_returns: bool = False
    needs_parameters: tuple[str, ...] = ()
    scale_invariant: bool = True     # multiplying all returns by k must not move weights
    author: str = "core"
    version: str = "1.0"

    def __call__(self, context: Context) -> np.ndarray:
        weights = np.asarray(self.solve(context), dtype=float)
        if weights.shape != (context.assets,):
            raise ValueError(
                f"Strategy '{self.name}' returned {weights.shape} weights for "
                f"{context.assets} assets."
            )
        return weights


def solve_with_diagnostics(strategy: Strategy, context: Context) -> dict:
    """Solve, then check the answer before anyone acts on it.

    The validity gate repeats what the conformance harness already tested. That
    is deliberate: the harness proves a methodology behaved on the data it was
    given, and this proves it behaved on the data it actually got. A solver that
    quietly returns its last iterate when it fails to converge is the failure
    this catches, and it is cheap next to the cost of trading on the result.
    """
    from .optimize import risk_contributions

    covariance, intensity = context.covariance()
    means, mean_intensity = context.expected_returns()
    weights = strategy(context)
    if not np.isfinite(weights).all():
        raise ValueError(f"Strategy '{strategy.name}' produced a non-finite weight.")
    if abs(float(weights.sum()) - 1) > 1e-4:
        raise ValueError(f"Strategy '{strategy.name}' returned weights summing to {weights.sum():.6f}.")
    if (weights < -1e-6).any():
        raise ValueError(f"Strategy '{strategy.name}' returned a negative weight in a long-only mandate.")
    cap = context.constraints.max_weight
    if weights.max() > cap + 1e-6:
        raise ValueError(
            f"Strategy '{strategy.name}' breached the {cap:.0%} cap with a weight of {weights.max():.2%}."
        )
    weights = np.clip(weights, 0.0, cap)
    weights = weights / weights.sum()
    decomposition = risk_contributions(weights, covariance)
    return {
        "weights": weights,
        "covariance": covariance,
        "means": means,
        "shrinkage_intensity": intensity,
        "mean_shrinkage_intensity": mean_intensity,
        "expected_volatility": decomposition["volatility"],
        "expected_return": float(weights @ means),
        "risk": decomposition,
        "observations": int(context.returns.shape[0]),
    }


REGISTRY: dict[str, Strategy] = {}


def register(strategy: Strategy) -> Strategy:
    if strategy.name in REGISTRY:
        raise ValueError(f"A strategy named '{strategy.name}' is already registered.")
    REGISTRY[strategy.name] = strategy
    return strategy


def get(name: str) -> Strategy:
    if name not in REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Registered: {', '.join(sorted(REGISTRY))}.")
    return REGISTRY[name]


def catalogue() -> list[dict]:
    """What the API and the interface advertise, straight from the registry."""
    return [
        {
            "name": s.name, "label": s.label, "description": s.description,
            "needs_expected_returns": s.needs_expected_returns,
            "parameters": list(s.needs_parameters), "author": s.author, "version": s.version,
        }
        for s in sorted(REGISTRY.values(), key=lambda s: s.name)
    ]


# --------------------------------------------------------------------------- #
# The methodologies that ship with the system
# --------------------------------------------------------------------------- #

register(Strategy(
    name="minimum_variance",
    label="Minimum variance",
    description="The lowest-variance long-only mix of the universe. Needs no return forecast.",
    solve=lambda ctx: minimum_variance(ctx.covariance()[0], ctx.constraints,
                                       **ctx.solver_options(convex=True)),
))

register(Strategy(
    name="risk_parity",
    label="Risk parity",
    description="Weights at which every holding contributes the same share of portfolio volatility.",
    solve=lambda ctx: risk_parity(ctx.covariance()[0], ctx.constraints, **ctx.solver_options()),
))

register(Strategy(
    name="maximum_diversification",
    label="Maximum diversification",
    description="Maximizes the ratio of the holdings' own volatility to the portfolio's.",
    solve=lambda ctx: maximum_diversification(ctx.covariance()[0], ctx.constraints, **ctx.solver_options()),
))


def _maximum_sharpe(ctx: Context) -> np.ndarray:
    covariance, _ = ctx.covariance()
    means, _ = ctx.expected_returns()
    if ctx.effort == "full" and ctx.previous_weights is None:
        return maximum_sharpe(covariance, means, ctx.risk_free_rate, ctx.constraints)
    starts = [np.full(ctx.assets, 1 / ctx.assets), inverse_volatility_weights(covariance, ctx.constraints)]
    if ctx.previous_weights is not None:
        starts.insert(0, np.asarray(ctx.previous_weights, dtype=float))
    return maximum_sharpe(covariance, means, ctx.risk_free_rate, ctx.constraints, starts=starts)


register(Strategy(
    name="maximum_sharpe",
    label="Maximum Sharpe",
    description="The tangency portfolio on the estimated frontier. Depends on return forecasts, "
                "which are far noisier than risk estimates, so its inputs are shrunk.",
    solve=_maximum_sharpe,
    needs_expected_returns=True,
    scale_invariant=False,  # the risk-free rate is a fixed level, so scaling returns does move it
))


def _target_volatility(ctx: Context) -> np.ndarray:
    covariance, _ = ctx.covariance()
    means, _ = ctx.expected_returns()
    target = ctx.volatility_target or ctx.parameters.get("volatility_target")
    if not target:
        raise ValueError("This strategy needs a volatility target.")
    return target_volatility(covariance, means, float(target), ctx.constraints)


register(Strategy(
    name="target_volatility",
    label="Volatility target",
    description="The highest expected return whose estimated volatility stays within a target.",
    solve=_target_volatility,
    needs_expected_returns=True,
    needs_parameters=("volatility_target",),
    scale_invariant=False,
))


# --------------------------------------------------------------------------- #
# A methodology added through the contract rather than by editing the engine
# --------------------------------------------------------------------------- #

def _turnover_aware_minimum_variance(ctx: Context) -> np.ndarray:
    """Minimum variance with the cost of getting there priced in.

    A plain variance minimizer re-solved every month rebuilds the portfolio
    around whichever estimate moved, and the trading bill is charged to the
    investor rather than to the estimate. This adds the round-trip cost of the
    move to the objective, so a change is only made when the variance it saves
    is worth paying for.

    The penalty is the traded notional times the cost, annualized across the
    holding period, against variance in the same units. ``lambda`` scales how
    literally that trade-off is taken; 1.0 means cost and variance are compared
    at face value.

    This is also the worked example of the registry: it was added without
    touching the engine, the API, the interface or the walk-forward loop.
    """
    from scipy.optimize import minimize

    covariance, _ = ctx.covariance()
    previous = ctx.previous_weights
    if previous is None:
        return minimum_variance(covariance, ctx.constraints, **ctx.solver_options())
    previous = np.asarray(previous, dtype=float)
    aversion = float(ctx.parameters.get("turnover_lambda", 1.0))
    periods = float(ctx.parameters.get("rebalances_per_year", 12))
    cost_rate = ctx.transaction_cost_bps / 10_000

    def objective(w):
        variance = w @ covariance @ w
        traded = np.abs(w - previous).sum()
        return variance + aversion * traded * cost_rate * periods

    result = minimize(objective, previous, method="SLSQP",
                      bounds=ctx.constraints.bounds(ctx.assets),
                      constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
                      options={"maxiter": 400, "ftol": 1e-12})
    if not result.success:
        return minimum_variance(covariance, ctx.constraints, **ctx.solver_options())
    weights = np.clip(result.x, ctx.constraints.min_weight, ctx.constraints.max_weight)
    return weights / weights.sum()


register(Strategy(
    name="minimum_variance_net_of_costs",
    label="Minimum variance, net of trading",
    description="Minimum variance with the round-trip cost of each rebalance added to the objective, "
                "so the portfolio only moves when the variance saved is worth the trade.",
    solve=_turnover_aware_minimum_variance,
    needs_parameters=("turnover_lambda",),
    author="example of the strategy contract",
))
