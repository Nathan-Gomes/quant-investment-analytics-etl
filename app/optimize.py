"""Portfolio construction: covariance estimation, objectives, and risk decomposition.

Three ideas run through this module, and they are the ones worth arguing about
in a review.

**Covariance is estimated, not observed.** With 25 assets and a year of daily
data a sample covariance matrix has 325 free parameters fitted on 252
observations, so its extreme eigenvalues are badly biased: the directions that
look calmest in sample are the ones an optimizer will pile into. Ledoit-Wolf
shrinkage pulls the matrix toward a structured target by an amount estimated
from the data itself, which is why it is the default here.

**Expected returns are far noisier than risk.** Estimating a mean return to the
precision an optimizer needs takes decades of data. So the objectives that need
no return forecast at all — minimum variance, equal risk contribution, maximum
diversification — are offered first, and the ones that do need forecasts say so
and shrink them.

**An optimizer evaluated on the data it was fitted to tells you nothing.** Every
weight here is computed from a trailing window and then applied forward, and
`analysis.py` reports the in-sample and walk-forward results side by side so the
gap between them is visible rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

TRADING_DAYS = 252
OBJECTIVES = (
    "minimum_variance",
    "maximum_sharpe",
    "risk_parity",
    "maximum_diversification",
    "target_volatility",
)


# --------------------------------------------------------------------------- #
# Covariance
# --------------------------------------------------------------------------- #

def sample_covariance(returns: np.ndarray, annualize: bool = True) -> np.ndarray:
    """Plain sample covariance. Included as the baseline to shrink away from."""
    if returns.ndim != 2 or returns.shape[0] < 2:
        raise ValueError("Covariance needs at least two observations.")
    covariance = np.cov(returns, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    return covariance * (TRADING_DAYS if annualize else 1)


def ledoit_wolf_covariance(returns: np.ndarray, annualize: bool = True) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage toward a scaled identity matrix.

    Follows Ledoit and Wolf (2004), *A well-conditioned estimator for
    large-dimensional covariance matrices*. The shrinkage intensity is the ratio
    of estimation error in the sample matrix to its distance from the target, so
    it rises when observations are scarce relative to assets and falls to zero
    when the sample matrix is already well determined. The result is always
    positive definite, which matters because an optimizer will otherwise find
    and exploit a near-zero eigenvalue.

    Returns the matrix and the intensity, so the interface can show how much
    structure was imposed rather than hiding it.
    """
    observations, assets = returns.shape
    if observations < 2:
        raise ValueError("Covariance needs at least two observations.")
    centred = returns - returns.mean(axis=0)
    sample = centred.T @ centred / observations

    mu = np.trace(sample) / assets
    target = mu * np.eye(assets)
    dispersion = np.sum((sample - target) ** 2) / assets

    # Mean squared error of the sample matrix entries, from the fourth moments.
    error = 0.0
    for t in range(observations):
        outer = np.outer(centred[t], centred[t])
        error += np.sum((outer - sample) ** 2)
    error = error / (observations ** 2) / assets
    error = min(error, dispersion)

    intensity = 0.0 if dispersion <= 0 else float(error / dispersion)
    intensity = float(np.clip(intensity, 0.0, 1.0))
    shrunk = intensity * target + (1 - intensity) * sample
    # Rescale to the unbiased sample convention so it matches sample_covariance.
    shrunk = shrunk * observations / (observations - 1)
    return shrunk * (TRADING_DAYS if annualize else 1), intensity


def shrink_means(returns: np.ndarray, intensity: float | None = None) -> tuple[np.ndarray, float]:
    """James-Stein shrinkage of asset means toward the cross-sectional average.

    An optimizer handed raw historical means will chase whichever asset happened
    to run hardest. Shrinking toward the grand mean concedes that the sample
    cannot tell the assets apart as confidently as the numbers suggest.
    """
    observations, assets = returns.shape
    means = returns.mean(axis=0) * TRADING_DAYS
    grand = means.mean()
    if intensity is None:
        spread = np.sum((means - grand) ** 2)
        variance = returns.var(axis=0, ddof=1).sum() * TRADING_DAYS ** 2 / observations
        intensity = 0.0 if spread <= 0 else float(min(1.0, (assets - 2) * variance / (observations * spread)))
    intensity = float(np.clip(intensity, 0.0, 1.0))
    return grand + (1 - intensity) * (means - grand), intensity


# --------------------------------------------------------------------------- #
# Risk decomposition
# --------------------------------------------------------------------------- #

def portfolio_volatility(weights: np.ndarray, covariance: np.ndarray) -> float:
    return float(np.sqrt(max(weights @ covariance @ weights, 0.0)))


def risk_contributions(weights: np.ndarray, covariance: np.ndarray) -> dict:
    """Split portfolio risk across holdings, since weight is not risk.

    Marginal contribution is the derivative of portfolio volatility with respect
    to a holding's weight; the contribution is that times the weight, and the
    contributions add up to total volatility exactly (Euler's theorem on the
    homogeneous-of-degree-one volatility function).
    """
    volatility = portfolio_volatility(weights, covariance)
    if volatility <= 0:
        zeros = np.zeros_like(weights)
        return {"volatility": 0.0, "marginal": zeros, "contribution": zeros,
                "share": zeros, "diversification_ratio": 1.0, "effective_bets": float(len(weights))}
    marginal = covariance @ weights / volatility
    contribution = weights * marginal
    share = contribution / volatility
    standalone = np.sqrt(np.diag(covariance))
    positive = share[share > 0]
    return {
        "volatility": volatility,
        "marginal": marginal,
        "contribution": contribution,
        "share": share,
        # How much of the holdings' standalone risk diversification removed.
        "diversification_ratio": float(weights @ standalone / volatility),
        # A concentration measure: 10 holdings can still be one bet.
        "effective_bets": float(1 / np.sum(positive ** 2)) if positive.size else 0.0,
    }


# --------------------------------------------------------------------------- #
# Optimization
# --------------------------------------------------------------------------- #

@dataclass
class Constraints:
    """Long-only and fully invested, with an optional cap on any one holding."""

    max_weight: float = 1.0
    min_weight: float = 0.0

    def validate(self, assets: int) -> None:
        if not 0 <= self.min_weight <= 1:
            raise ValueError("Minimum weight must be between 0 and 1.")
        if not 0 < self.max_weight <= 1:
            raise ValueError("Maximum weight must be above 0 and at most 1.")
        if self.max_weight * assets < 1 - 1e-9:
            raise ValueError(
                f"A cap of {self.max_weight:.0%} cannot be spread across {assets} holdings. "
                f"Raise the cap to at least {1 / assets:.0%} or add holdings."
            )
        if self.min_weight * assets > 1 + 1e-9:
            raise ValueError(f"A floor of {self.min_weight:.0%} across {assets} holdings exceeds the portfolio.")
        if self.min_weight > self.max_weight:
            raise ValueError("The weight floor cannot exceed the cap.")

    def bounds(self, assets: int):
        return [(self.min_weight, self.max_weight)] * assets


def _solve(objective, assets: int, constraints: Constraints, extra=(), starts=None,
           attempts: int = 5) -> np.ndarray:
    """Sequential least squares, from several starts, keeping the best result.

    SLSQP is a local method, so a single start can settle in a poor corner on the
    less convex objectives. Starting from equal weights, from a random simplex
    draw and from the corners keeps that from going unnoticed.
    """
    constraints.validate(assets)
    equality = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}] + list(extra)
    bounds = constraints.bounds(assets)
    if starts is None:
        rng = np.random.default_rng(0)
        starts = [np.full(assets, 1 / assets)]
        starts += [np.clip(rng.dirichlet(np.ones(assets)), constraints.min_weight, constraints.max_weight)
                   for _ in range(max(0, attempts - 1))]
    best, best_value = None, np.inf
    for start in starts:
        start = np.clip(np.asarray(start, dtype=float), constraints.min_weight, constraints.max_weight)
        start = start / start.sum()
        result = minimize(objective, start, method="SLSQP", bounds=bounds,
                          constraints=equality, options={"maxiter": 400, "ftol": 1e-12})
        if result.success and result.fun < best_value:
            best, best_value = result.x, result.fun
    if best is None:
        raise ValueError(
            "The optimizer could not satisfy these constraints. Loosen the weight cap or the target."
        )
    weights = np.clip(best, constraints.min_weight, constraints.max_weight)
    return weights / weights.sum()


def minimum_variance(covariance: np.ndarray, constraints: Constraints, **solver) -> np.ndarray:
    """Minimizing a positive-definite quadratic over a convex set has one optimum,
    so a single start finds it. Restarts are for the objectives below that are
    not convex, where a local search genuinely can settle short."""
    solver.setdefault("attempts", 1)
    return _solve(lambda w: w @ covariance @ w, len(covariance), constraints, **solver)


def maximum_diversification(covariance: np.ndarray, constraints: Constraints, **solver) -> np.ndarray:
    """Maximize the ratio of weighted standalone risk to portfolio risk."""
    standalone = np.sqrt(np.diag(covariance))
    def negative_ratio(w):
        volatility = np.sqrt(max(w @ covariance @ w, 1e-18))
        return -(w @ standalone) / volatility
    return _solve(negative_ratio, len(covariance), constraints, **solver)


def risk_parity(covariance: np.ndarray, constraints: Constraints, **solver) -> np.ndarray:
    """Equal risk contribution: every holding supplies the same share of risk.

    Needs no return forecast and no view on which asset is attractive, which is
    why it survives out of sample more often than mean-variance solutions.
    """
    assets = len(covariance)
    floor = max(constraints.min_weight, 1e-4)  # a zero weight has no risk share
    bounded = Constraints(max_weight=constraints.max_weight, min_weight=floor)
    def dispersion(w):
        volatility = np.sqrt(max(w @ covariance @ w, 1e-18))
        contribution = w * (covariance @ w) / volatility
        return float(np.sum((contribution - contribution.mean()) ** 2)) * 1e4
    return _solve(dispersion, assets, bounded, **solver)


def minimum_variance_at_return(covariance: np.ndarray, means: np.ndarray, target: float,
                               constraints: Constraints) -> np.ndarray:
    """Also convex: a quadratic under one more linear equality."""
    extra = [{"type": "eq", "fun": lambda w: float(w @ means - target)}]
    return _solve(lambda w: w @ covariance @ w, len(covariance), constraints, extra=extra, attempts=1)


def maximum_sharpe(covariance: np.ndarray, means: np.ndarray, risk_free: float,
                   constraints: Constraints, points: int = 40, starts=None) -> np.ndarray:
    """The tangency portfolio.

    The ratio is not concave under these constraints, so a single local search
    can settle short of the best point. Sweeping the frontier and polishing its
    best risk-adjusted point is the dependable route and is used when the answer
    is being inspected. Inside a walk-forward loop the same sweep runs at every
    rebalance, which is far too slow, so a seeded local search is used instead:
    the seeds are the equal-weight, inverse-volatility and previous portfolios,
    which bracket the sensible region well enough to make a poor local optimum
    unlikely.
    """
    assets = len(covariance)

    def negative_sharpe(w):
        volatility = np.sqrt(max(w @ covariance @ w, 1e-18))
        return -(w @ means - risk_free) / volatility

    if starts is None:
        frontier = efficient_frontier(covariance, means, constraints, points=points)
        if not frontier["weights"]:
            raise ValueError("No portfolio satisfies these constraints.")
        ratios = [(ret - risk_free) / vol if vol > 0 else -np.inf
                  for ret, vol in zip(frontier["returns"], frontier["volatilities"])]
        starts = [frontier["weights"][int(np.argmax(ratios))], np.full(assets, 1 / assets)]
    return _solve(negative_sharpe, assets, constraints, starts=starts)


def target_volatility(covariance: np.ndarray, means: np.ndarray, target: float,
                      constraints: Constraints) -> np.ndarray:
    """The highest-return portfolio whose volatility stays within the target.

    If even the minimum-variance portfolio is riskier than the target, that
    portfolio is returned: with no cash or leverage there is nothing calmer to
    hold, and the caller is told rather than silently given something else.
    """
    floor = minimum_variance(covariance, constraints)
    if portfolio_volatility(floor, covariance) >= target:
        return floor
    inequality = [{"type": "ineq", "fun": lambda w: target ** 2 - float(w @ covariance @ w)}]
    return _solve(lambda w: -float(w @ means), len(covariance), constraints, extra=inequality)


def efficient_frontier(covariance: np.ndarray, means: np.ndarray, constraints: Constraints,
                       points: int = 30) -> dict:
    """Minimum variance at each of a grid of target returns.

    Under a long-only constraint the reachable range runs from the
    minimum-variance portfolio's return to the highest return any feasible
    portfolio can reach, so the grid is built between those rather than between
    the smallest and largest asset means.
    """
    constraints.validate(len(covariance))
    floor_weights = minimum_variance(covariance, constraints)
    lowest = float(floor_weights @ means)
    top_weights = _solve(lambda w: -float(w @ means), len(covariance), constraints, attempts=1)
    highest = float(top_weights @ means)
    if highest <= lowest:
        return {"returns": [lowest], "volatilities": [portfolio_volatility(floor_weights, covariance)],
                "weights": [floor_weights]}
    grid = np.linspace(lowest, highest, points)
    returns, volatilities, weights = [], [], []
    for target in grid:
        try:
            solution = minimum_variance_at_return(covariance, means, float(target), constraints)
        except ValueError:
            continue
        returns.append(float(solution @ means))
        volatilities.append(portfolio_volatility(solution, covariance))
        weights.append(solution)
    return {"returns": returns, "volatilities": volatilities, "weights": weights}


def inverse_volatility_weights(covariance: np.ndarray, constraints: Constraints) -> np.ndarray:
    """A cheap, sensible starting point that needs no solver."""
    inverse = 1 / np.sqrt(np.diag(covariance))
    weights = np.clip(inverse / inverse.sum(), constraints.min_weight, constraints.max_weight)
    return weights / weights.sum()


def optimize(objective: str, returns: np.ndarray, constraints: Constraints,
             risk_free: float = 0.03, target: float | None = None,
             estimator: str = "ledoit_wolf", shrink_expected_returns: bool = True,
             effort: str = "full", previous: np.ndarray | None = None) -> dict:
    """Estimate inputs from a return window, then solve for weights.

    Everything the solver sees comes from the window passed in, so a caller that
    passes only trailing data gets a genuinely out-of-sample weight vector.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"Objective must be one of {', '.join(OBJECTIVES)}.")
    if returns.shape[0] < 30:
        raise ValueError("At least 30 observations are needed to estimate a covariance matrix.")
    if estimator == "sample":
        covariance, intensity = sample_covariance(returns), 0.0
    else:
        covariance, intensity = ledoit_wolf_covariance(returns)
    means, mean_intensity = shrink_means(returns, None if shrink_expected_returns else 0.0)

    fast = effort != "full"
    solver = {"attempts": 2 if fast else 5}
    # Re-solving from the last accepted answer is both quicker and steadier: a
    # tiny change in the estimate should not produce a wholly different portfolio.
    if previous is not None:
        seeds = [np.asarray(previous, dtype=float), np.full(len(covariance), 1 / len(covariance))]
        solver = {"starts": seeds}

    if objective == "minimum_variance":
        weights = minimum_variance(covariance, constraints, **solver)
    elif objective == "risk_parity":
        weights = risk_parity(covariance, constraints, **solver)
    elif objective == "maximum_diversification":
        weights = maximum_diversification(covariance, constraints, **solver)
    elif objective == "maximum_sharpe":
        if fast:
            starts = [np.full(len(covariance), 1 / len(covariance)),
                      inverse_volatility_weights(covariance, constraints)]
            if previous is not None:
                starts.insert(0, np.asarray(previous, dtype=float))
            weights = maximum_sharpe(covariance, means, risk_free, constraints, starts=starts)
        else:
            weights = maximum_sharpe(covariance, means, risk_free, constraints)
    else:
        if target is None:
            raise ValueError("A volatility target is required for this objective.")
        weights = target_volatility(covariance, means, target, constraints)

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
        "observations": int(returns.shape[0]),
    }
