"""Portfolio construction as convex programs.

Every problem here is written in disciplined convex form and handed to a conic
solver, which returns three things a local search cannot: a certificate that the
answer is the global optimum, dual variables saying what each constraint costs,
and a definite answer of "infeasible" when the mandate cannot be satisfied —
rather than whatever iterate the search happened to stop on.

The formulations worth knowing about:

* **Maximum Sharpe** is a ratio and not concave, but it has a classical convex
  reformulation (Schaible): optimize over an unnormalized ``y`` with the excess
  return pinned to one, then rescale. That works only while every constraint is
  homogeneous in ``y`` — a turnover budget or a tracking-error limit is not, so
  those cases fall back to scanning the frontier, which is convex at each point.
* **Maximum diversification** is the same ratio trick against the weighted
  volatility rather than against expected return.
* **Risk parity** looks non-convex written as "equalize the risk contributions",
  and is convex written as ``min ½wᵀΣw − Σ bᵢ log wᵢ``: the log barrier's first
  order condition *is* the equal-contribution condition, and the solution only
  needs rescaling to the budget. Spinu (2013), Maillard, Roncalli and Teïletche
  (2010).
* **Transaction costs** enter the objective directly. Spread is linear in traded
  weight; market impact is modelled as a power of it, which stays convex for any
  exponent above one, so the cost of getting to a portfolio is traded off against
  the risk it saves inside a single problem rather than bolted on afterwards.

The quadratic form is expressed through the factor structure when the risk model
has one — ``wᵀΣw = ‖Bᵀw‖² + Σ dᵢwᵢ²`` — so a 500-name problem costs a handful of
inner products instead of a 500×500 multiply.
"""

from __future__ import annotations

import time
import warnings
from collections import OrderedDict
from threading import RLock
from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from .riskmodel import RiskModel

SOLVERS = ("CLARABEL", "OSQP", "SCS")


def _tolerances(solver: str, tight: bool = False) -> dict:
    """Ask for more accuracy than the defaults.

    An interior-point solver stops when its duality gap is small enough, and the
    default "small enough" leaves the last few digits free to move when the same
    problem arrives with its columns in a different order. Tightening costs a
    handful of iterations and makes a run reproducible, which matters more here
    than the microseconds.
    """
    # Second-order and power cones — a tracking-error ball, a market-impact term —
    # converge more slowly than a plain quadratic, and asking for 1e-12 there makes
    # the solver report its own answer as inaccurate. The unconstrained log-barrier
    # program has no such trouble, so it gets the tighter setting.
    scale = 1e-12 if tight else 1e-10
    if solver == "CLARABEL":
        return {"tol_gap_abs": scale, "tol_gap_rel": scale, "tol_feas": scale}
    if solver == "OSQP":
        return {"eps_abs": scale, "eps_rel": scale, "max_iter": 40_000}
    if solver == "SCS":
        return {"eps": scale}
    return {}


# --------------------------------------------------------------------------- #
# The mandate
# --------------------------------------------------------------------------- #

@dataclass
class Group:
    """A limit on a set of holdings: a sector cap, a country floor, a sleeve."""

    name: str
    members: list[int]
    minimum: float = 0.0
    maximum: float = 1.0


@dataclass
class Mandate:
    """Everything the portfolio must satisfy, separate from what it optimizes.

    Splitting the mandate from the objective is the point: a researcher supplies
    an objective, and the mandate is what the desk is actually allowed to hold.
    """

    max_weight: float = 1.0
    min_weight: float = 0.0
    groups: list[Group] = field(default_factory=list)
    benchmark: np.ndarray | None = None
    tracking_error_limit: float | None = None
    turnover_budget: float | None = None
    previous_weights: np.ndarray | None = None
    spread_bps: float = 0.0
    impact_coefficient: float = 0.0      # cost = coefficient * |Δw| ** impact_exponent
    impact_exponent: float = 1.5
    long_only: bool = True

    def validate(self, assets: int) -> None:
        """Reject what cannot be satisfied, with the arithmetic that shows why."""
        if not 0 <= self.min_weight <= 1:
            raise ValueError("The weight floor must be between 0 and 1.")
        if not 0 < self.max_weight <= 1:
            raise ValueError("The weight cap must be above 0 and at most 1.")
        if self.min_weight > self.max_weight:
            raise ValueError("The weight floor cannot exceed the cap.")
        if self.max_weight * assets < 1 - 1e-9:
            raise ValueError(
                f"A cap of {self.max_weight:.0%} cannot be spread across {assets} holdings; "
                f"it allows at most {self.max_weight * assets:.0%} of the portfolio. "
                f"Raise the cap to at least {1 / assets:.1%} or add holdings."
            )
        if self.min_weight * assets > 1 + 1e-9:
            raise ValueError(
                f"A floor of {self.min_weight:.1%} across {assets} holdings requires "
                f"{self.min_weight * assets:.0%} of the portfolio."
            )
        for group in self.groups:
            if not group.members:
                raise ValueError(f"Group '{group.name}' has no members.")
            if max(group.members) >= assets or min(group.members) < 0:
                raise ValueError(f"Group '{group.name}' refers to a holding outside the universe.")
            if group.minimum > group.maximum:
                raise ValueError(f"Group '{group.name}' has a floor above its cap.")
            if group.maximum < len(group.members) * self.min_weight - 1e-9:
                raise ValueError(
                    f"Group '{group.name}' is capped at {group.maximum:.0%} but its "
                    f"{len(group.members)} holdings are each floored at {self.min_weight:.1%}."
                )
        floors = sum(g.minimum for g in self.groups)
        if floors > 1 + 1e-9:
            raise ValueError(f"Group floors add up to {floors:.0%} of the portfolio.")
        if self.turnover_budget is not None:
            if self.turnover_budget < 0:
                raise ValueError("The turnover budget cannot be negative.")
            if self.previous_weights is None:
                raise ValueError("A turnover budget needs the portfolio it is moving from.")
        if self.tracking_error_limit is not None:
            if self.tracking_error_limit <= 0:
                raise ValueError("The tracking-error limit must be above zero.")
            if self.benchmark is None:
                raise ValueError("A tracking-error limit needs a benchmark to track.")
        if self.benchmark is not None and len(self.benchmark) != assets:
            raise ValueError("The benchmark must cover the same universe as the portfolio.")

    @property
    def is_homogeneous(self) -> bool:
        """True when every constraint scales with the weights.

        The maximum-Sharpe reformulation optimizes an unnormalized vector, so it
        is only valid while the constraints survive that rescaling. A turnover
        budget or a tracking-error limit is stated in absolute terms and does
        not.
        """
        return (self.turnover_budget is None and self.tracking_error_limit is None
                and self.spread_bps == 0 and self.impact_coefficient == 0
                and self.min_weight == 0)


# --------------------------------------------------------------------------- #
# The result
# --------------------------------------------------------------------------- #

@dataclass
class Solution:
    weights: np.ndarray
    status: str
    objective_value: float
    solver: str
    solve_seconds: float
    duals: dict = field(default_factory=dict)
    binding: list[str] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def optimal(self) -> bool:
        return self.status in ("optimal", "optimal_inaccurate")


# A walk-forward backtest solves the same problem shape a hundred times with new
# numbers. cvxpy spends most of a small solve canonicalizing, so the compiled
# problem is kept and only its data replaced — the disciplined-parametrized-
# program rules make that valid, and it is three times faster.
_COMPILED: OrderedDict = OrderedDict()
_COMPILED_LOCK = RLock()


def _shape_key(model: RiskModel, mandate: Mandate) -> tuple | None:
    """A signature for problems that differ only in their numbers.

    Anything that changes the constraint structure between rebalances — a moving
    previous portfolio, a tracking-error ball, a cost term — is excluded, because
    then it is not the same problem with new data.
    """
    if (model.exposures is not None or mandate.turnover_budget is not None or mandate.tracking_error_limit is not None
            or mandate.spread_bps or mandate.impact_coefficient):
        return None
    groups = tuple(sorted((g.name, tuple(g.members), g.minimum, g.maximum) for g in mandate.groups))
    return (model.assets, mandate.min_weight, mandate.max_weight, mandate.long_only, groups)


def _covariance_root(model: RiskModel) -> np.ndarray:
    """Cᵀ where Σ = CCᵀ, so that wᵀΣw is ‖Cᵀw‖²."""
    matrix = model.covariance
    jitter = 1e-14 * float(np.trace(matrix)) / max(model.assets, 1)
    for attempt in range(4):
        try:
            return np.linalg.cholesky(matrix + np.eye(model.assets) * jitter).T
        except np.linalg.LinAlgError:
            jitter = max(jitter * 100, 1e-12)
    # Symmetric square root, for a matrix a Cholesky will not take.
    values, vectors = np.linalg.eigh(matrix)
    return (vectors * np.sqrt(np.clip(values, 0, None))) @ vectors.T


def _risk_expression(model: RiskModel, weights):
    """wᵀΣw, through the factor structure when there is one.

    ``‖Bᵀw‖² + Σ dᵢwᵢ²`` is the same number as the dense quadratic form, in
    ``k`` inner products rather than ``n²`` multiplications, and it is
    convex by construction so the solver never has to be told the matrix is
    positive semidefinite.
    """
    if model.exposures is not None and model.specific_variance is not None:
        factor = cp.sum_squares(model.exposures.T @ weights)
        specific = cp.sum(cp.multiply(model.specific_variance, cp.square(weights)))
        return factor + specific
    return cp.quad_form(weights, cp.psd_wrap(model.covariance))


def _base_constraints(weights, mandate: Mandate, assets: int) -> tuple[list, dict]:
    """Budget, bounds and group limits, kept addressable so duals can be read."""
    named = {}
    budget = cp.sum(weights) == 1
    named["budget"] = budget
    constraints = [budget]
    if mandate.long_only or mandate.min_weight >= 0:
        lower = weights >= mandate.min_weight
        named["floor"] = lower
        constraints.append(lower)
    upper = weights <= mandate.max_weight
    named["cap"] = upper
    constraints.append(upper)
    for group in mandate.groups:
        selector = np.zeros(assets)
        selector[group.members] = 1.0
        if group.maximum < 1:
            constraint = selector @ weights <= group.maximum
            named[f"group:{group.name}:max"] = constraint
            constraints.append(constraint)
        if group.minimum > 0:
            constraint = selector @ weights >= group.minimum
            named[f"group:{group.name}:min"] = constraint
            constraints.append(constraint)
    return constraints, named


def _mandate_constraints(weights, mandate: Mandate, model: RiskModel, assets: int):
    constraints, named = _base_constraints(weights, mandate, assets)
    if mandate.turnover_budget is not None:
        previous = np.asarray(mandate.previous_weights, dtype=float)
        constraint = cp.norm1(weights - previous) <= mandate.turnover_budget
        named["turnover"] = constraint
        constraints.append(constraint)
    if mandate.tracking_error_limit is not None:
        benchmark = np.asarray(mandate.benchmark, dtype=float)
        active = weights - benchmark
        constraint = _risk_expression(model, active) <= mandate.tracking_error_limit ** 2
        named["tracking_error"] = constraint
        constraints.append(constraint)
    return constraints, named


def _cost_expression(weights, mandate: Mandate):
    """Spread plus impact, both convex in the traded weight.

    Impact rises faster than linearly because filling a larger share of a day's
    volume moves the price against you; the square-root law in price terms is a
    3/2 power once multiplied by the size traded.
    """
    if mandate.previous_weights is None:
        return 0
    if mandate.spread_bps == 0 and mandate.impact_coefficient == 0:
        return 0
    traded = cp.abs(weights - np.asarray(mandate.previous_weights, dtype=float))
    cost = 0
    if mandate.spread_bps:
        cost = cost + (mandate.spread_bps / 10_000) * cp.sum(traded)
    if mandate.impact_coefficient:
        cost = cost + mandate.impact_coefficient * cp.sum(cp.power(traded, mandate.impact_exponent))
    return cost


def _solve(problem: cp.Problem, weights, mandate: Mandate, named: dict,
           label: str, assets: int) -> Solution:
    started = time.perf_counter()
    last_error = None
    warnings.simplefilter("ignore", UserWarning)
    for index, solver in enumerate(SOLVERS):
        try:
            problem.solve(solver=solver, **_tolerances(solver))
        except (cp.error.SolverError, cp.error.DCPError, Exception) as error:  # noqa: BLE001
            last_error = error
            continue
        if problem.status == "optimal":
            break
        # A solver reporting its own answer as inaccurate is worth one more
        # attempt elsewhere before it is accepted; the last one in the list is
        # taken as-is, because something is better than nothing and the status
        # travels with the result.
        if problem.status == "optimal_inaccurate" and index == len(SOLVERS) - 1:
            break
        if problem.status in ("infeasible", "infeasible_inaccurate"):
            break
    elapsed = time.perf_counter() - started

    if problem.status in ("infeasible", "infeasible_inaccurate"):
        raise ValueError(_diagnose(mandate, assets))
    if problem.status is None or problem.status not in ("optimal", "optimal_inaccurate"):
        raise ValueError(
            f"The optimizer returned '{problem.status or last_error}' for {label}. "
            "The mandate may be too tight to satisfy."
        )

    raw = np.asarray(weights.value, dtype=float).ravel()
    if not np.isfinite(raw).all():
        raise ValueError(f"The optimizer returned a non-finite weight for {label}.")
    # Conic solvers land a hair outside the bounds; clip and renormalize rather
    # than hand on a portfolio that breaches its own mandate by 1e-9.
    cleaned = np.clip(raw, max(mandate.min_weight, 0.0), mandate.max_weight)
    total = cleaned.sum()
    cleaned = cleaned / total if total > 0 else np.full(assets, 1 / assets)

    duals, binding = {}, []
    for name, constraint in named.items():
        value = getattr(constraint, "dual_value", None)
        if value is None:
            continue
        magnitude = float(np.max(np.abs(np.atleast_1d(value))))
        duals[name] = magnitude
        if magnitude > 1e-7 and name not in ("budget",):
            binding.append(name)

    return Solution(
        weights=cleaned,
        status=problem.status,
        objective_value=float(problem.value),
        solver=problem.solver_stats.solver_name if problem.solver_stats else "unknown",
        solve_seconds=elapsed,
        duals=duals,
        binding=binding,
        diagnostics={"iterations": getattr(problem.solver_stats, "num_iters", None),
                     "assets": assets},
    )


def _diagnose(mandate: Mandate, assets: int) -> str:
    """Say which requirement cannot be met, not merely that one cannot."""
    reasons = []
    if mandate.max_weight * assets < 1:
        reasons.append(f"a {mandate.max_weight:.0%} cap across {assets} holdings cannot reach 100%")
    if mandate.min_weight * assets > 1:
        reasons.append(f"a {mandate.min_weight:.1%} floor across {assets} holdings exceeds 100%")
    caps = sum(g.maximum for g in mandate.groups) if mandate.groups else None
    if caps is not None and mandate.groups and caps < 1 and _covers_universe(mandate, assets):
        reasons.append(f"group caps add up to {caps:.0%} but must cover the whole portfolio")
    if mandate.turnover_budget is not None and mandate.previous_weights is not None:
        previous = np.asarray(mandate.previous_weights, dtype=float)
        # Anything above the cap has to be sold, and the budget constraint means
        # the same amount has to be bought back somewhere, so the L1 distance
        # cannot be less than twice the forced sales.
        forced = float(np.maximum(previous - mandate.max_weight, 0).sum()
                       + np.maximum(mandate.min_weight - previous, 0).sum())
        if 2 * forced > mandate.turnover_budget + 1e-9:
            reasons.append(
                f"reaching a portfolio inside a {mandate.max_weight:.0%} cap from the current one "
                f"requires at least {2 * forced:.2f} of turnover, against a budget of "
                f"{mandate.turnover_budget:.2f}")
        elif mandate.turnover_budget < 1e-6:
            reasons.append("the turnover budget leaves no room to trade")
    if mandate.tracking_error_limit is not None and mandate.tracking_error_limit < 0.005:
        reasons.append(
            f"a {mandate.tracking_error_limit:.1%} tracking-error limit may be unreachable "
            "under the other constraints")
    detail = "; ".join(reasons) if reasons else "the constraints conflict"
    return f"No portfolio satisfies this mandate: {detail}."


def _covers_universe(mandate: Mandate, assets: int) -> bool:
    covered = set()
    for group in mandate.groups:
        covered.update(group.members)
    return len(covered) == assets


# --------------------------------------------------------------------------- #
# Objectives
# --------------------------------------------------------------------------- #

def minimum_variance(model: RiskModel, mandate: Mandate) -> Solution:
    # Parameters and solver state are mutable: a second request must not replace
    # them while a solve is running.
    with _COMPILED_LOCK:
        return _minimum_variance(model, mandate)


def _minimum_variance(model: RiskModel, mandate: Mandate) -> Solution:
    assets = model.assets
    mandate.validate(assets)
    key = _shape_key(model, mandate)
    if key is not None:
        compiled = _COMPILED.get(("minvar",) + key)
        if compiled is None:
            weights = cp.Variable(assets)
            root = cp.Parameter((assets, assets))
            constraints, named = _base_constraints(weights, mandate, assets)
            problem = cp.Problem(cp.Minimize(cp.sum_squares(root @ weights)), constraints)
            compiled = (problem, root, weights, named)
            _COMPILED[("minvar",) + key] = compiled
            while len(_COMPILED) > 24:
                _COMPILED.popitem(last=False)
        _COMPILED.move_to_end(("minvar",) + key)
        problem, root, weights, named = compiled
        root.value = _covariance_root(model)
        return _solve(problem, weights, mandate, named, "minimum variance", assets)

    weights = cp.Variable(assets)
    constraints, named = _mandate_constraints(weights, mandate, model, assets)
    objective = _risk_expression(model, weights) + _cost_expression(weights, mandate)
    return _solve(cp.Problem(cp.Minimize(objective), constraints), weights, mandate,
                  named, "minimum variance", assets)


def mean_variance(model: RiskModel, expected: np.ndarray, risk_aversion: float,
                  mandate: Mandate) -> Solution:
    """The Markowitz trade-off at a stated risk aversion."""
    assets = model.assets
    mandate.validate(assets)
    weights = cp.Variable(assets)
    constraints, named = _mandate_constraints(weights, mandate, model, assets)
    objective = (risk_aversion / 2) * _risk_expression(model, weights) \
        - expected @ weights + _cost_expression(weights, mandate)
    return _solve(cp.Problem(cp.Minimize(objective), constraints), weights, mandate,
                  named, "mean variance", assets)


def maximum_sharpe(model: RiskModel, expected: np.ndarray, risk_free: float,
                   mandate: Mandate) -> Solution:
    """The tangency portfolio, by convex reformulation where that is valid.

    Optimizing over an unnormalized ``y`` with ``(μ − rf)ᵀy = 1`` turns the ratio
    into a quadratic minimization. The rescaling is only sound while every
    constraint is homogeneous in ``y``; when it is not, the frontier is scanned
    instead and the best risk-adjusted point on it taken, which is a sequence of
    convex problems rather than one.
    """
    assets = model.assets
    mandate.validate(assets)
    excess = np.asarray(expected, dtype=float) - risk_free
    if excess.max() <= 0:
        raise ValueError(
            "No holding has an expected return above the risk-free rate over this window, "
            "so a tangency portfolio is not defined. Use minimum variance or risk parity."
        )
    if not mandate.is_homogeneous:
        return _sharpe_by_frontier(model, expected, risk_free, mandate)

    # Same normalization as maximum diversification, and for the same reason.
    scale = float(np.max(np.abs(excess))) or 1.0
    excess = excess / scale
    model = RiskModel(
        covariance=model.covariance / scale ** 2, kind=model.kind, observations=model.observations,
        exposures=None if model.exposures is None else model.exposures / scale,
        specific_variance=None if model.specific_variance is None else model.specific_variance / scale ** 2,
    )
    y = cp.Variable(assets)
    kappa = cp.Variable(nonneg=True)
    constraints = [excess @ y == 1, cp.sum(y) == kappa, y >= 0, y <= kappa * mandate.max_weight]
    named = {"scale": constraints[1]}
    for group in mandate.groups:
        selector = np.zeros(assets)
        selector[group.members] = 1.0
        if group.maximum < 1:
            constraint = selector @ y <= group.maximum * kappa
            named[f"group:{group.name}:max"] = constraint
            constraints.append(constraint)
        if group.minimum > 0:
            constraint = selector @ y >= group.minimum * kappa
            named[f"group:{group.name}:min"] = constraint
            constraints.append(constraint)

    problem = cp.Problem(cp.Minimize(_risk_expression(model, y)), constraints)
    solution = _solve(problem, y, Mandate(max_weight=1.0), named, "maximum Sharpe", assets)
    scale = solution.weights.sum()
    solution.weights = solution.weights / scale if scale > 0 else solution.weights
    solution.diagnostics["reformulation"] = "Schaible transform"
    return solution


def _sharpe_by_frontier(model: RiskModel, expected: np.ndarray, risk_free: float,
                        mandate: Mandate, points: int = 18) -> Solution:
    curve = efficient_frontier(model, expected, mandate, points=points)
    if not curve["solutions"]:
        raise ValueError("No portfolio satisfies this mandate.")
    ratios = [
        (ret - risk_free) / vol if vol > 0 else -np.inf
        for ret, vol in zip(curve["returns"], curve["volatilities"])
    ]
    best = curve["solutions"][int(np.argmax(ratios))]
    best.diagnostics["reformulation"] = "frontier scan (constraints are not homogeneous)"
    return best


def maximum_diversification(model: RiskModel, mandate: Mandate) -> Solution:
    """Maximize weighted standalone volatility over portfolio volatility.

    The same ratio reformulation as maximum Sharpe, against volatilities rather
    than expected returns, so it needs no return forecast at all.
    """
    assets = model.assets
    mandate.validate(assets)
    standalone = np.sqrt(np.diag(model.covariance))
    # The ratio transform pins a linear functional to one, so the optimizing
    # variable inherits the units of the input: quote returns in percent instead
    # of decimals and y shrinks a hundredfold, while the solver's tolerances stay
    # where they are. Normalizing the scale first keeps the variable near one and
    # makes the answer independent of the units, which it should be.
    scale = float(standalone.mean()) or 1.0
    standalone = standalone / scale
    model = RiskModel(
        covariance=model.covariance / scale ** 2, kind=model.kind, observations=model.observations,
        exposures=None if model.exposures is None else model.exposures / scale,
        specific_variance=None if model.specific_variance is None else model.specific_variance / scale ** 2,
    )
    y = cp.Variable(assets)
    constraints = [standalone @ y == 1, y >= 0]
    named = {"scale": constraints[0]}
    if mandate.max_weight < 1:
        constraint = y <= mandate.max_weight * cp.sum(y)
        named["cap"] = constraint
        constraints.append(constraint)
    problem = cp.Problem(cp.Minimize(_risk_expression(model, y)), constraints)
    solution = _solve(problem, y, Mandate(max_weight=1.0), named, "maximum diversification", assets)
    total = solution.weights.sum()
    solution.weights = solution.weights / total if total > 0 else solution.weights
    solution.diagnostics["reformulation"] = "ratio transform"
    return solution


def risk_parity(model: RiskModel, mandate: Mandate, budget: np.ndarray | None = None) -> Solution:
    """Equal risk contribution, as a convex program rather than a search.

    ``min ½wᵀΣw − Σ bᵢ log wᵢ`` has first-order condition ``Σw = b / w``, which
    says each holding's risk contribution is proportional to its budget. The
    solution is unique, needs no restarts, and only has to be rescaled to sum to
    one. Written instead as "minimize the dispersion of risk contributions" the
    same problem is not convex, which is why a local search was needed before.
    """
    assets = model.assets
    mandate.validate(assets)
    shares = np.full(assets, 1 / assets) if budget is None else np.asarray(budget, dtype=float)
    shares = shares / shares.sum()

    y = cp.Variable(assets, nonneg=True)
    objective = 0.5 * _risk_expression(model, y) - shares @ cp.log(y)
    problem = cp.Problem(cp.Minimize(objective), [])
    started = time.perf_counter()
    for solver in ("CLARABEL", "SCS"):
        try:
            problem.solve(solver=solver, **_tolerances(solver, tight=True))
            if problem.status in ("optimal", "optimal_inaccurate"):
                break
        except Exception:  # noqa: BLE001
            continue
    elapsed = time.perf_counter() - started
    if problem.status not in ("optimal", "optimal_inaccurate"):
        raise ValueError(f"The risk-parity program returned '{problem.status}'.")

    raw = np.asarray(y.value, dtype=float).ravel()
    # Polish the log-barrier first-order condition when conic termination leaves
    # a small residual. Newton steps retain positivity through backtracking.
    covariance = model.covariance
    for _ in range(12):
        gradient = covariance @ raw - shares / raw
        if np.max(np.abs(gradient)) < 1e-13:
            break
        direction = np.linalg.solve(covariance + np.diag(shares / raw ** 2), gradient)
        scale = 1.0
        residual = np.linalg.norm(gradient)
        while scale > 1e-8:
            candidate = raw - scale * direction
            if np.all(candidate > 0) and np.linalg.norm(covariance @ candidate - shares / candidate) < residual:
                raw = candidate
                break
            scale *= 0.5
        else:
            raise ValueError("Risk-parity refinement failed to reduce the optimality residual.")
    weights = raw / raw.sum()
    if weights.max() > mandate.max_weight + 1e-6:
        # The unconstrained solution breaches the cap, so re-solve as a
        # constrained least-squares fit to it. Exact equal contribution and a
        # binding cap cannot both hold; the cap is the mandate and wins.
        target = weights
        w = cp.Variable(assets)
        constraints, named = _mandate_constraints(w, mandate, model, assets)
        fit = cp.Problem(cp.Minimize(cp.sum_squares(w - target)), constraints)
        solution = _solve(fit, w, mandate, named, "risk parity under a cap", assets)
        solution.diagnostics["note"] = "projected onto the mandate; contributions are no longer exactly equal"
        return solution

    return Solution(weights=weights, status=problem.status, objective_value=float(problem.value),
                    solver=problem.solver_stats.solver_name if problem.solver_stats else "unknown",
                    solve_seconds=elapsed, duals={}, binding=[],
                    diagnostics={"reformulation": "log-barrier", "assets": assets})


def target_volatility(model: RiskModel, expected: np.ndarray, target: float,
                      mandate: Mandate) -> Solution:
    """The highest expected return inside a volatility budget."""
    assets = model.assets
    mandate.validate(assets)
    weights = cp.Variable(assets)
    constraints, named = _mandate_constraints(weights, mandate, model, assets)
    cap = _risk_expression(model, weights) <= target ** 2
    named["volatility_target"] = cap
    constraints.append(cap)
    objective = expected @ weights - _cost_expression(weights, mandate)
    try:
        return _solve(cp.Problem(cp.Maximize(objective), constraints), weights, mandate,
                      named, "volatility target", assets)
    except ValueError:
        # Nothing in the universe is calm enough; the floor portfolio is the
        # honest answer, and the caller is told rather than quietly given it.
        floor = minimum_variance(model, mandate)
        floor.diagnostics["note"] = (
            f"the {target:.1%} volatility target is below the minimum achievable "
            f"{model.volatility(floor.weights):.1%}, so the minimum-variance portfolio was returned")
        return floor


def minimum_tracking_error(model: RiskModel, mandate: Mandate,
                           expected: np.ndarray | None = None,
                           risk_aversion: float = 0.0) -> Solution:
    """Optimize against a benchmark rather than in absolute terms.

    Most mandates are relative: the risk that matters is the risk of differing
    from the index, not the risk of the index itself. This minimizes active
    variance, optionally trading it against expected active return.
    """
    if mandate.benchmark is None:
        raise ValueError("Benchmark-relative optimization needs a benchmark.")
    assets = model.assets
    mandate.validate(assets)
    benchmark = np.asarray(mandate.benchmark, dtype=float)
    weights = cp.Variable(assets)
    constraints, named = _mandate_constraints(weights, mandate, model, assets)
    active = weights - benchmark
    objective = _risk_expression(model, active) + _cost_expression(weights, mandate)
    if expected is not None and risk_aversion > 0:
        objective = objective - (1 / risk_aversion) * (expected @ active)
    return _solve(cp.Problem(cp.Minimize(objective), constraints), weights, mandate,
                  named, "minimum tracking error", assets)


def efficient_frontier(model: RiskModel, expected: np.ndarray, mandate: Mandate,
                       points: int = 24) -> dict:
    """Minimum variance at each of a grid of target returns."""
    assets = model.assets
    mandate.validate(assets)
    expected = np.asarray(expected, dtype=float)

    floor = minimum_variance(model, mandate)
    lowest = float(expected @ floor.weights)

    weights = cp.Variable(assets)
    constraints, named = _mandate_constraints(weights, mandate, model, assets)
    top = _solve(cp.Problem(cp.Maximize(expected @ weights), constraints), weights, mandate,
                 named, "maximum return", assets)
    highest = float(expected @ top.weights)
    if highest <= lowest + 1e-12:
        return {"returns": [lowest], "volatilities": [model.volatility(floor.weights)],
                "weights": [floor.weights], "solutions": [floor]}

    returns, volatilities, solutions = [], [], []
    for target in np.linspace(lowest, highest, points):
        w = cp.Variable(assets)
        cons, nm = _mandate_constraints(w, mandate, model, assets)
        pin = expected @ w >= float(target)
        nm["return_target"] = pin
        cons.append(pin)
        try:
            solution = _solve(cp.Problem(cp.Minimize(_risk_expression(model, w)), cons),
                              w, mandate, nm, "frontier point", assets)
        except ValueError:
            continue
        returns.append(float(expected @ solution.weights))
        volatilities.append(model.volatility(solution.weights))
        solutions.append(solution)
    return {"returns": returns, "volatilities": volatilities,
            "weights": [s.weights for s in solutions], "solutions": solutions}


def enforce_cardinality(model: RiskModel, mandate: Mandate, solve, holdings: int) -> Solution:
    """Hold at most ``holdings`` names, by a two-pass heuristic.

    Cardinality is a combinatorial constraint: written exactly it is a mixed
    integer program, and this system does not carry a MIP solver. The heuristic
    is to solve the relaxation, keep the largest positions, and re-solve on that
    subset. That is a sound way to get a sparse portfolio and it is **not**
    guaranteed optimal, which is why it says so here rather than being presented
    as one.
    """
    full = solve(model, mandate)
    if holdings >= model.assets or (full.weights > 1e-6).sum() <= holdings:
        return full
    keep = np.argsort(full.weights)[::-1][:holdings]
    keep.sort()
    reduced = RiskModel(
        covariance=model.covariance[np.ix_(keep, keep)],
        kind=model.kind, observations=model.observations,
        shrinkage_intensity=model.shrinkage_intensity,
        exposures=None if model.exposures is None else model.exposures[keep],
        specific_variance=None if model.specific_variance is None else model.specific_variance[keep],
    )
    trimmed = Mandate(max_weight=mandate.max_weight, min_weight=mandate.min_weight,
                      long_only=mandate.long_only)
    partial = solve(reduced, trimmed)
    weights = np.zeros(model.assets)
    weights[keep] = partial.weights
    partial.weights = weights
    partial.diagnostics["cardinality"] = {
        "requested": holdings, "method": "two-pass heuristic, not a proven optimum"}
    return partial
