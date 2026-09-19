"""Tests for convex construction and the factor risk models.

Three kinds of check. Against a closed form or a known structure, where one
exists: a two-asset minimum-variance weight, the number of factors in data that
was built with a known number, the exact adding-up of a variance decomposition.
Against the defining property, where no closed form exists: risk parity must
equalize risk contributions, a constrained optimum must satisfy its constraints,
a looser constraint can never give a worse objective. And against the previous
implementation, so that replacing a local search with a convex program is shown
to change the method rather than the answer.
"""

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app import convex, optimize, riskmodel, strategies  # noqa: E402
from app.convex import Group, Mandate  # noqa: E402
from app.optimize import Constraints, risk_contributions  # noqa: E402


def factor_returns(assets=40, days=750, factors=4, seed=5, noise=(0.004, 0.018)):
    """Returns with a known factor structure, so the model has a truth to find."""
    rng = np.random.default_rng(seed)
    loadings = rng.normal(0, 1, (assets, factors)) * 0.011
    common = rng.normal(0, 1, (days, factors))
    specific = rng.normal(0, 1, (days, assets)) * rng.uniform(*noise, assets)
    return common @ loadings.T + specific


@pytest.fixture(scope="module")
def returns():
    return factor_returns()


@pytest.fixture(scope="module")
def model(returns):
    return riskmodel.build(returns, "ledoit_wolf")


# --------------------------------------------------------------------------- #
# Risk models
# --------------------------------------------------------------------------- #

def test_the_factor_count_comes_from_the_noise_threshold_not_a_guess():
    """Marchenko-Pastur: components below the edge are indistinguishable from noise."""
    built = riskmodel.build(factor_returns(assets=200, days=500, factors=3, seed=9),
                            "statistical_factor")
    assert built.factors == 3
    leading = built.diagnostics["leading_eigenvalues"]
    assert leading[2] > built.noise_edge > leading[3]


def test_a_factor_model_stays_invertible_where_a_sample_matrix_does_not():
    """More assets than observations makes the sample estimate singular."""
    thin = factor_returns(assets=60, days=40, factors=2, seed=1)
    sample = riskmodel.build(thin, "sample")
    factor = riskmodel.build(thin, "statistical_factor")
    assert np.linalg.eigvalsh(sample.covariance).min() < 1e-12
    assert np.linalg.eigvalsh(factor.covariance).min() > 0
    assert factor.condition_number() < sample.condition_number()


def test_variance_splits_exactly_into_factor_and_specific(returns):
    built = riskmodel.build(returns, "statistical_factor")
    weights = np.full(built.assets, 1 / built.assets)
    parts = built.attribution(weights)
    assert parts["factor_variance"] + parts["specific_variance"] == pytest.approx(
        parts["total_variance"], rel=1e-10)
    assert sum(f["variance"] for f in parts["by_factor"]) == pytest.approx(
        parts["factor_variance"], rel=1e-10)
    assert built.volatility(weights) ** 2 == pytest.approx(parts["total_variance"], rel=1e-10)


def test_without_a_factor_structure_nothing_is_attributed_to_factors(model):
    parts = model.attribution(np.full(model.assets, 1 / model.assets))
    assert parts["factor_share"] == 0
    assert parts["specific_variance"] == pytest.approx(parts["total_variance"])


def test_the_factor_model_reproduces_the_sample_covariance_closely(returns):
    """A reduction has to stay near what it reduces, or it is a different matrix."""
    sample = riskmodel.build(returns, "sample").covariance
    factor = riskmodel.build(returns, "statistical_factor").covariance
    scale = np.sqrt(np.outer(np.diag(sample), np.diag(sample)))
    relative = np.abs(factor - sample) / scale
    assert np.median(relative) < 0.08
    assert np.allclose(np.diag(factor), np.diag(sample), rtol=0.05)


# --------------------------------------------------------------------------- #
# Convex programs versus the local search
# --------------------------------------------------------------------------- #

def test_convex_minimum_variance_matches_the_two_asset_closed_form():
    covariance = np.array([[0.04, 0.006], [0.006, 0.09]])
    built = riskmodel.RiskModel(covariance=covariance, kind="sample", observations=500)
    weights = convex.minimum_variance(built, Mandate()).weights
    v1, v2, cov = covariance[0, 0], covariance[1, 1], covariance[0, 1]
    assert weights[0] == pytest.approx((v2 - cov) / (v1 + v2 - 2 * cov), rel=1e-6)


def test_the_convex_program_agrees_with_the_local_search(model):
    """Same answer, different guarantees: this is a change of method, not of result."""
    for cap in (1.0, 0.15, 0.08):
        convexed = convex.minimum_variance(model, Mandate(max_weight=cap)).weights
        searched = optimize.minimum_variance(model.covariance, Constraints(max_weight=cap))
        assert model.volatility(convexed) == pytest.approx(model.volatility(searched), rel=1e-4)
        assert np.abs(convexed - searched).max() < 5e-3


def test_the_convex_form_equalizes_risk_contributions_where_the_search_drifts(returns):
    """The reason for the rewrite, stated as a test.

    Minimizing the dispersion of risk contributions is not a convex problem, and
    a local search on it degrades as the universe grows. The log-barrier form is
    convex, so it does not.
    """
    wide = riskmodel.build(returns, "ledoit_wolf")          # 40 assets
    convexed = convex.risk_parity(wide, Mandate()).weights
    searched = optimize.risk_parity(wide.covariance, Constraints())

    spread = lambda w: float(np.ptp(risk_contributions(w, wide.covariance)["share"]))  # noqa: E731
    assert spread(convexed) < 1e-5
    assert spread(convexed) < spread(searched)


def test_risk_parity_gives_the_calmer_asset_more_of_the_money():
    covariance = np.diag([0.04, 0.01])
    built = riskmodel.RiskModel(covariance=covariance, kind="sample", observations=500)
    weights = convex.risk_parity(built, Mandate()).weights
    assert weights[1] > weights[0]
    assert weights[1] / weights[0] == pytest.approx(2.0, rel=1e-4)   # inverse volatility when uncorrelated


def test_maximum_sharpe_is_solved_by_transform_when_it_can_be(model, returns):
    means, _ = optimize.shrink_means(returns)
    solution = convex.maximum_sharpe(model, means, 0.03, Mandate(max_weight=0.2))
    assert solution.diagnostics["reformulation"] == "Schaible transform"
    sharpe = lambda w: (w @ means - 0.03) / model.volatility(w)  # noqa: E731
    assert sharpe(solution.weights) >= sharpe(np.full(model.assets, 1 / model.assets))


def test_maximum_sharpe_falls_back_when_the_constraints_are_not_homogeneous(model, returns):
    """A turnover budget is an absolute limit, so the rescaling argument fails."""
    means, _ = optimize.shrink_means(returns)
    previous = np.full(model.assets, 1 / model.assets)
    solution = convex.maximum_sharpe(
        model, means, 0.03,
        Mandate(max_weight=0.2, previous_weights=previous, turnover_budget=0.4))
    assert "frontier" in solution.diagnostics["reformulation"]
    assert np.abs(solution.weights - previous).sum() <= 0.4 + 1e-4


# --------------------------------------------------------------------------- #
# The mandate
# --------------------------------------------------------------------------- #

def test_group_limits_are_respected(model):
    groups = [Group("alpha", list(range(0, 10)), maximum=0.25),
              Group("beta", list(range(10, 20)), minimum=0.15, maximum=0.5)]
    weights = convex.minimum_variance(model, Mandate(max_weight=0.1, groups=groups)).weights
    assert weights[:10].sum() <= 0.25 + 1e-6
    assert 0.15 - 1e-6 <= weights[10:20].sum() <= 0.5 + 1e-6


def test_a_turnover_budget_is_respected(model):
    previous = np.full(model.assets, 1 / model.assets)
    solution = convex.minimum_variance(
        model, Mandate(max_weight=0.1, previous_weights=previous, turnover_budget=0.25))
    assert np.abs(solution.weights - previous).sum() <= 0.25 + 1e-5
    assert "turnover" in solution.binding


def test_a_turnover_budget_that_cannot_be_met_says_why(model):
    """Moving off a concentrated book into a capped one forces a known amount of trading."""
    previous = np.zeros(model.assets)
    previous[:5] = 0.2                       # five names at 20%, against a 10% cap
    with pytest.raises(ValueError, match="requires at least 1.00 of turnover"):
        convex.minimum_variance(
            model, Mandate(max_weight=0.1, previous_weights=previous, turnover_budget=0.25))


def test_a_tracking_error_limit_is_respected(model):
    benchmark = np.full(model.assets, 1 / model.assets)
    solution = convex.minimum_variance(
        model, Mandate(max_weight=0.08, benchmark=benchmark, tracking_error_limit=0.02))
    assert model.volatility(solution.weights - benchmark) <= 0.02 + 1e-4


def test_tightening_a_constraint_can_only_cost(model):
    """A smaller feasible set cannot contain a better optimum."""
    loose = convex.minimum_variance(model, Mandate(max_weight=0.2))
    tight = convex.minimum_variance(model, Mandate(max_weight=0.06))
    assert model.volatility(tight.weights) >= model.volatility(loose.weights) - 1e-9


def test_a_binding_constraint_has_a_price_and_a_slack_one_does_not(model):
    binding = convex.minimum_variance(model, Mandate(max_weight=0.04))
    assert binding.duals["cap"] > 1e-7
    assert "cap" in binding.binding
    slack = convex.minimum_variance(model, Mandate(max_weight=0.95))
    assert slack.duals.get("cap", 0.0) < 1e-7


def test_costs_in_the_objective_reduce_trading(model):
    previous = np.full(model.assets, 1 / model.assets)
    free = convex.minimum_variance(model, Mandate(max_weight=0.1, previous_weights=previous))
    priced = convex.minimum_variance(model, Mandate(
        max_weight=0.1, previous_weights=previous, spread_bps=50, impact_coefficient=2.0))
    assert np.abs(priced.weights - previous).sum() < np.abs(free.weights - previous).sum()


def test_benchmark_relative_construction_beats_absolute_on_active_risk(model):
    benchmark = np.full(model.assets, 1 / model.assets)
    relative = convex.minimum_tracking_error(model, Mandate(max_weight=0.1, benchmark=benchmark))
    absolute = convex.minimum_variance(model, Mandate(max_weight=0.1))
    assert model.volatility(relative.weights - benchmark) <= model.volatility(absolute.weights - benchmark)


def test_an_unreachable_volatility_target_returns_the_calmest_portfolio(model, returns):
    means, _ = optimize.shrink_means(returns)
    floor = convex.minimum_variance(model, Mandate())
    solution = convex.target_volatility(model, means, model.volatility(floor.weights) * 0.5, Mandate())
    assert "note" in solution.diagnostics
    assert model.volatility(solution.weights) == pytest.approx(model.volatility(floor.weights), rel=1e-3)


@pytest.mark.parametrize("mandate,expected", [
    (Mandate(max_weight=0.01), "cannot be spread"),
    (Mandate(max_weight=0.2, min_weight=0.1), "requires"),
    (Mandate(groups=[Group("a", [0, 1], minimum=0.8), Group("b", [2, 3], minimum=0.8)]), "add up"),
])
def test_an_impossible_mandate_is_explained(model, mandate, expected):
    with pytest.raises(ValueError, match=expected):
        convex.minimum_variance(model, mandate)


def test_the_frontier_is_monotone_and_every_point_is_feasible(model, returns):
    means, _ = optimize.shrink_means(returns)
    curve = convex.efficient_frontier(model, means, Mandate(max_weight=0.2), points=10)
    volatilities = np.array(curve["volatilities"])
    assert len(volatilities) >= 5
    assert np.all(np.diff(np.array(curve["returns"])) > -1e-9)
    start = int(np.argmin(volatilities))
    assert np.all(np.diff(volatilities[start:]) > -1e-6)
    assert all(w.max() <= 0.2 + 1e-6 and abs(w.sum() - 1) < 1e-6 for w in curve["weights"])


def test_cardinality_is_a_heuristic_and_says_so(model):
    solution = convex.enforce_cardinality(model, Mandate(max_weight=0.3),
                                          convex.minimum_variance, holdings=8)
    assert (solution.weights > 1e-6).sum() <= 8
    assert "heuristic" in solution.diagnostics["cardinality"]["method"]


# --------------------------------------------------------------------------- #
# Scale
# --------------------------------------------------------------------------- #

def test_a_five_hundred_name_universe_solves_quickly():
    """The reason the factor model exists."""
    wide = factor_returns(assets=500, days=750, factors=6, seed=3)
    built = riskmodel.build(wide, "statistical_factor")
    started = time.perf_counter()
    solution = convex.minimum_variance(built, Mandate(max_weight=0.02))
    elapsed = time.perf_counter() - started
    assert solution.optimal
    assert elapsed < 5.0
    assert solution.weights.max() <= 0.02 + 1e-6
    assert solution.weights.sum() == pytest.approx(1)


def test_the_factor_form_and_the_dense_form_give_the_same_answer():
    """‖Bᵀw‖² + Σdᵢwᵢ² is the quadratic form, not an approximation of it."""
    wide = factor_returns(assets=80, days=600, factors=4, seed=8)
    built = riskmodel.build(wide, "statistical_factor")
    dense = riskmodel.RiskModel(covariance=built.covariance, kind="sample",
                                observations=built.observations)
    through_factors = convex.minimum_variance(built, Mandate(max_weight=0.05)).weights
    through_matrix = convex.minimum_variance(dense, Mandate(max_weight=0.05)).weights
    assert np.abs(through_factors - through_matrix).max() < 1e-5


# --------------------------------------------------------------------------- #
# Through the registry
# --------------------------------------------------------------------------- #

def test_the_convex_methodologies_are_registered_and_conform():
    from app import conformance

    names = ["minimum_variance_convex", "risk_parity_convex", "maximum_sharpe_convex",
             "maximum_diversification_convex", "minimum_variance_after_costs",
             "active_return_at_risk_budget"]
    assert set(names) <= set(strategies.REGISTRY)
    failures = {r.strategy: r.to_dict()["failures"]
                for r in conformance.run_all(names) if not r.passed}
    assert not failures, failures


def test_a_strategy_can_ask_for_a_factor_risk_model(returns):
    context = strategies.Context(returns=returns, constraints=Constraints(max_weight=0.1),
                                 parameters={"risk_model": "statistical_factor"})
    solved = strategies.solve_with_diagnostics(strategies.get("minimum_variance_convex"), context)
    assert context.risk_model().kind == "statistical_factor"
    assert context.risk_model().factors >= 1
    assert solved["weights"].max() <= 0.1 + 1e-6


def test_the_mandate_does_not_inherit_objectives_from_the_parameter_bag(returns):
    """A tracking-error limit changes what is being optimized, so it is opt-in.

    This caught a real defect: parameters meant for one methodology were being
    picked up by every other, so "minimum variance" was quietly solving a
    benchmark-relative problem.
    """
    context = strategies.Context(
        returns=returns, constraints=Constraints(max_weight=0.1),
        benchmark=np.full(returns.shape[1], 1 / returns.shape[1]),
        parameters={"tracking_error_limit": 0.01, "spread_bps": 100})
    mandate = context.mandate()
    assert mandate.tracking_error_limit is None
    assert mandate.spread_bps == 0


# --------------------------------------------------------------------------- #
# The browser optimizer must agree with cvxpy
# --------------------------------------------------------------------------- #

NODE = shutil.which("node")


def solve_in_browser(objective: str, returns: np.ndarray, cap: float,
                     estimator: str = "ledoit_wolf") -> np.ndarray:
    """Run app/static/optimize.js through Node and return its weights."""
    with tempfile.TemporaryDirectory() as folder:
        request = Path(folder) / "request.json"
        request.write_text(json.dumps({
            "objective": objective, "returns": returns.tolist(),
            "constraints": {"max_weight": cap, "min_weight": 0.0},
            "estimator": estimator,
        }))
        completed = subprocess.run(
            [NODE, str(ROOT / "tools/js_optimizer_harness.mjs"), str(request)],
            capture_output=True, text=True, check=True, cwd=ROOT,
        )
    return np.array(json.loads(completed.stdout)["weights"])


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("objective,tolerance", [
    ("minimum_variance", 2e-6),   # accelerated projected gradient
    ("risk_parity", 1e-9),        # closed-form coordinate updates
])
@pytest.mark.parametrize("assets,cap", [(8, 1.0), (8, 0.25), (20, 0.15), (5, 0.35)])
def test_the_browser_optimizer_matches_cvxpy(objective, tolerance, assets, cap):
    """The offline build solves the same programs, without a solver library.

    The browser cannot run cvxpy, so the demo implements these two objectives
    directly: minimum variance by accelerated projected gradient, risk parity by
    the closed-form coordinate updates of the log-barrier form. That is only
    worth doing if the answers agree, which is what this checks.
    """
    returns = factor_returns(assets=assets, days=504, factors=3, seed=assets * 7 + int(cap * 100))
    model = riskmodel.build(returns, "ledoit_wolf")
    mandate = Mandate(max_weight=cap)
    expected = (convex.minimum_variance(model, mandate) if objective == "minimum_variance"
                else convex.risk_parity(model, mandate)).weights
    actual = solve_in_browser(objective, returns, cap)

    assert np.abs(actual - expected).max() < tolerance
    assert actual.sum() == pytest.approx(1, abs=1e-9)
    assert actual.max() <= cap + 1e-6
    # And the portfolio it produces is as good, which is what actually matters.
    assert model.volatility(actual) == pytest.approx(model.volatility(expected), rel=1e-6)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_browser_shrinkage_matches_the_python_one():
    returns = factor_returns(assets=10, days=400, factors=3, seed=21)
    with tempfile.TemporaryDirectory() as folder:
        request = Path(folder) / "request.json"
        request.write_text(json.dumps({
            "objective": "minimum_variance", "returns": returns.tolist(),
            "constraints": {"max_weight": 1.0, "min_weight": 0.0}, "estimator": "ledoit_wolf"}))
        payload = json.loads(subprocess.run(
            [NODE, str(ROOT / "tools/js_optimizer_harness.mjs"), str(request)],
            capture_output=True, text=True, check=True, cwd=ROOT).stdout)
    _, intensity = riskmodel.ledoit_wolf_covariance(returns)
    assert payload["shrinkage_intensity"] == pytest.approx(intensity, rel=1e-9)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_browser_refuses_objectives_it_cannot_solve_exactly():
    """Better to decline than to return a different answer under the same name."""
    returns = factor_returns(assets=6, days=300, seed=4)
    with tempfile.TemporaryDirectory() as folder:
        request = Path(folder) / "request.json"
        request.write_text(json.dumps({
            "objective": "maximum_sharpe_convex", "returns": returns.tolist(),
            "constraints": {"max_weight": 0.5, "min_weight": 0.0}, "estimator": "ledoit_wolf"}))
        completed = subprocess.run(
            [NODE, str(ROOT / "tools/js_optimizer_harness.mjs"), str(request)],
            capture_output=True, text=True, cwd=ROOT)
    assert completed.returncode != 0
    assert "Python app" in completed.stderr
