"""Tests for the strategy contract and the gate in front of it.

A harness nobody has tried to fool is decoration. Most of this file defines
methodologies that are broken in specific, realistic ways — one that ignores the
weight cap, one whose answer changes between calls, one that assumes the assets
arrive in a fixed order, one that returns NaN on a bad data day — and asserts
that the harness fails each of them, on the check that names the fault.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app import analysis, conformance, strategies  # noqa: E402
from app.engine import Portfolio, Settings  # noqa: E402
from app.optimize import Constraints, minimum_variance  # noqa: E402
from app.strategies import Context, Strategy  # noqa: E402

from tests.application.test_optimize import synthetic_prices  # noqa: E402


@pytest.fixture
def returns():
    return conformance.sample_returns()


def context(returns, **kwargs):
    return Context(returns=returns, constraints=Constraints(),
                   parameters={"volatility_target": 0.12, "turnover_lambda": 1.0}, **kwargs)


def strategy(solve, **kwargs) -> Strategy:
    return Strategy(name=kwargs.pop("name", "candidate"), label="Candidate",
                    description="A methodology under test.", solve=solve, **kwargs)


# --------------------------------------------------------------------------- #
# The registry
# --------------------------------------------------------------------------- #

def test_the_shipped_methodologies_are_registered():
    names = set(strategies.REGISTRY)
    assert {"minimum_variance", "risk_parity", "maximum_diversification",
            "maximum_sharpe", "minimum_variance_net_of_costs"} <= names


def test_the_catalogue_says_which_methodologies_need_return_forecasts():
    listed = {entry["name"]: entry for entry in strategies.catalogue()}
    assert listed["maximum_sharpe"]["needs_expected_returns"] is True
    assert listed["minimum_variance"]["needs_expected_returns"] is False
    assert all(entry["label"] and entry["description"] for entry in listed.values())


def test_an_unknown_methodology_names_what_is_available():
    with pytest.raises(ValueError, match="Registered:"):
        strategies.get("no_such_method")


def test_registering_a_duplicate_name_is_refused():
    with pytest.raises(ValueError, match="already registered"):
        strategies.register(strategy(lambda ctx: np.full(ctx.assets, 1 / ctx.assets),
                                     name="minimum_variance"))


def test_every_registered_methodology_conforms():
    """The gate applies to what shipped, not only to what arrives next."""
    reports = conformance.run_all()
    failures = {r.strategy: r.to_dict()["failures"] for r in reports if not r.passed}
    assert not failures, failures


# --------------------------------------------------------------------------- #
# The validity gate at the boundary
# --------------------------------------------------------------------------- #

def test_weights_that_do_not_add_up_are_rejected_before_anyone_trades(returns):
    half = strategy(lambda ctx: np.full(ctx.assets, 0.5 / ctx.assets))
    with pytest.raises(ValueError, match="summing to"):
        strategies.solve_with_diagnostics(half, context(returns))


def test_a_negative_weight_in_a_long_only_mandate_is_rejected(returns):
    def shorting(ctx):
        weights = np.full(ctx.assets, 1.2 / (ctx.assets - 1))
        weights[0] = 1 - 1.2          # fully invested, but short the first name
        return weights
    with pytest.raises(ValueError, match="negative weight"):
        strategies.solve_with_diagnostics(strategy(shorting), context(returns))


def test_breaching_the_cap_is_rejected_rather_than_clipped_silently(returns):
    def concentrated(ctx):
        weights = np.zeros(ctx.assets)
        weights[0] = 1.0
        return weights
    ctx = Context(returns=returns, constraints=Constraints(max_weight=0.3))
    with pytest.raises(ValueError, match="breached"):
        strategies.solve_with_diagnostics(strategy(concentrated), ctx)


def test_a_non_finite_weight_is_rejected(returns):
    broken = strategy(lambda ctx: np.full(ctx.assets, np.nan))
    with pytest.raises(ValueError, match="non-finite"):
        strategies.solve_with_diagnostics(broken, context(returns))


def test_the_wrong_number_of_weights_is_caught_by_the_contract(returns):
    stumpy = strategy(lambda ctx: np.array([0.5, 0.5]))
    with pytest.raises(ValueError, match="weights for"):
        stumpy(context(returns))


# --------------------------------------------------------------------------- #
# The harness must fail methodologies that deserve to fail
# --------------------------------------------------------------------------- #

def failed_checks(report) -> set[str]:
    return {check.name for check in report.checks if check.required and not check.passed}


def test_the_harness_catches_a_methodology_that_ignores_the_cap():
    def ignores_cap(ctx):
        weights = np.zeros(ctx.assets)
        weights[:2] = 0.5
        return weights
    report = conformance.evaluate(strategy(ignores_cap))
    assert not report.passed
    assert "honours a weight cap" in failed_checks(report)


def test_the_harness_catches_a_methodology_that_is_not_deterministic():
    def wanders(ctx):
        noise = np.random.default_rng().normal(0, 0.01, ctx.assets)
        weights = np.abs(np.full(ctx.assets, 1 / ctx.assets) + noise)
        return weights / weights.sum()
    report = conformance.evaluate(strategy(wanders))
    assert not report.passed
    assert "deterministic" in failed_checks(report)


def test_the_harness_catches_a_methodology_that_assumes_the_asset_order():
    """The classic production bug: an index carried in a comment, not in the code."""
    def positional(ctx):
        weights = np.full(ctx.assets, 0.5 / (ctx.assets - 1))
        weights[0] = 0.5          # "the first column is always the defensive one"
        return weights / weights.sum()
    report = conformance.evaluate(strategy(positional))
    assert not report.passed
    assert "independent of asset order" in failed_checks(report)


def test_the_harness_catches_a_methodology_that_breaks_on_degenerate_data():
    """Shrinkage hides this bug, which is why the check uses the raw sample estimate:
    a rule that only survives because of the risk model it happens to be paired
    with has not been shown to be safe."""
    def inverts_blindly(ctx):
        # The realistic version of this bug: the researcher's notebook used the
        # raw sample covariance, so the regularization that would have saved it
        # never happens. A halted holding has exactly zero variance here.
        from app.optimize import sample_covariance

        covariance = sample_covariance(ctx.returns)
        raw = np.linalg.pinv(covariance) @ np.ones(ctx.assets)
        weights = np.abs(raw / np.sqrt(np.diag(covariance)))
        return weights / weights.sum()
    report = conformance.evaluate(strategy(inverts_blindly))
    assert not report.passed
    assert any(name.startswith("survives") for name in failed_checks(report))


def test_the_harness_catches_a_methodology_that_is_too_slow():
    import time

    def dawdles(ctx):
        time.sleep(1.0)
        return np.full(ctx.assets, 1 / ctx.assets)
    report = conformance.evaluate(strategy(dawdles))
    assert not report.passed
    assert "runs within budget" in failed_checks(report)


def test_a_methodology_that_raises_is_reported_not_swallowed():
    def explodes(ctx):
        raise RuntimeError("bad input")
    report = conformance.evaluate(strategy(explodes))
    assert not report.passed
    assert "produces weights" in failed_checks(report)
    assert "RuntimeError" in report.checks[0].detail


def test_scale_invariance_is_only_required_where_it_is_claimed(returns):
    """A methodology that depends on a fixed rate should declare it, not be failed for it."""
    def level_dependent(ctx):
        means, _ = ctx.expected_returns()
        excess = np.maximum(means - ctx.risk_free_rate, 0)
        weights = excess if excess.sum() > 0 else np.ones(ctx.assets)
        return weights / weights.sum()
    honest = conformance.evaluate(strategy(level_dependent, scale_invariant=False))
    overclaiming = conformance.evaluate(strategy(level_dependent, scale_invariant=True))
    assert "scale invariant" not in {c.name for c in honest.checks}
    assert "scale invariant" in failed_checks(overclaiming)


# --------------------------------------------------------------------------- #
# Estimation error, measured
# --------------------------------------------------------------------------- #

def test_resampling_shows_forecast_based_weights_are_the_least_stable(returns):
    """Risk-based rules should move less under resampling than one that needs means."""
    stability = {
        name: conformance.weight_stability(strategies.get(name), returns, Constraints())
        for name in ("risk_parity", "minimum_variance", "maximum_sharpe")
    }
    assert stability["risk_parity"]["widest_range"] < stability["minimum_variance"]["widest_range"]
    assert stability["minimum_variance"]["widest_range"] < stability["maximum_sharpe"]["widest_range"]
    assert all(0 <= s["mean_absolute_move"] <= 1 for s in stability.values())


def test_stability_is_reported_alongside_the_weights():
    payload = analysis.analyze(
        synthetic_prices(),
        [Portfolio("Optimized", {t: 1 for t in "ABCD"}, scheme="optimized",
                   objective="minimum_variance", estimation_days=120)],
        Settings(benchmark="M", paths=200, horizon_years=1),
    )
    report = payload["portfolios"][0]["optimization"]
    assert report["stability"]["draws"] >= 10
    for holding in report["holdings"]:
        assert holding["weight_p05"] <= holding["weight"] + 0.25
        assert holding["weight_p95"] >= holding["weight"] - 0.25


# --------------------------------------------------------------------------- #
# The worked example of the contract
# --------------------------------------------------------------------------- #

def test_pricing_the_trade_into_the_objective_reduces_trading():
    """The methodology added through the registry does what it claims."""
    prices = synthetic_prices()
    universe = {t: 1 for t in "ABCD"}
    payload = analysis.analyze(
        prices,
        [Portfolio("Plain", universe, scheme="optimized", objective="minimum_variance",
                   estimation_days=120),
         Portfolio("Net of costs", universe, scheme="optimized",
                   objective="minimum_variance_net_of_costs", estimation_days=120,
                   parameters={"turnover_lambda": 5.0})],
        Settings(paths=200, horizon_years=1, transaction_cost_bps=25),
    )
    plain, aware = payload["portfolios"]
    assert aware["summary"]["total_turnover"] < plain["summary"]["total_turnover"]
    assert aware["summary"]["total_cost"] < plain["summary"]["total_cost"]
    # And it is still a valid portfolio, not a frozen one.
    assert aware["optimization"]["reoptimizations"] == plain["optimization"]["reoptimizations"]


def test_a_methodology_can_be_added_without_touching_the_engine():
    """The point of the registry: register, and the whole system picks it up."""
    def equal_risk_budget(ctx):
        deviation = np.sqrt(np.diag(ctx.covariance()[0]))
        weights = (1 / deviation) ** 2
        return weights / weights.sum()

    added = strategy(equal_risk_budget, name="inverse_variance", author="test")
    strategies.register(added)
    try:
        assert "inverse_variance" in {entry["name"] for entry in strategies.catalogue()}
        assert conformance.evaluate(added).passed
        payload = analysis.analyze(
            synthetic_prices(),
            [Portfolio("Added", {t: 1 for t in "ABCD"}, scheme="optimized",
                       objective="inverse_variance", estimation_days=120)],
            Settings(paths=200, horizon_years=1),
        )
        assert payload["portfolios"][0]["optimization"]["objective"] == "inverse_variance"
    finally:
        strategies.REGISTRY.pop("inverse_variance", None)
