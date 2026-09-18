"""Tests for portfolio construction.

Where a closed form exists, the solver is checked against it. Where one does
not, the optimum is checked against the property that defines it: risk parity
must equalize risk contributions, minimum variance must beat portfolios drawn at
random, the frontier must be monotone above its own minimum. The shrinkage
estimator is checked against scikit-learn's implementation of the same paper.

The last test is the one that matters most for trusting any of it: an optimized
backtest must produce identical weights and NAV when the future is changed,
because a walk-forward rule that can see forward is not walk-forward.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.covariance import ledoit_wolf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app import analysis, engine, optimize  # noqa: E402
from app.engine import Portfolio, Settings  # noqa: E402
from app.optimize import Constraints  # noqa: E402


@pytest.fixture
def returns():
    """Six correlated assets with deliberately unequal volatilities."""
    rng = np.random.default_rng(11)
    loadings = rng.normal(0, 1, (6, 3))
    market = rng.normal(0, 0.01, (600, 3))
    idiosyncratic = rng.normal(0, 1, (600, 6)) * np.array([0.004, 0.006, 0.02, 0.012, 0.003, 0.009])
    return market @ loadings.T + idiosyncratic


# --------------------------------------------------------------------------- #
# Covariance estimation
# --------------------------------------------------------------------------- #

def test_shrinkage_matches_scikit_learn(returns):
    mine, intensity = optimize.ledoit_wolf_covariance(returns, annualize=False)
    theirs, their_intensity = ledoit_wolf(returns)
    observations = len(returns)
    assert intensity == pytest.approx(their_intensity, rel=1e-9)
    # scikit-learn reports the maximum-likelihood convention; ours is unbiased.
    assert np.allclose(mine * (observations - 1) / observations, theirs, rtol=1e-9, atol=1e-18)


def test_shrinkage_rises_when_observations_are_scarce(returns):
    _, plenty = optimize.ledoit_wolf_covariance(returns)
    _, scarce = optimize.ledoit_wolf_covariance(returns[:40])
    assert 0 <= plenty <= scarce <= 1
    assert scarce > plenty


def test_shrunk_covariance_is_positive_definite_when_the_sample_is_not():
    """More assets than observations makes the sample matrix singular."""
    rng = np.random.default_rng(3)
    thin = rng.normal(0, 0.01, (20, 30))
    sample = optimize.sample_covariance(thin)
    shrunk, intensity = optimize.ledoit_wolf_covariance(thin)
    assert np.linalg.eigvalsh(sample).min() < 1e-12
    assert np.linalg.eigvalsh(shrunk).min() > 0
    assert intensity > 0


def test_mean_shrinkage_pulls_toward_the_average(returns):
    raw = returns.mean(axis=0) * optimize.TRADING_DAYS
    shrunk, intensity = optimize.shrink_means(returns)
    assert 0 <= intensity <= 1
    assert shrunk.mean() == pytest.approx(raw.mean())
    assert np.std(shrunk) <= np.std(raw) + 1e-12


# --------------------------------------------------------------------------- #
# Objectives
# --------------------------------------------------------------------------- #

def test_minimum_variance_matches_the_two_asset_closed_form():
    covariance = np.array([[0.04, 0.006], [0.006, 0.09]])
    weights = optimize.minimum_variance(covariance, Constraints())
    v1, v2, cov = covariance[0, 0], covariance[1, 1], covariance[0, 1]
    assert weights[0] == pytest.approx((v2 - cov) / (v1 + v2 - 2 * cov), rel=1e-6)


def test_minimum_variance_beats_portfolios_drawn_at_random(returns):
    covariance = optimize.sample_covariance(returns)
    solution = optimize.minimum_variance(covariance, Constraints())
    best = solution @ covariance @ solution
    rng = np.random.default_rng(5)
    for _ in range(300):
        candidate = rng.dirichlet(np.ones(len(covariance)))
        assert candidate @ covariance @ candidate >= best - 1e-12


def test_risk_parity_equalizes_risk_contributions(returns):
    covariance = optimize.sample_covariance(returns)
    weights = optimize.risk_parity(covariance, Constraints())
    shares = optimize.risk_contributions(weights, covariance)["share"]
    assert shares.max() - shares.min() < 1e-5
    assert shares.sum() == pytest.approx(1)


def test_risk_contributions_add_up_to_volatility(returns):
    """Euler's theorem on a homogeneous function: the split is exact, not approximate."""
    covariance = optimize.sample_covariance(returns)
    weights = np.array([0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
    decomposition = optimize.risk_contributions(weights, covariance)
    assert decomposition["contribution"].sum() == pytest.approx(decomposition["volatility"], rel=1e-12)
    assert decomposition["volatility"] == pytest.approx(optimize.portfolio_volatility(weights, covariance))


def test_at_the_minimum_variance_optimum_weight_share_equals_risk_share(returns):
    """The defining first-order condition: marginal risks are equal there."""
    covariance = optimize.sample_covariance(returns)
    weights = optimize.minimum_variance(covariance, Constraints())
    shares = optimize.risk_contributions(weights, covariance)["share"]
    assert np.allclose(shares, weights, atol=1e-4)


def test_maximum_diversification_raises_the_diversification_ratio(returns):
    covariance = optimize.sample_covariance(returns)
    equal = np.full(len(covariance), 1 / len(covariance))
    solved = optimize.maximum_diversification(covariance, Constraints())
    ratio = lambda w: optimize.risk_contributions(w, covariance)["diversification_ratio"]  # noqa: E731
    assert ratio(solved) >= ratio(equal) - 1e-9


def test_maximum_sharpe_beats_equal_weight_on_the_data_it_was_given(returns):
    covariance, _ = optimize.ledoit_wolf_covariance(returns)
    means, _ = optimize.shrink_means(returns)
    equal = np.full(len(covariance), 1 / len(covariance))
    solved = optimize.maximum_sharpe(covariance, means, 0.03, Constraints())
    sharpe = lambda w: (w @ means - 0.03) / optimize.portfolio_volatility(w, covariance)  # noqa: E731
    assert sharpe(solved) >= sharpe(equal) - 1e-9


def test_weight_cap_is_respected_and_spreads_the_portfolio(returns):
    covariance = optimize.sample_covariance(returns)
    uncapped = optimize.minimum_variance(covariance, Constraints())
    capped = optimize.minimum_variance(covariance, Constraints(max_weight=0.25))
    assert uncapped.max() > 0.25
    assert capped.max() <= 0.25 + 1e-9
    assert capped.sum() == pytest.approx(1)
    assert (capped >= -1e-12).all()
    # A constraint can only cost variance, never save it.
    assert capped @ covariance @ capped >= uncapped @ covariance @ uncapped - 1e-12


def test_an_impossible_cap_is_explained_not_silently_ignored():
    covariance = np.eye(4) * 0.04
    with pytest.raises(ValueError, match="cannot be spread"):
        optimize.minimum_variance(covariance, Constraints(max_weight=0.2))


def test_volatility_target_is_met_or_the_calmest_portfolio_is_returned(returns):
    covariance = optimize.sample_covariance(returns)
    means, _ = optimize.shrink_means(returns)
    floor = optimize.portfolio_volatility(optimize.minimum_variance(covariance, Constraints()), covariance)

    reachable = optimize.target_volatility(covariance, means, floor * 1.3, Constraints())
    assert optimize.portfolio_volatility(reachable, covariance) <= floor * 1.3 + 1e-6

    impossible = optimize.target_volatility(covariance, means, floor * 0.5, Constraints())
    assert optimize.portfolio_volatility(impossible, covariance) == pytest.approx(floor, rel=1e-6)


def test_the_frontier_is_monotone_above_its_own_minimum(returns):
    covariance, _ = optimize.ledoit_wolf_covariance(returns)
    means, _ = optimize.shrink_means(returns)
    curve = optimize.efficient_frontier(covariance, means, Constraints(), points=15)
    volatilities = np.array(curve["volatilities"])
    expected = np.array(curve["returns"])
    assert len(volatilities) > 5
    assert np.all(np.diff(expected) > -1e-9)                       # returns increase along the grid
    start = int(np.argmin(volatilities))
    assert np.all(np.diff(volatilities[start:]) > -1e-6)            # and so does risk, above the minimum
    assert all(abs(w.sum() - 1) < 1e-8 for w in curve["weights"])


# --------------------------------------------------------------------------- #
# Walk-forward behaviour
# --------------------------------------------------------------------------- #

def synthetic_prices(seed: int = 4, days: int = 900, shock: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=days)
    frames = []
    for i, ticker in enumerate(["A", "B", "C", "D", "M"]):
        steps = rng.normal(0.0004, 0.006 + 0.004 * i, days)
        if shock:
            steps[-30:] += 0.05  # a violent, obvious future
        frames.append(pd.DataFrame({"date": dates, "ticker": ticker,
                                    "adjusted_price": 100 * np.cumprod(1 + steps)}))
    return pd.concat(frames, ignore_index=True)


def optimized_run(prices: pd.DataFrame) -> dict:
    return analysis.analyze(
        prices,
        [Portfolio("Optimized", {t: 1 for t in "ABCD"}, scheme="optimized",
                   objective="minimum_variance", estimation_days=120, max_weight=0.5)],
        Settings(benchmark="M", paths=200, horizon_years=1),
        start=None, end=None,
    )


def test_the_optimizer_cannot_see_the_future():
    """The decisive test: rewrite the last 30 days and nothing earlier may move."""
    calm = optimized_run(synthetic_prices())
    shocked = optimized_run(synthetic_prices(shock=True))

    calm_weights = calm["portfolios"][0]["optimization"]["weight_history"]
    shocked_weights = shocked["portfolios"][0]["optimization"]["weight_history"]
    shared = min(len(calm_weights["dates"]), len(shocked_weights["dates"])) - 2
    assert shared > 10
    assert calm_weights["dates"][:shared] == shocked_weights["dates"][:shared]
    assert np.allclose(calm_weights["weights"][:shared], shocked_weights["weights"][:shared])

    # And the realized path before the change is untouched.
    assert np.allclose(calm["portfolios"][0]["curve"]["nav"][:-40],
                       shocked["portfolios"][0]["curve"]["nav"][:-40])


def test_an_optimized_portfolio_reports_what_it_did():
    payload = optimized_run(synthetic_prices())
    report = payload["portfolios"][0]["optimization"]
    assert report["objective"] == "minimum_variance"
    assert report["estimation_days"] == 120
    assert report["reoptimizations"] > 20
    assert 0 <= report["shrinkage_intensity"] <= 1
    assert 1 <= report["effective_bets"] <= 4
    assert report["diversification_ratio"] >= 1
    assert sum(h["weight"] for h in report["holdings"]) == pytest.approx(1)
    assert sum(h["risk_share"] for h in report["holdings"]) == pytest.approx(1)
    # Rounded to six decimals for transport, so a few units in the last place.
    assert all(abs(sum(row) - 1) < 1e-5 for row in report["weight_history"]["weights"])


def test_the_reported_frontier_is_labelled_as_hindsight():
    payload = optimized_run(synthetic_prices())
    frontier = payload["frontier"]
    assert frontier["in_sample"] is True
    assert len(frontier["volatilities"]) == len(frontier["returns"]) > 5
    assert {"Optimized", "M benchmark"} <= {row["name"] for row in frontier["realized"]}


def test_an_optimized_portfolio_needs_a_rebalance_schedule():
    with pytest.raises(ValueError, match="re-estimates"):
        analysis.analyze(
            synthetic_prices(),
            [Portfolio("Optimized", {t: 1 for t in "ABCD"}, scheme="optimized")],
            Settings(rebalance="none", paths=200),
        )


def test_optimization_needs_at_least_two_holdings():
    with pytest.raises(ValueError, match="at least two holdings"):
        analysis.analyze(
            synthetic_prices(),
            [Portfolio("Optimized", {"A": 1}, scheme="optimized")],
            Settings(paths=200),
        )


def test_a_walk_forward_rule_trades_more_than_a_fixed_one():
    """Re-estimating costs money; the app should show that rather than hide it."""
    prices = synthetic_prices()
    payload = analysis.analyze(
        prices,
        [Portfolio("Optimized", {t: 1 for t in "ABCD"}, scheme="optimized",
                   objective="minimum_variance", estimation_days=120),
         Portfolio("Equal", {t: 1 for t in "ABCD"}, scheme="equal")],
        Settings(paths=200, horizon_years=1),
    )
    optimized, equal = payload["portfolios"]
    assert optimized["summary"]["total_turnover"] > equal["summary"]["total_turnover"]
    assert optimized["summary"]["total_cost"] > equal["summary"]["total_cost"]
