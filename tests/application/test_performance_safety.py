"""Regression checks for cache isolation and unchanged scenario accounting."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy.stats import binom

from app import convex, engine, riskmodel, server
from tests.application.test_app_engine import BAKE_OFF, client


def test_compiled_cache_matches_fresh_solves_under_concurrency(monkeypatch):
    models = [riskmodel.RiskModel(np.diag([0.01 + i * 0.02, 0.16, 0.09]),
                                  "sample", 252) for i in range(6)]
    mandate = convex.Mandate(max_weight=0.8)
    with monkeypatch.context() as patch:
        patch.setattr(convex, "_shape_key", lambda *args: None)
        expected = [convex.minimum_variance(m, mandate).weights.copy() for m in models]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda m: convex.minimum_variance(m, mandate).weights.copy(), models))
    for left, right in zip(expected, actual):
        np.testing.assert_allclose(left, right, atol=1e-6, rtol=1e-6)


def test_rising_paths_include_initial_balance_as_trough():
    result = engine.scenario_statistics(np.array([0.01, 0.02]), np.zeros((40, 5), dtype=int),
                                        100_000, floor_fraction=1.005)
    assert result["summary"]["probability_ever_below_floor"] == 1
    assert result["summary"]["median_max_drawdown"] == 0


@pytest.mark.parametrize("q", [0.05, 0.5, 0.95])
def test_order_statistic_interval_has_declared_coverage(q):
    n = 5000
    values = np.arange(1, n + 1, dtype=float)
    for confidence in (0.9, 0.95, 0.99):
        interval = engine.quantile_interval(values, q, confidence)
        lower_rank, upper_rank = interval["low"], interval["high"]
        coverage = binom.cdf(upper_rank - 1, n, q) - binom.cdf(lower_rank - 1, n, q)
        assert coverage >= confidence


def test_result_cache_is_bounded_and_can_be_disabled(client, monkeypatch):
    server._RESULT_CACHE.clear()
    monkeypatch.setattr(server, "RESULT_CACHE_SIZE", 1)
    for seed in (42, 43):
        response = client.post("/api/analyze", json={**BAKE_OFF, "seed": seed})
        assert response.status_code == 200
    assert len(server._RESULT_CACHE) == 1
    assert client.post("/api/analyze", json={**BAKE_OFF, "seed": 43}).json()["meta"]["result_cache_hit"]
    monkeypatch.setattr(server, "RESULT_CACHE_SIZE", 0)
    assert not client.post("/api/analyze", json={**BAKE_OFF, "seed": 43}).json()["meta"]["result_cache_hit"]


def test_result_cache_includes_source_and_security_metadata(client, monkeypatch):
    server._RESULT_CACHE.clear()
    original_load = server.marketdata.load
    first = client.post("/api/analyze", json=BAKE_OFF).json()

    def changed_load(*args):
        prices = original_load(*args)
        prices.note = "Different documented source"
        prices.metadata = prices.metadata.copy()
        prices.metadata["sector"] = "Updated sector"
        return prices

    monkeypatch.setattr(server.marketdata, "load", changed_load)
    second = client.post("/api/analyze", json=BAKE_OFF).json()
    assert not second["meta"]["result_cache_hit"]
    assert second["meta"]["source"] != first["meta"]["source"]
