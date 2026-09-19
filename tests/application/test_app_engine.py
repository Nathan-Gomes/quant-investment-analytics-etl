"""Tests for the web app engine.

The first group checks the calculations by hand. The second checks that the app
reproduces the research pipeline's published results exactly, so the two cannot
drift apart. The third checks that the browser engine agrees with Python, which
is what lets the offline demo claim the same numbers.
"""

import json
import time
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app import analysis, engine, marketdata  # noqa: E402
from app.engine import Portfolio, Settings  # noqa: E402
from app.rng import Mulberry32  # noqa: E402
from src.analytics import metrics as pipeline_metrics  # noqa: E402


# --------------------------------------------------------------------------- #
# Calculations
# --------------------------------------------------------------------------- #

def frame(series: dict, dates=None) -> pd.DataFrame:
    dates = dates if dates is not None else pd.bdate_range("2021-01-04", periods=len(next(iter(series.values()))))
    rows = []
    for ticker, prices in series.items():
        rows.append(pd.DataFrame({"date": dates, "ticker": ticker, "adjusted_price": prices}))
    return pd.concat(rows, ignore_index=True)


def test_metrics_match_the_pipeline_definitions():
    returns = [0.01, -0.02, 0.015, 0.004, -0.001, 0.02]
    ours = engine.metrics(returns, 0.03)
    theirs = pipeline_metrics(returns, 0.03)
    for key in ("cumulative_return", "annualized_return", "volatility", "sharpe", "max_drawdown"):
        assert ours[key] == pytest.approx(theirs[key])


def test_hand_calculated_compounding_and_costs():
    # Two assets, one rebalance at the start of the second month.
    prices = frame({
        "A": [100, 110, 110, 110, 110],
        "B": [100, 100, 100, 100, 100],
    }, dates=pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06", "2021-02-01", "2021-02-02"]))
    wide = engine.to_wide(prices)
    target = pd.Series({"A": 0.5, "B": 0.5})
    settings = Settings(initial_capital=1000, transaction_cost_bps=100, rebalance="monthly")
    run = engine.run_backtest(wide, target, settings)
    # Day 2: A gains 10%, portfolio gains 5%.
    assert run["net_returns"][1] == pytest.approx(0.05)
    assert run["nav"][1] == pytest.approx(1050)
    # A is now 550 of the 1050 NAV: a weight of 11/21, which is 1/42 above target.
    # The February rebalance sells 1/42 of NAV in A and buys the same in B, so both
    # sides count and turnover is 1/21.
    assert run["turnover"][3] == pytest.approx(1 / 21)
    assert run["cost"][3] == pytest.approx(1050 * (1 / 21) * 0.01)  # 100 bps on the traded notional
    assert run["cost"][3] == pytest.approx(0.5)
    assert run["nav"][3] == pytest.approx(1049.5)


def test_weights_are_lagged_so_a_return_cannot_set_its_own_weight():
    prices = frame({"A": [100, 100, 100, 130], "B": [100, 100, 100, 100]})
    wide = engine.to_wide(prices)
    settings = Settings(initial_capital=1000, transaction_cost_bps=0, rebalance="none")
    run = engine.run_backtest(wide, pd.Series({"A": 0.5, "B": 0.5}), settings)
    assert run["net_returns"][3] == pytest.approx(0.15)  # half of A's 30%, not more


def test_later_prices_never_change_earlier_values():
    base = {"A": [100, 101, 103, 102, 105, 106], "B": [50, 51, 50, 52, 53, 52]}
    altered = {"A": base["A"][:-1] + [300], "B": base["B"]}
    settings = Settings(initial_capital=1000, rebalance="none")
    target = pd.Series({"A": 0.6, "B": 0.4})
    first = engine.run_backtest(engine.to_wide(frame(base)), target, settings)
    second = engine.run_backtest(engine.to_wide(frame(altered)), target, settings)
    assert np.allclose(first["nav"][:-1], second["nav"][:-1])


@pytest.mark.parametrize("schedule,expected", [("monthly", 4), ("quarterly", 2), ("annual", 1), ("none", 0)])
def test_rebalance_schedules(schedule, expected):
    """A rebalance lands on the first session observed in a new period, never on day one."""
    index = pd.to_datetime(["2021-01-04", "2021-01-29", "2021-02-01", "2021-02-26",
                            "2021-03-01", "2021-04-01", "2022-01-03"])
    mask = engine.rebalance_mask(index, schedule)
    assert mask.sum() == expected
    assert not mask[0]


def test_alignment_drops_days_a_holding_did_not_trade():
    prices = pd.concat([
        frame({"A": [100, 101, 102, 103]}),
        frame({"B": [10, 11, 12]}, dates=pd.bdate_range("2021-01-04", periods=3)),
    ], ignore_index=True)
    wide, quality = engine.align(engine.to_wide(prices))
    assert quality["shared_trading_days"] == 3
    assert quality["days_dropped_for_missing_prices"] == 1


@pytest.mark.parametrize("weights", [{"A": -1, "B": 2}, {"A": 0, "B": 0}, {"A": float("inf")}])
def test_bad_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        Portfolio("Bad", weights).resolve(["A", "B"], pd.DataFrame({"A": [0.01], "B": [0.01]}))


def test_weights_are_normalized_whichever_convention_is_used():
    calibration = pd.DataFrame({"A": [0.01, 0.02], "B": [0.01, 0.02]})
    as_percent = Portfolio("P", {"A": 60, "B": 40}).resolve(["A", "B"], calibration)
    as_fraction = Portfolio("P", {"A": 0.6, "B": 0.4}).resolve(["A", "B"], calibration)
    assert np.allclose(as_percent.to_numpy(), as_fraction.to_numpy())
    assert as_percent.sum() == pytest.approx(1)


def test_inverse_volatility_prefers_the_calmer_asset():
    calibration = pd.DataFrame({"A": np.random.default_rng(0).normal(0, 0.02, 200),
                                "B": np.random.default_rng(1).normal(0, 0.005, 200)})
    weights = Portfolio("LV", {"A": 1, "B": 1}, scheme="inverse_volatility").resolve(["A", "B"], calibration)
    assert weights["B"] > weights["A"]
    assert weights.sum() == pytest.approx(1)


# --------------------------------------------------------------------------- #
# Scenarios
# --------------------------------------------------------------------------- #

def test_every_portfolio_is_sampled_on_the_same_blocks():
    first = engine.block_positions(500, 252, 20, 100, seed=7)
    second = engine.block_positions(500, 252, 20, 100, seed=7)
    assert np.array_equal(first, second)
    assert first.max() < 500 and first.min() >= 0


def test_a_flat_return_history_produces_a_flat_scenario():
    returns = np.zeros(300)
    index = engine.block_positions(300, 252, 20, 200, seed=3)
    stats = engine.scenario_statistics(returns, index, 100000.0)
    assert stats["summary"]["terminal_median"] == pytest.approx(100000)
    assert stats["summary"]["probability_terminal_loss"] == 0
    assert stats["summary"]["median_max_drawdown"] == pytest.approx(0)


def test_scenario_start_counts_as_the_first_drawdown_peak():
    returns = np.full(60, -0.01)
    index = engine.block_positions(60, 40, 20, 50, seed=1)
    stats = engine.scenario_statistics(returns, index, 100000.0)
    assert stats["summary"]["median_max_drawdown"] < -0.3
    assert stats["summary"]["probability_terminal_loss"] == 1


def test_a_block_longer_than_the_history_is_refused():
    with pytest.raises(ValueError):
        engine.block_positions(10, 252, 20, 100, seed=1)


def test_paired_comparison_is_path_by_path():
    left = np.array([120.0, 90.0, 150.0, 100.0])
    right = np.array([100.0, 100.0, 100.0, 100.0])
    result = engine.paired_comparison(left, right, "L", "R")
    # Differences are +20, -10, +50, 0: ahead in two of the four shared paths.
    assert result["probability_ahead"] == pytest.approx(0.5)
    assert result["difference_median"] == pytest.approx(10.0)
    assert result["difference_p95"] == pytest.approx(45.5)


# --------------------------------------------------------------------------- #
# Agreement with the research pipeline
# --------------------------------------------------------------------------- #

PUBLISHED = {
    "Growth": (0.27795034970873567, 0.8973744566813312, -0.487162506530592),
    "Balanced": (0.19121469622038378, 0.9745724957615799, -0.2873012892542631),
    "Income": (0.16134441577932557, 0.8278604022119785, -0.3305609883288516),
    "Low volatility": (0.17566153928551365, 0.9420329567461985, -0.29709104365227024),
    "XIC.TO benchmark": (0.16222273915530638, 0.820370062636065, -0.37211737229120),
}


@pytest.fixture(scope="module")
def published_run():
    priceset = marketdata.load_bundled(None)
    holdings = pd.read_csv(ROOT / "data/holdings.csv")
    portfolios = [Portfolio(name, dict(zip(group.ticker, group.target_weight)))
                  for name, group in holdings.groupby("portfolio_id")]
    assets = [t for t in priceset.metadata.ticker if t != "XIC.TO"]
    portfolios.append(Portfolio("Low volatility", {t: 1 for t in assets}, scheme="inverse_volatility"))
    return analysis.analyze(
        priceset.prices, portfolios,
        Settings(benchmark="XIC.TO", paths=1000, calibration_days=252),
        metadata=priceset.metadata, start="2018-01-01", end="2026-08-31",
    )


def test_app_reproduces_the_published_pipeline_results(published_run):
    """The app is the same research, so it must give the same answers."""
    assert published_run["meta"]["window_start"] == "2019-01-03"
    for portfolio in published_run["portfolios"]:
        annualized, sharpe, drawdown = PUBLISHED[portfolio["name"]]
        assert portfolio["summary"]["annualized_return"] == pytest.approx(annualized, rel=1e-9)
        assert portfolio["summary"]["sharpe"] == pytest.approx(sharpe, rel=1e-9)
        assert portfolio["summary"]["max_drawdown"] == pytest.approx(drawdown, rel=1e-9)


def test_payload_carries_the_caveats_a_reviewer_needs(published_run):
    meta = published_run["meta"]
    assert meta["calibration_days"] == 252
    assert meta["settings"]["block_days"] == 20
    assert published_run["quality"]["shared_trading_days"] > 1900
    assert all(p["scenario"]["summary"]["start_value"] == 100000 for p in published_run["portfolios"])
    assert all(np.isfinite(v) for v in published_run["correlation"]["matrix"][0])


def test_payload_is_json_serializable_with_no_nan(published_run):
    text = json.dumps(published_run)
    assert "NaN" not in text and "Infinity" not in text


def test_benchmark_pays_no_trading_costs(published_run):
    benchmark = next(p for p in published_run["portfolios"] if p["is_benchmark"])
    assert benchmark["summary"]["total_cost"] == 0


def test_short_history_is_refused_rather_than_extrapolated():
    prices = frame({"A": np.linspace(100, 110, 30), "B": np.linspace(50, 55, 30)})
    with pytest.raises(ValueError, match="60 shared trading days"):
        analysis.analyze(prices, [Portfolio("P", {"A": 0.5, "B": 0.5})], Settings())


# --------------------------------------------------------------------------- #
# Python and the browser must agree
# --------------------------------------------------------------------------- #

NODE = shutil.which("node")
REQUEST = {
    "portfolios": [
        {"name": "Growth", "weights": {"SHOP.TO": 35, "CSU.TO": 30, "CNR.TO": 20, "RY.TO": 15}, "scheme": "custom"},
        {"name": "Equal", "weights": {"RY.TO": 1, "ENB.TO": 1, "FTS.TO": 1, "CNR.TO": 1}, "scheme": "equal"},
    ],
    "benchmark": "XIC.TO", "start": "2019-01-01", "end": "2026-08-31",
    "initial_capital": 100000, "transaction_cost_bps": 10, "risk_free_rate": 0.03,
    "rebalance": "monthly", "horizon_years": 5, "paths": 800, "block_days": 20,
    "seed": 42, "scenario_basis": "equal",
}


def test_prng_matches_javascript():
    """Bit-for-bit equality with static/engine.js, so seeds mean the same thing."""
    expected = [
        0.6011037519201636, 0.44829055899754167, 0.8524657934904099,
        0.6697340414393693, 0.17481389874592423, 0.5265925421845168,
    ]
    generator = Mulberry32(42)
    assert [generator.next_float() for _ in range(6)] == pytest.approx(expected, rel=1e-15)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
@pytest.mark.parametrize("objective", [None, "minimum_variance_convex", "risk_parity_convex"])
def test_browser_engine_agrees_with_python(objective):
    from tools.build_demo import dataset_js
    study = json.loads(json.dumps(REQUEST))
    if objective:
        study["portfolios"] = [{**study["portfolios"][1], "scheme": "optimized", "objective": objective}]

    with tempfile.TemporaryDirectory() as folder:
        dataset = Path(folder) / "dataset.js"
        request = Path(folder) / "request.json"
        dataset.write_text(dataset_js())
        request.write_text(json.dumps(study))
        completed = subprocess.run(
            [NODE, str(ROOT / "tools/js_engine_harness.mjs"), str(dataset), str(request)],
            capture_output=True, text=True, check=True, cwd=ROOT,
        )
    browser = json.loads(completed.stdout)

    priceset = marketdata.load_bundled(None)
    portfolios = [Portfolio(**p) for p in study["portfolios"]]
    python = analysis.analyze(
        priceset.prices, portfolios,
        Settings(benchmark="XIC.TO", paths=REQUEST["paths"], seed=REQUEST["seed"]),
        metadata=priceset.metadata, start=REQUEST["start"], end=REQUEST["end"],
    )

    assert python["meta"]["trading_days"] == browser["meta"]["trading_days"]
    for left, right in zip(python["portfolios"], browser["portfolios"]):
        assert left["name"] == right["name"]
        for key in ("annualized_return", "volatility", "sharpe", "max_drawdown", "final_value"):
            # The demo dataset stores prices at four decimals, hence a loose relative tolerance.
            assert left["summary"][key] == pytest.approx(right["summary"][key], rel=2e-5)
        for key in ("terminal_median", "terminal_p05", "terminal_p95", "median_max_drawdown"):
            assert left["scenario"]["summary"][key] == pytest.approx(
                right["scenario"]["summary"][key], rel=2e-5)
        # The same seed must select the same blocks, so the loss counts are identical.
        assert left["scenario"]["summary"]["probability_terminal_loss"] == pytest.approx(
            right["scenario"]["summary"]["probability_terminal_loss"], abs=1e-12)


# --------------------------------------------------------------------------- #
# The Yahoo Finance path, exercised without a network
# --------------------------------------------------------------------------- #

class FakeTicker:
    """Stands in for yfinance.Ticker so the download path can be tested offline."""

    calls: list[str] = []

    def __init__(self, symbol):
        self.symbol = symbol
        FakeTicker.calls.append(symbol)

    def history(self, period="max", auto_adjust=True):
        if self.symbol == "NOSUCH":
            return pd.DataFrame()
        index = pd.date_range("2015-01-02", periods=900, freq="B", tz="America/Toronto")
        drift = 1 + 0.0004 * (1 if self.symbol == "AAPL" else 0.5)
        return pd.DataFrame({"Close": 100 * drift ** np.arange(len(index)), "Volume": 1000}, index=index)

    def get_info(self):
        return {"shortName": f"{self.symbol} Inc", "sector": "Technology",
                "currency": "USD" if "." not in self.symbol else "CAD", "quoteType": "EQUITY"}


@pytest.fixture
def fake_yahoo(monkeypatch, tmp_path):
    module = type(sys)("yfinance")
    module.Ticker = FakeTicker
    monkeypatch.setitem(sys.modules, "yfinance", module)
    monkeypatch.setattr(marketdata, "CACHE", tmp_path / "cache")
    FakeTicker.calls = []
    return module


def test_download_writes_prices_and_a_provenance_record(fake_yahoo, tmp_path):
    prices, info = marketdata.download("AAPL", "2016-01-01", "2026-01-01")
    assert list(prices.columns) == ["date", "ticker", "adjusted_price"]
    assert prices.adjusted_price.gt(0).all()
    assert info["source"].startswith("Yahoo Finance")
    assert info["currency"] == "USD"
    assert (marketdata.CACHE / "AAPL.csv").exists()
    assert json.loads((marketdata.CACHE / "AAPL.json").read_text())["price_basis"]


def test_a_second_request_is_served_from_the_cache(fake_yahoo):
    marketdata.download("AAPL", "2016-01-01", "2026-01-01")
    marketdata.download("AAPL", "2016-01-01", "2026-01-01")
    assert FakeTicker.calls.count("AAPL") == 1


def test_an_unknown_symbol_is_reported_not_guessed(fake_yahoo):
    with pytest.raises(ValueError, match="no price history for NOSUCH"):
        marketdata.download("NOSUCH", "2016-01-01", "2026-01-01")


def test_a_downloaded_universe_runs_end_to_end(fake_yahoo):
    priceset = marketdata.load(["AAPL", "MSFT", "SPY"], "2015-01-01", "2026-01-01", source="yahoo")
    payload = analysis.analyze(
        priceset.prices,
        [Portfolio("Mine", {"AAPL": 60, "MSFT": 40}),
         Portfolio("Equal", {"AAPL": 1, "MSFT": 1}, scheme="equal")],
        Settings(benchmark="SPY", paths=300),
        metadata=priceset.metadata,
    )
    assert [p["name"] for p in payload["portfolios"]] == ["Mine", "Equal", "SPY benchmark"]
    assert payload["assets"][0]["sector"] == "Technology"
    assert payload["portfolios"][0]["summary"]["final_value"] > 0


def test_one_bad_symbol_does_not_sink_the_whole_request(fake_yahoo):
    with pytest.raises(ValueError, match="NOSUCH"):
        marketdata.load(["AAPL", "NOSUCH"], "2015-01-01", "2026-01-01", source="yahoo")


def test_auto_source_rejects_partial_downloads(fake_yahoo):
    with pytest.raises(ValueError, match="NOSUCH"):
        marketdata.load(["AAPL", "NOSUCH"], "2015-01-01", "2026-01-01", source="auto")


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

def test_the_same_study_always_carries_the_same_run_id(published_run):
    """A fingerprint, not a serial number: reproducibility you can check by eye."""
    priceset = marketdata.load_bundled(None)
    portfolios = [Portfolio("G", {"SHOP.TO": 35, "RY.TO": 65})]
    first = analysis.analyze(priceset.prices, portfolios, Settings(benchmark="XIC.TO", paths=300),
                             metadata=priceset.metadata, start="2019-01-01")
    second = analysis.analyze(priceset.prices, [Portfolio("G", {"RY.TO": 65, "SHOP.TO": 35})],
                              Settings(benchmark="XIC.TO", paths=300),
                              metadata=priceset.metadata, start="2019-01-01")
    assert first["meta"]["run_id"] == second["meta"]["run_id"]   # order of holdings is not a difference
    assert len(first["meta"]["run_id"]) == 16


@pytest.mark.parametrize("change", ["seed", "cost", "window", "paths"])
def test_any_change_to_the_inputs_changes_the_run_id(change):
    priceset = marketdata.load_bundled(None)
    portfolios = [Portfolio("G", {"SHOP.TO": 35, "RY.TO": 65})]
    base = dict(benchmark="XIC.TO", paths=300, seed=42, transaction_cost_bps=10)
    altered = dict(base)
    start = "2019-01-01"
    if change == "seed":
        altered["seed"] = 43
    elif change == "cost":
        altered["transaction_cost_bps"] = 12
    elif change == "paths":
        altered["paths"] = 400
    else:
        start = "2019-02-01"
    first = analysis.analyze(priceset.prices, portfolios, Settings(**base),
                             metadata=priceset.metadata, start="2019-01-01")
    second = analysis.analyze(priceset.prices, portfolios, Settings(**altered),
                              metadata=priceset.metadata, start=start)
    assert first["meta"]["run_id"] != second["meta"]["run_id"]


def test_revised_prices_produce_a_different_run_even_with_the_same_request():
    """Providers revise adjusted history; the manifest digests the values used."""
    priceset = marketdata.load_bundled(None)
    revised = priceset.prices.copy()
    # Inside the study window: a revision before it would not enter the calculation,
    # and the manifest digests the values that did.
    inside = revised.index[revised.date >= "2020-01-01"][0]
    revised.loc[inside, "adjusted_price"] *= 1.001
    settings = Settings(benchmark="XIC.TO", paths=300)
    portfolios = [Portfolio("G", {"SHOP.TO": 35, "RY.TO": 65})]
    original = analysis.analyze(priceset.prices, portfolios, settings,
                                metadata=priceset.metadata, start="2019-01-01")
    changed = analysis.analyze(revised, portfolios, settings,
                               metadata=priceset.metadata, start="2019-01-01")
    assert original["manifest"]["data"]["sha256"] != changed["manifest"]["data"]["sha256"]
    assert original["meta"]["run_id"] != changed["meta"]["run_id"]


def test_the_manifest_records_the_code_and_environment_that_ran(published_run):
    record = published_run["manifest"]
    assert len(record["code_sha256"]) == 64
    assert "engine.py" in record["code_files"] and "optimize.py" in record["code_files"]
    assert record["environment"]["packages"]["numpy"]
    assert record["data"]["tickers"] and record["data"]["rows"] > 1000


# --------------------------------------------------------------------------- #
# The service, including the progress stream
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.server import api

    return TestClient(api)


BAKE_OFF = {
    "portfolios": [
        {"name": "Min var", "weights": {t: 1 for t in ["RY.TO", "TD.TO", "ENB.TO", "FTS.TO"]},
         "scheme": "optimized", "objective": "minimum_variance", "max_weight": 0.5},
        {"name": "Equal", "weights": {t: 1 for t in ["RY.TO", "TD.TO", "ENB.TO", "FTS.TO"]},
         "scheme": "equal"},
    ],
    "benchmark": "XIC.TO", "source": "bundled", "paths": 400,
    "start": "2019-01-01", "end": "2026-08-31",
}


def stream_events(client, body) -> list[dict]:
    with client.stream("POST", "/api/analyze/stream", json=body) as response:
        assert response.status_code == 200
        return [json.loads(line) for line in response.iter_lines() if line.strip()]


def test_the_stream_reports_each_stage_before_the_result(client):
    """The point of the endpoint: movement long before the answer arrives."""
    events = stream_events(client, BAKE_OFF)
    phases = [event["phase"] for event in events]
    assert phases[0] == "accepted"
    assert phases[-1] == "done"
    for expected in ("prices", "align", "backtest", "scenarios", "diagnostics"):
        assert expected in phases, f"no {expected} event in {phases}"
    # Progress within a phase, so a bar can move while one portfolio is running.
    steps = [e for e in events if e.get("steps")]
    assert steps and all(1 <= e["step"] <= e["steps"] for e in steps)


def test_the_streamed_result_matches_the_plain_endpoint(client):
    """Two ways in, one answer: the stream is a delivery detail, not a variant."""
    streamed = next(e["payload"] for e in stream_events(client, BAKE_OFF) if e["phase"] == "done")
    plain = client.post("/api/analyze", json=BAKE_OFF).json()
    assert streamed["meta"]["run_id"] == plain["meta"]["run_id"]
    for left, right in zip(streamed["portfolios"], plain["portfolios"]):
        assert left["summary"]["sharpe"] == pytest.approx(right["summary"]["sharpe"])
        assert left["scenario"]["summary"] == right["scenario"]["summary"]


def test_a_rejected_request_is_reported_on_the_stream_not_dropped(client):
    body = {**BAKE_OFF, "portfolios": [{"name": "X", "weights": {"NOSUCH.TO": 1}}]}
    events = stream_events(client, body)
    assert events[-1]["phase"] == "error"
    assert "NOSUCH.TO" in events[-1]["message"]


def test_the_payload_reports_where_the_time_went(client):
    payload = client.post("/api/analyze", json=BAKE_OFF).json()
    timings = payload["meta"]["timings_ms"]
    assert {"align", "backtest", "scenarios", "diagnostics", "total"} <= set(timings)
    assert timings["total"] >= sum(v for k, v in timings.items() if k != "total") - 1


# --------------------------------------------------------------------------- #
# Speed, and what it buys
# --------------------------------------------------------------------------- #

def test_the_scenario_rewrite_did_not_change_the_numbers():
    """The in-place version is an optimization, so it must be arithmetically identical."""
    generator = np.random.default_rng(1)
    returns = generator.normal(0.0005, 0.011, 2500)
    index = engine.block_positions(2500, 1260, 20, 2000, 42)
    stats = engine.scenario_statistics(returns, index, 100_000.0)
    # Recomputed the slow, obvious way.
    sampled = returns[index]
    values = np.column_stack([np.full(2000, 100_000.0), 100_000.0 * np.cumprod(1 + sampled, axis=1)])
    peak = np.maximum.accumulate(values, axis=1)
    assert stats["summary"]["terminal_median"] == pytest.approx(np.median(values[:, -1]))
    assert stats["summary"]["median_max_drawdown"] == pytest.approx(np.median((values / peak - 1).min(axis=1)))
    assert stats["bands"]["median"][0] == pytest.approx(100_000.0)


@pytest.mark.parametrize("q", [0.05, 0.5, 0.95])
def test_a_quoted_percentile_comes_with_its_sampling_error(q):
    generator = np.random.default_rng(3)
    values = np.sort(generator.lognormal(11.5, 0.6, 5000))
    band = engine.quantile_interval(values, q)
    estimate = float(np.quantile(values, q))
    assert band["low"] <= estimate <= band["high"]
    assert 0 < band["width"] < 0.3


def test_the_interval_narrows_as_paths_are_added():
    """The whole justification for a large path count, stated as a test."""
    generator = np.random.default_rng(3)
    widths = []
    for paths in (1000, 5000, 20000):
        values = np.sort(generator.lognormal(11.5, 0.6, paths))
        widths.append(engine.quantile_interval(values, 0.5)["width"])
    assert widths[0] > widths[1] > widths[2]


def test_an_identical_request_is_served_from_the_cache(client):
    from app import server

    server._RESULT_CACHE.clear()
    first = client.post("/api/analyze", json=BAKE_OFF).json()
    started = time.perf_counter()
    second = client.post("/api/analyze", json=BAKE_OFF).json()
    elapsed = time.perf_counter() - started
    assert first["meta"]["run_id"] == second["meta"]["run_id"]
    assert first["portfolios"][0]["scenario"]["summary"] == second["portfolios"][0]["scenario"]["summary"]
    assert elapsed < 0.2          # returned, not recomputed
    # A different request must not collide with it.
    changed = client.post("/api/analyze", json={**BAKE_OFF, "seed": 99}).json()
    assert changed["meta"]["run_id"] != first["meta"]["run_id"]
