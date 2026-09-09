import sqlite3

import numpy as np
import pandas as pd
import pytest

from src.analytics import metrics, portfolio_analytics
from src.load import load_mart
from src.models import feature_frame, FEATURES
from src.validation import clean_inputs, validate_outputs


@pytest.fixture
def inputs():
    dates = pd.bdate_range("2020-01-01", periods=360)
    frames = []
    for i, ticker in enumerate(["A", "B", "M"]):
        r = 0.0002 + (i + 1) * 0.002 * np.sin(np.arange(len(dates)) / 7)
        frames.append(pd.DataFrame({"date": dates, "ticker": ticker,
                                    "adjusted_price": 100 * np.cumprod(1 + r), "volume": 1000}))
    prices = pd.concat(frames, ignore_index=True)
    holdings = pd.DataFrame({"portfolio_id": ["Balanced"] * 2, "ticker": ["A", "B"], "target_weight": [0.6, 0.4]})
    securities = pd.DataFrame({"ticker": ["A", "B", "M"], "sector": ["Tech", "Banks", "Benchmark"], "currency": ["CAD"] * 3})
    config = {"warmup_days": 30, "benchmark": "M", "initial_capital": 100000,
              "transaction_cost_bps": 10, "risk_free_rate": 0.03}
    return prices, holdings, securities, config


def test_compounding_and_initial_loss_drawdown():
    result = metrics([-0.1, 0.1], 0)
    assert result["cumulative_return"] == pytest.approx(-0.01)
    assert result["max_drawdown"] == pytest.approx(-0.1)
    assert result["annualized_return"] == pytest.approx(0.99 ** 126 - 1)


def test_zero_volatility_sharpe_is_undefined():
    assert np.isnan(metrics([0., 0., 0.])["sharpe"])


@pytest.mark.parametrize("fault", ["negative", "missing", "infinite", "conflict", "gap", "weights", "unknown"])
def test_reject_bad_inputs(inputs, fault):
    prices, holdings, securities, _ = inputs
    if fault == "negative":
        prices.loc[0, "adjusted_price"] = -1
    elif fault == "missing":
        prices.loc[0, "volume"] = np.nan
    elif fault == "infinite":
        prices.loc[0, "adjusted_price"] = np.inf
    elif fault == "conflict":
        prices = pd.concat([prices, prices.iloc[:1].assign(adjusted_price=999)])
    elif fault == "gap":
        prices = prices.iloc[1:]
    elif fault == "weights":
        holdings.loc[0, "target_weight"] = 0.8
    elif fault == "unknown":
        holdings.loc[0, "ticker"] = "UNKNOWN"
    with pytest.raises(ValueError):
        clean_inputs(prices, holdings, securities)


def test_exact_duplicates_are_audited(inputs):
    p, h, s, _ = inputs
    cleaned, audit = clean_inputs(pd.concat([p, p.iloc[:1]]), h, s)
    assert len(cleaned) == len(p)
    assert audit["exact_duplicates_removed"] == 1


def test_previous_weights_and_reconciliation(inputs):
    p, h, s, c = inputs
    daily, positions, *_ = portfolio_analytics(p, h, s, c)
    validate_outputs(daily, positions)
    wide = p.pivot(index="date", columns="ticker", values="adjusted_price")
    r = wide.pct_change().iloc[31]
    observed = daily[daily.portfolio_id == "Balanced"].iloc[1]
    assert observed.gross_return == pytest.approx(0.6 * r.A + 0.4 * r.B)
    assert daily[daily.portfolio_id == "Benchmark"].cost.sum() == 0


def test_costs_reduce_nav(inputs):
    p, h, s, c = inputs
    cost = portfolio_analytics(p, h, s, c)[0]
    free = portfolio_analytics(p, h, s, dict(c, transaction_cost_bps=0))[0]
    a = cost[cost.portfolio_id == "Balanced"]
    b = free[free.portfolio_id == "Balanced"]
    assert a.cost.sum() > 0
    assert a.nav.iloc[-1] < b.nav.iloc[-1]


def test_low_vol_weights_ignore_future_prices(inputs):
    p, h, s, c = inputs
    first = portfolio_analytics(p, h, s, c)[-1]
    altered = p.copy()
    mask = altered.date > sorted(p.date.unique())[c["warmup_days"]]
    altered.loc[mask & (altered.ticker == "A"), "adjusted_price"] *= 2
    second = portfolio_analytics(altered, h, s, c)[-1]
    pd.testing.assert_frame_equal(first, second)


def test_forecast_label_and_features_use_correct_dates():
    dates = pd.bdate_range("2020-01-01", periods=120)
    r = pd.Series(np.sin(np.arange(120)) * 0.01, index=dates)
    group = pd.DataFrame({"date": dates, "net_return": r.values})
    frame = feature_frame(group, r, pd.Series(1000, index=dates), 20)
    t = frame.index[10]
    i = dates.get_loc(t)
    assert frame.loc[t, "target"] == pytest.approx(r.iloc[i + 1:i + 21].std() * np.sqrt(252))
    altered = group.copy()
    altered.loc[altered.date > t, "net_return"] = 0.2
    after = feature_frame(altered, r, pd.Series(1000, index=dates), 20)
    np.testing.assert_allclose(frame.loc[t, FEATURES].astype(float), after.loc[t, FEATURES].astype(float))


def test_atomic_load_preserves_previous_database(tmp_path):
    with sqlite3.connect(tmp_path / "investment_analytics.db") as conn:
        conn.execute("CREATE TABLE original (value INTEGER)")
        conn.execute("INSERT INTO original VALUES (7)")
    with pytest.raises(sqlite3.Error):
        load_mart(tmp_path, {"new_table": pd.DataFrame({"x": [1]})}, "INVALID SQL;")
    with sqlite3.connect(tmp_path / "investment_analytics.db") as conn:
        assert conn.execute("SELECT value FROM original").fetchone()[0] == 7
