import numpy as np
import pandas as pd


def clean_inputs(prices, holdings, securities):
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices.date, errors="raise")
    for col in ("adjusted_price", "volume"):
        prices[col] = pd.to_numeric(prices[col], errors="raise")
    if prices.isna().any().any() or not np.isfinite(prices[["adjusted_price", "volume"]]).all().all():
        raise ValueError("Missing or non-finite price fields")
    if not prices.adjusted_price.gt(0).all() or not prices.volume.ge(0).all():
        raise ValueError("Prices must be positive and volume nonnegative")
    before = len(prices)
    prices = prices.drop_duplicates()
    if prices.duplicated(["date", "ticker"]).any():
        raise ValueError("Conflicting duplicate ticker-date records")
    if securities.ticker.duplicated().any() or securities.isna().any().any():
        raise ValueError("Invalid security master")
    if set(prices.ticker) != set(securities.ticker):
        raise ValueError("Price universe does not match security master")
    if holdings.isna().any().any() or holdings.duplicated(["portfolio_id", "ticker"]).any():
        raise ValueError("Invalid or duplicate holdings")
    if not set(holdings.ticker) <= set(securities.ticker):
        raise ValueError("Unknown security in holdings")
    if not np.isfinite(holdings.target_weight).all() or not holdings.target_weight.ge(0).all():
        raise ValueError("Holdings must have finite nonnegative weights")
    if not np.allclose(holdings.groupby("portfolio_id").target_weight.sum(), 1):
        raise ValueError("Portfolio weights must sum to one")
    if set(securities.currency) != {"CAD"}:
        raise ValueError("This pipeline requires a single CAD currency basis")
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    wide = prices.pivot(index="date", columns="ticker", values="adjusted_price")
    if wide.isna().any().any():
        raise ValueError("Missing ticker-date prices; repair the source instead of silently filling")
    return prices, {"exact_duplicates_removed": before - len(prices), "missing_prices": 0,
                    "price_records": len(prices), "securities": len(securities), "dates": len(wide)}


def validate_outputs(daily, positions):
    if not np.isfinite(daily[["net_return", "nav"]]).all().all() or not daily.nav.gt(0).all():
        raise ValueError("Invalid portfolio results")
    if not np.allclose(positions.groupby(["date", "portfolio_id"]).weight.sum(), 1):
        raise ValueError("Position weights do not reconcile")
    totals = positions.groupby(["date", "portfolio_id"]).market_value.sum()
    expected = daily.set_index(["date", "portfolio_id"]).nav.reindex(totals.index)
    if not np.allclose(totals, expected):
        raise ValueError("Position values do not reconcile to NAV")
