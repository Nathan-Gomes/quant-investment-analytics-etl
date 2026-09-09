import numpy as np
import pandas as pd


def metrics(returns, risk_free_rate=0.03):
    returns = pd.Series(returns).dropna()
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax().clip(lower=1) - 1
    std = returns.std(ddof=1)
    rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1
    return {"cumulative_return": wealth.iloc[-1] - 1,
            "annualized_return": wealth.iloc[-1] ** (252 / len(returns)) - 1,
            "volatility": std * np.sqrt(252),
            "sharpe": (returns.mean() - rf_daily) / std * np.sqrt(252) if std > 0 else np.nan,
            "max_drawdown": drawdown.min()}


def security_analytics(prices):
    groups = []
    for _, group in prices.groupby("ticker", sort=True):
        group = group.sort_values("date").copy()
        group["daily_return"] = group.adjusted_price.pct_change(fill_method=None)
        group["cumulative_return"] = group.adjusted_price / group.adjusted_price.iloc[0] - 1
        group["moving_average_20"] = group.adjusted_price.rolling(20).mean()
        group["rolling_volatility_20"] = group.daily_return.rolling(20).std() * np.sqrt(252)
        group["drawdown"] = group.adjusted_price / group.adjusted_price.cummax() - 1
        group["max_drawdown_to_date"] = group.drawdown.cummin()
        group["average_volume_20"] = group.volume.rolling(20).mean()
        groups.append(group)
    return pd.concat(groups, ignore_index=True)


def portfolio_analytics(prices, holdings, securities, config):
    wide = prices.pivot(index="date", columns="ticker", values="adjusted_price").sort_index()
    returns = wide.pct_change(fill_method=None)
    warmup = config["warmup_days"]
    if len(wide) < warmup + 300:
        raise ValueError("At least 300 evaluation days plus the warmup period are required")
    assets = sorted(set(wide.columns) - {config["benchmark"]})
    targets = {name: group.set_index("ticker").target_weight.reindex(wide.columns, fill_value=0)
               for name, group in holdings.groupby("portfolio_id")}
    # Freeze inverse-volatility weights using only pre-investment observations.
    inv = 1 / returns.iloc[1:warmup + 1][assets].std()
    if not np.isfinite(inv).all():
        raise ValueError("Low-volatility calibration requires nonzero finite volatility")
    targets["Low volatility"] = (inv / inv.sum()).reindex(wide.columns, fill_value=0)
    targets["Benchmark"] = pd.Series({config["benchmark"]: 1.0}).reindex(wide.columns, fill_value=0)
    rows, position_rows, trades = [], [], []
    for name, target in targets.items():
        nav, weights = config["initial_capital"], target.copy()
        initial_date = wide.index[warmup]
        # Initial positions are assumed funded at the calibration close; entry cost excluded for all strategies.
        rows.append({"date": initial_date, "portfolio_id": name, "nav": nav,
                     "gross_return": 0., "net_return": 0., "turnover": 0., "cost": 0.})
        for date in wide.index[warmup + 1:]:
            r = returns.loc[date]
            gross = float(weights @ r)
            before_cost = nav * (1 + gross)
            drifted = weights * (1 + r) / (1 + gross)
            index = wide.index.get_loc(date)
            rebalance = date.month != wide.index[index - 1].month and name != "Benchmark"
            turnover = float((target - drifted).abs().sum()) if rebalance else 0.
            cost = before_cost * turnover * config["transaction_cost_bps"] / 10000
            new_nav = before_cost - cost
            rows.append({"date": date, "portfolio_id": name, "nav": new_nav,
                         "gross_return": gross, "net_return": new_nav / nav - 1,
                         "turnover": turnover, "cost": cost})
            if rebalance:
                trades.append({"date": date, "portfolio_id": name, "turnover": turnover, "cost": cost})
            weights = target.copy() if rebalance else drifted
            nav = new_nav
            for ticker in wide.columns:
                if weights[ticker] > 0:
                    position_rows.append({"date": date, "portfolio_id": name, "ticker": ticker,
                                          "adjusted_units": nav * weights[ticker] / wide.loc[date, ticker],
                                          "market_value": nav * weights[ticker], "weight": weights[ticker]})
    daily = pd.DataFrame(rows)
    daily["cumulative_return"] = daily.nav / config["initial_capital"] - 1
    daily["drawdown"] = daily.nav / daily.groupby("portfolio_id").nav.cummax() - 1
    positions = pd.DataFrame(position_rows).merge(securities, on="ticker", validate="many_to_one")
    sectors = positions.groupby(["date", "portfolio_id", "sector"], as_index=False).agg(
        market_value=("market_value", "sum"), weight=("weight", "sum"))
    summary = []
    for name, group in daily.groupby("portfolio_id"):
        row = metrics(group.net_return.iloc[1:], config["risk_free_rate"])
        row.update(portfolio_id=name, total_cost=group.cost.sum(), turnover=group.turnover.sum())
        summary.append(row)
    summary = pd.DataFrame(summary)
    benchmark_return = summary.loc[summary.portfolio_id == "Benchmark", "annualized_return"].iloc[0]
    summary["excess_annualized_return"] = summary.annualized_return - benchmark_return
    target_table = pd.concat([v.rename("weight").rename_axis("ticker").reset_index().assign(portfolio_id=k)
                              for k, v in targets.items()], ignore_index=True)
    return daily, positions, sectors, summary, pd.DataFrame(trades), target_table
