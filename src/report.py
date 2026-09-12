import html
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


COLORS = ["#177565", "#386cb0", "#b44955", "#8969a9", "#646e72"]


def transparent(hex_color, alpha=0.13):
    red, green, blue = (int(hex_color[index:index + 2], 16) for index in (1, 3, 5))
    return f"rgba({red}, {green}, {blue}, {alpha})"


def figures(tables):
    daily, summary = tables["portfolio_daily_summary"], tables["portfolio_summary"]
    sectors, predictions = tables["sector_exposures"], tables["model_predictions"]
    latest = sectors[sectors.date == sectors.date.max()]
    latest = latest[latest.portfolio_id != "Benchmark"]
    charts = {
        "performance": px.line(daily, x="date", y="nav", color="portfolio_id", render_mode="svg", title=f"Portfolio value<br><sup>CAD {daily.nav.iloc[0]:,.0f} initial capital</sup>", color_discrete_sequence=COLORS),
        "drawdown": px.line(daily, x="date", y="drawdown", color="portfolio_id", render_mode="svg", title="Loss from previous peak", color_discrete_sequence=COLORS),
        "risk": px.scatter(summary, x="volatility", y="annualized_return", color="portfolio_id", size=[18] * len(summary), title="Annualized risk and return", color_discrete_sequence=COLORS),
        "sectors": px.bar(latest, x="portfolio_id", y="weight", color="sector", title="Latest sector exposure", color_discrete_sequence=COLORS),
    }
    projection = tables["forward_projection_bands"]
    projected_chart = go.Figure()
    portfolio_ids = [name for name in daily.portfolio_id.unique() if name != "Benchmark"]
    for index, portfolio_id in enumerate(portfolio_ids):
        color = COLORS[index]
        history = daily[daily.portfolio_id == portfolio_id]
        future = projection[projection.portfolio_id == portfolio_id]
        projected_chart.add_trace(go.Scatter(x=history.date, y=history.nav, mode="lines", name=f"{portfolio_id} actual",
                                             line=dict(color=color, width=2)))
        projected_chart.add_trace(go.Scatter(x=list(future.date) + list(future.date[::-1]),
                                             y=list(future.p95) + list(future.p05[::-1]), fill="toself",
                                             fillcolor=transparent(color), line=dict(color="rgba(0,0,0,0)"),
                                             hoverinfo="skip", showlegend=False, name=f"{portfolio_id} 5-95% range"))
        projected_chart.add_trace(go.Scatter(x=future.date, y=future["median"], mode="lines", name=f"{portfolio_id} median projection",
                                             line=dict(color=color, width=2, dash="dash")))
    projected_chart.add_vline(x=projection.date.min(), line_width=1, line_dash="dot", line_color="#646e72")
    projected_chart.add_annotation(x=projection.date.min(), y=1.02, yref="paper", text="Projection begins", showarrow=False,
                                   xanchor="left", font=dict(color="#55636b"))
    projected_chart.update_layout(title="Historical portfolio value and five-year scenario range<br><sup>Solid: completed backtest · Dashed: median simulation · Shaded: 5th-95th percentile</sup>")
    charts["forward_scenarios"] = projected_chart
    returns = daily.pivot(index="date", columns="portfolio_id", values="net_return").iloc[1:]
    charts["correlation"] = px.imshow(returns.corr(), zmin=-1, zmax=1, color_continuous_scale="RdBu", text_auto=".2f", title="Portfolio return correlation", aspect="auto")
    scores = tables["model_scores"]
    selected = scores[scores.selected_by_cv][["portfolio_id", "model"]]
    plot = predictions.merge(selected, on=["portfolio_id", "model"], validate="many_to_one")
    long = plot.melt(id_vars=["date", "portfolio_id"], value_vars=["actual", "prediction"], var_name="series", value_name="volatility")
    charts["forecasts"] = px.line(long, x="date", y="volatility", color="series", render_mode="svg", facet_row="portfolio_id",
                                   title="Held-out volatility forecasts", color_discrete_sequence=COLORS)
    charts["forecasts"].for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    charts["model_error"] = px.bar(scores, x="portfolio_id", y="rmse", color="model", barmode="group", title="Volatility forecast error<br><sup>Test RMSE · lower is better</sup>", color_discrete_sequence=COLORS)
    for key, chart in charts.items():
        chart.update_layout(template="plotly_white", font=dict(family="Arial", size=13, color="#222c30"),
                            margin=dict(l=50, r=25, t=70, b=80), height=460,
                            legend=dict(orientation="h", y=-0.19, x=0, title_text=""), paper_bgcolor="white")
        if key in ("drawdown", "sectors", "forecasts", "model_error"):
            chart.update_yaxes(tickformat=".0%")
        if key == "risk":
            chart.update_xaxes(tickformat=".0%")
            chart.update_yaxes(tickformat=".0%")
        if key == "forecasts":
            chart.update_layout(height=1100, legend=dict(y=-0.07))
        charts[key] = chart
    return charts


def render_report(output, tables, provenance, config, quality):
    charts = figures(tables)
    summary = tables["portfolio_summary"].copy()
    for col in ["cumulative_return", "annualized_return", "volatility", "max_drawdown", "excess_annualized_return"]:
        summary[col] = summary[col].map(lambda value: f"{value:.2%}")
    summary["sharpe"] = summary.sharpe.map(lambda value: f"{value:.2f}")
    summary["total_cost"] = summary.total_cost.map(lambda value: f"${value:,.2f}")
    summary = summary[["portfolio_id", "cumulative_return", "annualized_return", "volatility", "sharpe", "max_drawdown", "total_cost", "excess_annualized_return"]]
    summary.columns = [name.replace("_", " ").title() for name in summary.columns]
    scores = tables["model_scores"].copy()
    score_display = scores[["portfolio_id", "model", "rmse", "r2", "selected_by_cv"]].copy()
    score_display.rmse = score_display.rmse.map(lambda x: f"{x:.4f}")
    score_display.r2 = score_display.r2.map(lambda x: f"{x:.3f}")
    scenario_display = tables["forward_projection_summary"].copy()
    scenario_display["as_of_date"] = pd.to_datetime(scenario_display.as_of_date).dt.strftime("%d %b %Y")
    for column in ["start_value", "terminal_p05", "terminal_median", "terminal_p95"]:
        scenario_display[column] = scenario_display[column].map(lambda value: f"${value:,.0f}")
    scenario_display["probability_terminal_loss"] = scenario_display.probability_terminal_loss.map(lambda value: f"{value:.1%}")
    scenario_display["median_max_drawdown"] = scenario_display.median_max_drawdown.map(lambda value: f"{value:.1%}")
    scenario_display.columns = ["Portfolio", "Historical end", "Starting value", "5th percentile terminal value",
                                "Median terminal value", "95th percentile terminal value", "Probability of loss after horizon",
                                "Median maximum drawdown", "Paths", "Years", "Block days"]
    panels = "".join(f'<section class="chart" id="{key}">{chart.to_html(full_html=False, include_plotlyjs=True if i == 0 else False, config={"responsive": True, "displaylogo": False})}</section>'
                     for i, (key, chart) in enumerate(charts.items()))
    daily = tables["portfolio_daily_summary"]
    source = html.escape(provenance["source"])
    text = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Investment Analytics | Research Report</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f7f7;color:#202a2e;font:15px/1.6 Arial,sans-serif;letter-spacing:0}}main{{max-width:1320px;margin:auto;padding:32px 24px}}nav{{display:flex;gap:24px;flex-wrap:wrap;border-bottom:1px solid #ccd5d8;padding-bottom:16px}}a{{color:#176d60}}h1{{font-size:32px;line-height:1.2;margin:28px 0 12px}}h2{{font-size:22px}}.muted{{color:#55636b}}.stats{{display:flex;gap:36px;flex-wrap:wrap;padding:20px 0;border-bottom:1px solid #ccd5d8}}.stats strong{{display:block;font-size:23px}}.table{{overflow-x:auto;margin:24px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:right;border-bottom:1px solid #dce2e4;padding:12px;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{background:#eaf0f1}}.chart{{min-width:0;background:white;margin:24px 0;border:1px solid #dce2e4}}details{{border-top:1px solid #ccd5d8;padding:18px 0}}summary{{cursor:pointer;font-weight:bold}}pre{{overflow:auto;font-size:12px}}@media(max-width:600px){{main{{padding:20px 12px}}h1{{font-size:26px}}}}
th:first-child,td:first-child{{position:sticky;left:0;background:#f5f7f7;z-index:1}}th:first-child{{background:#eaf0f1}}
</style></head><body><main><nav><strong>INVESTMENT ANALYTICS</strong><a href="#comparison">Portfolios</a><a href="#forward_scenarios">Five-year scenarios</a><a href="#forecasts">Forecasts</a><a href="#method">Methodology</a><a href="portfolio_summary.csv">Download metrics</a><a href="forward_projection_summary.csv">Scenario results</a></nav>
<h1>Portfolio research &amp; risk analytics</h1><p class="muted">{source} · CAD · {daily.date.min():%d %b %Y} to {daily.date.max():%d %b %Y}</p><p><a href="notebook.html">Open research notebook</a> · <a href="../notebooks/portfolio_exploration.ipynb" download>Download notebook</a></p>
<div class="stats"><div><strong>4</strong>Research portfolios</div><div><strong>{quality['price_records']:,}</strong>Validated price records</div><div><strong>{quality['missing_prices']}</strong>Missing prices</div><div><strong>{config['transaction_cost_bps']} bps</strong>Cost per traded dollar</div></div>
<h2 id="comparison">Portfolio comparison</h2><p>Growth emphasizes technology. Income emphasizes banks, energy and utilities. Balanced diversifies across these equity sectors. Low volatility uses inverse historical volatility weights. All four remain equity portfolios.</p>
<div class="table">{summary.to_html(index=False, border=0)}</div>{panels}
<h2>Five-year scenario analysis</h2><p>The scenario engine begins after the final completed backtest date. It resamples consecutive {config['bootstrap_block_days']}-session blocks of each portfolio's completed net returns, including modeled rebalancing costs, across {config['simulation_paths']:,} paths. This preserves some short-run return clustering while showing a distribution of plausible outcomes rather than a single expected-value promise.</p>
<p>These are conditional historical scenarios, not price targets or investment advice. They use no returns after the projection start date and do not claim the historical sample will repeat.</p>
<div class="table">{scenario_display.to_html(index=False, border=0)}</div>
<h2>Forecast validation</h2><p>Models predict the next {config['forecast_days']} sessions' realized annualized volatility. Selection uses training-period cross-validation only. The last {config['test_fraction']:.0%} is held out, with a {config['forecast_days']}-session gap to exclude overlapping training labels. A negative R² means the model is worse than predicting the test mean. Compare RMSE with persistence before concluding that machine learning helps.</p>
<div class="table">{score_display.to_html(index=False, border=0)}</div>
<details id="method" open><summary>Methodology and limitations</summary><p>Adjusted closes approximate reinvested total returns. Position units are synthetic adjusted-price units, not broker share balances. Portfolios begin after a {config['warmup_days']}-session calibration period and rebalance at the close of each month's first trading session. The benchmark is {html.escape(config['benchmark'])}, a broad Canadian equity ETF held without rebalancing. Returns use previous-close weights and deduct {config['transaction_cost_bps']} bps on total absolute traded weight. Initial entry costs, taxes and additional market impact are excluded.</p><p><strong>Backtest timing and look-ahead controls.</strong> The Low Volatility target weights are calibrated once using only the first {config['warmup_days']} completed sessions; subsequent prices cannot affect those weights. On each later date, the portfolio return is calculated from the weights held at the previous close. A monthly rebalance occurs only after that day's return and transaction cost have been booked, so it applies to future returns rather than the same day's already-known move. The forecast models use features available through date t and labels composed only of returns from t+1 onward. The final {config['test_fraction']:.0%} is held out, and a {config['forecast_days']}-session purge separates the last training label from the test period. Scaling and parameter selection occur within training folds only.</p><p>Sharpe uses arithmetic mean daily excess return divided by daily standard deviation, annualized with √252. Annualized return uses compound growth. The constant annual risk-free assumption is {config['risk_free_rate']:.1%}. Income is a sector allocation label: dividend yield and cash distributions are not modeled separately. Today's chosen security universe creates selection and survivorship bias. Low-volatility weights are calibrated once and need not deliver the lowest future risk. Overlapping forecast targets mean test errors are dependent; no statistical significance or trading profitability is claimed.</p><p>Forecasts are a separate research experiment and do not drive portfolio allocations. Forward scenarios start after the final backtest date and resample only completed portfolio returns; they are not an out-of-sample forecast of returns. Financial data and model outputs remain reviewable in the SQL mart and exported CSV files.</p></details>
<details><summary>Run configuration and provenance</summary><pre>{html.escape(json.dumps({'configuration': config, 'provenance': provenance, 'validation': quality}, indent=2))}</pre></details>
</main></body></html>'''
    (output / "report.html").write_text(text)
    return charts
