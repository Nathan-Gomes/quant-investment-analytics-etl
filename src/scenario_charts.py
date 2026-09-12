"""Consistent historical and conditional scenario comparisons."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go

PALETTE = {"Balanced": "#1f77b4", "Benchmark": "#ff7f0e", "Growth": "#2ca02c",
           "Income": "#d62728", "Low volatility": "#9467bd"}


def comparison(daily, bands, normalized=False):
    figure = go.Figure()
    for name, color in PALETTE.items():
        history = daily[daily.portfolio_id == name].sort_values("date")
        future = bands[bands.portfolio_id == name].sort_values("date")
        scale = 1 if normalized else history.nav.iloc[-1] / future["median"].iloc[0]
        if not normalized:
            figure.add_trace(go.Scatter(x=history.date, y=history.nav, name=name,
                legendgroup=name, line=dict(color=color, width=2), mode="lines"))
        figure.add_trace(go.Scatter(x=future.date, y=future["median"] * scale,
            name=name if normalized else name + " scenario median", legendgroup=name,
            showlegend=normalized, line=dict(color=color, width=2, dash="dash"), mode="lines"))
    start = bands.date.min()
    figure.add_vline(x=start, line_dash="dot", line_color="#657077")
    figure.update_layout(title="Equal investment: CAD 100,000 each" if normalized else "Historical wealth continuation (unequal starting balances)",
        yaxis_title="CAD", hovermode="x unified")
    return figure


def previews(output, daily, bands):
    start, end = bands.date.min(), bands.date.max()
    for normalized, filename in [(False, "forward_scenarios.png"), (True, "forward_indexed.png")]:
        fig, ax = plt.subplots(figsize=(12, 5.6), dpi=180)
        for name, color in PALETTE.items():
            history = daily[daily.portfolio_id == name].sort_values("date")
            future = bands[bands.portfolio_id == name].sort_values("date")
            scale = 1 if normalized else history.nav.iloc[-1] / future["median"].iloc[0]
            if not normalized:
                ax.plot(history.date, history.nav, color=color, lw=1.6)
            ax.plot(future.date, future["median"] * scale, color=color, lw=1.8, ls="--", label=name)
        ax.axvline(start, color="#657077", ls=":", lw=1)
        if not normalized:
            ax.text(.02, .96, "Historical backtest", transform=ax.transAxes, va="top", color="#52616b")
            ax.text(.70, .96, "Conditional scenario medians", transform=ax.transAxes, va="top", color="#52616b")
        ax.set_title("Five-year strategy comparison: CAD 100,000 invested in each" if normalized else "Historical wealth continuation: unequal balances at August 2026", loc="left", fontsize=14, pad=44)
        ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, fontsize=9)
        ax.set_ylabel("Portfolio value (CAD)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.grid(alpha=.18)
        ax.set_xlim(start if normalized else daily.date.min(), end)
        fig.tight_layout()
        fig.savefig(output / filename, facecolor="white")
        plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), dpi=160, sharex=True, sharey=True)
    for ax, (name, color) in zip(axes.flat, PALETTE.items()):
        future = bands[bands.portfolio_id == name]
        scale = 1
        ax.fill_between(future.date, future.p05 * scale, future.p95 * scale, color=color, alpha=.13, label="5th-95th percentile")
        ax.fill_between(future.date, future.p25 * scale, future.p75 * scale, color=color, alpha=.24, label="25th-75th percentile")
        ax.plot(future.date, future["median"] * scale, color=color, ls="--", label="Median")
        ax.axhline(future["median"].iloc[0], color="#657077", lw=.8, ls=":")
        ax.set_title(name, loc="left", fontsize=12)
        ax.set_ylabel("Portfolio value (CAD)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.grid(alpha=.18)
        ax.tick_params(labelbottom=True)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center left", frameon=False)
    fig.suptitle("Uncertainty around the medians | identical scales", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / "forward_ranges.png", facecolor="white")
    plt.close(fig)
