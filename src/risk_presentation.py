"""Shared risk charts and source-backed explanation for the report and website."""

import html
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

from .scenario_charts import PALETTE


def risk_charts(output, downside):
    data = downside.set_index('portfolio_id').reindex(PALETTE).dropna()
    y = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=180)
    for field, label, marker, color, offset in [
        ('terminal_p01', '1st percentile', 'v', '#b43a43', -.18),
        ('worst5_mean', 'Average of worst 5%', 'D', '#7b529d', 0),
        ('terminal_p05', '5th-percentile cutoff', 'o', '#256f94', .18)]:
        ax.scatter(data[field], y+offset, marker=marker, color=color, label=label, s=45, zorder=3)
        for i, value in enumerate(data[field]):
            ax.annotate(f'${value:,.0f}', (value, i+offset), xytext=(7,0), textcoords='offset points', va='center', fontsize=9)
    capital = data.start_value.iloc[0]
    ax.axvline(capital, color='#252d32', ls='--', lw=1, label=f'Initial ${capital:,.0f}')
    ax.set_xlim(0, max(capital, data.terminal_p05.max()) * 1.25)
    ax.set_yticks(y, data.index)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v,_: f'${v:,.0f}'))
    ax.set_xlabel('Ending portfolio value (CAD)')
    ax.set_title('Adverse five-year outcomes from equal investments', loc='left', pad=45)
    ax.legend(loc='lower left', bbox_to_anchor=(0,1.01), ncol=2, frameon=False, fontsize=9)
    ax.grid(axis='x', alpha=.18)
    fig.tight_layout()
    fig.savefig(output/'scenario_downside.png', facecolor='white')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11,4.8), dpi=180)
    ax.barh(y-.16, -data.median_max_drawdown*100, height=.3, color='#347f80', label='Median maximum drawdown')
    ax.barh(y+.16, -data.severe_max_drawdown_p05*100, height=.3, color='#b43a43', label='Severe drawdown threshold (5% of paths are worse)')
    for i,row in enumerate(data.itertuples()):
        for offset,value in [(-.16,-row.median_max_drawdown*100),(.16,-row.severe_max_drawdown_p05*100)]:
            ax.text(value+1,i+offset,f'{value:.1f}%',va='center',fontsize=10)
    ax.set_yticks(y,data.index)
    ax.invert_yaxis()
    ax.set_xlim(0,min(110,max(-data.severe_max_drawdown_p05*100)*1.2))
    ax.set_xlabel('Size of peak-to-trough loss during the five years (%)')
    ax.set_title('Losses along the way, even when ending wealth recovers',loc='left',pad=42)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,fontsize=9)
    ax.grid(axis='x',alpha=.18)
    fig.tight_layout()
    fig.savefig(output/'scenario_drawdowns.png',facecolor='white')
    plt.close(fig)


def risk_section(tables, config, prefix=''):
    d = tables['scenario_downside'].copy()
    for key in ['terminal_p01','terminal_p05','worst5_mean']:
        d[key] = d[key].map(lambda v:f'${v:,.0f}')
    for key in ['probability_terminal_loss','probability_ever_below_80pct']:
        d[key] = d[key].map(lambda v:f'{v:.2%}')
    d = d[['portfolio_id','terminal_p01','terminal_p05','worst5_mean','probability_terminal_loss','probability_ever_below_80pct']]
    d.columns=['Portfolio','1st percentile','5th percentile','Worst-5% average','Finish below initial capital','Ever lose over 20% of initial capital']
    paired = tables['scenario_paired']
    paired_text = ''
    base = paired[paired.experiment == 'Published model'] if not paired.empty else paired
    if not base.empty:
        r = base.iloc[0]
        paired_text = f'''<h3>Growth versus Balanced in the same market scenarios</h3>
<p>Subtract Balanced's ending value from Growth's in each shared simulated market sequence. Growth finishes below Balanced in <strong>{r.probability_underperformance:.2%}</strong> of paths. The median difference is <strong>${r.difference_median:+,.0f}</strong>; the 5th-percentile difference is <strong>${r.difference_p05:+,.0f}</strong>. These are conditional simulation results.</p>
<p>Subtracting the portfolios' separate 5th percentiles does not give the 5th percentile of their difference. Their individual adverse outcomes need not occur in the same path. The paired shortfall is not a maximum loss.</p>'''
    s = tables['scenario_sensitivity'].copy()
    s = s[s.portfolio_id.isin(['Growth','Balanced'])][['experiment','portfolio_id','terminal_p05','terminal_median','probability_terminal_loss']]
    for key in ['terminal_p05','terminal_median']:
        s[key] = s[key].map(lambda v:f'${v:,.0f}')
    s.probability_terminal_loss = s.probability_terminal_loss.map(lambda v:f'{v:.2%}')
    s.columns=['Sampling assumption','Portfolio','5th percentile','Median final value','Finish below initial capital']
    return f'''<section id="scenario-risk"><h3>How much could the investment lose?</h3>
<p>The lower fan boundary is a 5th-percentile cutoff: 5% of simulated ending balances lie below it. The average of that worst 5% describes the tail beyond the cutoff. Values below the initial CAD {config['initial_capital']:,.0f} represent losses of principal.</p>
<figure><a href="{prefix}scenario_downside.png" title="Open full-size downside chart"><img src="{prefix}scenario_downside.png" alt="Adverse ending balances compared with the initial investment" style="width:100%;height:auto"></a><figcaption>Same dollar scale across all portfolios. This view focuses on downside without compression from the large upside range.</figcaption></figure>
<div class="table-wrap table" style="overflow-x:auto" tabindex="0" role="region" aria-label="Scenario downside metrics">{d.to_html(index=False,border=0)}</div>
{paired_text}
<h3>What can happen before year five?</h3>
<figure><a href="{prefix}scenario_drawdowns.png" title="Open full-size drawdown chart"><img src="{prefix}scenario_drawdowns.png" alt="Median and severe simulated maximum drawdown for each portfolio" style="width:100%;height:auto"></a><figcaption>Each path's largest fall from a prior peak, including the initial balance. A peak can be above starting capital. Larger bars mean deeper losses.</figcaption></figure>
<h3>Does the conclusion survive different sampling assumptions?</h3>
<p>All comparisons use {config['simulation_paths']:,} paths, seed {config['seed']}, equal capital and the same completed history. Longer moving blocks retain longer sequences but underweight dates near the sample boundaries. Circular blocks wrap the end to the beginning, reducing that weighting effect while introducing artificial joins. The choice affects both dependence and sampled returns. No variant is selected because it produces a preferred ranking.</p>
<div class="table-wrap table" style="overflow-x:auto" tabindex="0" role="region" aria-label="Sampling sensitivity">{s.to_html(index=False,border=0)}</div>
<p><strong>Research finding:</strong> Growth's historical return advantage makes its median attractive under these assumptions, while several measures show greater downside than Balanced. A positive 5th percentile does not rule out principal losses. Higher volatility alone does not require every lower percentile to rank worst. The future return distribution remains unvalidated, and changing sampling assumptions can change the conclusion.</p>
<p>The bootstrap can generate worse compounded outcomes than any observed historical path by repeating adverse blocks. It cannot establish probabilities for risks absent from the sample. The separate volatility-model holdout does not validate these five-year return scenarios.</p>
<p><a href="{prefix}scenario_downside.csv">Download downside metrics</a> · <a href="{prefix}scenario_paired.csv">Download paired comparisons</a> · <a href="{prefix}scenario_sensitivity.csv">Download sampling sensitivity</a></p></section>'''
