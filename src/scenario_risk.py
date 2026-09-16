"""Downside and paired comparisons under declared sampling assumptions."""

import numpy as np
import pandas as pd

from .forward import scenario_paths


def paired_outcomes(left, right, left_name='Growth', right_name='Balanced'):
    """Compare two portfolios on the same simulated market sequences."""
    difference = np.asarray(left) - np.asarray(right)
    return dict(left_portfolio=left_name, right_portfolio=right_name,
                probability_underperformance=float(np.mean(difference < 0)),
                difference_p05=float(np.quantile(difference, .05)),
                difference_median=float(np.median(difference)),
                difference_p95=float(np.quantile(difference, .95)))


def risk_review(daily, config):
    """Report tail outcomes and sensitivity without selecting a preferred model."""
    experiments = [('Published model', config['bootstrap_block_days'], False),
                   ('126-session moving blocks', 126, False),
                   ('252-session moving blocks', 252, False),
                   ('126-session circular blocks', 126, True),
                   ('252-session circular blocks', 252, True)]
    downside, sensitivity, paired = [], [], []
    capital = config['initial_capital']
    history_days = daily.date.nunique() - 1
    for label, block, circular in experiments:
        if block > history_days:
            continue
        terminals = {}
        for name, dates, values in scenario_paths(daily, dict(config, bootstrap_block_days=block), circular):
            final = values[:, -1]
            terminals[name] = final
            p05 = np.quantile(final, .05)
            sensitivity.append(dict(experiment=label, portfolio_id=name, block_days=block,
                circular=circular, paths=len(final), seed=config['seed'],
                terminal_p05=p05, terminal_median=np.median(final),
                probability_terminal_loss=np.mean(final < capital)))
            if label == 'Published model':
                worst = (values / np.maximum.accumulate(values, axis=1) - 1).min(axis=1)
                downside.append(dict(portfolio_id=name, start_value=capital,
                    terminal_p01=np.quantile(final, .01), terminal_p05=p05,
                    worst5_mean=np.mean(final[final <= p05]),
                    probability_terminal_loss=np.mean(final < capital),
                    terminal_loss_count=np.sum(final < capital), paths=len(final),
                    median_max_drawdown=np.median(worst), severe_max_drawdown_p05=np.quantile(worst,.05),
                    probability_ever_below_80pct=np.mean(values.min(axis=1) < .8 * capital)))
        if 'Growth' in terminals and 'Balanced' in terminals:
            paired.append(dict(experiment=label, **paired_outcomes(terminals['Growth'], terminals['Balanced'])))
    return pd.DataFrame(downside), pd.DataFrame(sensitivity), pd.DataFrame(paired)
