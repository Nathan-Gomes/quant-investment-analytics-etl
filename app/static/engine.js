/* The same engine as app/engine.py, in the browser.

   The offline demo has no server, so the backtest and the block bootstrap run
   here instead. Both sides share the mulberry32 generator, so the same seed
   produces the same scenarios in Python and in the browser, and the parity test
   in tests/application/test_app_engine.py checks that the numbers agree. */
(function (PL) {
  "use strict";

  const TRADING_DAYS = 252;
  const MASK = 0xffffffff;

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = ((t + Math.imul(t ^ (t >>> 7), 61 | t)) >>> 0) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function quantile(sorted, q) {
    if (!sorted.length) return NaN;
    const position = (sorted.length - 1) * q;
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    if (lower === upper) return sorted[lower];
    return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
  }

  const sortedCopy = (values) => Float64Array.from(values).sort();

  function metrics(returns, riskFree) {
    let wealth = 1;
    let peak = 1;
    let worst = 0;
    let sum = 0;
    let best = -Infinity;
    let worstDay = Infinity;
    let positive = 0;
    for (let i = 0; i < returns.length; i += 1) {
      const r = returns[i];
      wealth *= 1 + r;
      peak = Math.max(peak, wealth);
      worst = Math.min(worst, wealth / peak - 1);
      sum += r;
      best = Math.max(best, r);
      worstDay = Math.min(worstDay, r);
      if (r > 0) positive += 1;
    }
    const n = returns.length;
    const mean = sum / n;
    let variance = 0;
    const rfDaily = Math.pow(1 + riskFree, 1 / TRADING_DAYS) - 1;
    let downsideSum = 0;
    let downsideCount = 0;
    let downsideMean = 0;
    for (let i = 0; i < n; i += 1) {
      variance += (returns[i] - mean) ** 2;
      if (returns[i] < rfDaily) { downsideCount += 1; downsideMean += returns[i]; }
    }
    variance /= n - 1;
    const deviation = Math.sqrt(variance);
    if (downsideCount > 1) {
      downsideMean /= downsideCount;
      for (let i = 0; i < n; i += 1) {
        if (returns[i] < rfDaily) downsideSum += (returns[i] - downsideMean) ** 2;
      }
    }
    const downsideDeviation = downsideCount > 1 ? Math.sqrt(downsideSum / (downsideCount - 1)) : NaN;
    return {
      cumulative_return: wealth - 1,
      annualized_return: Math.pow(wealth, TRADING_DAYS / n) - 1,
      volatility: deviation * Math.sqrt(TRADING_DAYS),
      sharpe: deviation > 0 ? ((mean - rfDaily) / deviation) * Math.sqrt(TRADING_DAYS) : NaN,
      sortino: downsideDeviation > 0 ? ((mean - rfDaily) / downsideDeviation) * Math.sqrt(TRADING_DAYS) : NaN,
      max_drawdown: worst,
      best_day: best,
      worst_day: worstDay,
      positive_days: positive / n,
      observations: n,
    };
  }

  function rebalanceMask(dates, schedule) {
    const flags = new Array(dates.length).fill(false);
    if (schedule === "none") return flags;
    const keyOf = (iso) => {
      const [y, m] = iso.split("-").map(Number);
      if (schedule === "monthly") return y * 12 + m;
      if (schedule === "quarterly") return y * 4 + Math.ceil(m / 3);
      return y;
    };
    let previous = keyOf(dates[0]);
    for (let i = 1; i < dates.length; i += 1) {
      const key = keyOf(dates[i]);
      flags[i] = key !== previous;
      previous = key;
    }
    return flags;
  }

  function backtest(matrix, dates, target, settings, schedule, targetProvider) {
    const assets = target.length;
    const days = dates.length;
    const flags = rebalanceMask(dates, schedule);
    if (targetProvider) target = Float64Array.from(targetProvider(0));
    const weights = Float64Array.from(target);
    const history = [{ index: 0, weights: Array.from(target) }];
    const nav = new Float64Array(days);
    const net = new Float64Array(days);
    const drawdown = new Float64Array(days);
    const contribution = new Float64Array(assets);
    const exposure = new Float64Array(assets);
    const returns = new Float64Array(assets);
    let totalCost = 0;
    let totalTurnover = 0;
    let rebalances = 0;
    nav[0] = settings.initial_capital;
    let peak = nav[0];

    for (let i = 1; i < days; i += 1) {
      let step = 0;
      for (let a = 0; a < assets; a += 1) {
        returns[a] = matrix[a][i] / matrix[a][i - 1] - 1;
        step += weights[a] * returns[a];
        contribution[a] += weights[a] * returns[a];
        exposure[a] += weights[a];
      }
      const beforeCost = nav[i - 1] * (1 + step);
      const drifted = new Float64Array(assets);
      for (let a = 0; a < assets; a += 1) {
        drifted[a] = (weights[a] * (1 + returns[a])) / (1 + step);
      }
      if (flags[i] && targetProvider) {
        // Re-estimated from data through this close, never beyond it.
        target = Float64Array.from(targetProvider(i));
        history.push({ index: i, weights: Array.from(target) });
      }
      let traded = 0;
      if (flags[i]) for (let a = 0; a < assets; a += 1) traded += Math.abs(target[a] - drifted[a]);
      const charge = flags[i] ? (beforeCost * traded * settings.transaction_cost_bps) / 10000 : 0;
      nav[i] = beforeCost - charge;
      net[i] = nav[i] / nav[i - 1] - 1;
      peak = Math.max(peak, nav[i]);
      drawdown[i] = nav[i] / peak - 1;
      totalCost += charge;
      totalTurnover += traded;
      if (flags[i]) { rebalances += 1; weights.set(target); } else { weights.set(drifted); }
    }
    for (let a = 0; a < assets; a += 1) exposure[a] += weights[a];

    const summary = metrics(net.subarray(1), settings.risk_free_rate);
    summary.final_value = nav[days - 1];
    summary.total_cost = totalCost;
    summary.total_turnover = totalTurnover;
    summary.rebalances = rebalances;
    summary.years = (days - 1) / TRADING_DAYS;
    return {
      nav, net, drawdown, summary, targetHistory: history,
      finalWeights: Array.from(weights),
      averageWeights: Array.from(exposure, (v) => v / days),
      contribution: Array.from(contribution),
    };
  }

  function rollingReturns(nav, window) {
    if (nav.length <= window) return {};
    const values = [];
    for (let i = window; i < nav.length; i += 1) {
      values.push(Math.pow(nav[i] / nav[i - window], TRADING_DAYS / window) - 1);
    }
    const sorted = sortedCopy(values);
    return {
      worst: sorted[0],
      median: quantile(sorted, 0.5),
      best: sorted[sorted.length - 1],
      share_negative: values.filter((v) => v < 0).length / values.length,
      windows: values.length,
    };
  }

  function blockPositions(sampleLength, days, block, paths, seed) {
    const random = mulberry32(seed);
    const high = sampleLength - block + 1;
    const blocksNeeded = Math.ceil(days / block);
    const starts = new Int32Array(paths * blocksNeeded);
    for (let i = 0; i < starts.length; i += 1) {
      starts[i] = Math.min(Math.floor(random() * high), high - 1);
    }
    return { starts, blocksNeeded, block, days, paths };
  }

  function scenarioStatistics(net, plan, startValue, gridPoints) {
    const { starts, blocksNeeded, block, days, paths } = plan;
    const gridSet = [];
    const count = Math.min(gridPoints, days + 1);
    for (let i = 0; i < count; i += 1) {
      const point = Math.round((i * days) / (count - 1));
      if (!gridSet.length || gridSet[gridSet.length - 1] !== point) gridSet.push(point);
    }
    const grid = Int32Array.from(gridSet);
    const banded = [];
    for (let g = 0; g < grid.length; g += 1) banded.push(new Float64Array(paths));
    const terminal = new Float64Array(paths);
    const worstDrawdown = new Float64Array(paths);
    const trough = new Float64Array(paths);

    const walk = new Float64Array(days + 1);
    for (let p = 0; p < paths; p += 1) {
      let value = startValue;
      let peak = startValue;
      let worst = 0;
      let low = startValue;
      walk[0] = startValue;
      let day = 0;
      for (let b = 0; b < blocksNeeded && day < days; b += 1) {
        const origin = starts[p * blocksNeeded + b];
        for (let k = 0; k < block && day < days; k += 1) {
          value *= 1 + net[origin + k];
          day += 1;
          walk[day] = value;
          if (value > peak) peak = value;
          const dd = value / peak - 1;
          if (dd < worst) worst = dd;
          if (value < low) low = value;
        }
      }
      terminal[p] = value;
      worstDrawdown[p] = worst;
      trough[p] = low;
      for (let g = 0; g < grid.length; g += 1) banded[g][p] = walk[grid[g]];
    }

    const bands = { p05: [], p25: [], median: [], p75: [], p95: [] };
    for (let g = 0; g < grid.length; g += 1) {
      const sorted = sortedCopy(banded[g]);
      bands.p05.push(quantile(sorted, 0.05));
      bands.p25.push(quantile(sorted, 0.25));
      bands.median.push(quantile(sorted, 0.5));
      bands.p75.push(quantile(sorted, 0.75));
      bands.p95.push(quantile(sorted, 0.95));
    }
    const sortedTerminal = sortedCopy(terminal);
    const sortedDrawdown = sortedCopy(worstDrawdown);
    const cutoff = quantile(sortedTerminal, 0.05);
    let tailSum = 0;
    let tailCount = 0;
    let losses = 0;
    let belowFloor = 0;
    for (let p = 0; p < paths; p += 1) {
      if (terminal[p] <= cutoff) { tailSum += terminal[p]; tailCount += 1; }
      if (terminal[p] < startValue) losses += 1;
      if (trough[p] < 0.8 * startValue) belowFloor += 1;
    }
    return {
      grid: Array.from(grid),
      bands,
      terminal,
      summary: {
        start_value: startValue,
        terminal_p05: cutoff,
        terminal_p25: quantile(sortedTerminal, 0.25),
        terminal_median: quantile(sortedTerminal, 0.5),
        terminal_p75: quantile(sortedTerminal, 0.75),
        terminal_p95: quantile(sortedTerminal, 0.95),
        worst5_mean: tailCount ? tailSum / tailCount : NaN,
        probability_terminal_loss: losses / paths,
        probability_ever_below_floor: belowFloor / paths,
        median_max_drawdown: quantile(sortedDrawdown, 0.5),
        severe_max_drawdown_p05: quantile(sortedDrawdown, 0.05),
        paths,
      },
    };
  }

  function pairedComparison(left, right, leftName, rightName) {
    const difference = new Float64Array(left.length);
    const ratios = new Float64Array(left.length);
    let ahead = 0;
    for (let i = 0; i < left.length; i += 1) {
      difference[i] = left[i] - right[i];
      ratios[i] = left[i] / right[i];
      if (difference[i] > 0) ahead += 1;
    }
    const sorted = sortedCopy(difference);
    return {
      portfolio: leftName,
      versus: rightName,
      probability_ahead: ahead / left.length,
      difference_p05: quantile(sorted, 0.05),
      difference_median: quantile(sorted, 0.5),
      difference_p95: quantile(sorted, 0.95),
      ratio_median: quantile(sortedCopy(ratios), 0.5),
    };
  }

  PL.engine = {
    TRADING_DAYS, mulberry32, quantile, sortedCopy, metrics, rebalanceMask,
    backtest, rollingReturns, blockPositions, scenarioStatistics, pairedComparison,
  };
})(window.PL = window.PL || {});
