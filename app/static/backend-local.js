/* Offline backend: builds the same payload as app/analysis.py, in the page.
   Used by the standalone demo build, which carries the frozen research dataset
   with it and therefore needs no server and no network. */
(function (PL) {
  "use strict";

  const MAX_CURVE_POINTS = 900;
  const engine = () => PL.engine;

  function thin(length, limit) {
    if (length <= limit) return Array.from({ length }, (_, i) => i);
    const out = [];
    for (let i = 0; i < limit; i += 1) {
      const point = Math.round((i * (length - 1)) / (limit - 1));
      if (!out.length || out[out.length - 1] !== point) out.push(point);
    }
    return out;
  }

  const round = (values, digits) => {
    const factor = Math.pow(10, digits);
    return Array.from(values, (v) => (isFinite(v) ? Math.round(v * factor) / factor : null));
  };
  const clean = (value) => (typeof value === "number" && !isFinite(value) ? null : value);
  const cleanObject = (object) => {
    const out = {};
    for (const [key, value] of Object.entries(object)) out[key] = clean(value);
    return out;
  };

  function addYears(iso, years) {
    const date = new Date(`${iso}T00:00:00Z`);
    date.setUTCDate(date.getUTCDate() + Math.round(years * 365.25));
    return date.toISOString().slice(0, 10);
  }

  function scenarioDates(lastDate, days, years) {
    const start = Date.parse(`${lastDate}T00:00:00Z`);
    const end = Date.parse(`${addYears(lastDate, years)}T00:00:00Z`);
    const out = [];
    for (let i = 0; i <= days; i += 1) {
      out.push(new Date(start + ((end - start) * i) / days).toISOString().slice(0, 10));
    }
    return out;
  }

  function resolveWeights(portfolio, tickers, calibration) {
    const chosen = Object.keys(portfolio.weights);
    if (!chosen.length) throw new Error(`${portfolio.name}: needs at least one holding.`);
    const missing = chosen.filter((t) => !tickers.includes(t));
    if (missing.length) throw new Error(`${portfolio.name}: no price data loaded for ${missing.join(", ")}.`);
    let raw;
    if (portfolio.scheme === "equal") {
      raw = chosen.map(() => 1 / chosen.length);
    } else if (portfolio.scheme === "inverse_volatility") {
      const inverse = chosen.map((ticker) => {
        const series = calibration[tickers.indexOf(ticker)];
        const mean = series.reduce((a, b) => a + b, 0) / series.length;
        const variance = series.reduce((a, b) => a + (b - mean) ** 2, 0) / (series.length - 1);
        const deviation = Math.sqrt(variance);
        if (!(deviation > 0)) {
          throw new Error(`${portfolio.name}: inverse-volatility weights need a non-zero volatility for ${ticker}.`);
        }
        return 1 / deviation;
      });
      const total = inverse.reduce((a, b) => a + b, 0);
      raw = inverse.map((v) => v / total);
    } else {
      const values = chosen.map((t) => Math.max(portfolio.weights[t], 0));
      const total = values.reduce((a, b) => a + b, 0);
      if (!(total > 0)) throw new Error(`${portfolio.name}: weights must add up to more than zero.`);
      raw = values.map((v) => v / total);
    }
    const target = tickers.map(() => 0);
    chosen.forEach((ticker, i) => { target[tickers.indexOf(ticker)] = raw[i]; });
    return target;
  }

  const canonical = (value) => {
    // Sorted keys and no spaces, matching the Python side's digest input.
    if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
    if (value && typeof value === "object") {
      return `{${Object.keys(value).sort().map((k) => `${JSON.stringify(k)}:${canonical(value[k])}`).join(",")}}`;
    }
    return JSON.stringify(value === undefined ? null : value);
  };

  async function sha256(text) {
    if (!globalThis.crypto?.subtle) return null;
    const bytes = new TextEncoder().encode(text);
    const buffer = await globalThis.crypto.subtle.digest("SHA-256", bytes);
    return [...new Uint8Array(buffer)].map((b) => b.toString(16).padStart(2, "0")).join("");
  }

  async function localManifest(request, data) {
    // The browser build can digest the request it was given and name the frozen
    // dataset it carries. It cannot digest the Python source or the environment
    // it does not have, and it says so rather than reporting a partial record as
    // if it were the full one.
    const requestDigest = await sha256(canonical(request));
    return {
      run_id: requestDigest ? requestDigest.slice(0, 16) : null,
      generated_at: new Date().toISOString().slice(0, 19),
      engine_version: `${PL.ENGINE_VERSION || "1.0.0"} (in-browser)`,
      request_sha256: requestDigest,
      request,
      data: {
        rows: data.dates.length * Object.keys(data.prices).length,
        tickers: Object.keys(data.prices).sort(),
        first_date: data.dates[0],
        last_date: data.dates[data.dates.length - 1],
        sha256: null,
        source: data.source,
      },
      code_sha256: null,
      environment: { python: null, packages: {} },
      partial: "Issued by the browser build. Source and environment digests are "
        + "recorded by the Python service, which is what a shared result should cite.",
    };
  }

  function optimizerReport(portfolio, state, run) {
    if (!state || !state.last) return null;
    const solved = state.last;
    const risk = solved.risk;
    const rebalances = Math.max(state.history.length - 1, 0);
    const turnover = run.summary.total_turnover;
    const stability = weightStability(portfolio, state);
    return {
      objective: portfolio.objective,
      label: portfolio.label || portfolio.objective,
      estimator: portfolio.estimator || "ledoit_wolf",
      estimation_days: portfolio.estimation_days || 252,
      max_weight: portfolio.max_weight || 1,
      reoptimizations: state.history.length,
      turnover_per_rebalance: rebalances ? turnover / rebalances : 0,
      annual_turnover: turnover / Math.max(run.summary.years, 1e-9),
      shrinkage_intensity: solved.shrinkage_intensity,
      mean_shrinkage_intensity: 0,
      expected_volatility: risk.volatility,
      expected_return: null,
      diversification_ratio: risk.diversification_ratio,
      effective_bets: risk.effective_bets,
      risk_model: {
        kind: portfolio.estimator === "sample" ? "sample" : "ledoit_wolf",
        assets: state.universe.length,
        shrinkage_intensity: solved.shrinkage_intensity,
        factors: 0,
        explained_variance: 0,
        condition_number: null,
      },
      solver: {
        name: "browser (projected gradient / coordinate descent)",
        status: "computed",
        seconds: 0,
        binding_constraints: solved.weights.some((w) => w > (portfolio.max_weight || 1) - 1e-6) ? ["cap"] : [],
        shadow_prices: {},
        reformulation: portfolio.objective.startsWith("risk_parity") ? "log-barrier" : "projected gradient",
        note: null,
      },
      stability,
      holdings: state.universe.map((ticker, i) => ({
        ticker,
        weight: solved.weights[i],
        risk_share: risk.share[i],
        marginal_risk: null,
        weight_p05: stability ? stability.percentiles.p05[i] : null,
        weight_p95: stability ? stability.percentiles.p95[i] : null,
      })),
      weight_history: {
        dates: state.history.map((row) => row.date),
        tickers: state.universe,
        weights: state.history.map((row) => row.weights.map((w) => Math.round(w * 1e6) / 1e6)),
      },
    };
  }

  /** Re-solve on bootstrapped samples to size the estimation error in the weights. */
  function weightStability(portfolio, state, draws = 12) {
    const context = state.lastSample;
    if (!context) return null;
    const rows = context.length;
    let seed = 9;
    const random = () => {
      seed = (seed * 1103515245 + 12345) & 0x7fffffff;
      return seed / 0x7fffffff;
    };
    const solutions = [];
    for (let draw = 0; draw < draws; draw += 1) {
      const sample = [];
      for (let i = 0; i < rows; i += 1) sample.push(context[Math.floor(random() * rows)]);
      try {
        solutions.push(PL.optimize.solve(portfolio.objective, sample, state.constraints,
                                         portfolio.estimator).weights);
      } catch (error) { /* a failed draw is skipped, as in Python */ }
    }
    if (solutions.length < 3) return null;
    const n = state.universe.length;
    const quantile = (values, q) => {
      const sorted = [...values].sort((a, b) => a - b);
      const position = (sorted.length - 1) * q;
      const low = Math.floor(position);
      const high = Math.ceil(position);
      return low === high ? sorted[low] : sorted[low] + (sorted[high] - sorted[low]) * (position - low);
    };
    const p05 = [], p95 = [], median = [];
    let move = 0;
    for (let i = 0; i < n; i += 1) {
      const column = solutions.map((w) => w[i]);
      p05.push(quantile(column, 0.05));
      p95.push(quantile(column, 0.95));
      median.push(quantile(column, 0.5));
      move += column.reduce((sum, v) => sum + Math.abs(v - state.last.weights[i]), 0) / column.length / n;
    }
    return {
      mean_absolute_move: move,
      widest_range: Math.max(...p95.map((v, i) => v - p05[i])),
      draws: solutions.length,
      percentiles: { p05, median, p95 },
    };
  }

  function analyze(request) {
    const data = PL.DATA;
    const unsupported = request.portfolios.find(
      (p) => p.scheme === "optimized" && !PL.optimize.objectives.includes(p.objective));
    if (unsupported) {
      throw new Error(
        `${unsupported.objective} is solved by the Python app. This browser build runs minimum `
        + "variance and risk parity. The other "
        + "objectives require the Python app.",
      );
    }
    const E = engine();
    const warnings = [];
    const names = [];
    request.portfolios.forEach((p) => Object.keys(p.weights).forEach((t) => names.push(t.toUpperCase())));
    if (request.benchmark) names.push(request.benchmark.toUpperCase());
    const tickers = [...new Set(names)];
    const unknown = tickers.filter((t) => !data.prices[t]);
    if (unknown.length) {
      throw new Error(
        `This demo carries a frozen dataset of ${Object.keys(data.prices).join(", ")}. ` +
        `No prices for ${unknown.join(", ")}. Run the app locally against Yahoo Finance for any other ticker.`,
      );
    }

    const keep = [];
    data.dates.forEach((date, i) => {
      if (request.start && date < request.start) return;
      if (request.end && date > request.end) return;
      if (tickers.every((t) => isFinite(data.prices[t][i]) && data.prices[t][i] > 0)) keep.push(i);
    });
    if (keep.length < 60) throw new Error("Fewer than 60 shared trading days are available. Widen the date range.");
    const dates = keep.map((i) => data.dates[i]);
    const matrix = tickers.map((t) => Float64Array.from(keep, (i) => data.prices[t][i]));

    const dailyReturns = matrix.map((series) => {
      const out = new Float64Array(series.length - 1);
      for (let i = 1; i < series.length; i += 1) out[i - 1] = series[i] / series[i - 1] - 1;
      return out;
    });

    const needsCalibration = request.portfolios.some((p) => p.scheme === "inverse_volatility");
    const estimationNeed = Math.max(0, ...request.portfolios
      .filter((p) => p.scheme === "optimized")
      .map((p) => p.estimation_days || 252));
    const requested = Math.max(needsCalibration ? 252 : 0, estimationNeed);
    const warmup = requested ? Math.min(requested, Math.floor(dates.length / 3)) : 0;
    if (requested && warmup < requested) {
      warnings.push(
        `Weights were estimated on ${warmup} sessions rather than the ${requested} requested, because the ` +
        "window is short. Those sessions are used only to set the opening weights and are excluded from " +
        "every reported result.",
      );
    }
    const calibration = dailyReturns.map((series) => series.subarray(0, warmup || series.length));
    const windowDates = dates.slice(warmup);
    const windowMatrix = matrix.map((series) => series.subarray(warmup));
    if (windowDates.length < 40) throw new Error("Too little history remains after the calibration window.");

    const currencies = new Set(tickers.map((t) => data.securities[t]?.currency).filter(Boolean));
    if (currencies.size > 1) {
      warnings.push(
        `The universe mixes ${[...currencies].sort().join(" and ")}. Returns are measured in each security's own ` +
        "currency and no FX conversion is applied.",
      );
    }

    const settings = {
      initial_capital: request.initial_capital,
      transaction_cost_bps: request.transaction_cost_bps,
      risk_free_rate: request.risk_free_rate,
      rebalance: request.rebalance,
      horizon_years: request.horizon_years,
      paths: request.paths,
      block_days: request.block_days,
      seed: request.seed,
      scenario_basis: request.scenario_basis,
    };

    const portfolios = request.portfolios.map((p) => ({
      name: p.name,
      scheme: p.scheme,
      // The optimizer settings travel with the portfolio; dropping them here
      // left the solver without an objective.
      objective: p.objective,
      estimation_days: p.estimation_days,
      max_weight: p.max_weight,
      estimator: p.estimator,
      weights: Object.fromEntries(Object.entries(p.weights).map(([k, v]) => [k.toUpperCase(), v])),
      is_benchmark: false,
    }));
    let benchmarkName = null;
    if (request.benchmark) {
      benchmarkName = `${request.benchmark.toUpperCase()} benchmark`;
      portfolios.push({
        name: benchmarkName, scheme: "custom",
        weights: { [request.benchmark.toUpperCase()]: 1 }, is_benchmark: true,
      });
    }

    // Returns over the whole aligned window, for the walk-forward estimator.
    const fullReturns = [];
    for (let i = 1; i < dates.length; i += 1) {
      const row = new Float64Array(tickers.length);
      for (let t = 0; t < tickers.length; t += 1) row[t] = matrix[t][i] / matrix[t][i - 1] - 1;
      fullReturns.push(row);
    }

    const optimizerState = {};

    function makeProvider(portfolio) {
      // The same contract as the Python side: the slice ends at the session being
      // traded and never reaches past it, which is what makes it walk-forward.
      const chosen = tickers.filter((t) => portfolio.weights[t] !== undefined);
      if (chosen.length < 2) throw new Error(`${portfolio.name}: optimization needs at least two holdings.`);
      const positions = chosen.map((t) => tickers.indexOf(t));
      const span = portfolio.estimation_days || 252;
      const constraints = { max_weight: portfolio.max_weight || 1, min_weight: 0 };
      const state = { universe: chosen, positions, history: [], last: null, constraints };
      optimizerState[portfolio.name] = state;

      return (position) => {
        const end = warmup + position + 1;
        const start = Math.max(1, end - span);
        const sample = [];
        for (let i = start; i < end; i += 1) {
          const row = new Array(positions.length);
          for (let c = 0; c < positions.length; c += 1) row[c] = fullReturns[i - 1][positions[c]];
          sample.push(row);
        }
        const solved = PL.optimize.solve(portfolio.objective, sample, constraints, portfolio.estimator);
        state.last = solved;
        state.lastSample = sample;
        state.history.push({ date: dates[end - 1], weights: solved.weights });
        const full = new Float64Array(tickers.length);
        positions.forEach((p, c) => { full[p] = solved.weights[c]; });
        return full;
      };
    }

    const runs = portfolios.map((portfolio) => {
      const total = portfolio.scheme === "custom"
        ? Object.values(portfolio.weights).reduce((a, b) => a + Math.max(b, 0), 0) : 1;
      if (portfolio.scheme === "custom" && Math.min(Math.abs(total - 1), Math.abs(total - 100)) > 0.005) {
        warnings.push(
          `${portfolio.name}: the weights you entered do not add up to a whole portfolio, so they were ` +
          "rescaled to 100% while keeping their proportions.",
        );
      }
      const optimizing = portfolio.scheme === "optimized";
      if (optimizing && (portfolio.is_benchmark || settings.rebalance === "none")) {
        throw new Error(
          `${portfolio.name}: an optimized portfolio needs a rebalance schedule, because that is when it `
          + "re-estimates. Choose monthly, quarterly or annual.");
      }
      const provider = optimizing ? makeProvider(portfolio) : null;
      const target = optimizing ? new Float64Array(tickers.length)
        : resolveWeights(portfolio, tickers, calibration);
      const schedule = portfolio.is_benchmark ? "none" : settings.rebalance;
      const run = E.backtest(windowMatrix, windowDates, target, settings, schedule, provider);
      return { portfolio, target: provider ? Array.from(run.targetHistory[0].weights) : target, run };
    });

    const horizonDays = Math.round(settings.horizon_years * E.TRADING_DAYS);
    const sampleLength = windowDates.length - 1;
    const block = Math.min(settings.block_days, sampleLength);
    if (block < settings.block_days) {
      warnings.push(`The bootstrap block was shortened to ${block} sessions to fit the window.`);
    }
    settings.block_days = block;
    const plan = E.blockPositions(sampleLength, horizonDays, block, settings.paths, settings.seed);
    const forwardDates = scenarioDates(windowDates[windowDates.length - 1], horizonDays, settings.horizon_years);

    const scenarios = runs.map(({ portfolio, run }) => {
      const startValue = settings.scenario_basis === "continuation"
        ? run.summary.final_value : settings.initial_capital;
      return E.scenarioStatistics(run.net.subarray(1), plan, startValue, 240);
    });

    const pooled = [];
    scenarios.forEach((s) => s.terminal.forEach((v) => pooled.push(v)));
    const sortedPool = E.sortedCopy(pooled);
    const low = E.quantile(sortedPool, 0.005);
    const high = Math.max(E.quantile(sortedPool, 0.98), low * 1.01);
    const edges = Array.from({ length: 41 }, (_, i) => low + ((high - low) * i) / 40);

    const benchmarkIndex = portfolios.findIndex((p) => p.is_benchmark);
    const benchmarkAnnual = benchmarkIndex >= 0 ? runs[benchmarkIndex].run.summary.annualized_return : null;
    const curveIndex = thin(windowDates.length, MAX_CURVE_POINTS);

    const payloadPortfolios = runs.map(({ portfolio, target, run }, index) => {
      const scenario = scenarios[index];
      const held = tickers.map((t, i) => [t, target[i]]).filter(([, w]) => w > 0).sort((a, b) => b[1] - a[1]);
      const counts = new Array(40).fill(0);
      scenario.terminal.forEach((value) => {
        const slot = Math.floor(((value - low) / (high - low)) * 40);
        if (slot >= 0 && slot < 40) counts[slot] += 1;
      });
      const sectorTotals = {};
      held.forEach(([ticker]) => {
        const sector = data.securities[ticker]?.sector || "Unclassified";
        sectorTotals[sector] = (sectorTotals[sector] || 0) + run.finalWeights[tickers.indexOf(ticker)];
      });
      const summary = cleanObject(run.summary);
      summary.excess_annualized_return = benchmarkAnnual === null || portfolio.is_benchmark
        ? null : run.summary.annualized_return - benchmarkAnnual;
      summary.tracking_error = null;
      if (benchmarkIndex >= 0 && !portfolio.is_benchmark) {
        const other = runs[benchmarkIndex].run.net;
        let mean = 0;
        for (let i = 1; i < run.net.length; i += 1) mean += run.net[i] - other[i];
        mean /= run.net.length - 1;
        let variance = 0;
        for (let i = 1; i < run.net.length; i += 1) variance += (run.net[i] - other[i] - mean) ** 2;
        summary.tracking_error = Math.sqrt(variance / (run.net.length - 2)) * Math.sqrt(E.TRADING_DAYS);
      }
      return {
        name: portfolio.name,
        is_benchmark: portfolio.is_benchmark,
        scheme: portfolio.scheme,
        order: index,
        weights: Object.fromEntries(held.map(([t, w]) => [t, Math.round(w * 1e6) / 1e6])),
        summary,
        rolling: {
          one_year: cleanObject(E.rollingReturns(run.nav, E.TRADING_DAYS)),
          three_year: cleanObject(E.rollingReturns(run.nav, E.TRADING_DAYS * 3)),
        },
        curve: {
          nav: round(curveIndex.map((i) => run.nav[i]), 2),
          drawdown: round(curveIndex.map((i) => run.drawdown[i]), 5),
        },
        contribution: held.map(([ticker]) => {
          const i = tickers.indexOf(ticker);
          return {
            ticker,
            contribution: Math.round(run.contribution[i] * 1e5) / 1e5,
            average_weight: Math.round(run.averageWeights[i] * 1e5) / 1e5,
            final_weight: Math.round(run.finalWeights[i] * 1e5) / 1e5,
          };
        }),
        sectors: Object.entries(sectorTotals)
          .sort((a, b) => b[1] - a[1])
          .map(([sector, weight]) => ({ sector, weight: Math.round(weight * 1e5) / 1e5 })),
        scenario: {
          bands: Object.fromEntries(Object.entries(scenario.bands).map(([k, v]) => [k, round(v, 2)])),
          summary: cleanObject(scenario.summary),
          histogram: { counts },
        },
        optimization: optimizerReport(portfolio, optimizerState[portfolio.name], run),
        versus_benchmark: benchmarkIndex >= 0 && !portfolio.is_benchmark
          ? cleanObject(E.pairedComparison(
            scenario.terminal, scenarios[benchmarkIndex].terminal, portfolio.name, benchmarkName,
          )) : null,
      };
    });

    const windowReturns = windowMatrix.map((series) => {
      const out = new Float64Array(series.length - 1);
      for (let i = 1; i < series.length; i += 1) out[i - 1] = series[i] / series[i - 1] - 1;
      return out;
    });
    const assets = tickers.map((ticker, i) => {
      const stats = E.metrics(windowReturns[i], settings.risk_free_rate);
      const profile = data.securities[ticker] || {};
      return cleanObject({
        ticker,
        name: profile.name || ticker,
        sector: profile.sector || "Unclassified",
        currency: profile.currency || null,
        annualized_return: stats.annualized_return,
        volatility: stats.volatility,
        sharpe: stats.sharpe,
        max_drawdown: stats.max_drawdown,
        first_price: windowMatrix[i][0],
        last_price: windowMatrix[i][windowMatrix[i].length - 1],
      });
    });

    const matrixOut = windowReturns.map((left) => windowReturns.map((right) => {
      const n = left.length;
      let mx = 0;
      let my = 0;
      for (let i = 0; i < n; i += 1) { mx += left[i]; my += right[i]; }
      mx /= n; my /= n;
      let cov = 0;
      let vx = 0;
      let vy = 0;
      for (let i = 0; i < n; i += 1) {
        cov += (left[i] - mx) * (right[i] - my);
        vx += (left[i] - mx) ** 2;
        vy += (right[i] - my) ** 2;
      }
      return Math.round((cov / Math.sqrt(vx * vy)) * 1000) / 1000;
    }));

    return {
      meta: {
        generated_at: new Date().toISOString().slice(0, 19),
        engine_version: `${PL.ENGINE_VERSION || "1.0.0"} (in-browser)`,
        source: data.source,
        window_start: windowDates[0],
        window_end: windowDates[windowDates.length - 1],
        trading_days: windowDates.length,
        years: Math.round(((windowDates.length - 1) / E.TRADING_DAYS) * 100) / 100,
        calibration_days: warmup,
        benchmark: benchmarkName,
        settings,
      },
      quality: {
        calendar_days_in_window: dates.length,
        shared_trading_days: dates.length,
        days_dropped_for_missing_prices: 0,
        first_shared_date: dates[0],
        last_shared_date: dates[dates.length - 1],
      },
      warnings,
      history: { dates: curveIndex.map((i) => windowDates[i]) },
      forward: {
        dates: scenarios[0].grid.map((i) => forwardDates[i]),
        histogram_edges: round(edges, 2),
      },
      assets,
      correlation: { tickers, matrix: matrixOut },
      portfolios: payloadPortfolios,
    };
  }

  PL.backend = {
    sources: ["bundled"],
    // Minimum variance and risk parity translate exactly to JavaScript and run
    // here; the ratio-based and mandate-constrained objectives stay in Python.
    supportsOptimization: true,
    // Both spellings: the registry carries a local-search and a convex version of
    // each, and this file implements the convex answer for either name.
    objectives: ["minimum_variance", "minimum_variance_convex",
                 "risk_parity", "risk_parity_convex"],
    defaultSource: "bundled",
    allowsSourceChoice: false,
    autorun: true,
    defaultStart: null,
    defaultEnd: null,
    // The demo knows every symbol it holds, so the interface can check a ticker
    // as it is typed instead of waiting for the run to fail.
    universe: null,
    async ready() {
      PL.backend.defaultStart = PL.DATA.dates[0];
      PL.backend.defaultEnd = PL.DATA.dates[PL.DATA.dates.length - 1];
      PL.backend.universe = Object.keys(PL.DATA.prices).sort();
    },
    async analyze(request, onProgress) {
      // Fast enough that nobody waits, but the same phases are reported so the
      // interface behaves identically in both builds.
      const step = async (phase, detail) => {
        if (onProgress) onProgress({ phase, detail });
        await new Promise((resolve) => setTimeout(resolve, 0));
      };
      await step("prices", "the frozen research dataset");
      await step("align", "checking the shared trading calendar");
      await step("backtest", `${request.portfolios.length} portfolios`);
      const payload = analyze(request);
      await step("scenarios", `${request.paths.toLocaleString()} paths`);
      payload.manifest = await localManifest(request, PL.DATA);
      payload.meta.run_id = payload.manifest.run_id;
      return payload;
    },
  };
})(window.PL = window.PL || {});
