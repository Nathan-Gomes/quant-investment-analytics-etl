/* Portfolio optimization in the browser.

   The Python app solves these as convex programs with cvxpy. This file solves
   the same two programs without a solver library, so the offline build can run
   walk-forward optimization with no server behind it.

   Both are numerical solvers checked against Python within test tolerances:

   * **Minimum variance** is a quadratic over a box intersected with the budget
     constraint. Accelerated projected gradient converges to machine precision at
     this size, and the projection onto {sum w = 1, l <= w <= u} has a one
     dimensional root that bisection finds in about fifty iterations.

   * **Risk parity** uses the log-barrier form, whose coordinate updates have a
     closed form: fixing every other weight leaves a quadratic in w_i with one
     positive root. Cyclical coordinate descent on it converges in tens of
     iterations. Spinu (2013).

   Maximum Sharpe and maximum diversification are deliberately absent. Their
   convex form pins a ratio and rescales, and the per-holding cap does not
   survive that translation cleanly; approximating them here would make the
   browser disagree with the Python answer, and agreement is the point.

   tests/application/test_convex.py checks this file against cvxpy through a Node
   harness. */
(function (PL) {
  "use strict";

  const TRADING_DAYS = 252;

  // ---------------------------------------------------------------- linear algebra

  function multiply(matrix, vector) {
    const n = vector.length;
    const out = new Float64Array(n);
    for (let i = 0; i < n; i += 1) {
      const row = matrix[i];
      let total = 0;
      for (let j = 0; j < n; j += 1) total += row[j] * vector[j];
      out[i] = total;
    }
    return out;
  }

  const dot = (a, b) => {
    let total = 0;
    for (let i = 0; i < a.length; i += 1) total += a[i] * b[i];
    return total;
  };

  /** The largest eigenvalue, by power iteration: the step size needs it. */
  function spectralNorm(matrix, iterations = 80) {
    const n = matrix.length;
    let vector = new Float64Array(n).fill(1 / Math.sqrt(n));
    let value = 0;
    for (let step = 0; step < iterations; step += 1) {
      const next = multiply(matrix, vector);
      const norm = Math.sqrt(dot(next, next));
      if (!(norm > 0)) return 1;
      for (let i = 0; i < n; i += 1) next[i] /= norm;
      value = norm;
      vector = next;
    }
    return value;
  }

  // ---------------------------------------------------------------- projection

  /** Project onto {w : sum w = 1, low <= w <= high}.
   *
   * Clamping at a shifted level is monotone in the shift, so there is exactly
   * one shift where the clamped vector sums to one, and bisection finds it.
   */
  function projectToBudget(vector, low, high) {
    const n = vector.length;
    if (low * n > 1 + 1e-12 || high * n < 1 - 1e-12) {
      throw new Error("These bounds cannot hold a fully invested portfolio.");
    }
    const clamp = (shift) => {
      const out = new Float64Array(n);
      let total = 0;
      for (let i = 0; i < n; i += 1) {
        const value = Math.min(Math.max(vector[i] - shift, low), high);
        out[i] = value;
        total += value;
      }
      return { out, total };
    };
    let lower = Math.min(...vector) - high - 1;
    let upper = Math.max(...vector) - low + 1;
    let result = clamp(0);
    for (let step = 0; step < 200 && Math.abs(result.total - 1) > 1e-13; step += 1) {
      const middle = (lower + upper) / 2;
      result = clamp(middle);
      if (result.total > 1) lower = middle; else upper = middle;
    }
    return result.out;
  }

  // ---------------------------------------------------------------- objectives

  /** Minimum variance, by accelerated projected gradient (FISTA). */
  function minimumVariance(covariance, constraints) {
    const n = covariance.length;
    const low = constraints.min_weight || 0;
    const high = constraints.max_weight === undefined ? 1 : constraints.max_weight;
    const step = 1 / (2 * spectralNorm(covariance));

    let current = projectToBudget(new Float64Array(n).fill(1 / n), low, high);
    let momentum = current.slice();
    let theta = 1;

    for (let iteration = 0; iteration < 20000; iteration += 1) {
      const gradient = multiply(covariance, momentum);
      const candidate = new Float64Array(n);
      for (let i = 0; i < n; i += 1) candidate[i] = momentum[i] - 2 * step * gradient[i];
      const next = projectToBudget(candidate, low, high);

      let movement = 0;
      for (let i = 0; i < n; i += 1) movement = Math.max(movement, Math.abs(next[i] - current[i]));

      const nextTheta = (1 + Math.sqrt(1 + 4 * theta * theta)) / 2;
      const weight = (theta - 1) / nextTheta;
      const lookahead = new Float64Array(n);
      for (let i = 0; i < n; i += 1) lookahead[i] = next[i] + weight * (next[i] - current[i]);

      current = next;
      momentum = lookahead;
      theta = nextTheta;
      if (movement < 1e-13 && iteration > 50) break;
    }
    return current;
  }

  /** Equal risk contribution, by cyclical coordinate descent on the log barrier.
   *
   * Minimizing ½wᵀΣw − Σ bᵢ log wᵢ over w > 0 has first-order condition
   * Σw = b / w, which says each holding contributes risk in proportion to its
   * budget. Holding the other weights fixed leaves Σᵢᵢwᵢ² + cᵢwᵢ − bᵢ = 0, whose
   * positive root is the update. The solution is then rescaled to sum to one.
   */
  function riskParity(covariance, constraints) {
    const n = covariance.length;
    const budget = new Float64Array(n).fill(1 / n);
    let weights = new Float64Array(n).fill(1 / Math.sqrt(n));

    for (let sweep = 0; sweep < 3000; sweep += 1) {
      let movement = 0;
      for (let i = 0; i < n; i += 1) {
        let cross = 0;
        for (let j = 0; j < n; j += 1) if (j !== i) cross += covariance[i][j] * weights[j];
        const a = covariance[i][i];
        if (!(a > 0)) continue;
        const updated = (-cross + Math.sqrt(cross * cross + 4 * a * budget[i])) / (2 * a);
        movement = Math.max(movement, Math.abs(updated - weights[i]));
        weights[i] = updated;
      }
      if (movement < 1e-14) break;
    }

    let total = 0;
    for (let i = 0; i < n; i += 1) total += weights[i];
    for (let i = 0; i < n; i += 1) weights[i] /= total;

    const high = constraints.max_weight === undefined ? 1 : constraints.max_weight;
    if (weights.reduce((most, w) => Math.max(most, w), 0) > high + 1e-9) {
      // Exact equal contribution and a binding cap cannot both hold. The cap is
      // the mandate, so the solution is projected onto it — the same least
      // squares projection the Python version uses, so the two agree.
      return projectToBudget(weights, constraints.min_weight || 0, high);
    }
    return weights;
  }

  // ---------------------------------------------------------------- risk model

  /** Ledoit-Wolf shrinkage toward a scaled identity, annualized. */
  function ledoitWolf(returns) {
    const observations = returns.length;
    const n = returns[0].length;
    const means = new Float64Array(n);
    for (const row of returns) for (let i = 0; i < n; i += 1) means[i] += row[i] / observations;

    const centred = returns.map((row) => {
      const out = new Float64Array(n);
      for (let i = 0; i < n; i += 1) out[i] = row[i] - means[i];
      return out;
    });

    const sample = [];
    for (let i = 0; i < n; i += 1) sample.push(new Float64Array(n));
    for (const row of centred) {
      for (let i = 0; i < n; i += 1) {
        const value = row[i];
        for (let j = 0; j < n; j += 1) sample[i][j] += value * row[j] / observations;
      }
    }

    let trace = 0;
    for (let i = 0; i < n; i += 1) trace += sample[i][i];
    const mu = trace / n;

    let dispersion = 0;
    for (let i = 0; i < n; i += 1) {
      for (let j = 0; j < n; j += 1) {
        const target = i === j ? mu : 0;
        dispersion += (sample[i][j] - target) ** 2;
      }
    }
    dispersion /= n;

    // Mean squared error of the sample entries, from the fourth moments.
    let error = 0;
    for (const row of centred) {
      for (let i = 0; i < n; i += 1) {
        const value = row[i] * row[i];
        for (let j = 0; j < n; j += 1) error += value * row[j] * row[j];
      }
    }
    let squared = 0;
    for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) squared += sample[i][j] ** 2;
    error = Math.max(error / observations - squared, 0) / (observations * n);
    error = Math.min(error, dispersion);

    const intensity = dispersion <= 0 ? 0 : Math.min(Math.max(error / dispersion, 0), 1);
    const scale = observations / (observations - 1) * TRADING_DAYS;
    const shrunk = [];
    for (let i = 0; i < n; i += 1) {
      const row = new Float64Array(n);
      for (let j = 0; j < n; j += 1) {
        const target = i === j ? mu : 0;
        row[j] = (intensity * target + (1 - intensity) * sample[i][j]) * scale;
      }
      shrunk.push(row);
    }
    return { covariance: shrunk, intensity };
  }

  function sampleCovariance(returns) {
    const observations = returns.length;
    const n = returns[0].length;
    const means = new Float64Array(n);
    for (const row of returns) for (let i = 0; i < n; i += 1) means[i] += row[i] / observations;
    const out = [];
    for (let i = 0; i < n; i += 1) out.push(new Float64Array(n));
    for (const row of returns) {
      for (let i = 0; i < n; i += 1) {
        const a = row[i] - means[i];
        for (let j = 0; j < n; j += 1) out[i][j] += a * (row[j] - means[j]) / (observations - 1);
      }
    }
    for (let i = 0; i < n; i += 1) for (let j = 0; j < n; j += 1) out[i][j] *= TRADING_DAYS;
    return { covariance: out, intensity: 0 };
  }

  function riskContributions(weights, covariance) {
    const product = multiply(covariance, weights);
    const variance = dot(weights, product);
    const volatility = Math.sqrt(Math.max(variance, 0));
    const share = new Float64Array(weights.length);
    if (volatility > 0) {
      for (let i = 0; i < weights.length; i += 1) share[i] = weights[i] * product[i] / volatility;
    }
    const standalone = covariance.map((row, i) => Math.sqrt(row[i]));
    let weighted = 0;
    for (let i = 0; i < weights.length; i += 1) weighted += weights[i] * standalone[i];
    let concentration = 0;
    for (let i = 0; i < share.length; i += 1) {
      const fraction = volatility > 0 ? share[i] / volatility : 0;
      if (fraction > 0) concentration += fraction * fraction;
    }
    return {
      volatility,
      share: Array.from(share, (value) => (volatility > 0 ? value / volatility : 0)),
      contribution: Array.from(share),
      diversification_ratio: volatility > 0 ? weighted / volatility : 1,
      effective_bets: concentration > 0 ? 1 / concentration : weights.length,
    };
  }

  const OBJECTIVES = {
    minimum_variance: minimumVariance,
    minimum_variance_convex: minimumVariance,
    risk_parity: riskParity,
    risk_parity_convex: riskParity,
  };

  function solve(objective, returns, constraints, estimator) {
    const model = estimator === "sample" ? sampleCovariance(returns) : ledoitWolf(returns);
    const routine = OBJECTIVES[objective];
    if (!routine) {
      throw new Error(
        `${objective} is solved by the Python app. This browser build solves minimum variance `
        + "and risk parity. Other objectives are not implemented in this browser build.");
    }
    if (!model.covariance.every(row => row.every(Number.isFinite)) ||
        model.covariance.some((row, i) => !(row[i] > 0))) {
      throw new Error("Optimization requires finite, nonzero return variance for every security.");
    }
    const weights = routine(model.covariance, constraints);
    const total = weights.reduce((sum, value) => sum + value, 0);
    const normalized = Array.from(weights, (value) => value / total);
    if (normalized.some(value => !Number.isFinite(value) ||
        value < (constraints.min_weight || 0) - 1e-8 ||
        value > (constraints.max_weight === undefined ? 1 : constraints.max_weight) + 1e-8)) {
      throw new Error("The browser solver did not produce feasible weights.");
    }
    return {
      weights: normalized,
      covariance: model.covariance,
      shrinkage_intensity: model.intensity,
      risk: riskContributions(normalized, model.covariance),
    };
  }

  PL.optimize = {
    solve, minimumVariance, riskParity, ledoitWolf, sampleCovariance,
    riskContributions, projectToBudget, objectives: Object.keys(OBJECTIVES),
  };
})(window.PL = window.PL || {});
