/* Strata interface.
   State lives in one object, the rail writes to it, and every render reads from
   it. The analysis itself comes from `backend`, which is the FastAPI service in
   the served app and an in-page engine in the offline demo build. */

(function (PL) {
"use strict";
const charts = PL.charts;
const backend = PL.backend;

const SERIES = ["--s1", "--s2", "--s3", "--s5", "--s4", "--s6"];
// The interface offers objectives; the engine takes a scheme plus an objective.
const OBJECTIVES = {
  minimum_variance: "the lowest-variance mix of these holdings",
  risk_parity: "weights where every holding supplies the same share of risk",
  maximum_diversification: "the largest gap between the holdings' own risk and the portfolio's",
  maximum_sharpe: "the best estimated risk-adjusted return, which needs a return forecast",
};
const isObjective = (scheme) => Object.prototype.hasOwnProperty.call(OBJECTIVES, scheme);
const describeObjective = (scheme) => OBJECTIVES[scheme] || "the registered methodology";
const BENCHMARK_COLOR = "--benchmark";

const PRESETS = [
  {
    label: "Research portfolios",
    source: "bundled",
    benchmark: "XIC.TO",
    portfolios: [
      { name: "Growth", scheme: "custom", holdings: [["SHOP.TO", 35], ["CSU.TO", 30], ["CNR.TO", 20], ["RY.TO", 15]] },
      { name: "Balanced", scheme: "custom", holdings: [["RY.TO", 15], ["TD.TO", 10], ["SHOP.TO", 10], ["CSU.TO", 10], ["ENB.TO", 15], ["FTS.TO", 15], ["EMA.TO", 10], ["CNR.TO", 15]] },
      { name: "Income", scheme: "custom", holdings: [["RY.TO", 25], ["TD.TO", 20], ["ENB.TO", 20], ["FTS.TO", 20], ["EMA.TO", 15]] },
    ],
  },
  {
    // The comparison worth running first: three construction rules on one universe.
    label: "Optimizer bake-off",
    source: "bundled",
    benchmark: "XIC.TO",
    portfolios: [
      { name: "Minimum variance", scheme: "minimum_variance", maxWeight: 0.35, estimationDays: 252,
        holdings: [["RY.TO", 0], ["TD.TO", 0], ["SHOP.TO", 0], ["CSU.TO", 0], ["ENB.TO", 0], ["FTS.TO", 0], ["EMA.TO", 0], ["CNR.TO", 0]] },
      { name: "Risk parity", scheme: "risk_parity", maxWeight: 0.35, estimationDays: 252,
        holdings: [["RY.TO", 0], ["TD.TO", 0], ["SHOP.TO", 0], ["CSU.TO", 0], ["ENB.TO", 0], ["FTS.TO", 0], ["EMA.TO", 0], ["CNR.TO", 0]] },
      { name: "Equal weight", scheme: "equal",
        holdings: [["RY.TO", 0], ["TD.TO", 0], ["SHOP.TO", 0], ["CSU.TO", 0], ["ENB.TO", 0], ["FTS.TO", 0], ["EMA.TO", 0], ["CNR.TO", 0]] },
    ],
  },
  {
    label: "Five US large caps",
    source: "auto",
    benchmark: "SPY",
    portfolios: [
      { name: "Equal weight five", scheme: "equal", holdings: [["AAPL", 20], ["MSFT", 20], ["NVDA", 20], ["AMZN", 20], ["GOOGL", 20]] },
      { name: "Concentrated", scheme: "custom", holdings: [["NVDA", 40], ["MSFT", 30], ["AAPL", 30]] },
    ],
  },
  {
    label: "Canadian five",
    source: "auto",
    benchmark: "XIC.TO",
    portfolios: [
      { name: "My five", scheme: "custom", holdings: [["RY.TO", 25], ["ENB.TO", 20], ["CNR.TO", 20], ["FTS.TO", 15], ["SHOP.TO", 20]] },
      { name: "Same five, equal", scheme: "equal", holdings: [["RY.TO", 20], ["ENB.TO", 20], ["CNR.TO", 20], ["FTS.TO", 20], ["SHOP.TO", 20]] },
    ],
  },
];

const state = {
  portfolios: [],
  active: 0,
  settings: {
    start: "", end: "", benchmark: "XIC.TO", capital: 100000, cost: 10, rf: 0.03,
    rebalance: "monthly", horizon: 5, paths: 5000, block: 20, seed: 42,
    basis: "equal", source: "bundled",
  },
  result: null,
  selected: null,
  log: false,
  renderers: [],
};

const $ = (id) => document.getElementById(id);

/* A run is reproducible only if someone else can start from the same inputs.
   The whole setup encodes into the link, so a colleague opening it sees the
   identical study rather than a description of one. */
function encodeSetup() {
  const setup = {
    v: 1,
    p: state.portfolios.map((portfolio) => ({
      n: portfolio.name, s: portfolio.scheme,
      e: portfolio.estimationDays, m: portfolio.maxWeight, c: portfolio.estimator,
      h: portfolio.holdings.filter((holding) => holding.ticker)
        .map((holding) => [holding.ticker, holding.weight]),
    })),
    g: state.settings,
  };
  return btoa(unescape(encodeURIComponent(JSON.stringify(setup)))).replace(/=+$/, "");
}

function decodeSetup(encoded) {
  const setup = JSON.parse(decodeURIComponent(escape(atob(encoded))));
  if (!setup || setup.v !== 1 || !Array.isArray(setup.p) || !setup.p.length) {
    throw new Error("That link does not carry a readable setup.");
  }
  state.portfolios = setup.p.map((portfolio) => ({
    id: uid(),
    name: String(portfolio.n || "Portfolio").slice(0, 40),
    scheme: portfolio.s || "custom",
    estimationDays: Number(portfolio.e) || 252,
    maxWeight: Number(portfolio.m) || 1,
    estimator: portfolio.c === "sample" ? "sample" : "ledoit_wolf",
    holdings: (portfolio.h || []).map(([ticker, weight]) => ({
      ticker: String(ticker).toUpperCase().slice(0, 16), weight: Number(weight) || 0,
    })),
  }));
  Object.assign(state.settings, setup.g || {});
  state.active = 0;
}

function shareSetup() {
  const link = `${location.origin}${location.pathname}#setup=${encodeSetup()}`;
  const done = (message) => {
    const status = $("run-status");
    const previous = status.innerHTML;
    status.textContent = message;
    setTimeout(() => { status.innerHTML = previous; }, 2600);
  };
  history.replaceState(null, "", `#setup=${encodeSetup()}`);
  if (navigator.clipboard?.writeText) {
    navigator.clipboard.writeText(link)
      .then(() => done("Link copied. It reproduces this exact setup."))
      .catch(() => done("Link is in the address bar."));
  } else {
    done("Link is in the address bar.");
  }
}
const colorOf = (index, benchmark) => `var(${benchmark ? BENCHMARK_COLOR : SERIES[index % SERIES.length]})`;
const uid = () => Math.random().toString(36).slice(2, 8);

/* ----------------------------- rail ----------------------------- */

function loadPreset(preset) {
  state.portfolios = preset.portfolios.map((p) => ({
    id: uid(), name: p.name, scheme: p.scheme,
    estimationDays: p.estimationDays || 252, maxWeight: p.maxWeight || 1, estimator: "ledoit_wolf",
    holdings: p.holdings.map(([ticker, weight]) => ({ ticker, weight })),
  }));
  state.active = 0;
  state.settings.benchmark = preset.benchmark;
  // Every preset uses real symbols, so they all work against Yahoo. Only fall
  // back to the frozen dataset when this build has no other option, and switch
  // away from it when the preset reaches for a symbol it does not hold.
  const bundle = backend.universe || backend.bundledUniverse;
  const symbols = preset.portfolios
    .flatMap((p) => p.holdings.map(([ticker]) => ticker))
    .concat(preset.benchmark ? [preset.benchmark] : []);
  if (!backend.allowsSourceChoice) {
    state.settings.source = "bundled";
  } else if (state.settings.source === "bundled" && bundle && !symbols.every((t) => bundle.includes(t))) {
    state.settings.source = "auto";
  }
  backend.universe = state.settings.source === "bundled" ? backend.bundledUniverse : null;
  writeRail();
  showMode();
}

function currentPortfolio() { return state.portfolios[state.active]; }

/* The commonest confusion with this app is not knowing which prices are behind
   the numbers, so the masthead says it outright and updates after every run. */
function showMode(sourceNote) {
  const badge = $("mode");
  const live = backend.allowsSourceChoice && state.settings.source !== "bundled";
  badge.classList.toggle("live", live);
  badge.textContent = live ? "Live prices: Yahoo Finance" : "Frozen dataset: 9 securities";
  badge.title = sourceNote || (live
    ? "Adjusted closes fetched through yfinance when you run an analysis"
    : "The 2018–2026 research dataset bundled with the project. Run the app locally for any other ticker.");
}

function drawTabs() {
  const tabs = $("portfolio-tabs");
  tabs.innerHTML = "";
  state.portfolios.forEach((portfolio, index) => {
    const tab = document.createElement("button");
    tab.type = "button";
    tab.className = "tab";
    tab.setAttribute("aria-selected", String(index === state.active));
    tab.innerHTML = `<span class="swatch" style="background:${colorOf(index)}"></span>${portfolio.name || "Untitled"}`;
    tab.onclick = () => { state.active = index; writeRail(); };
    tabs.appendChild(tab);
  });
  if (state.portfolios.length < 4) {
    const add = document.createElement("button");
    add.type = "button";
    add.className = "tab add";
    add.textContent = "+ Compare another portfolio";
    add.onclick = () => {
      const source = currentPortfolio();
      state.portfolios.push({
        id: uid(),
        name: `Portfolio ${String.fromCharCode(65 + state.portfolios.length)}`,
        scheme: "equal",
        estimationDays: 252, maxWeight: 1, estimator: "ledoit_wolf",
        holdings: source.holdings.map((h) => ({ ...h })),
      });
      state.active = state.portfolios.length - 1;
      writeRail();
    };
    tabs.appendChild(add);
  }
  $("portfolio-count").textContent = state.portfolios.length > 1
    ? `${state.portfolios.length} compared` : "1 portfolio";
}

function checkTicker(input, ticker) {
  const known = !backend.universe || !ticker || backend.universe.includes(ticker);
  input.classList.toggle("unknown", !known);
  input.setAttribute("aria-invalid", String(!known));
  return known;
}

function drawUniverseNote() {
  const note = $("universe-note");
  if (!backend.universe) { note.textContent = ""; return; }
  const unknown = state.portfolios
    .flatMap((p) => p.holdings)
    .map((h) => h.ticker)
    .filter((t) => t && !backend.universe.includes(t));
  note.textContent = unknown.length
    ? `${[...new Set(unknown)].join(", ")} ${unknown.length > 1 ? "are" : "is"} not in this demo's dataset. It holds ${backend.universe.join(", ")} — run the app locally for any other symbol.`
    : "";
}

function drawOptimizerSettings() {
  const portfolio = currentPortfolio();
  const panel = $("optimizer-settings");
  const optimizing = isObjective(portfolio.scheme);
  panel.hidden = !optimizing;
  if (!optimizing) return;
  $("estimation").value = String(portfolio.estimationDays);
  $("maxweight").value = String(portfolio.maxWeight);
  $("estimator").value = portfolio.estimator;
  const years = (portfolio.estimationDays / 252).toFixed(portfolio.estimationDays % 252 ? 1 : 0);
  $("optimizer-note").textContent =
    `${describeObjective(portfolio.scheme)} Re-estimated at every rebalance from the previous `
    + `${years} year${years === "1" ? "" : "s"} of returns only. The holdings below set the universe; `
    + "the weights you type are ignored.";
}

function drawHoldings() {
  const portfolio = currentPortfolio();
  const host = $("holdings");
  host.innerHTML = "";
  const derived = portfolio.scheme !== "custom";
  portfolio.holdings.forEach((holding, index) => {
    const row = document.createElement("div");
    row.className = "holding";
    row.innerHTML = `
      <input class="ticker" value="${holding.ticker}" placeholder="TICKER" aria-label="Ticker ${index + 1}" spellcheck="false"${backend.universe ? ' list="universe"' : ""}>
      <button class="step" type="button" data-delta="-1" aria-label="Decrease ${holding.ticker || "holding"} weight" title="Hold shift for 5" ${derived ? "disabled" : ""}>&minus;</button>
      <input class="weight" type="number" step="1" min="0" value="${holding.weight}" aria-label="Weight ${index + 1}" ${derived ? "disabled" : ""}>
      <button class="step" type="button" data-delta="1" aria-label="Increase ${holding.ticker || "holding"} weight" title="Hold shift for 5" ${derived ? "disabled" : ""}>+</button>
      <span class="unit">%</span>
      <button class="remove" type="button" aria-label="Remove ${holding.ticker || "holding"}">&times;</button>`;
    const [ticker, down, weight, up, , remove] = row.children;
    ticker.oninput = () => {
      holding.ticker = ticker.value.trim().toUpperCase();
      checkTicker(ticker, holding.ticker);
      drawUniverseNote();
    };
    ticker.onchange = () => { ticker.value = holding.ticker; drawTabs(); };
    checkTicker(ticker, holding.ticker);
    weight.oninput = () => { holding.weight = Number(weight.value) || 0; drawWeightBar(); };
    const nudge = (event, delta) => {
      // Shift steps by five, which is how most allocations are actually written.
      const size = event.shiftKey ? 5 : 1;
      holding.weight = Math.max(0, Math.round((holding.weight + delta * size) * 100) / 100);
      weight.value = holding.weight;
      drawWeightBar();
    };
    down.onclick = (event) => nudge(event, -1);
    up.onclick = (event) => nudge(event, 1);
    remove.onclick = () => {
      portfolio.holdings.splice(index, 1);
      if (!portfolio.holdings.length) portfolio.holdings.push({ ticker: "", weight: 0 });
      writeRail();
    };
    host.appendChild(row);
  });
  drawWeightBar();
  drawUniverseNote();
  drawOptimizerSettings();
}

function drawWeightBar() {
  const portfolio = currentPortfolio();
  const bar = $("weightbar");
  bar.innerHTML = "";
  const derived = portfolio.scheme !== "custom";
  const entries = portfolio.holdings.filter((h) => h.ticker);
  const weights = derived ? entries.map(() => 1) : entries.map((h) => Math.max(h.weight, 0));
  const total = weights.reduce((a, b) => a + b, 0);
  // Each holding gets its own colour rather than a fade of one, so the split is
  // readable at a glance before anything has been run.
  weights.forEach((weight, index) => {
    const span = document.createElement("span");
    span.style.width = `${total ? (weight / total) * 100 : 0}%`;
    span.style.background = `var(${SERIES[index % SERIES.length]})`;
    span.title = `${entries[index].ticker}: ${total ? ((weight / total) * 100).toFixed(1) : "0.0"}%`;
    bar.appendChild(span);
  });
  const raw = portfolio.holdings.reduce((a, h) => a + (h.ticker ? Math.max(h.weight, 0) : 0), 0);
  $("weight-total").textContent = derived
    ? `${entries.length} holdings, weights computed`
    : `Total: ${raw.toFixed(1)}%${Math.abs(raw - 100) > 0.05 ? " — rescaled to 100%" : ""}`;
}

function writeRail() {
  const portfolio = currentPortfolio();
  drawTabs();
  $("portfolio-name").value = portfolio.name;
  $("scheme").value = portfolio.scheme;
  drawHoldings();
  const s = state.settings;
  $("start").value = s.start; $("end").value = s.end;
  $("benchmark").value = s.benchmark; $("capital").value = s.capital;
  $("cost").value = s.cost; $("rebalance").value = s.rebalance;
  $("horizon").value = s.horizon; $("paths").value = s.paths;
  $("block").value = s.block; $("basis").value = s.basis;
  $("rf").value = s.rf; $("seed").value = s.seed; $("source").value = s.source;
}

function readRail() {
  const s = state.settings;
  s.start = $("start").value; s.end = $("end").value;
  s.benchmark = $("benchmark").value.trim().toUpperCase();
  s.capital = Number($("capital").value) || 100000;
  s.cost = Number($("cost").value);
  s.rf = Number($("rf").value);
  s.rebalance = $("rebalance").value;
  s.horizon = Number($("horizon").value) || 5;
  s.paths = Number($("paths").value) || 5000;
  s.block = Number($("block").value) || 20;
  s.seed = Number($("seed").value) || 0;
  s.basis = $("basis").value;
  s.source = $("source").value;
}

function bindRail() {
  $("portfolio-name").oninput = (e) => { currentPortfolio().name = e.target.value; drawTabs(); };
  $("scheme").onchange = (e) => { currentPortfolio().scheme = e.target.value; drawHoldings(); };
  $("estimation").onchange = (e) => { currentPortfolio().estimationDays = Number(e.target.value); drawOptimizerSettings(); };
  $("maxweight").onchange = (e) => { currentPortfolio().maxWeight = Number(e.target.value); };
  $("estimator").onchange = (e) => { currentPortfolio().estimator = e.target.value; };
  $("add-holding").onclick = () => { currentPortfolio().holdings.push({ ticker: "", weight: 0 }); drawHoldings(); };
  $("normalize").onclick = () => {
    const holdings = currentPortfolio().holdings.filter((h) => h.ticker);
    const total = holdings.reduce((a, h) => a + Math.max(h.weight, 0), 0);
    if (!total) holdings.forEach((h) => { h.weight = Number((100 / holdings.length).toFixed(2)); });
    else holdings.forEach((h) => { h.weight = Number(((Math.max(h.weight, 0) / total) * 100).toFixed(2)); });
    drawHoldings();
  };
  document.querySelectorAll("[data-years]").forEach((chip) => {
    chip.onclick = () => {
      const end = new Date();
      $("end").value = end.toISOString().slice(0, 10);
      if (chip.dataset.years === "max") $("start").value = "";
      else {
        const start = new Date(end);
        start.setFullYear(start.getFullYear() - Number(chip.dataset.years));
        $("start").value = start.toISOString().slice(0, 10);
      }
      readRail();
    };
  });
  const presets = $("presets");
  PRESETS
    .filter((p) => backend.sources.includes(p.source) || p.source === "auto")
    .filter((p) => backend.supportsOptimization || !p.portfolios.some((q) => isObjective(q.scheme)))
    .forEach((preset) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = preset.label;
    chip.onclick = () => loadPreset(preset);
    presets.appendChild(chip);
  });
  $("source").onchange = () => {
    readRail();
    // Only a closed dataset can be checked as you type; Yahoo is open ended.
    backend.universe = state.settings.source === "bundled" ? backend.bundledUniverse : null;
    drawHoldings();
    showMode();
  };
  // Cards fold away once a decision is settled. The header is the control, so
  // it carries the keyboard affordances a button would.
  document.querySelectorAll(".card > header").forEach((header) => {
    const card = header.parentElement;
    const toggle = () => card.setAttribute("data-open", card.getAttribute("data-open") === "false" ? "true" : "false");
    header.tabIndex = 0;
    header.setAttribute("role", "button");
    header.onclick = toggle;
    header.onkeydown = (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); toggle(); }
    };
  });
  $("reset-setup").onclick = () => {
    const preset = PRESETS.find((p) => backend.sources.includes(p.source)) || PRESETS[0];
    history.replaceState(null, "", location.pathname);
    loadPreset(preset);
  };
  $("rail").onsubmit = (event) => { event.preventDefault(); run(); };
  $("theme-toggle").onclick = () => {
    const prefersDark = typeof window.matchMedia === "function"
      && window.matchMedia("(prefers-color-scheme: dark)").matches;
    const dark = document.documentElement.getAttribute("data-theme") === "dark"
      || (!document.documentElement.getAttribute("data-theme") && prefersDark);
    document.documentElement.setAttribute("data-theme", dark ? "light" : "dark");
    $("theme-toggle").textContent = dark ? "Dark" : "Light";
    rerender();
  };
  $("share-setup").onclick = shareSetup;
  $("export-json").onclick = () => {
    if (!state.result) return;
    const blob = new Blob([JSON.stringify(state.result, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `strata-${state.result.meta.run_id || "results"}.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  };
  if (!backend.allowsSourceChoice) $("source-field").style.display = "none";
  if (backend.supportsOptimization && backend.strategies) {
    const group = $("scheme").querySelector('optgroup[label^="Re-optimized"]');
    if (group) {
      group.innerHTML = "";
      backend.strategies.forEach((strategy) => {
        const option = document.createElement("option");
        option.value = strategy.name;
        option.textContent = strategy.label;
        option.title = strategy.description;
        group.appendChild(option);
        OBJECTIVES[strategy.name] = strategy.description;
      });
    }
  }
  if (!backend.supportsOptimization) {
    // Say why the options are gone rather than leaving a shorter list unexplained.
    const group = $("scheme").querySelector('optgroup[label^="Re-optimized"]');
    if (group) group.remove();
    const note = document.createElement("p");
    // The weighting control sits in a two-column grid, so the note has to span
    // it; dropped in as a plain sibling it becomes a narrow second column.
    note.className = "note span2";
    note.style.margin = "-2px 0 0";
    note.textContent = "Minimum variance, risk parity, maximum diversification and maximum Sharpe are "
      + "solved in the Python app; this browser demo runs the fixed rules only.";
    const cell = $("scheme").closest(".span2");
    if (cell) cell.after(note);
    else ($("scheme").closest(".field") || $("scheme")).after(note);
  }
  if (backend.universe) {
    const list = document.createElement("datalist");
    list.id = "universe";
    backend.universe.forEach((ticker) => {
      const option = document.createElement("option");
      option.value = ticker;
      list.appendChild(option);
    });
    document.body.appendChild(list);
  }
}

/* ----------------------------- run ----------------------------- */

function buildRequest() {
  readRail();
  const s = state.settings;
  const portfolios = state.portfolios.map((portfolio) => {
    const weights = {};
    portfolio.holdings.forEach((h) => {
      if (!h.ticker) return;
      // A row left at zero is not a holding: keep its ticker out of the universe
      // so a half-typed symbol cannot fail the whole run.
      if (portfolio.scheme === "custom" && !(h.weight > 0)) return;
      weights[h.ticker] = portfolio.scheme === "custom" ? h.weight : 1;
    });
    const optimizing = isObjective(portfolio.scheme);
    return {
      name: portfolio.name || "Portfolio",
      weights,
      scheme: optimizing ? "optimized" : portfolio.scheme,
      objective: optimizing ? portfolio.scheme : undefined,
      estimation_days: portfolio.estimationDays,
      max_weight: portfolio.maxWeight,
      estimator: portfolio.estimator,
    };
  });
  const empty = portfolios.find((p) => !Object.keys(p.weights).length);
  if (empty) {
    throw new Error(`${empty.name} needs at least one holding with a weight above zero.`);
  }
  return {
    portfolios,
    start: s.start || null,
    end: s.end || null,
    benchmark: s.benchmark || null,
    initial_capital: s.capital,
    transaction_cost_bps: s.cost,
    risk_free_rate: s.rf,
    rebalance: s.rebalance,
    horizon_years: s.horizon,
    paths: s.paths,
    block_days: s.block,
    seed: s.seed,
    scenario_basis: s.basis,
    source: s.source,
  };
}

let first = true;

/* The work takes seconds, and on a sleeping free-tier host the first request
   takes longer still. Silence for that long is indistinguishable from a hang, so
   the button reports the stage it is in, how long it has been going, and — if
   nothing has come back at all — that the server is probably waking up. */
const PHASES = {
  accepted: { label: "Request accepted", share: 0.02 },
  prices: { label: "Loading prices", share: 0.12 },
  align: { label: "Checking and aligning the data", share: 0.05 },
  backtest: { label: "Walking the backtest", share: 0.45 },
  scenarios: { label: "Simulating markets", share: 0.26 },
  diagnostics: { label: "Risk decomposition and frontier", share: 0.10 },
};

function progressReporter(status) {
  const started = performance.now();
  let reached = 0;
  let latest = "Starting";
  let woken = false;

  const bar = document.createElement("div");
  bar.className = "progress";
  bar.innerHTML = '<span class="progress-fill"></span>';
  const line = document.createElement("div");
  line.className = "progress-line";
  status.innerHTML = "";
  status.append(bar, line);

  const paint = () => {
    const seconds = (performance.now() - started) / 1000;
    if (!woken && seconds > 4 && reached <= 0.02) {
      latest = "Waking the server — a free host sleeps when idle, so the first run is slow";
    }
    bar.firstChild.style.width = `${Math.min(reached, 0.97) * 100}%`;
    line.textContent = `${latest} · ${seconds.toFixed(1)}s`;
  };
  const timer = setInterval(paint, 100);
  paint();

  return {
    update(event) {
      woken = true;
      const phase = PHASES[event.phase];
      if (!phase) return;
      let done = 0;
      for (const [name, spec] of Object.entries(PHASES)) {
        if (name === event.phase) break;
        done += spec.share;
      }
      // Within a phase, step/steps says how far along it is.
      const within = event.steps ? (event.step - 1) / event.steps : 0;
      reached = Math.max(reached, done + phase.share * within);
      latest = event.detail ? `${phase.label}: ${event.detail}` : phase.label;
      paint();
    },
    finish() { clearInterval(timer); },
    elapsed() { return performance.now() - started; },
  };
}

async function run() {
  const button = $("run");
  const status = $("run-status");
  button.disabled = true;
  const reporter = progressReporter(status);
  try {
    const request = buildRequest();
    const result = await backend.analyze(request, (event) => reporter.update(event));
    reporter.finish();
    state.result = result;
    state.selected = result.portfolios.find((p) => !p.is_benchmark)?.name || result.portfolios[0].name;
    const timings = result.meta.timings_ms;
    status.textContent = `Done in ${(reporter.elapsed() / 1000).toFixed(1)}s · `
      + `${result.meta.trading_days.toLocaleString()} shared sessions · `
      + `${result.meta.settings.paths.toLocaleString()} simulated markets`
      + (result.meta.result_cache_hit ? " · reused verified result"
        : timings ? ` · engine ${(timings.total / 1000).toFixed(1)}s` : "");
    showMode(result.meta.source);
    renderResults();
    if (!first) revealResults();
    first = false;
  } catch (error) {
    reporter.finish();
    renderError(error.message || String(error), status);
  } finally {
    button.disabled = false;
  }
}

/* --------------------------- rendering --------------------------- */

function seriesColor(portfolio) {
  if (portfolio.is_benchmark) return `var(${BENCHMARK_COLOR})`;
  const order = state.result.portfolios.filter((p) => !p.is_benchmark).indexOf(portfolio);
  return colorOf(order);
}

function renderError(message, status) {
  $("results").innerHTML = `<div class="flags"><div class="flag error">${escapeHtml(message)}</div></div>`;
  if (status) status.innerHTML = `<span class="rail-error">${escapeHtml(message)}</span>`;
  revealResults();
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function rerender() { if (state.result) renderResults(); }

function revealResults() {
  const results = $("results");
  const box = results.getBoundingClientRect();
  const offScreen = box.top > window.innerHeight * 0.6 || box.bottom < 80;
  if (!offScreen || typeof results.scrollIntoView !== "function") return;
  const still = typeof window.matchMedia === "function"
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  results.scrollIntoView({ behavior: still ? "auto" : "smooth", block: "start" });
}

function renderResults() {
  const result = state.result;
  const host = $("results");
  host.innerHTML = "";
  state.renderers = [];

  if (result.warnings?.length) {
    const flags = document.createElement("div");
    flags.className = "flags";
    result.warnings.forEach((warning) => {
      const flag = document.createElement("div");
      flag.className = "flag";
      flag.textContent = warning;
      flags.appendChild(flag);
    });
    host.appendChild(flags);
  }

  host.appendChild(heroPlate(result));
  host.appendChild(metricsPlate(result));
  host.appendChild(drawdownPlate(result));
  host.appendChild(scenarioPlate(result));
  const optimization = optimizationPlate(result);
  if (optimization) host.appendChild(optimization);
  host.appendChild(compositionPlate(result));
  host.appendChild(assumptionsPlate(result));

  const observer = charts.observeResize(state.renderers);
  observer.observe(host);
}

function plate(title, method) {
  const node = document.createElement("section");
  node.className = "plate";
  node.innerHTML = `<header><h2>${escapeHtml(title)}</h2><span class="method">${method}</span></header>`;
  return node;
}

function heroPlate(result) {
  const node = plate(
    "One continuous picture",
    `Realized value through ${result.meta.window_end}, then ${result.meta.settings.paths.toLocaleString()} scenarios over ${result.meta.settings.horizon_years} years. The shaded fan is the selected portfolio's 5th–95th and 25th–75th percentile range; dashed lines are the other portfolios' medians.`,
  );
  const hero = document.createElement("div");
  hero.className = "hero";

  const legend = document.createElement("div");
  legend.className = "legend";
  result.portfolios.forEach((portfolio) => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-pressed", String(portfolio.name === state.selected));
    button.innerHTML = `<span class="swatch" style="background:${seriesColor(portfolio)}"></span>${escapeHtml(portfolio.name)}`;
    button.onclick = () => { state.selected = portfolio.name; renderResults(); };
    legend.appendChild(button);
  });
  const scaleToggle = document.createElement("button");
  scaleToggle.type = "button";
  scaleToggle.style.marginLeft = "auto";
  scaleToggle.setAttribute("aria-pressed", String(state.log));
  scaleToggle.textContent = state.log ? "Log scale" : "Linear scale";
  scaleToggle.onclick = () => { state.log = !state.log; renderResults(); };
  legend.appendChild(scaleToggle);
  hero.appendChild(legend);

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "chart");
  hero.appendChild(svg);
  const readout = document.createElement("div");
  readout.className = "readout";
  readout.textContent = "Hover the chart to read values.";
  hero.appendChild(readout);
  node.appendChild(hero);

  const draw = charts.timeline(svg, {
    log: state.log,
    selected: state.selected,
    seamLabel: result.meta.settings.scenario_basis === "equal"
      ? `scenarios restart at ${charts.money(result.meta.settings.initial_capital)}`
      : "scenarios continue from here",
    history: {
      dates: result.history.dates,
      series: result.portfolios.map((p) => ({
        key: p.name, color: seriesColor(p), values: p.curve.nav, benchmark: p.is_benchmark,
      })),
    },
    forward: {
      dates: result.forward.dates,
      series: result.portfolios.map((p) => ({
        key: p.name, color: seriesColor(p), bands: p.scenario.bands, benchmark: p.is_benchmark,
      })),
    },
    onHover: (point) => {
      if (!point) { readout.textContent = "Hover the chart to read values."; return; }
      const parts = [`<span class="k">${point.date}</span>`];
      result.portfolios.forEach((p) => {
        const value = point.future ? p.scenario.bands.median[point.index] : p.curve.nav[point.index];
        parts.push(`<span><span class="swatch" style="display:inline-block;width:8px;height:8px;background:${seriesColor(p)};margin-right:5px"></span>${escapeHtml(p.name)} <span class="k">${charts.money(value)}</span></span>`);
      });
      if (point.future) parts.push('<span class="faint">median of simulated paths</span>');
      readout.innerHTML = parts.join(" ");
    },
  });
  state.renderers.push(draw);
  return node;
}

function metricsPlate(result) {
  const node = plate(
    "What the window actually did",
    `${result.meta.window_start} to ${result.meta.window_end}, ${result.meta.years} years. Costs of ${result.meta.settings.transaction_cost_bps} bps are charged on every rebalance. Click a row to make it the focus.`,
  );
  const columns = [
    ["Portfolio", (p) => `<span class="swatch" style="background:${seriesColor(p)}"></span>${escapeHtml(p.name)}`, "label"],
    ["Final value", (p) => charts.money(p.summary.final_value)],
    ["Total return", (p) => charts.percent(p.summary.cumulative_return, 0)],
    ["Annualized", (p) => charts.percent(p.summary.annualized_return)],
    ["Volatility", (p) => charts.percent(p.summary.volatility)],
    ["Sharpe", (p) => charts.ratio(p.summary.sharpe)],
    ["Sortino", (p) => charts.ratio(p.summary.sortino)],
    ["Worst drawdown", (p) => charts.percent(p.summary.max_drawdown)],
    ["vs benchmark", (p) => (p.summary.excess_annualized_return === null || p.is_benchmark ? "—" : charts.percent(p.summary.excess_annualized_return))],
    ["Costs paid", (p) => charts.money(p.summary.total_cost)],
  ];
  const wrap = document.createElement("div");
  wrap.className = "scroll";
  wrap.style.overflowX = "auto";
  const table = document.createElement("table");
  table.className = "data";
  table.innerHTML = `<thead><tr>${columns.map(([name]) => `<th>${name}</th>`).join("")}</tr></thead>`;
  const body = document.createElement("tbody");
  result.portfolios.forEach((portfolio) => {
    const row = document.createElement("tr");
    row.setAttribute("aria-selected", String(portfolio.name === state.selected));
    row.innerHTML = columns.map(([, render, cls]) => `<td class="${cls || ""}">${render(portfolio)}</td>`).join("");
    row.onclick = () => { state.selected = portfolio.name; renderResults(); };
    body.appendChild(row);
  });
  table.appendChild(body);
  wrap.appendChild(table);
  node.appendChild(wrap);

  const selected = result.portfolios.find((p) => p.name === state.selected);
  if (selected?.rolling?.one_year?.windows) {
    const rolling = document.createElement("p");
    rolling.className = "note";
    rolling.style.marginTop = "10px";
    const one = selected.rolling.one_year;
    const three = selected.rolling.three_year;
    rolling.innerHTML = `Across every rolling one-year window inside this history, ${escapeHtml(selected.name)} ranged from ${charts.percent(one.worst)} to ${charts.percent(one.best)}, with ${charts.percent(one.share_negative, 0)} of those windows ending down.` +
      (three?.windows ? ` Over rolling three-year windows the range was ${charts.percent(three.worst)} to ${charts.percent(three.best)}.` : "");
    node.appendChild(rolling);
  }
  return node;
}

function drawdownPlate(result) {
  const node = plate("Depth of the holes", "Value against its own running peak. The starting balance counts as the first peak, so an early loss shows as a drawdown.");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "chart");
  node.appendChild(svg);
  state.renderers.push(charts.drawdown(svg, {
    dates: result.history.dates,
    selected: state.selected,
    series: result.portfolios.map((p) => ({
      key: p.name, color: seriesColor(p), values: p.curve.drawdown, benchmark: p.is_benchmark,
    })),
  }));
  return node;
}

function scenarioPlate(result) {
  const settings = result.meta.settings;
  const node = plate(
    `Where ${settings.horizon_years} years could land`,
    `Every portfolio is compounded through the same ${settings.paths.toLocaleString()} sampled sequences of ${settings.block_days}-session blocks drawn from its own completed returns. This repeats the window's return distribution; it is not a forecast.`,
  );
  const selected = result.portfolios.find((p) => p.name === state.selected) || result.portfolios[0];
  const scenario = selected.scenario.summary;

  const callouts = document.createElement("div");
  callouts.className = "callouts";
  // A percentile from a finite number of paths carries sampling error. Quoting
  // it without one invites the reader to believe the last digits.
  const error = scenario.sampling_error || {};
  const noise = (key) => {
    const band = error[key];
    return band && band.low != null && band.high != null
      ? ` · 95% sampling interval for this percentile: ${charts.money(band.low)}–${charts.money(band.high)}` : "";
  };
  const cells = [
    ["Median outcome", charts.money(scenario.terminal_median),
      `from ${charts.money(scenario.start_value)}${noise("terminal_median")}`],
    ["Weak case (5th pct)", charts.money(scenario.terminal_p05),
      `mean of the worst 5%: ${charts.money(scenario.worst5_mean)}${noise("terminal_p05")}`],
    ["Strong case (95th pct)", charts.money(scenario.terminal_p95),
      `25th–75th: ${charts.money(scenario.terminal_p25)} to ${charts.money(scenario.terminal_p75)}${noise("terminal_p95")}`],
    ["Ends below start", charts.percent(scenario.probability_terminal_loss, 1), `of ${scenario.paths.toLocaleString()} paths`],
    ["Typical worst drop", charts.percent(scenario.median_max_drawdown), `1 in 20 paths fall ${charts.percent(scenario.severe_max_drawdown_p05)} or worse`],
    ["Dips under 80% of start", charts.percent(scenario.probability_ever_below_floor, 1), "at any point along the way"],
  ];
  cells.forEach(([k, v, sub]) => {
    const cell = document.createElement("div");
    cell.className = "callout";
    cell.innerHTML = `<span class="k">${k}</span><span class="v">${v}</span><span class="sub">${sub}</span>`;
    callouts.appendChild(cell);
  });
  node.appendChild(callouts);
  const caption = document.createElement("p");
  caption.className = "note";
  caption.style.margin = "10px 0 18px";
  caption.textContent = `Figures above are for ${selected.name}.`;
  node.appendChild(caption);

  const grid = document.createElement("div");
  grid.className = "grid2";

  const left = document.createElement("div");
  left.innerHTML = '<h3 style="margin-bottom:8px">Range of final values</h3>';
  const whiskerSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  whiskerSvg.setAttribute("class", "chart");
  left.appendChild(whiskerSvg);
  state.renderers.push(charts.whiskers(whiskerSvg, {
    start: result.portfolios[0].scenario.summary.start_value,
    rows: result.portfolios.map((p) => ({
      name: p.name, color: seriesColor(p), start: p.scenario.summary.start_value,
      p05: p.scenario.summary.terminal_p05, p25: p.scenario.summary.terminal_p25,
      median: p.scenario.summary.terminal_median, p75: p.scenario.summary.terminal_p75,
      p95: p.scenario.summary.terminal_p95,
    })),
  }));
  grid.appendChild(left);

  const right = document.createElement("div");
  right.innerHTML = `<h3 style="margin-bottom:8px">Where the ${settings.paths.toLocaleString()} outcomes cluster after ${settings.horizon_years} years</h3>`;
  const distSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  distSvg.setAttribute("class", "chart");
  right.appendChild(distSvg);
  state.renderers.push(charts.distribution(distSvg, {
    edges: result.forward.histogram_edges,
    start: result.portfolios[0].scenario.summary.start_value,
    selected: state.selected,
    horizonLabel: `${settings.horizon_years} years`,
    series: result.portfolios.map((p) => ({ key: p.name, color: seriesColor(p), counts: p.scenario.histogram.counts })),
  }));
  grid.appendChild(right);
  node.appendChild(grid);

  const paired = result.portfolios.filter((p) => p.versus_benchmark);
  if (paired.length) {
    const compare = document.createElement("div");
    compare.style.marginTop = "22px";
    compare.innerHTML = `<h3>Head to head in the same markets</h3><p class="note" style="margin:4px 0 12px">Each path is one sampled market sequence. This counts the paths where the portfolio finishes ahead of ${escapeHtml(result.meta.benchmark)}, rather than comparing two separately drawn samples.</p>`;
    const bars = document.createElement("div");
    bars.className = "bars";
    paired.forEach((portfolio) => {
      const stat = portfolio.versus_benchmark;
      const row = document.createElement("div");
      row.className = "row";
      row.innerHTML = `<span>${escapeHtml(portfolio.name)}</span>
        <span class="track"><span class="fill" style="left:0;width:${(stat.probability_ahead * 100).toFixed(1)}%;background:${seriesColor(portfolio)}"></span></span>
        <span class="value">${charts.percent(stat.probability_ahead, 1)}</span>`;
      bars.appendChild(row);
    });
    compare.appendChild(bars);
    const median = paired[0].versus_benchmark;
    const detail = document.createElement("p");
    detail.className = "note";
    detail.style.marginTop = "10px";
    detail.textContent = `Median gap for ${paired[0].portfolio ?? paired[0].name}: ${charts.money(median.difference_median)}. In the weakest twentieth of paths it trails by ${charts.money(Math.abs(Math.min(median.difference_p05, 0)))}.`;
    compare.appendChild(detail);
    node.appendChild(compare);
  }
  return node;
}

function compositionPlate(result) {
  const selected = result.portfolios.find((p) => p.name === state.selected) || result.portfolios[0];
  const node = plate(
    `Inside ${selected.name}`,
    "Contribution is the sum of each holding's weight on the previous close times its return that day, so it adds up close to the portfolio's own return.",
  );
  const grid = document.createElement("div");
  grid.className = "grid2";

  const holdings = document.createElement("div");
  holdings.innerHTML = "<h3>Holdings</h3>";
  const table = document.createElement("table");
  table.className = "data";
  table.style.marginTop = "8px";
  table.innerHTML = `<thead><tr><th>Ticker</th><th>Target</th><th>Drifted to</th><th>Contribution</th></tr></thead>`;
  const body = document.createElement("tbody");
  selected.contribution.forEach((row) => {
    const asset = result.assets.find((a) => a.ticker === row.ticker);
    const tr = document.createElement("tr");
    tr.style.cursor = "default";
    tr.innerHTML = `<td class="label" title="${escapeHtml(asset?.name || "")}">${row.ticker}</td>
      <td>${charts.percent(selected.weights[row.ticker] || 0, 1)}</td>
      <td>${charts.percent(row.final_weight, 1)}</td>
      <td class="${row.contribution >= 0 ? "gain" : "loss"}">${charts.percent(row.contribution, 1)}</td>`;
    body.appendChild(tr);
  });
  table.appendChild(body);
  holdings.appendChild(table);
  grid.appendChild(holdings);

  const right = document.createElement("div");
  right.innerHTML = "<h3>Sector mix at the end of the window</h3>";
  const bars = document.createElement("div");
  bars.className = "bars";
  bars.style.marginTop = "10px";
  const top = Math.max(...selected.sectors.map((s) => s.weight), 0.01);
  selected.sectors.forEach((sector, index) => {
    const row = document.createElement("div");
    row.className = "row";
    row.innerHTML = `<span>${escapeHtml(sector.sector)}</span>
      <span class="track"><span class="fill" style="left:0;width:${((sector.weight / top) * 100).toFixed(1)}%;background:color-mix(in srgb, ${seriesColor(selected)} ${90 - index * 8}%, transparent)"></span></span>
      <span class="value">${charts.percent(sector.weight, 1)}</span>`;
    bars.appendChild(row);
  });
  right.appendChild(bars);

  if (result.correlation.tickers.length > 1) {
    right.insertAdjacentHTML("beforeend", '<h3 style="margin-top:20px">Daily return correlation</h3><p class="note" style="margin:4px 0 8px">Holdings that move together give less diversification than their count suggests.</p>');
    const matrix = document.createElement("div");
    matrix.style.overflowX = "auto";
    const tickers = result.correlation.tickers;
    let html = `<table class="matrix"><thead><tr><th></th>${tickers.map((t) => `<th>${t.replace(".TO", "")}</th>`).join("")}</tr></thead><tbody>`;
    result.correlation.matrix.forEach((row, i) => {
      html += `<tr><th class="side">${tickers[i].replace(".TO", "")}</th>`;
      row.forEach((value) => {
        const strength = Math.max(0, Math.min(1, (value + 0.2) / 1.2));
        html += `<td style="background:color-mix(in srgb, var(--accent) ${(strength * 45).toFixed(0)}%, transparent)">${value.toFixed(2)}</td>`;
      });
      html += "</tr>";
    });
    matrix.innerHTML = `${html}</tbody></table>`;
    right.appendChild(matrix);
  }
  grid.appendChild(right);
  node.appendChild(grid);
  return node;
}

function optimizationPlate(result) {
  const optimized = result.portfolios.filter((p) => p.optimization);
  if (!optimized.length && !result.frontier) return null;
  const focus = optimized.find((p) => p.name === state.selected) || optimized[0];
  const node = plate(
    "Optimization, out of sample",
    focus
      ? `Weights are re-solved at every rebalance from the previous ${(focus.optimization.estimation_days / 252).toFixed(focus.optimization.estimation_days % 252 ? 1 : 0)} year(s) of returns and then held forward. Nothing here was fitted to the returns it earned.`
      : "The frontier below was fitted to the whole window, so every point on it is a decision made with hindsight.",
  );

  if (focus) {
    const o = focus.optimization;
    const callouts = document.createElement("div");
    callouts.className = "callouts";
    [
      ["Shrinkage applied", charts.percent(o.shrinkage_intensity, 1),
        "toward a scaled identity, chosen by Ledoit-Wolf"],
      ["Effective bets", charts.ratio(o.effective_bets, 1),
        `across ${o.holdings.length} holdings`],
      ["Diversification ratio", charts.ratio(o.diversification_ratio, 2),
        "holdings' own risk versus the portfolio's"],
      ["Estimated volatility", charts.percent(o.expected_volatility),
        `realized was ${charts.percent(focus.summary.volatility)}`],
      ["Turnover a year", charts.ratio(o.annual_turnover, 2),
        `${o.reoptimizations} re-optimizations, costing ${charts.money(focus.summary.total_cost)}`],
      ["Cap per holding", o.max_weight >= 1 ? "none" : charts.percent(o.max_weight, 0),
        `covariance: ${o.estimator === "sample" ? "sample" : "Ledoit-Wolf"}`],
    ].forEach(([k, v, sub]) => {
      const cell = document.createElement("div");
      cell.className = "callout";
      cell.innerHTML = `<span class="k">${k}</span><span class="v">${v}</span><span class="sub">${sub}</span>`;
      callouts.appendChild(cell);
    });
    node.appendChild(callouts);

    const gap = document.createElement("p");
    gap.className = "note";
    gap.style.margin = "10px 0 20px";
    const ratio = focus.summary.volatility / Math.max(o.expected_volatility, 1e-9);
    gap.textContent = `${focus.name}: the risk model expected ${charts.percent(o.expected_volatility)} volatility `
      + `and the portfolio realized ${charts.percent(focus.summary.volatility)}, `
      + `${ratio > 1.15 ? `${charts.ratio(ratio, 1)}x the estimate — estimated covariance understates risk when correlations rise in stress, which is exactly when it matters` : "close to the estimate"}.`;
    node.appendChild(gap);
  }

  if (focus && (focus.optimization.risk_model || focus.optimization.solver)) {
    const model = focus.optimization.risk_model;
    const solver = focus.optimization.solver;
    const detail = document.createElement("div");
    detail.style.marginBottom = "20px";
    const rows = [];
    if (model) {
      const kind = { statistical_factor: "Statistical factor model", ledoit_wolf: "Ledoit-Wolf shrinkage", sample: "Sample covariance" }[model.kind] || model.kind;
      rows.push(["Risk model", model.factors
        ? `${kind}, ${model.factors} factor${model.factors > 1 ? "s" : ""} explaining ${charts.percent(model.explained_variance, 0)} of the correlation structure`
        : `${kind}, shrinkage ${charts.percent(model.shrinkage_intensity, 1)}`]);
      rows.push(["Condition number", `${Math.round(model.condition_number).toLocaleString()} — how close the covariance is to singular, and how far an optimizer can be misled by its calmest-looking direction`]);
      if (model.attribution && model.attribution.factor_share > 0) {
        rows.push(["Variance split", `${charts.percent(model.attribution.factor_share, 0)} common factors, ${charts.percent(1 - model.attribution.factor_share, 0)} holding-specific`]);
      }
    }
    if (solver) {
      rows.push(["Solver", `${solver.name}, ${solver.status} in ${(solver.seconds * 1000).toFixed(0)} ms`
        + (solver.reformulation ? ` · ${solver.reformulation}` : "")]);
      const prices = Object.entries(solver.shadow_prices || {}).filter(([k]) => k !== "budget");
      rows.push(["Binding constraints", solver.binding_constraints.length
        ? solver.binding_constraints.join(", ") + (prices.length
          ? ` — shadow prices ${prices.map(([k, v]) => `${k} ${v.toFixed(5)}`).join(", ")}`
          : "")
        : "none; the optimum is interior, so every limit has room"]);
      if (solver.note) rows.push(["Note", solver.note]);
    }
    detail.innerHTML = '<h3 style="margin-bottom:8px">How the answer was reached</h3>'
      + '<table class="data"><tbody>' + rows.map(([k, v]) =>
        `<tr style="cursor:default"><td class="label" style="white-space:nowrap">${escapeHtml(k)}</td>`
        + `<td class="label faint" style="text-align:left">${escapeHtml(String(v))}</td></tr>`).join("")
      + "</tbody></table>";
    node.appendChild(detail);
  }

  const grid = document.createElement("div");
  grid.className = "grid2";

  if (result.frontier) {
    const left = document.createElement("div");
    left.innerHTML = '<h3>Risk and return, promised and delivered</h3>'
      + '<p class="note" style="margin:4px 0 8px">The dashed curve is the frontier fitted to this whole window, so it is what a perfect forecast would have allowed. The filled dots are what each portfolio actually realized. The distance between them is the cost of not knowing the future.</p>';
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "chart");
    left.appendChild(svg);
    state.renderers.push(charts.frontier(svg, {
      curve: { volatilities: result.frontier.volatilities, returns: result.frontier.returns },
      assets: result.frontier.assets,
      realized: result.frontier.realized.map((point) => ({
        ...point,
        color: seriesColor(result.portfolios.find((p) => p.name === point.name) || {}),
      })),
    }));
    grid.appendChild(left);
  }

  if (focus) {
    const right = document.createElement("div");
    const stability = focus.optimization.stability;
    right.innerHTML = '<h3>Weight, risk, and how much of either is the sample</h3>'
      + '<p class="note" style="margin:4px 0 10px">A holding\'s share of the money, its share of portfolio volatility, and the range its weight covers when the estimation window is resampled. At a minimum-variance solution the first two line up, because every holding\'s marginal risk is equal there.</p>';
    const bars = document.createElement("div");
    bars.className = "bars";
    const holdings = [...focus.optimization.holdings].sort((a, b) => b.weight - a.weight);
    const top = Math.max(...holdings.flatMap((h) => [h.weight, h.risk_share, h.weight_p95 || 0]), 0.01);
    const pct = (v) => ((Math.max(v, 0) / top) * 100).toFixed(1);
    holdings.forEach((holding) => {
      const row = document.createElement("div");
      row.className = "row";
      const range = holding.weight_p05 !== null && holding.weight_p95 !== null
        ? `<span style="display:block;height:3px;margin:2px 0 0;margin-left:${pct(holding.weight_p05)}%;width:${(Number(pct(holding.weight_p95)) - Number(pct(holding.weight_p05))).toFixed(1)}%;background:${seriesColor(focus)};opacity:.9"></span>`
        : "";
      row.innerHTML = `<span>${holding.ticker}</span>
        <span class="track" style="height:auto;padding:2px 0">
          <span style="display:block;height:6px;margin:1px 0;width:${pct(holding.weight)}%;background:${seriesColor(focus)}"></span>
          <span style="display:block;height:6px;margin:1px 0;width:${pct(holding.risk_share)}%;background:${seriesColor(focus)};opacity:.45"></span>
          ${range}
        </span>
        <span class="value">${charts.percent(holding.weight, 0)} / ${charts.percent(holding.risk_share, 0)}</span>`;
      bars.appendChild(row);
    });
    right.appendChild(bars);
    const key = document.createElement("p");
    key.className = "note";
    key.style.marginTop = "8px";
    key.innerHTML = "Solid bar: share of money. Faded bar: share of risk. Thin line: 5th to 95th percentile of the weight across resampled windows."
      + (stability ? ` Across ${stability.draws} resamples the weights moved by ${charts.percent(stability.mean_absolute_move, 1)} on average, and the widest holding spanned ${charts.percent(stability.widest_range, 0)} — that span is estimation error, not a view.` : "");
    right.appendChild(key);
    grid.appendChild(right);
  }
  node.appendChild(grid);

  if (focus && backend.conformance) {
    const panel = document.createElement("details");
    panel.className = "advanced";
    panel.style.marginTop = "20px";
    panel.innerHTML = `<summary>Conformance report for ${escapeHtml(focus.optimization.label || focus.optimization.objective)}</summary>`
      + '<p class="note" style="margin:8px 0">Every registered methodology runs the same battery before it ships: validity of the weights it returns, the constraints it honours, whether two identical calls agree, whether it depends on the order the assets arrive in, how it behaves on degenerate data, and whether it solves inside a walk-forward time budget.</p>'
      + '<div class="conformance">Loading…</div>';
    let loaded = false;
    panel.addEventListener("toggle", async () => {
      if (!panel.open || loaded) return;
      loaded = true;
      const host = panel.querySelector(".conformance");
      try {
        const report = await backend.conformance(focus.optimization.objective);
        host.innerHTML = `<table class="data"><thead><tr><th>Check</th><th>Result</th><th>Measured</th></tr></thead><tbody>`
          + report.checks.map((check) => `<tr style="cursor:default"><td class="label">${escapeHtml(check.name)}${check.required ? "" : ' <span class="faint">advisory</span>'}</td>`
            + `<td class="${check.passed ? "gain" : "loss"}">${check.passed ? "pass" : "fail"}</td>`
            + `<td class="label faint" style="text-align:left">${escapeHtml(check.detail)}</td></tr>`).join("")
          + `</tbody></table>`;
      } catch (error) {
        host.innerHTML = `<p class="note">The harness could not be reached: ${escapeHtml(error.message)}</p>`;
      }
    });
    node.appendChild(panel);
  }

  if (focus && focus.optimization.weight_history.dates.length > 2) {
    const history = document.createElement("div");
    history.style.marginTop = "22px";
    history.innerHTML = `<h3>How much ${escapeHtml(focus.name)} moved</h3>`
      + '<p class="note" style="margin:4px 0 10px">Every re-estimation shifts the weights. A rule that jumps around each month pays for it in trading costs, and it is a sign the estimates are noisier than the differences they are acting on.</p>';
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("class", "chart");
    history.appendChild(svg);
    const tickers = focus.optimization.weight_history.tickers;
    const rows = focus.optimization.weight_history.weights;
    state.renderers.push(charts.stacked(svg, {
      dates: focus.optimization.weight_history.dates,
      series: tickers.map((ticker, i) => ({
        name: ticker,
        color: `var(${SERIES[i % SERIES.length]})`,
        values: rows.map((row) => row[i]),
      })),
    }));
    node.appendChild(history);
  }
  return node;
}

function assumptionsPlate(result) {
  const settings = result.meta.settings;
  const quality = result.quality;
  const node = plate("Assumptions behind these numbers", "Everything the engine applied, so a reviewer can argue with it.");
  const wrap = document.createElement("div");
  wrap.className = "assumptions";
  wrap.innerHTML = `
    <dl>
      <dt>Prices</dt><dd>${escapeHtml(result.meta.source)}. Split and dividend adjusted closes, which approximate a reinvested total return.</dd>
      <dt>Window</dt><dd>${result.meta.window_start} to ${result.meta.window_end}: ${quality.shared_trading_days.toLocaleString()} sessions every holding traded${quality.days_dropped_for_missing_prices ? `, ${quality.days_dropped_for_missing_prices} dropped for a missing price` : ""}.${result.meta.calibration_days ? ` A further ${result.meta.calibration_days} sessions were used only to calibrate weights.` : ""}</dd>
      <dt>Return convention</dt><dd>Previous close weights times current returns, so no holding is set using the return it earns.</dd>
      <dt>Rebalancing</dt><dd>${settings.rebalance === "none" ? "None: weights drift for the whole window." : `${settings.rebalance[0].toUpperCase()}${settings.rebalance.slice(1)}, on the first session of each new period.`} Cost of ${settings.transaction_cost_bps} bps on the traded notional, both sides.</dd>
      <dt>Sharpe</dt><dd>Risk-free rate of ${charts.percent(settings.risk_free_rate)} a year, converted to a daily equivalent. Constant across the window.</dd>
      <dt>Sampling error</dt><dd>Live-server results include 95% intervals for the estimated percentiles, using binomial order statistics. These measure finite-simulation error under the chosen model, not uncertainty about future markets or model assumptions. More paths generally improve precision; no fixed percentage applies to every portfolio. The offline demo does not calculate these intervals.</dd>
      <dt>Scenario engine</dt><dd>Block bootstrap: ${settings.paths.toLocaleString()} paths of ${settings.block_days}-session blocks resampled from this window's completed net returns, seed ${settings.seed}. Every portfolio is given the identical block positions.</dd>
      <dt>Starting balance for scenarios</dt><dd>${settings.scenario_basis === "equal" ? `The same ${charts.money(settings.initial_capital)} for every portfolio, so the comparison is about construction.` : "Each portfolio continues from its own final value, which answers a wealth-continuation question instead."}</dd>
      ${result.portfolios.some((p) => p.optimization) ? `
      <dt>Optimizer inputs</dt><dd>Covariance is estimated with Ledoit-Wolf shrinkage toward a scaled identity, which keeps the matrix well conditioned; an optimizer will otherwise load into whichever direction the sample happens to call calm. Expected returns, where an objective needs them, are shrunk toward the cross-sectional mean.</dd>
      <dt>Out of sample</dt><dd>Every optimized weight was solved from returns before the day it was applied. The efficient frontier shown is the exception and is fitted to the whole window; it is drawn as the hindsight benchmark it is.</dd>
      <dt>What optimization cannot fix</dt><dd>Estimation error. With a handful of assets and a few years of daily data, differences between candidate portfolios are often smaller than the error in the inputs that produced them, which is why equal weight is a serious competitor rather than a naive baseline.</dd>` : ""}
      <dt>Not modelled</dt><dd>Taxes, FX, inflation, dividends paid in cash, delistings, market impact beyond the fixed spread, and any change in the companies themselves.</dd>
      <dt>Survivorship</dt><dd>The tickers were chosen today, knowing which survived. That flatters any backtest and no correction is applied.</dd>
      <dt>What the fan is not</dt><dd>It repeats this window's return distribution, including its luck. It is not an estimate of future returns and the median at each date is not one investable path.</dd>
    </dl>`;
  node.appendChild(wrap);

  const record = result.manifest;
  if (record) {
    const panel = document.createElement("details");
    panel.className = "advanced";
    panel.style.marginTop = "16px";
    panel.innerHTML = `<summary>Provenance: run ${escapeHtml(record.run_id)}</summary>`
      + '<p class="note" style="margin:8px 0">The run identifier is derived from the request, the price data, and the code that ran, so the same study on the same data always carries the same identifier. A different identifier means something changed, and these digests say what.</p>'
      + '<table class="data"><tbody>'
      + [["Run", record.run_id],
         ["Request digest", record.request_sha256 ? record.request_sha256.slice(0, 24) : "unavailable"],
         ["Price data", `${record.data.rows.toLocaleString()} rows, ${record.data.tickers.length} tickers, ${record.data.first_date} to ${record.data.last_date}`],
         ["Data digest", record.data.sha256 ? record.data.sha256.slice(0, 24) : "server only"],
         ["Source", record.data.source],
         ["Engine", record.code_sha256
           ? `${record.engine_version}, code ${record.code_sha256.slice(0, 16)}`
           : record.engine_version],
         ["Python", record.environment.python
           ? `${record.environment.python} · ` + Object.entries(record.environment.packages)
             .filter(([, v]) => v).map(([k, v]) => `${k} ${v}`).join(", ")
           : "server only"],
         ["Generated", record.generated_at]]
        .map(([k, v]) => `<tr style="cursor:default"><td class="label">${k}</td><td class="label faint" style="text-align:left">${escapeHtml(String(v))}</td></tr>`).join("")
      + "</tbody></table>"
      + (record.partial ? `<p class="note" style="margin-top:8px">${escapeHtml(record.partial)}</p>` : "");
    node.appendChild(panel);
  }

  const footer = document.createElement("p");
  footer.className = "footer";
  footer.style.marginTop = "18px";
  footer.innerHTML = `Run ${escapeHtml(result.meta.run_id || "—")}, generated ${escapeHtml(result.meta.generated_at)} by engine ${escapeHtml(result.meta.engine_version)}. Research tooling for studying historical data and stated assumptions — not investment advice, and not a recommendation to buy or sell anything. Past results do not establish future suitability.`;
  node.appendChild(footer);
  return node;
}

/* ----------------------------- boot ----------------------------- */

async function boot() {
  await backend.ready();
  state.settings.source = backend.defaultSource;
  if (backend.bundledUniverse && backend.defaultSource === "bundled") {
    backend.universe = backend.bundledUniverse;
  }
  const preset = PRESETS.find((p) => backend.sources.includes(p.source)) || PRESETS[0];
  const end = new Date();
  const start = new Date(end);
  start.setFullYear(start.getFullYear() - 10);
  state.settings.end = backend.defaultEnd || end.toISOString().slice(0, 10);
  state.settings.start = backend.defaultStart || start.toISOString().slice(0, 10);
  const shared = location.hash.match(/setup=([A-Za-z0-9+/=_-]+)/);
  let restored = false;
  if (shared) {
    try {
      decodeSetup(shared[1]);
      restored = true;
    } catch (error) {
      console.warn("Ignoring an unreadable setup link:", error.message);
    }
  }
  if (!restored) loadPreset(preset);
  bindRail();
  writeRail();
  showMode();
  if (backend.autorun || restored) run();
}

boot();
})(window.PL = window.PL || {});
