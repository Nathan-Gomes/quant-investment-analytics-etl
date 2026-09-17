/* Portfolio Lab interface.
   State lives in one object, the rail writes to it, and every render reads from
   it. The analysis itself comes from `backend`, which is the FastAPI service in
   the served app and an in-page engine in the offline demo build. */

(function (PL) {
"use strict";
const charts = PL.charts;
const backend = PL.backend;

const SERIES = ["--s1", "--s2", "--s3", "--s5", "--s4", "--s6"];
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
const colorOf = (index, benchmark) => `var(${benchmark ? BENCHMARK_COLOR : SERIES[index % SERIES.length]})`;
const uid = () => Math.random().toString(36).slice(2, 8);

/* ----------------------------- rail ----------------------------- */

function loadPreset(preset) {
  state.portfolios = preset.portfolios.map((p) => ({
    id: uid(), name: p.name, scheme: p.scheme,
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
    add.textContent = "+ Compare another";
    add.onclick = () => {
      const source = currentPortfolio();
      state.portfolios.push({
        id: uid(),
        name: `Portfolio ${String.fromCharCode(65 + state.portfolios.length)}`,
        scheme: "equal",
        holdings: source.holdings.map((h) => ({ ...h })),
      });
      state.active = state.portfolios.length - 1;
      writeRail();
    };
    tabs.appendChild(add);
  }
  $("portfolio-count").textContent = state.portfolios.length > 1 ? `${state.portfolios.length} compared` : "";
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
      <input class="weight" type="number" step="0.5" min="0" value="${holding.weight}" aria-label="Weight ${index + 1}" ${derived ? "disabled" : ""}>
      <button class="remove" type="button" aria-label="Remove ${holding.ticker || "holding"}">&times;</button>`;
    const [ticker, weight, remove] = row.children;
    ticker.oninput = () => {
      holding.ticker = ticker.value.trim().toUpperCase();
      checkTicker(ticker, holding.ticker);
      drawUniverseNote();
    };
    ticker.onchange = () => { ticker.value = holding.ticker; drawTabs(); };
    checkTicker(ticker, holding.ticker);
    weight.oninput = () => { holding.weight = Number(weight.value) || 0; drawWeightBar(); };
    remove.onclick = () => {
      portfolio.holdings.splice(index, 1);
      if (!portfolio.holdings.length) portfolio.holdings.push({ ticker: "", weight: 0 });
      writeRail();
    };
    host.appendChild(row);
  });
  drawWeightBar();
  drawUniverseNote();
}

function drawWeightBar() {
  const portfolio = currentPortfolio();
  const bar = $("weightbar");
  bar.innerHTML = "";
  const derived = portfolio.scheme !== "custom";
  const entries = portfolio.holdings.filter((h) => h.ticker);
  const weights = derived ? entries.map(() => 1) : entries.map((h) => Math.max(h.weight, 0));
  const total = weights.reduce((a, b) => a + b, 0);
  weights.forEach((weight, index) => {
    const span = document.createElement("span");
    span.style.width = `${total ? (weight / total) * 100 : 0}%`;
    span.style.background = `color-mix(in srgb, ${colorOf(state.active)} ${95 - index * 9}%, transparent)`;
    bar.appendChild(span);
  });
  const raw = portfolio.holdings.reduce((a, h) => a + (h.ticker ? Math.max(h.weight, 0) : 0), 0);
  $("weight-total").textContent = derived
    ? `${entries.length} holdings, weights computed`
    : `Entered: ${raw.toFixed(1)}%${Math.abs(raw - 100) > 0.05 ? " — rescaled to 100%" : ""}`;
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
  PRESETS.filter((p) => backend.sources.includes(p.source) || p.source === "auto").forEach((preset) => {
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
  $("export-json").onclick = () => {
    if (!state.result) return;
    const blob = new Blob([JSON.stringify(state.result, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "portfolio-lab-results.json";
    link.click();
    URL.revokeObjectURL(link.href);
  };
  if (!backend.allowsSourceChoice) $("source-field").style.display = "none";
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
    return { name: portfolio.name || "Portfolio", weights, scheme: portfolio.scheme };
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

async function run() {
  const button = $("run");
  const status = $("run-status");
  button.disabled = true;
  status.innerHTML = '<span class="spinner"></span> Running backtest and scenarios…';
  try {
    const request = buildRequest();
    const started = performance.now();
    const result = await backend.analyze(request);
    state.result = result;
    state.selected = result.portfolios.find((p) => !p.is_benchmark)?.name || result.portfolios[0].name;
    status.textContent = `Done in ${((performance.now() - started) / 1000).toFixed(1)}s · ${result.meta.trading_days.toLocaleString()} shared sessions · ${result.meta.settings.paths.toLocaleString()} simulated markets`;
    showMode(result.meta.source);
    renderResults();
    if (!first) revealResults();
    first = false;
  } catch (error) {
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
  const cells = [
    ["Median outcome", charts.money(scenario.terminal_median), `from ${charts.money(scenario.start_value)}`],
    ["Weak case (5th pct)", charts.money(scenario.terminal_p05), `mean of the worst 5%: ${charts.money(scenario.worst5_mean)}`],
    ["Strong case (95th pct)", charts.money(scenario.terminal_p95), `25th–75th: ${charts.money(scenario.terminal_p25)} to ${charts.money(scenario.terminal_p75)}`],
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
      <dt>Scenario engine</dt><dd>Block bootstrap: ${settings.paths.toLocaleString()} paths of ${settings.block_days}-session blocks resampled from this window's completed net returns, seed ${settings.seed}. Every portfolio is given the identical block positions.</dd>
      <dt>Starting balance for scenarios</dt><dd>${settings.scenario_basis === "equal" ? `The same ${charts.money(settings.initial_capital)} for every portfolio, so the comparison is about construction.` : "Each portfolio continues from its own final value, which answers a wealth-continuation question instead."}</dd>
      <dt>Not modelled</dt><dd>Taxes, FX, inflation, dividends paid in cash, delistings, market impact beyond the fixed spread, and any change in the companies themselves.</dd>
      <dt>Survivorship</dt><dd>The tickers were chosen today, knowing which survived. That flatters any backtest and no correction is applied.</dd>
      <dt>What the fan is not</dt><dd>It repeats this window's return distribution, including its luck. It is not an estimate of future returns and the median at each date is not one investable path.</dd>
    </dl>`;
  node.appendChild(wrap);

  const footer = document.createElement("p");
  footer.className = "footer";
  footer.style.marginTop = "18px";
  footer.innerHTML = `Generated ${escapeHtml(result.meta.generated_at)} by engine ${escapeHtml(result.meta.engine_version)}. Research tooling for studying historical data and stated assumptions — not investment advice, and not a recommendation to buy or sell anything. Past results do not establish future suitability.`;
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
  loadPreset(preset);
  bindRail();
  writeRail();
  showMode();
  if (backend.autorun) run();
}

boot();
})(window.PL = window.PL || {});
