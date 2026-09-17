/* eslint-env browser */
(function (PL) {
"use strict";
/* Small SVG chart set built for this app: a continuous history-to-scenario
   timeline, a drawdown panel, and a terminal-value distribution. No chart
   library, so the page works offline and every mark is themeable with CSS. */

const SVG_NS = "http://www.w3.org/2000/svg";

function el(name, attrs = {}, parent) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    if (value !== null && value !== undefined) node.setAttribute(key, value);
  }
  if (parent) parent.appendChild(node);
  return node;
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

function money(value, digits = 0) {
  if (value === null || !isFinite(value)) return "—";
  const sign = value < 0 ? "-" : "";
  const abs = Math.abs(value);
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(2)}M`;
  if (abs >= 10000) return `${sign}$${Math.round(abs / 1000)}k`;
  return `${sign}$${abs.toLocaleString(undefined, { maximumFractionDigits: digits })}`;
}

function percent(value, digits = 1) {
  if (value === null || value === undefined || !isFinite(value)) return "—";
  const scaled = Math.abs(value) < 5e-7 ? 0 : value * 100;  // keep "0%" from printing as "-0%"
  return `${scaled.toFixed(digits)}%`;
}

function ratio(value, digits = 2) {
  if (value === null || value === undefined || !isFinite(value)) return "—";
  return value.toFixed(digits);
}

function niceTicks(min, max, count = 5) {
  if (!isFinite(min) || !isFinite(max) || min === max) return [min];
  const span = max - min;
  const step = Math.pow(10, Math.floor(Math.log10(span / count)));
  const candidates = [1, 2, 2.5, 5, 10].map((m) => m * step);
  const chosen = candidates.find((c) => span / c <= count * 1.4) || candidates[candidates.length - 1];
  const ticks = [];
  for (let v = Math.ceil(min / chosen) * chosen; v <= max + chosen * 0.001; v += chosen) ticks.push(v);
  return ticks;
}

function logTicks(min, max) {
  const ticks = [];
  const lo = Math.floor(Math.log10(min));
  const hi = Math.ceil(Math.log10(max));
  for (let e = lo; e <= hi; e += 1) {
    for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, e);
      if (v >= min && v <= max) ticks.push(v);
    }
  }
  return ticks.length > 2 ? ticks : niceTicks(min, max, 4);
}

function scale(domain, range, log) {
  const [d0, d1] = log ? [Math.log(domain[0]), Math.log(domain[1])] : domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (value) => {
    const v = log ? Math.log(Math.max(value, 1e-9)) : value;
    return r0 + ((v - d0) / span) * (r1 - r0);
  };
}

function path(points) {
  let d = "";
  for (let i = 0; i < points.length; i += 1) {
    const [x, y] = points[i];
    if (!isFinite(x) || !isFinite(y)) continue;
    d += `${d ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`;
  }
  return d;
}

function measure(node, fallback = 760) {
  const width = node.getBoundingClientRect().width;
  return width > 80 ? width : fallback;
}

function yearTicks(stamps) {
  const seen = new Set();
  const ticks = [];
  stamps.forEach((stamp, i) => {
    const year = new Date(stamp).getUTCFullYear();
    if (!seen.has(year)) { seen.add(year); ticks.push({ i, stamp, label: String(year) }); }
  });
  return ticks.length > 12 ? ticks.filter((_, i) => i % 2 === 0) : ticks;
}

const stampOf = (iso) => Date.parse(`${iso}T00:00:00Z`);

/* ------------------------------------------------------------------ */
/* Timeline: realized path on the left, scenario fan on the right      */
/* ------------------------------------------------------------------ */

function timeline(node, spec) {
  const render = () => {
    clear(node);
    const width = measure(node);
    const height = spec.height || 372;
    node.setAttribute("viewBox", `0 0 ${width} ${height}`);
    node.setAttribute("height", height);
    const pad = { top: 14, right: 16, bottom: 24, left: 62 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;

    const histStamps = spec.history.dates.map(stampOf);
    const fwdStamps = (spec.forward ? spec.forward.dates : []).map(stampOf);
    const allStamps = histStamps.concat(fwdStamps);
    const x = scale([allStamps[0], allStamps[allStamps.length - 1]], [pad.left, pad.left + plotW], false);

    let lo = Infinity;
    let hi = -Infinity;
    const consider = (values) => values.forEach((v) => {
      if (v === null || !isFinite(v)) return;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    });
    spec.history.series.forEach((s) => consider(s.values));
    (spec.forward ? spec.forward.series : []).forEach((s) => {
      if (s.key === spec.selected) { consider(s.bands.p05); consider(s.bands.p95); }
      else consider(s.bands.median);
    });
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    const log = spec.log;
    if (log) { lo = Math.max(lo * 0.92, 1); hi *= 1.08; } else { const pad2 = (hi - lo) * 0.06; lo -= pad2; hi += pad2; if (lo < 0) lo = 0; }
    const y = scale([lo, hi], [pad.top + plotH, pad.top], log);

    const ticks = log ? logTicks(lo, hi) : niceTicks(lo, hi, 5);
    ticks.forEach((t) => {
      const yy = y(t);
      if (yy < pad.top - 1 || yy > pad.top + plotH + 1) return;
      el("line", { class: "gridline", x1: pad.left, x2: pad.left + plotW, y1: yy, y2: yy }, node);
      el("text", { x: pad.left - 8, y: yy + 3.5, "text-anchor": "end" }, node).textContent = money(t);
    });
    yearTicks(allStamps).forEach(({ stamp, label }) => {
      el("text", { x: x(stamp), y: height - 7, "text-anchor": "middle" }, node).textContent = label;
    });
    el("line", { class: "axis", x1: pad.left, x2: pad.left + plotW, y1: pad.top + plotH, y2: pad.top + plotH, stroke: "currentColor", "stroke-opacity": 0.35 }, node);

    // Scenario fan for the selected portfolio, medians for the rest.
    if (spec.forward) {
      const seam = x(histStamps[histStamps.length - 1]);
      el("rect", {
        x: seam, y: pad.top, width: pad.left + plotW - seam, height: plotH,
        fill: "currentColor", "fill-opacity": 0.022,
      }, node);
      spec.forward.series.forEach((s) => {
        const chosen = s.key === spec.selected;
        const pts = (values) => fwdStamps.map((stamp, i) => [x(stamp), y(values[i])]);
        if (chosen) {
          const area = (a, b) => path(pts(a)) + "L" + pts(b).reverse().map(([px, py]) => `${px.toFixed(1)} ${py.toFixed(1)}`).join("L") + "Z";
          el("path", { class: "band", d: area(s.bands.p95, s.bands.p05), fill: s.color, "fill-opacity": 0.13 }, node);
          el("path", { class: "band", d: area(s.bands.p75, s.bands.p25), fill: s.color, "fill-opacity": 0.22 }, node);
        }
        el("path", {
          class: `series${chosen ? "" : " dim"}${s.benchmark ? " bench" : ""}`,
          d: path(pts(s.bands.median)), stroke: s.color, "stroke-dasharray": chosen ? null : "4 3",
        }, node);
      });
      el("line", { class: "seam", x1: seam, x2: seam, y1: pad.top, y2: pad.top + plotH }, node);
      el("text", { class: "seam-label", x: seam + 5, y: pad.top + 10 }, node).textContent =
        spec.seamLabel || "scenarios start";
    }

    spec.history.series.forEach((s) => {
      const pts = histStamps.map((stamp, i) => [x(stamp), y(s.values[i])]);
      el("path", {
        class: `series${s.key === spec.selected || !spec.selected ? "" : " dim"}${s.benchmark ? " bench" : ""}`,
        d: path(pts), stroke: s.color,
      }, node);
    });

    // Hover readout
    const hair = el("line", { class: "crosshair", y1: pad.top, y2: pad.top + plotH, opacity: 0 }, node);
    const hit = el("rect", { x: pad.left, y: pad.top, width: plotW, height: plotH, fill: "transparent" }, node);
    const nearest = (stamps, target) => {
      let best = 0;
      let gap = Infinity;
      stamps.forEach((s, i) => { const d = Math.abs(s - target); if (d < gap) { gap = d; best = i; } });
      return best;
    };
    const move = (event) => {
      const box = node.getBoundingClientRect();
      const px = ((event.clientX - box.left) / box.width) * width;
      const stamp = x.invert ? x.invert(px) : allStamps[0] + ((px - pad.left) / plotW) * (allStamps[allStamps.length - 1] - allStamps[0]);
      hair.setAttribute("x1", px); hair.setAttribute("x2", px); hair.setAttribute("opacity", 0.5);
      const future = stamp > histStamps[histStamps.length - 1];
      if (future && spec.forward) {
        const i = nearest(fwdStamps, stamp);
        spec.onHover?.({ future: true, date: spec.forward.dates[i], index: i });
      } else {
        const i = nearest(histStamps, stamp);
        spec.onHover?.({ future: false, date: spec.history.dates[i], index: i });
      }
    };
    hit.addEventListener("mousemove", move);
    hit.addEventListener("mouseleave", () => { hair.setAttribute("opacity", 0); spec.onHover?.(null); });
  };
  render();
  return render;
}

/* ------------------------------------------------------------------ */

function drawdown(node, spec) {
  const render = () => {
    clear(node);
    const width = measure(node);
    const height = spec.height || 190;
    node.setAttribute("viewBox", `0 0 ${width} ${height}`);
    node.setAttribute("height", height);
    const pad = { top: 10, right: 16, bottom: 22, left: 62 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const stamps = spec.dates.map(stampOf);
    const x = scale([stamps[0], stamps[stamps.length - 1]], [pad.left, pad.left + plotW], false);
    let lo = 0;
    spec.series.forEach((s) => s.values.forEach((v) => { if (v < lo) lo = v; }));
    const y = scale([lo * 1.05, 0], [pad.top + plotH, pad.top], false);

    niceTicks(lo * 1.05, 0, 4).forEach((t) => {
      const yy = y(t);
      el("line", { class: t === 0 ? "zeroline" : "gridline", x1: pad.left, x2: pad.left + plotW, y1: yy, y2: yy }, node);
      el("text", { x: pad.left - 8, y: yy + 3.5, "text-anchor": "end" }, node).textContent = percent(t, 0);
    });
    yearTicks(stamps).forEach(({ stamp, label }) => {
      el("text", { x: x(stamp), y: height - 6, "text-anchor": "middle" }, node).textContent = label;
    });
    spec.series.forEach((s) => {
      const pts = stamps.map((stamp, i) => [x(stamp), y(s.values[i])]);
      const chosen = s.key === spec.selected || !spec.selected;
      if (chosen) {
        el("path", {
          d: path(pts) + `L${x(stamps[stamps.length - 1]).toFixed(1)} ${y(0).toFixed(1)}L${x(stamps[0]).toFixed(1)} ${y(0).toFixed(1)}Z`,
          fill: s.color, "fill-opacity": 0.12, stroke: "none",
        }, node);
      }
      el("path", { class: `series${chosen ? "" : " dim"}${s.benchmark ? " bench" : ""}`, d: path(pts), stroke: s.color }, node);
    });
  };
  render();
  return render;
}

/* ------------------------------------------------------------------ */

function distribution(node, spec) {
  const render = () => {
    clear(node);
    const width = measure(node);
    const height = spec.height || 210;
    node.setAttribute("viewBox", `0 0 ${width} ${height}`);
    node.setAttribute("height", height);
    const pad = { top: 12, right: 14, bottom: 22, left: 40 };
    const plotW = width - pad.left - pad.right;
    const plotH = height - pad.top - pad.bottom;
    const edges = spec.edges;
    const x = scale([edges[0], edges[edges.length - 1]], [pad.left, pad.left + plotW], false);
    let peak = 0;
    spec.series.forEach((s) => s.counts.forEach((c) => { if (c > peak) peak = c; }));
    const y = scale([0, peak * 1.06 || 1], [pad.top + plotH, pad.top], false);

    el("line", { class: "axis", x1: pad.left, x2: pad.left + plotW, y1: pad.top + plotH, y2: pad.top + plotH, stroke: "currentColor", "stroke-opacity": 0.35 }, node);
    niceTicks(edges[0], edges[edges.length - 1], 5).forEach((t) => {
      if (t < edges[0] || t > edges[edges.length - 1]) return;
      el("text", { x: x(t), y: height - 7, "text-anchor": "middle" }, node).textContent = money(t);
    });
    if (spec.start >= edges[0] && spec.start <= edges[edges.length - 1]) {
      el("line", { class: "seam", x1: x(spec.start), x2: x(spec.start), y1: pad.top, y2: pad.top + plotH }, node);
      el("text", { class: "seam-label", x: x(spec.start) + 5, y: pad.top + 9 }, node).textContent = "starting amount";
    }
    spec.series.forEach((s) => {
      const pts = [];
      s.counts.forEach((count, i) => { pts.push([x(edges[i]), y(count)], [x(edges[i + 1]), y(count)]); });
      const chosen = s.key === spec.selected;
      if (chosen) {
        el("path", {
          d: path(pts) + `L${x(edges[edges.length - 1]).toFixed(1)} ${y(0).toFixed(1)}L${x(edges[0]).toFixed(1)} ${y(0).toFixed(1)}Z`,
          fill: s.color, "fill-opacity": 0.18, stroke: "none",
        }, node);
      }
      el("path", { class: `series${chosen ? "" : " dim"}`, d: path(pts), stroke: s.color, "stroke-width": chosen ? 1.8 : 1.1 }, node);
    });
  };
  render();
  return render;
}

/* ------------------------------------------------------------------ */

function whiskers(node, spec) {
  const render = () => {
    clear(node);
    const width = measure(node);
    const rowH = 30;
    const height = spec.rows.length * rowH + 26;
    node.setAttribute("viewBox", `0 0 ${width} ${height}`);
    node.setAttribute("height", height);
    const pad = { left: 150, right: 74 };
    const plotW = width - pad.left - pad.right;
    let lo = Infinity;
    let hi = -Infinity;
    spec.rows.forEach((r) => { lo = Math.min(lo, r.p05, r.start); hi = Math.max(hi, r.p95); });
    lo = Math.min(lo, spec.start);
    const x = scale([lo, hi], [pad.left, pad.left + plotW], false);

    const shared = spec.rows.every((r) => Math.abs(r.start - spec.rows[0].start) < 1);
    if (shared && spec.start >= lo) {
      el("line", { class: "seam", x1: x(spec.start), x2: x(spec.start), y1: 2, y2: spec.rows.length * rowH + 4 }, node);
    }
    spec.rows.forEach((row, i) => {
      const cy = i * rowH + rowH / 2;
      el("text", { x: pad.left - 10, y: cy + 4, "text-anchor": "end", fill: "var(--ink)" }, node).textContent = row.name;
      el("line", { x1: x(row.p05), x2: x(row.p95), y1: cy, y2: cy, stroke: row.color, "stroke-width": 1.4, "stroke-opacity": 0.55 }, node);
      el("rect", { x: x(row.p25), y: cy - 6, width: Math.max(x(row.p75) - x(row.p25), 1), height: 12, fill: row.color, "fill-opacity": 0.3, stroke: row.color, "stroke-opacity": 0.6 }, node);
      el("circle", { cx: x(row.median), cy, r: 4, fill: row.color }, node);
      if (!shared) {
        el("line", { class: "seam", x1: x(row.start), x2: x(row.start), y1: cy - 9, y2: cy + 9 }, node);
      }
      el("text", { x: width - 6, y: cy + 4, "text-anchor": "end", fill: "var(--ink)" }, node).textContent = money(row.median);
    });
    niceTicks(lo, hi, 4).forEach((t) => {
      el("text", { x: x(t), y: height - 4, "text-anchor": "middle" }, node).textContent = money(t);
    });
  };
  render();
  return render;
}

function observeResize(renderers) {
  let frame = null;
  const redraw = () => { frame = null; renderers.forEach((fn) => fn && fn()); };
  const schedule = () => {
    if (frame) return;
    frame = typeof requestAnimationFrame === "function" ? requestAnimationFrame(redraw) : setTimeout(redraw, 16);
  };
  if (typeof ResizeObserver === "function") return new ResizeObserver(schedule);
  window.addEventListener("resize", schedule);
  return { observe() {}, disconnect() { window.removeEventListener("resize", schedule); } };
}

PL.charts = { money, percent, ratio, timeline, drawdown, distribution, whiskers, observeResize };
})(window.PL = window.PL || {});
