/* Server backend: the analysis runs in Python, this only carries the request. */
(function (PL) {
  "use strict";

  // Serve the interface from anywhere and point it at the API with
  // <script>window.PORTFOLIO_LAB_API = "https://your-api.example.com/";</script>
  const base = (window.PORTFOLIO_LAB_API || "").replace(/\/?$/, "/");
  const endpoint = (path) => (base ? base + path : path);

  const backend = {
    sources: ["auto", "bundled"],
    defaultSource: "auto",
    allowsSourceChoice: true,
    autorun: false,
    defaultStart: null,
    defaultEnd: null,
    supportsOptimization: true,
    strategies: null,
    async conformance(name) {
      const response = await fetch(endpoint(`api/conformance?strategy=${encodeURIComponent(name)}`));
      if (!response.ok) throw new Error(`The conformance harness returned ${response.status}.`);
      return (await response.json()).reports[0];
    },
    universe: null,        // Yahoo accepts any symbol it knows, so nothing to check against
    bundledUniverse: null,  // the frozen dataset is a closed list, so it can be checked

    async ready() {
      try {
        const response = await fetch(endpoint("api/health"));
        if (!response.ok) throw new Error(String(response.status));
        const health = await response.json();
        backend.defaultSource = health.default_source === "bundled" ? "bundled" : "auto";
        try {
          const listed = await (await fetch(endpoint("api/universe"))).json();
          backend.bundledUniverse = listed.bundled.map((row) => row.ticker).sort();
        } catch (error) {
          backend.bundledUniverse = null;
        }
        try {
          // The menu is built from the registry, so a methodology registered in
          // Python appears here with nothing in the interface to edit.
          const listed = await (await fetch(endpoint("api/strategies"))).json();
          backend.strategies = listed.strategies;
        } catch (error) {
          backend.strategies = null;
        }
      } catch (error) {
        // A connection failure must not silently replace Yahoo with frozen prices.
        backend.defaultSource = "auto";
      }
    },

    async analyze(request) {
      const response = await fetch(endpoint("api/analyze"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || `The server returned ${response.status}.`);
      }
      return payload;
    },
  };

  PL.backend = backend;
})(window.PL = window.PL || {});
