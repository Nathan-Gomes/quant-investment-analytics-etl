/* Server backend: the analysis runs in Python, this only carries the request. */
(function (PL) {
  "use strict";

  // Serve the interface from anywhere and point it at the API with
  // <script>window.STRATA_API = "https://your-api.example.com/";</script>
  const base = (window.STRATA_API || window.PORTFOLIO_LAB_API || "").replace(/\/?$/, "/");
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
        // Connection failures must not silently replace Yahoo with frozen prices.
        backend.defaultSource = "auto";
      }
    },

    async analyze(request, onProgress) {
      // The streaming endpoint reports each stage as it finishes. If anything
      // about it fails — an old server, a proxy that buffers the body — fall
      // back to the plain endpoint rather than leaving the person with nothing.
      if (onProgress && globalThis.ReadableStream) {
        try {
          return await streamAnalysis(request, onProgress);
        } catch (error) {
          if (error.fromServer) throw error;
        }
      }
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

  async function streamAnalysis(request, onProgress) {
    const response = await fetch(endpoint("api/analyze/stream"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const failure = new Error(`The server returned ${response.status}.`);
      failure.fromServer = response.status >= 400 && response.status < 500
        && response.status !== 404 && response.status !== 405;
      throw failure;
    }
    if (!response.body) throw new Error("The server did not return a stream.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let result = null;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.phase === "error") {
          const failure = new Error(event.message);
          failure.fromServer = true;   // a real rejection, not a transport problem
          throw failure;
        }
        if (event.phase === "done") result = event.payload;
        else onProgress(event);
      }
    }
    if (!result) throw new Error("The server closed the connection before finishing.");
    return result;
  }

  PL.backend = backend;
})(window.PL = window.PL || {});
