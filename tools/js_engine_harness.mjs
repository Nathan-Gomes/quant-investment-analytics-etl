/* Runs the browser engine outside a browser so the Python test suite can
   compare the two implementations number by number.

   Usage: node tools/js_engine_harness.mjs <dataset.js> <request.json>
   Prints the analysis payload as JSON on stdout. */

import { readFileSync } from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const staticDir = path.join(here, "..", "app", "static");
const [datasetPath, requestPath] = process.argv.slice(2);

const sandbox = { window: {}, console, setTimeout, performance };
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

for (const source of [datasetPath, path.join(staticDir, "engine.js"), path.join(staticDir, "backend-local.js")]) {
  vm.runInContext(readFileSync(source, "utf8"), sandbox, { filename: source });
}

const request = JSON.parse(readFileSync(requestPath, "utf8"));
const PL = sandbox.window.PL;
await PL.backend.ready();
const payload = await PL.backend.analyze(request);
process.stdout.write(JSON.stringify(payload));
