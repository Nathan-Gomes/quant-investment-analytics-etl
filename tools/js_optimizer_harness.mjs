/* Runs the browser optimizer outside a browser so the Python suite can compare
   it with cvxpy. Usage: node tools/js_optimizer_harness.mjs <input.json> */
import { readFileSync } from "node:fs";
import vm from "node:vm";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const sandbox = { window: {}, console };
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(readFileSync(path.join(here, "..", "app", "static", "optimize.js"), "utf8"), sandbox);

const request = JSON.parse(readFileSync(process.argv[2], "utf8"));
const PL = sandbox.window.PL;
const out = PL.optimize.solve(request.objective, request.returns, request.constraints, request.estimator);
process.stdout.write(JSON.stringify({
  weights: out.weights,
  shrinkage_intensity: out.shrinkage_intensity,
  risk_share: out.risk.share,
  volatility: out.risk.volatility,
}));
