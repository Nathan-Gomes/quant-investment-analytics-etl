# Handoff brief: deploying Strata

Paste this whole file to whoever picks the work up. It describes what exists,
what must not change, and the three ways this can reach a public site.

## What this is

A research web app on top of an existing Python pipeline that compares equity
allocations. A person enters any set of tickers and weights, picks a window, and
gets a historical backtest plus a range of forward scenarios built by resampling
the portfolio's own completed returns. It is a study tool, not a trading system
and not advice.

The engine already works and is covered by 50 passing tests. Yahoo Finance is
already wired up and is the default price source. **Nothing needs to be enabled.**
`python -m app.server` fetches live prices today. The only reason the shareable
single-file demo uses a frozen dataset is that a static page cannot run Python.

## The one real constraint

The interface is static files and will run anywhere. The price fetching and the
analysis are Python and need a server that can run processes and make outbound
HTTPS requests. GitHub Pages, Netlify's static tier, S3 and similar cannot do
that. So the choice is about where the Python runs.

## Three options

**A. Frozen demo on the existing static site.** Copy
`dist/strata-demo.html` (built with `python tools/build_demo.py`) next to
the other project pages. One file, no infrastructure, no running costs, works
offline. It carries nine Canadian securities and cannot accept any other ticker.
Good as a case-study exhibit next to the existing report.

**B. Full app on its own host, linked from the portfolio.** Deploy this repo to
Render, Fly.io, Railway or any container host using the included `Dockerfile`.
The server serves both the API and the interface, so there is no CORS and no
cross-origin wiring. The portfolio page links to it. This is the simplest way to
get live tickers in front of someone.

**C. Interface on the static site, API on a small host.** Copy `app/static/*`
into the site and point it at the API:

```html
<script>window.STRATA_API = "https://portfolio-lab-api.example.com/";</script>
<script src="charts.js"></script>
<script src="backend.js"></script>
<script src="app.js"></script>
```

Then start the API with the site's origin allowed:

```sh
STRATA_ALLOWED_ORIGINS="https://www.nathan-gomes.com" uvicorn app.server:api
```

Both pieces are already built for this; `backend.js` reads
`window.STRATA_API` and the server adds CORS only when that environment
variable is set. Use this only if the interface has to live inside the existing
site's navigation. Otherwise B is less to go wrong.

**Recommendation: A and B together.** The frozen demo stays on the portfolio
site so a reader always sees something instantly, with a link to the live app.
If the live host sleeps or yfinance breaks, the exhibit still works.

## Deploying option B

```sh
docker build -t portfolio-lab .
docker run -p 8000:8000 portfolio-lab
```

On a platform, point it at the `Dockerfile` and set:

| Variable | Value | Why |
| --- | --- | --- |
| `STRATA_SOURCE` | `auto` | Yahoo Finance. Use `bundled` to force the frozen dataset. |
| `PORT` | platform-provided | The image already reads it. |
| `STRATA_ALLOWED_ORIGINS` | unset | Only needed for option C. |

Mount a persistent volume at `/srv/app/cache` if the platform offers one. That
directory holds each ticker's downloaded history plus a provenance record. It is
a cache, so the app works without it, but every cold start will re-fetch.

Resource shape: a request runs in about one second of CPU for 5,000 scenario
paths across four portfolios, and peak memory stays well under 512 MB. The
smallest paid tier anywhere is enough. Free tiers that sleep are acceptable
if the first request is allowed to take ten seconds.

## Things that will bite

- **yfinance from cloud IPs.** Yahoo rate-limits datacentre addresses far more
  aggressively than home connections, and periodically changes its endpoints,
  which breaks yfinance until it is updated. Plan for it: keep the cache volume,
  raise `CACHE_MAX_AGE_HOURS` in `app/marketdata.py` from 12 to 72 on a public
  deployment, and pin then routinely bump the yfinance version. If it becomes a
  habitual problem, move to a keyed provider such as Tiingo or Alpha Vantage by
  rewriting only `download()` in `app/marketdata.py` — nothing else touches the
  provider.
- **Abuse.** The analyse endpoint is unauthenticated and does real work. Before
  it is public, put a rate limit in front of it (the platform's own, or
  `slowapi`). The request caps already in place are 25 tickers, 6 portfolios and
  20,000 paths.
- **Terms of use.** Yahoo's terms govern this data. It is fine for a personal
  research demo; do not redistribute bulk price history or resell access.
- **Cold starts** on free tiers make the first run look broken. Either use a
  tier that stays warm or add a short "waking up" note in the interface.

## Rules for anyone changing the code

1. **Do not touch the return maths in `app/engine.py`** — previous-close
   weights, drift between rebalances, turnover-based costs — without running
   `pytest tests/`. `test_app_reproduces_the_published_pipeline_results` checks
   the app against the research pipeline's published figures to nine significant
   figures. If Growth's annualized return stops being 27.795%, something broke.
2. **Optimized weights must stay walk-forward.** Every weight is solved from a
   trailing window and applied forward. `test_the_optimizer_cannot_see_the_future`
   rewrites the last month of prices and asserts nothing earlier moves. Do not
   "improve" results by fitting weights on the full sample.
3. **The forward scenarios are a block bootstrap, not a forecast.** Do not
   replace them with an ML price predictor, and do not relabel percentile bands
   as predictions. The honesty of this project is the point of it.
4. **Never substitute simulated data for a failed download.** A failure is
   reported, naming the ticker. Silent fallbacks are how a demo becomes a lie.
5. **Keep the assumptions plate and the disclaimers.** Survivorship bias, no FX,
   no taxes, "not investment advice". They are in `app/static/app.js` and
   `docs/APP_GUIDE.md`.
6. **`app/rng.py` and `app/static/engine.js` must stay in step.** They implement
   the same generator so a seed means the same thing on the server and in the
   browser demo. `test_browser_engine_agrees_with_python` enforces it.
7. **A new construction rule goes through the registry**, never by editing the
   engine, and must pass `python -m app.conformance` before it ships. That
   command is in CI and exits non-zero on failure.
8. Run `pytest tests/` before shipping. All 92 should pass. If node is
   installed, that includes the cross-language check.

## Acceptance checklist

- [ ] `AAPL, MSFT, NVDA` at 40/30/30 over the last 10 years returns a result
- [ ] A nonsense symbol returns a clear message naming it, not a blank screen
- [ ] A second portfolio can be added and both appear in the same chart
- [ ] The scenario fan and the callouts change when a different portfolio is selected
- [ ] Export results downloads a JSON file containing every displayed number
- [ ] The page is usable on a phone: results scroll into view after Run
- [ ] Dark and light both readable
- [ ] A repeat run with the same seed gives identical scenario numbers

## File map

```
app/engine.py        backtest and block-bootstrap maths
app/optimize.py      covariance estimation, objectives, frontier, risk decomposition
app/strategies.py    the registry: the contract a methodology implements
app/conformance.py   the battery every methodology passes; runs in CI
app/strategies.py    the registry: the contract a methodology implements
app/conformance.py   the battery every methodology passes before it ships (CLI + CI)
app/analysis.py      assembles the response payload
app/marketdata.py    yfinance download, disk cache, frozen dataset
app/server.py        FastAPI: POST /api/analyze, GET /api/health, /api/universe
app/rng.py           seeded generator shared with the browser
app/static/          interface: index.html, styles.css, charts.js, app.js,
                     backend.js (server mode), engine.js + backend-local.js
                     (browser mode, used only by the standalone demo)
tools/build_demo.py  builds dist/strata-demo.html
tests/               92 tests: parity with the research pipeline, solver
                     correctness against closed forms, no-look-ahead checks, and
                     deliberately broken methodologies the harness must reject
docs/APP_GUIDE.md    how the app works, the API, and its limits
docs/ADDING_A_METHODOLOGY.md  the research-to-production path for a new rule
Dockerfile           container for option B
```
