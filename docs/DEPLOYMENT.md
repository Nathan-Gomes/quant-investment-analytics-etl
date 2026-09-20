# Deploying it

The interface is static files and will run anywhere. The price fetching and the
analysis are Python, and need a host that can run processes and make outbound
HTTPS requests — so GitHub Pages, Netlify's static tier and S3 are out. The only
real question is where the Python runs.

## Three ways to do it

**A. Frozen demo on a static site.** Build `dist/strata-demo.html` with
`python tools/build_demo.py` and drop it next to any other page. One file, no
infrastructure, no running costs, works offline. It carries the nine Canadian
securities and cannot accept any other ticker.

**B. Full app on its own host.** Deploy the repo to Render, Fly.io, Railway or
any container host using the included `Dockerfile`. The server serves both the
API and the interface, so there's no CORS and no cross-origin wiring. This is
the simplest way to get live tickers working.

**C. Interface on a static site, API elsewhere.** Copy `app/static/*` into the
site and point it at the API:

```html
<script>window.STRATA_API = "https://portfolio-lab-api.example.com/";</script>
<script src="charts.js"></script>
<script src="backend.js"></script>
<script src="app.js"></script>
```

Then start the API with that origin allowed:

```sh
STRATA_ALLOWED_ORIGINS="https://www.nathan-gomes.com" uvicorn app.server:api
```

Both halves already support this — `backend.js` reads `window.STRATA_API`, and
the server adds CORS only when that variable is set. Worth it only if the
interface has to sit inside an existing site's navigation; otherwise B is less
to go wrong.

A and B together work well: the frozen demo loads instantly and keeps working
even if the live host sleeps or yfinance breaks.

## Running option B

```sh
docker build -t portfolio-lab .
docker run -p 8000:8000 portfolio-lab
```

On a platform, point it at the `Dockerfile` and set:

| Variable | Value | Why |
| --- | --- | --- |
| `STRATA_SOURCE` | `auto` | Yahoo Finance. `bundled` forces the frozen dataset. |
| `PORT` | platform-provided | The image already reads it. |
| `STRATA_LOG_FILE` | unset | Set it to also write a rotating log file; stdout otherwise. |
| `STRATA_ALLOWED_ORIGINS` | unset | Only needed for option C. |

Mount a persistent volume at `/srv/app/cache` if the platform offers one. It
holds each ticker's downloaded history plus a provenance record. It's only a
cache, so the app works without it, but every cold start will re-fetch.

A request costs roughly one second of CPU for 5,000 scenario paths across four
portfolios, and peak memory stays well under 512 MB. The smallest paid tier
anywhere is enough, and a free tier that sleeps is fine if the first request is
allowed to take ten seconds.

## Things that will bite

- **yfinance from cloud IPs.** Yahoo rate-limits datacentre addresses much more
  aggressively than home connections, and periodically changes endpoints, which
  breaks yfinance until it's updated. Keep the cache volume, raise
  `CACHE_MAX_AGE_HOURS` in `app/marketdata.py` from 12 to 72 on anything public,
  and pin then bump the yfinance version. If it turns into a habit, moving to a
  keyed provider like Tiingo or Alpha Vantage means rewriting only `download()`
  in `app/marketdata.py` — nothing else touches the provider.
- **Abuse.** The analyse endpoint is unauthenticated and does real work. Put a
  rate limit in front of it before making it public — the platform's own, or
  `slowapi`. The caps already in place are 25 tickers, 6 portfolios, 20,000 paths.
- **Terms of use.** Yahoo's terms govern this data. Fine for a personal research
  demo; don't redistribute bulk price history or resell access.
- **Cold starts** on free tiers make the first run look broken. Either use a tier
  that stays warm, or add a short "waking up" note to the interface.

## Things not to break

These are the invariants the tests exist to protect, and the ones easiest to
undo by accident later.

1. **The return maths in `app/engine.py`** — previous-close weights, drift
   between rebalances, turnover-based costs. `test_app_reproduces_the_published_pipeline_results`
   checks the app against the pipeline's published figures to nine significant
   figures; if Growth's annualized return stops being 27.795%, something broke.
2. **Optimized weights stay walk-forward.** Every weight is solved from a
   trailing window and applied forward. `test_the_optimizer_cannot_see_the_future`
   rewrites the last month of prices and asserts nothing earlier moves. Fitting
   weights on the full sample would "improve" results by cheating.
3. **The forward scenarios are a block bootstrap, not a forecast.** Not an ML
   price predictor, and percentile bands aren't predictions. The honesty is the
   point of the project.
4. **Never substitute simulated data for a failed download.** A failure is
   reported and names the ticker. Silent fallbacks are how a demo becomes a lie —
   the ones that are unavoidable are logged and surfaced in `meta.degradations`
   (see [`OBSERVABILITY.md`](OBSERVABILITY.md)).
5. **Keep the assumptions plate and the disclaimers** — survivorship bias, no FX,
   no taxes, not investment advice. They live in `app/static/app.js` and
   [`APP_GUIDE.md`](APP_GUIDE.md).
6. **`app/rng.py` and `app/static/engine.js` stay in step.** They implement the
   same generator so a seed means the same thing on the server and in the browser
   demo. `test_browser_engine_agrees_with_python` enforces it.
7. **A new construction rule goes through the registry**, never by editing the
   engine, and passes `python -m app.conformance` first. That command runs in CI
   and exits non-zero on failure.

Run `pytest tests/` before deploying. With node installed, that includes the
cross-language check.

## Worth checking after a deploy

- `AAPL, MSFT, NVDA` at 40/30/30 over the last 10 years returns a result
- A nonsense symbol gives a clear message naming it, not a blank screen
- A second portfolio can be added and both appear in the same chart
- The scenario fan and callouts change when a different portfolio is selected
- Export downloads a JSON file containing every displayed number
- Usable on a phone: results scroll into view after Run
- Dark and light both readable
- A repeat run with the same seed gives identical scenario numbers
