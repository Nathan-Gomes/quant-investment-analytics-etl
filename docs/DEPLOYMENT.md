# Deploying it

The front end is just static files, so it'll run anywhere. The Python side, which
is the price fetching and the analysis, needs a host that can run processes and
make outbound HTTPS calls. That rules out GitHub Pages, Netlify's static tier and
S3. So the question is really just where the Python lives.

## Three ways to do it

**A. Frozen demo on a static site.** Run `python tools/build_demo.py` to get
`dist/strata-demo.html`, then drop that one file next to any other page. No
infrastructure, nothing to pay for, works offline. The catch is that it only has
the nine Canadian securities baked into it and won't accept any other ticker.

**B. Full app on its own host.** Push the repo to Render, Fly.io, Railway or any
container host using the `Dockerfile` that's already here. The server hands out
both the API and the interface, so there's no CORS to deal with. This is the
easiest way to get live tickers working.

**C. Interface on a static site, API somewhere else.** Copy `app/static/*` into
the site and point it at the API:

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

Both sides already handle this. `backend.js` reads `window.STRATA_API`, and the
server only switches CORS on when that variable is set. I'd only bother with this
if the interface has to sit inside an existing site's navigation. Otherwise B is
less to go wrong.

Running A and B together works well. The frozen demo loads instantly, and it
keeps working even when the live host is asleep or yfinance is broken.

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
| `STRATA_LOG_FILE` | unset | Set it to also write a rotating log file. Otherwise stdout. |
| `STRATA_ALLOWED_ORIGINS` | unset | Only needed for option C. |

If the platform offers a persistent volume, mount it at `/srv/app/cache`. That's
where each ticker's downloaded history goes, along with a provenance record. It's
only a cache so nothing breaks without it, but every cold start will re-fetch
everything.

For sizing: a request takes about a second of CPU for 5,000 scenario paths across
four portfolios, and memory peaks well under 512 MB. The cheapest paid tier
anywhere handles that. A free tier that sleeps is fine too, as long as I'm okay
with the first request taking ten seconds.

## Things that will bite

- **yfinance from cloud IPs.** Yahoo is much harsher about rate-limiting
  datacentre addresses than home connections. It also changes its endpoints every
  so often, which breaks yfinance until someone updates it. So keep the cache
  volume, bump `CACHE_MAX_AGE_HOURS` in `app/marketdata.py` from 12 to 72 on
  anything public, and pin the yfinance version but actually update it now and
  then. If it keeps happening, switching to a keyed provider like Tiingo or Alpha
  Vantage only means rewriting `download()` in `app/marketdata.py`. Nothing else
  touches the provider.
- **Abuse.** The analyse endpoint has no auth and does real work on every call.
  It needs a rate limit in front of it before it goes public, either the
  platform's own or `slowapi`. The caps already there are 25 tickers, 6
  portfolios and 20,000 paths.
- **Terms of use.** Yahoo's terms cover this data. A personal research demo is
  fine. Redistributing bulk price history or reselling access is not.
- **Cold starts.** On a free tier the first run just looks broken. Either pay for
  a tier that stays warm, or put a short "waking up" note in the interface.

## Things not to break

The tests cover all of these. They're just the ones that would be easy to undo by
accident later.

1. **The return maths in `app/engine.py`**, meaning previous-close weights, drift
   between rebalances, and turnover-based costs.
   `test_app_reproduces_the_published_pipeline_results` compares the app against
   the pipeline's published numbers to nine significant figures. If Growth's
   annualized return stops being 27.795%, I broke something.
2. **Optimized weights have to stay walk-forward.** Every weight is solved on a
   trailing window and applied forward.
   `test_the_optimizer_cannot_see_the_future` rewrites the last month of prices
   and checks that nothing earlier moved. Fitting on the full sample would make
   the results look better by cheating.
3. **The forward scenarios are a block bootstrap, not a forecast.** Not an ML
   price predictor, and the percentile bands aren't predictions. That's most of
   the point of the project.
4. **Never swap in simulated data when a download fails.** Failures get reported
   with the ticker named. Where a fallback genuinely can't be avoided it gets
   logged and shows up in `meta.degradations`, which
   [`OBSERVABILITY.md`](OBSERVABILITY.md) covers.
5. **Keep the assumptions plate and the disclaimers**, meaning survivorship bias,
   no FX, no taxes, not investment advice. They're in `app/static/app.js` and
   [`APP_GUIDE.md`](APP_GUIDE.md).
6. **`app/rng.py` and `app/static/engine.js` have to stay in step.** Same
   generator on both sides, so a seed means the same thing on the server and in
   the browser demo. `test_browser_engine_agrees_with_python` checks it.
7. **New construction rules go through the registry**, not by editing the engine,
   and have to pass `python -m app.conformance` first. That runs in CI and exits
   non-zero on failure.

Run `pytest tests/` before deploying. If node is installed that includes the
cross-language check.

## Worth checking after a deploy

- `AAPL, MSFT, NVDA` at 40/30/30 over the last 10 years returns a result
- A nonsense symbol gives a clear message naming it instead of a blank screen
- A second portfolio can be added and both show up in the same chart
- The scenario fan and callouts change when a different portfolio is selected
- Export downloads a JSON file with every displayed number in it
- Usable on a phone, and results scroll into view after Run
- Dark and light are both readable
- A repeat run with the same seed gives identical scenario numbers
