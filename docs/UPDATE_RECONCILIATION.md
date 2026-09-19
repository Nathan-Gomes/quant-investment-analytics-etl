# September 2026 update reconciliation

The offline-optimizer update was merged into the existing organized repository,
not copied over it. Existing result-cache locking, sampling intervals, research
outputs, source-selection safeguards and deployment URLs were retained.

## Included

- Browser minimum-variance and risk-parity solvers, with walk-forward rebalancing.
- Python/browser weight, covariance and full-run numerical comparison tests.
- Risk-parity first-order refinement when conic termination leaves a residual.
- Four total Yahoo price-fetch attempts for detected rate limits, with backoff.
- Known security metadata reused without a second provider profile request.
- Three download workers by default and in the deployment specification.
- Rebuilt standalone demo and matching portfolio-site descriptions.

## Intentional differences from the supplied package

- Yahoo failures remain errors. Frozen prices are available only by explicit
  selection, never automatic substitution. Unknown symbols are not dropped.
- No scheduled keep-warm workflow was enabled. Hosting can still cold-start.
- No network downloads were added to Docker builds. Cache warming does not
  guarantee fresh coverage, successful downloads or persistent free-tier storage.
- Numerical agreement is stated within tested tolerances, not as exact equality.
- The demo is limited to nine frozen securities; arbitrary supported Yahoo
  symbols and the other objectives require the live Python service.

## Verification

Run `python -m pytest -q`, `python -m app.conformance`, then
`python tools/build_demo.py`. Browser solver tests require Node.js.
Runtime environment variables override `render.yaml`; existing services must
also set `STRATA_DOWNLOAD_WORKERS=3` to use the reduced concurrency.
