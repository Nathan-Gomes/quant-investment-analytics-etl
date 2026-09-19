# Repository Architecture

Strata contains two related products: an interactive application and its original
reproducible research pipeline. They share financial conventions, but keep their
entry points and outputs distinct. Regression tests verify their agreement.

## Directory Map

```text
app/                    Deployed Python application
  server.py             FastAPI routes, streaming responses, interface serving
  marketdata.py         Yahoo access and explicitly selected frozen data
  analysis.py           Coordinates a complete study
  engine.py             Historical accounting and backtests
  optimize.py           Original optimization routines
  convex.py             Convex portfolio construction and mandates
  riskmodel.py          Covariance and statistical factor models
  strategies.py         Methodology registry and contract
  conformance.py        Shared strategy validation gate
  provenance.py         Input, code and environment fingerprints
  rng.py                Reproducible cross-language random generator
  static/               Browser interface and offline JavaScript engine
src/                    Original batch research pipeline
sql/                    Research reporting queries
tests/
  application/          API, engines, optimizers, strategies and parity checks
  research/             Pipeline calculations and data validation
data/                   Versioned research inputs used by the frozen demo
output/                 Published research evidence and generated reports
notebooks/              Research walkthrough
scripts/                Research reporting and notebook utilities
tools/                  Application demo builder and browser parity harness
docs/                   Architecture, methodology, review and operations guides
  assets/               GitHub banner, screenshots and research illustrations
.github/workflows/      Continuous integration
```

## Execution Boundaries

The live interface submits a study to `app/server.py`. Market data is loaded
through `marketdata.py`, then `analysis.py` coordinates historical accounting,
portfolio construction, scenarios and diagnostics. `provenance.py` records
what produced the result. Interface and API share an origin in production.

The offline demo is built by `tools/build_demo.py`. It bundles browser assets and
the frozen dataset into `dist/strata-demo.html`. Browser solvers support minimum
variance and risk parity with walk-forward rebalancing. It cannot run the other
Python objectives or download Yahoo prices. It identifies its frozen source explicitly.

The batch workflow starts at `python -m src.pipeline`. It validates inputs,
calculates research results, loads SQLite and writes reports. It is not a
background task required by the live server.

## Stable Paths and Generated Files

`src/` is a historical Python package name, not an installable-package source
layout. Both entry points run from the checkout or Docker working directory.
Keeping the package name preserves existing commands and notebook imports.

The research input, SQL, notebook and output paths are retained because the
published portfolio and tests reference them. `output/` is deliberately a
versioned research exhibit; `dist/`, `app/cache/` and local environments are not.
Do not move published evidence merely to make the repository root shorter.

## Deployment and Limits

The Dockerfile packages `app/`, `src/` and `data/`. README images and generated
reports are not needed in the runtime image. Render settings remain at the root
so hosting tools can discover them. `make check` runs the same correctness gates
as CI, without regenerating the published research snapshot.

Repository organization is not production hardening: the public API still needs
rate limiting, bounded concurrent analyses and request timeouts before broad use.
Yahoo throttling and free-host cold starts remain operational constraints.
