# Developing Strata

## Setup

Run commands from the repository root. Python 3.11+ and Node.js are required
for the complete test suite; Node checks parity with the browser engine.

```sh
python3 -m venv .venv
source .venv/bin/activate
make install
make run
```

Open http://127.0.0.1:8000. The app uses Yahoo by default. For an explicitly
offline session, run `STRATA_SOURCE=bundled make run`. Environment settings
are documented in [the app guide](docs/APP_GUIDE.md).

`make help` lists the supported development commands. Without Make, run
the equivalent Python commands shown in the Makefile.

## Where changes belong

- Application and API: `app/`.
- Browser interface and offline engine: `app/static/`.
- Original research pipeline: `src/` and `sql/`.
- Application, API and parity tests: `tests/application/`.
- Research regression tests: `tests/research/`.
- Methodology and operational documentation: `docs/`.

Read [the architecture guide](docs/ARCHITECTURE.md) before changing boundaries.
New strategies follow [the methodology contract](docs/ADDING_A_METHODOLOGY.md).

## Before a pull request

```sh
make check
make demo
git diff --check
```

Keep regression assertions intact. Changes to calculations need independent
expected values, not updated snapshots alone. Preserve previous-close weights,
trailing-only optimization inputs, explicit data sources, and Python/browser
parity. Do not substitute frozen or simulated data after a failed Yahoo request.

For interface changes, check desktop and mobile, both themes, progress and error
states, and the standalone demo. Capture screenshots from the actual app.

## Outputs and deployments

`make research` rewrites `output/`; `make notebook` updates the research notebook.
Do not commit regenerated research evidence as part of unrelated UI work.
The saved outputs are a published research snapshot, not runtime app state.

GitHub Actions runs the tests, strategy checks, a synthetic pipeline run and
artifact builds. The Dockerfile serves both the interface and API; `render.yaml`
describes the hosting settings. Keep production service URLs stable. Never
commit credentials, `.env` files, local price caches or virtual environments.
