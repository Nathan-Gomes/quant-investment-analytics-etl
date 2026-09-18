# Test Suites

- `application/`: API delivery, historical accounting, optimizers, strategy
  contracts, provenance, and parity with the research and browser engines.
- `research/`: the batch pipeline's financial calculations, validation and
  scenario reporting.

Run both with `make test`, or use `make test-app` and `make test-research`.
`make check` also runs the strategy conformance gate. Node.js is needed for
cross-language parity coverage; install it before running the full suite.

This split describes the products under test, not isolation level: application
tests include both focused numerical checks and integration tests. Keep tests
that compare the research and app together in the application suite.
