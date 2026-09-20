# Knowing when something went wrong

The uncomfortable question this answers: someone asks why a number looks wrong,
and the honest reply cannot be "we don't know". Three things make that reply
avoidable — a durable record of every run, log lines that can be tied to one run,
and fallbacks that announce themselves instead of quietly substituting.

## Every run leaves a record, including the ones that failed

`pipeline_runs` is appended to on both paths. A successful run writes its row as
part of the data mart; a failed run writes straight to the live database from the
exception handler, because a run that could not build the mart still has to leave
a trace.

| Column | On success | On failure |
| --- | --- | --- |
| `status` | `success` | `failed` |
| `stage` | `complete` | the step it died at, e.g. `fit_models` |
| `error_type` / `error` | null | the exception class and its message |
| `duration_seconds` | wall clock | wall clock up to the failure |
| `price_rows` | rows loaded | null |
| `manifest` | the full run manifest | null |

The stage names are the ones an analyst would recognise — `extract`,
`clean_inputs`, `portfolio_analytics`, `fit_models`, `load_mart` — not function
names. A traceback says where the code broke; the stage says what was being done.

Read it back through the `run_history` view:

```sql
SELECT timestamp, status, stage, error_type, error
FROM run_history
WHERE status = 'failed';
```

Recording a failure never masks it. If the write itself fails, that is logged and
the original exception is still what propagates.

A database written before these columns existed is widened in place on the next
run, and its existing rows keep their values, with nulls where they have nothing
to say.

## Log lines say which run they belong to

`src/observability.py` stamps every record with the run in progress:

```
2026-09-20 01:02:47,682 INFO     [run=10465754-...] root: [2/9] Cleaning and validating inputs
2026-09-20 01:03:32,731 ERROR    [run=09ca323c-...] root: Pipeline failed during portfolio_analytics
```

So `grep 'run=09ca323c'` returns one run's history and nothing else. The file
rotates at 5 MB with five generations kept, because a log that grows without
bound is one full disk away from losing the incident it was kept for.

The API mints a short reference per request and puts it in both the log line and
the error message, so a user reporting "it failed and said reference 4f2a91c0e33b"
points directly at a log entry. A run that succeeds carries its `run_id` instead.

## Fallbacks announce themselves

Some failures should not stop a run: a missing issuer profile is not a reason to
discard a valid price series. But a substitution nobody is told about is the worst
case of all, because the run succeeds and the number gets acted on.

Every fallback in `app/` reports through `app/observability.py`. Each one is logged
as a warning and attached to the response at `meta.degradations`:

```json
"degradations": [
  {"kind": "profile_unavailable",
   "detail": "RY.TO: no issuer profile from the provider (...); sector recorded as Unclassified, which affects sector exposure"}
]
```

Currently reported:

| `kind` | Raised when | What it changes |
| --- | --- | --- |
| `security_master_unavailable` | the bundled security master cannot be read | name and sector fall back to the provider |
| `profile_unavailable` | the provider returns no profile for a ticker | sector becomes `Unclassified`, which moves sector exposure |
| `solver_failed` | a convex solver declines and the next one is tried | the answer comes from a different solver than intended |
| `diagnostics_unavailable` | weight stability cannot be computed | the analysis is unaffected; the diagnostic is missing |

An empty list is the normal case, and it is the one worth being able to prove.

## Adding a fallback

If you write a handler that keeps a run alive by substituting something, record
it. The rule is narrow: if the run returns a number that differs from what it
would have returned without the fallback, it is a degradation.

```python
from . import observability

try:
    ...
except Exception as error:  # noqa: BLE001
    observability.record("what_happened", f"{context}: {error}; what it changes")
    return fallback
```

Collection is thread-local and started per request, so concurrent analyses cannot
inherit each other's degradations.
