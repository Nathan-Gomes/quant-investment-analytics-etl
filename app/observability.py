"""Recording the fallbacks that keep a run alive but change its answer.

A degradation is not an error. The run finishes, returns a number, and nothing
in the result says the number was built on a substitute profile or a
second-choice solver. That silence is the problem: an answer nobody can
interrogate is worse than one that failed loudly, because it gets acted on.

Every fallback in this package reports here. The warning is always logged, and
when a request is listening the degradation is also attached to its payload, so
a finished analysis can still be asked what it had to work around.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger("strata.degradation")

# Analyses run on a worker thread per request; the collector is thread-local so
# concurrent requests cannot inherit each other's degradations.
_state = threading.local()


def start() -> None:
    """Begin collecting degradations for the work about to run on this thread."""
    _state.events = []


def record(kind: str, detail: str) -> None:
    """Note one fallback. Always logged; collected when a run is listening."""
    logger.warning("degraded [%s] %s", kind, detail)
    events = getattr(_state, "events", None)
    if events is not None:
        events.append({"kind": kind, "detail": detail})


def drain() -> list[dict]:
    """Return what was collected since ``start`` and stop collecting."""
    events = getattr(_state, "events", None)
    _state.events = None
    return list(events) if events else []
