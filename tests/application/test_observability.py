"""Degradations: the fallbacks that finish a run instead of failing it.

These are the dangerous ones. The request succeeds and returns a number, so
nothing prompts anyone to look; the only protection is that the substitution
announced itself.
"""

import logging
import threading

from app import observability


def test_a_recorded_degradation_reaches_the_payload():
    observability.start()
    observability.record("profile_unavailable", "XIU.TO: sector recorded as Unclassified")
    drained = observability.drain()
    assert drained == [{"kind": "profile_unavailable",
                        "detail": "XIU.TO: sector recorded as Unclassified"}]


def test_a_clean_run_reports_nothing():
    observability.start()
    assert observability.drain() == []


def test_draining_twice_does_not_repeat_the_first_run_s_degradations():
    observability.start()
    observability.record("solver_failed", "CLARABEL failed")
    assert len(observability.drain()) == 1
    # A second analysis that never called start() must not inherit the first.
    assert observability.drain() == []


def test_a_degradation_is_logged_even_when_nobody_is_collecting(caplog):
    observability.drain()  # ensure no collector is active
    with caplog.at_level(logging.WARNING, logger="strata.degradation"):
        observability.record("security_master_unavailable", "bundled master unreadable")
    assert "security_master_unavailable" in caplog.text
    assert "bundled master unreadable" in caplog.text


def test_concurrent_requests_do_not_see_each_other_s_degradations():
    seen = {}

    def request(name):
        observability.start()
        observability.record("solver_failed", f"{name} hit a fallback")
        seen[name] = observability.drain()

    threads = [threading.Thread(target=request, args=(name,)) for name in ("a", "b", "c")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert {name: [event["detail"] for event in events] for name, events in seen.items()} == {
        "a": ["a hit a fallback"], "b": ["b hit a fallback"], "c": ["c hit a fallback"]}
