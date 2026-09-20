"""What the pipeline is able to say about a run after it has ended.

A run that fails is the run someone will ask about, so the record of it has to
outlive the exception rather than being written only on the happy path.
"""

import logging
import sqlite3

import pytest

from src import observability
from src.load import ensure_run_table, record_run, RUN_COLUMNS


def _rows(path):
    with sqlite3.connect(path) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(pipeline_runs)")]
        return columns, [dict(zip(columns, row)) for row in conn.execute("SELECT * FROM pipeline_runs")]


def test_a_failed_run_is_recorded_without_the_rest_of_the_mart(tmp_path):
    record_run(tmp_path, {"run_id": "r1", "timestamp": "2026-01-01T00:00:00Z", "status": "failed",
                          "stage": "fit_models", "price_rows": None, "source": "cached",
                          "duration_seconds": 1.5, "error_type": "ValueError",
                          "error": "Invalid portfolio results", "manifest": None})
    columns, rows = _rows(tmp_path / "investment_analytics.db")
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["stage"] == "fit_models"
    assert rows[0]["error_type"] == "ValueError"
    assert rows[0]["error"] == "Invalid portfolio results"


def test_failures_accumulate_rather_than_replacing_each_other(tmp_path):
    for index in range(3):
        record_run(tmp_path, {"run_id": f"r{index}", "timestamp": "2026-01-01T00:00:00Z",
                              "status": "failed", "stage": "extract", "error_type": "OSError",
                              "error": "provider refused", "source": "download"})
    _, rows = _rows(tmp_path / "investment_analytics.db")
    assert [row["run_id"] for row in rows] == ["r0", "r1", "r2"]


def test_a_database_written_before_these_columns_existed_is_widened(tmp_path):
    path = tmp_path / "investment_analytics.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE pipeline_runs (run_id TEXT, timestamp TEXT, status TEXT, "
                     "price_rows INTEGER, source TEXT, manifest TEXT)")
        conn.execute("INSERT INTO pipeline_runs VALUES ('old', 'then', 'success', 10, 'cached', '{}')")

    record_run(tmp_path, {"run_id": "new", "timestamp": "now", "status": "failed",
                          "stage": "load_mart", "error_type": "ValueError", "error": "boom"})

    columns, rows = _rows(path)
    assert set(RUN_COLUMNS) <= set(columns)
    assert len(rows) == 2
    # The pre-existing row survives the widening, with nulls where it had nothing to say.
    old = next(row for row in rows if row["run_id"] == "old")
    assert old["status"] == "success" and old["stage"] is None


def test_every_log_line_carries_the_run_it_belongs_to(tmp_path, capsys):
    log = tmp_path / "pipeline.log"
    observability.configure(log)
    try:
        observability.set_run_id("abc-123")
        logging.info("extracting")
        observability.set_run_id("def-456")
        logging.warning("something drifted")
    finally:
        for handler in list(logging.getLogger().handlers):
            handler.close()
            logging.getLogger().removeHandler(handler)

    lines = log.read_text().splitlines()
    assert "[run=abc-123]" in lines[0] and "extracting" in lines[0]
    assert "[run=def-456]" in lines[1] and "WARNING" in lines[1]


def test_the_log_rotates_rather_than_growing_without_bound(tmp_path):
    log = tmp_path / "pipeline.log"
    observability.configure(log, max_bytes=2_000, backups=2)
    try:
        observability.set_run_id("noisy")
        for index in range(200):
            logging.info("stage %s of a very repetitive pipeline run", index)
    finally:
        for handler in list(logging.getLogger().handlers):
            handler.close()
            logging.getLogger().removeHandler(handler)

    assert log.exists()
    assert (tmp_path / "pipeline.log.1").exists(), "rotation never happened"
    assert log.stat().st_size < 10_000
