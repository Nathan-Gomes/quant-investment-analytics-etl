import os
import sqlite3

# Success and failure rows share one schema, so "which runs failed, and where"
# is a single query rather than a join against something that was never written.
RUN_COLUMNS = {
    "run_id": "TEXT", "timestamp": "TEXT", "status": "TEXT", "stage": "TEXT",
    "price_rows": "INTEGER", "source": "TEXT", "duration_seconds": "REAL",
    "error_type": "TEXT", "error": "TEXT", "manifest": "TEXT",
}


def ensure_run_table(conn):
    """Create pipeline_runs, or widen a database written before these columns existed."""
    columns = ", ".join(f'"{name}" {kind}' for name, kind in RUN_COLUMNS.items())
    conn.execute(f"CREATE TABLE IF NOT EXISTS pipeline_runs ({columns})")
    present = {row[1] for row in conn.execute("PRAGMA table_info(pipeline_runs)")}
    for name, kind in RUN_COLUMNS.items():
        if name not in present:
            conn.execute(f'ALTER TABLE pipeline_runs ADD COLUMN "{name}" {kind}')


def record_run(output, row):
    """Append one run record straight to the live mart, without rebuilding it.

    A failed run has no tables to load, but it still has to leave a trace. This
    writes directly to the database that already exists so the record survives
    the exception that is about to propagate.
    """
    output.mkdir(parents=True, exist_ok=True)
    names = [name for name in RUN_COLUMNS if name in row]
    with sqlite3.connect(output / "investment_analytics.db") as conn:
        ensure_run_table(conn)
        conn.execute(f"INSERT INTO pipeline_runs ({', '.join(names)}) "
                     f"VALUES ({', '.join('?' for _ in names)})",
                     [row[name] for name in names])


def load_mart(output, tables, queries):
    path = output / "investment_analytics.db"
    temporary = output / "investment_analytics.tmp.db"
    try:
        with sqlite3.connect(temporary) as conn:
            if path.exists():
                with sqlite3.connect(path) as old:
                    old.backup(conn)
            for view in ("latest_sector_exposure", "monthly_portfolio_returns", "volatility_ranking", "rolling_portfolio_returns", "run_history"):
                conn.execute(f'DROP VIEW IF EXISTS "{view}"')
            ensure_run_table(conn)
            for name, frame in tables.items():
                frame.to_sql(name, conn, if_exists="append" if name == "pipeline_runs" else "replace", index=False)
            conn.executescript(queries)
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS security_date ON security_daily_analytics(ticker, date)")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS portfolio_date ON portfolio_daily_summary(portfolio_id, date)")
            if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("SQLite integrity check failed")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
