import os
import sqlite3


def load_mart(output, tables, queries):
    path = output / "investment_analytics.db"
    temporary = output / "investment_analytics.tmp.db"
    try:
        with sqlite3.connect(temporary) as conn:
            if path.exists():
                with sqlite3.connect(path) as old:
                    old.backup(conn)
            for view in ("latest_sector_exposure", "monthly_portfolio_returns", "volatility_ranking", "rolling_portfolio_returns"):
                conn.execute(f'DROP VIEW IF EXISTS "{view}"')
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
