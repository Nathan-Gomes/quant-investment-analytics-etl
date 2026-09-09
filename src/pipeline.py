import argparse
import hashlib
import json
import logging
import time
import uuid
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .analytics import portfolio_analytics, security_analytics
from .extract import extract
from .load import load_mart
from .models import fit_models
from .report import render_report
from .validation import clean_inputs, validate_outputs


def run(root, source="cached", config_path=None):
    started = time.perf_counter()
    output = root / "output"
    output.mkdir(exist_ok=True)
    config = json.loads((config_path or root / "config.json").read_text())
    if not (0 < config["test_fraction"] < 0.5 and config["forecast_days"] >= 2
            and config["initial_capital"] > 0 and config["transaction_cost_bps"] >= 0
            and config["risk_free_rate"] > -1 and config["warmup_days"] >= 20):
        raise ValueError("Invalid research configuration")
    run_id = str(uuid.uuid4())
    logging.info("[1/8] Extracting prices, holdings and security metadata")
    prices, holdings, securities, provenance = extract(root, config, source)
    prices = prices[(pd.to_datetime(prices.date) >= config["start"]) & (pd.to_datetime(prices.date) < config["end"])]
    logging.info("[2/8] Cleaning and validating inputs")
    prices, quality = clean_inputs(prices, holdings, securities)
    logging.info("[3/8] Calculating security analytics")
    security = security_analytics(prices)
    logging.info("[4/8] Simulating portfolios and transaction costs")
    daily, positions, sectors, summary, trades, targets = portfolio_analytics(prices, holdings, securities, config)
    logging.info("[5/8] Validating NAV and allocation reconciliation")
    validate_outputs(daily, positions)
    logging.info("[6/8] Training models with purged time-series validation")
    scores, predictions, coefficients, audits = fit_models(daily, prices, config)
    tables = {"securities": securities, "holdings": holdings, "security_daily_analytics": security,
              "portfolio_positions": positions, "portfolio_daily_summary": daily,
              "sector_exposures": sectors, "portfolio_summary": summary, "rebalancing_history": trades,
              "target_weights": targets, "model_scores": scores, "model_predictions": predictions,
              "model_coefficients": coefficients, "validation_splits": audits}
    logging.info("[7/8] Generating offline report and CSV exports")
    charts = render_report(output, tables, provenance, config, quality)
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    manifest = {"run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": provenance, "config": config, "validation": quality,
                "package_versions": {package: version(package) for package in ("pandas", "numpy", "scipy", "scikit-learn", "plotly", "yfinance")},
                "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (root / "src").glob("*.py")},
                "input_hashes": {name: hashlib.sha256((root / "data" / name).read_bytes()).hexdigest()
                                 for name in ("holdings.csv", "securities.csv")}}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    tables["pipeline_runs"] = pd.DataFrame([{"run_id": run_id, "timestamp": manifest["timestamp"],
                                            "status": "success", "price_rows": len(prices),
                                            "source": provenance["source"], "manifest": json.dumps(manifest)}])
    logging.info("[8/8] Loading SQL data mart")
    load_mart(output, tables, (root / "sql/analysis_queries.sql").read_text())
    logging.info("Pipeline complete: %s prices; %.1f seconds; report: %s", len(prices), time.perf_counter() - started, output / "report.html")
    return tables, charts


def main():
    parser = argparse.ArgumentParser(description="Investment analytics ETL and volatility forecasting")
    parser.add_argument("--source", choices=["cached", "download", "synthetic"], default="cached")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    (root / "output").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(root / "output/pipeline.log")])
    try:
        run(root, args.source, args.config)
    except Exception:
        logging.exception("Pipeline failed")
        raise


if __name__ == "__main__":
    main()
