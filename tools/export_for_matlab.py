"""Export the optimizer inputs and answers for the MATLAB cross-check.

The cross-check is only meaningful if MATLAB solves the same problem the Python
service solved, so this writes the covariance the weights were actually fitted
from — not a full-sample matrix that would describe a different problem.

    python -m tools.export_for_matlab
    octave --no-gui matlab/verify_against_python.m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import convex, marketdata, riskmodel  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=float, default=0.35, help="maximum weight per holding")
    parser.add_argument("--window", type=int, default=252, help="estimation window in sessions")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "matlab")
    arguments = parser.parse_args()

    priceset = marketdata.load_bundled(None)
    wide = priceset.prices.pivot(index="date", columns="ticker", values="adjusted_price").sort_index()
    universe = [ticker for ticker in wide.columns if ticker != "XIC.TO"]
    returns = wide[universe].pct_change(fill_method=None).dropna().tail(arguments.window)

    model = riskmodel.build(returns.to_numpy(dtype=float), "ledoit_wolf")
    mandate = convex.Mandate(max_weight=arguments.cap)
    minimum_variance = convex.minimum_variance(model, mandate).weights
    risk_parity = convex.risk_parity(model, mandate).weights

    arguments.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(model.covariance, index=universe, columns=universe).to_csv(
        arguments.out / "covariance.csv", index_label="ticker")
    pd.DataFrame(
        {"minimum_variance": minimum_variance, "risk_parity": risk_parity}, index=universe
    ).to_csv(arguments.out / "python_weights.csv", index_label="ticker")
    pd.DataFrame(
        {"maximum_weight": [arguments.cap], "observations": [len(returns)],
         "shrinkage_intensity": [model.shrinkage_intensity]}, index=["settings"]
    ).to_csv(arguments.out / "settings.csv", index_label="name")

    print(f"Exported {len(universe)} holdings and {len(returns)} observations to {arguments.out}")
    print(f"  covariance shrinkage applied: {model.shrinkage_intensity:.1%}")
    print("  now run: octave --no-gui matlab/verify_against_python.m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
