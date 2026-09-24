"""Bundle the interface, the browser engine and the frozen dataset into one file.

    python tools/build_demo.py

The result, ``dist/strata-demo.html``, opens straight from disk and needs
no server: the backtest and the block bootstrap run in the page. It carries the
nine securities in ``data/cached_prices.csv``. For any other ticker, run the
served app, which fetches prices through yfinance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "app" / "static"
DIST = ROOT / "dist"


def dataset() -> dict:
    prices = pd.read_csv(ROOT / "data/cached_prices.csv", usecols=["date", "ticker", "adjusted_price"])
    securities = pd.read_csv(ROOT / "data/securities.csv")
    wide = prices.pivot(index="date", columns="ticker", values="adjusted_price").sort_index()
    if wide.isna().any().any():
        raise SystemExit("The frozen dataset has gaps; the demo needs a complete price matrix.")
    return {
        "source": "Frozen research dataset: Yahoo Finance adjusted closes, "
                  f"{wide.index.min()} to {wide.index.max()}, checksum "
                  f"{hashlib.sha256((ROOT / 'data/cached_prices.csv').read_bytes()).hexdigest()[:12]}",
        "dates": list(wide.index),
        "prices": {ticker: [round(float(v), 4) for v in wide[ticker]] for ticker in wide.columns},
        "securities": {
            row.ticker: {"name": row.name, "sector": row.sector, "currency": row.currency}
            for row in securities.itertuples()
        },
    }


def dataset_js() -> str:
    payload = json.dumps(dataset(), separators=(",", ":"))
    return f"(function(PL){{PL.DATA={payload};PL.ENGINE_VERSION=\"1.0.0\";}})(window.PL=window.PL||{{}});"


def build() -> Path:
    html = (STATIC / "index.html").read_text()
    css = (STATIC / "styles.css").read_text()
    scripts = [dataset_js()] + [
        (STATIC / name).read_text() for name in
        ("optimize.js", "engine.js", "charts.js", "backend-local.js", "app.js")
    ]

    html = html.replace(
        '<link rel="stylesheet" href="styles.css">',
        f"<style>\n{css}\n</style>",
    )
    html = re.sub(r'\n<script src="[^"]+"></script>', "", html)
    html = html.replace("</body>", "<script>\n" + "\n".join(scripts) + "\n</script>\n</body>")

    # The demo cannot reach a price provider, so say so where the control sits.
    html = html.replace(
        '<span class="wordmark-sub">Portfolio construction bench</span>',
        '<span class="wordmark-sub" title="The full app fetches any ticker through Yahoo Finance '
        'and runs the Python solvers.">Offline demo · frozen 2018–2026 data</span>',
    )

    DIST.mkdir(exist_ok=True)
    out = DIST / "strata-demo.html"
    out.write_text(html)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-dataset", type=Path, help="write only the dataset script, for tests")
    args = parser.parse_args()
    if args.emit_dataset:
        args.emit_dataset.write_text(dataset_js())
        print(f"dataset -> {args.emit_dataset}")
        return
    out = build()
    print(f"{out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
