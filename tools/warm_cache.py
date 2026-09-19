"""Fill the price cache before the first visitor arrives.

A deployed instance starts with an empty cache, so the first person to press Run
waits for every ticker to download — and on a shared cloud address that burst is
exactly what Yahoo refuses. Running this at build time means the common tickers
are already on disk and the first run needs no network at all.

    python -m tools.warm_cache                    # the bundled universe
    python -m tools.warm_cache AAPL MSFT NVDA     # anything else

Failure is not fatal. If the provider refuses during a build, the image still
ships and the app still works from the frozen dataset.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from app import marketdata  # noqa: E402


def main(tickers: list[str]) -> int:
    if not tickers:
        tickers = sorted(marketdata.bundled_universe().ticker)
    print(f"Warming the cache for {len(tickers)} tickers")
    warmed, refused = [], []
    for ticker in tickers:
        started = time.perf_counter()
        try:
            marketdata.download(ticker, "2010-01-01", time.strftime("%Y-%m-%d"))
            warmed.append(ticker)
            print(f"  {ticker:<10} ok        {(time.perf_counter() - started) * 1000:6.0f} ms")
        except Exception as error:  # noqa: BLE001
            refused.append(ticker)
            print(f"  {ticker:<10} skipped   {type(error).__name__}: {str(error)[:60]}")
        # Deliberately unhurried: this runs once at build time, and a burst here
        # is what gets the address rate-limited in the first place.
        time.sleep(0.4)
    print(f"\n{len(warmed)} cached, {len(refused)} skipped")
    if refused:
        print("Skipped tickers will be fetched on demand; the frozen dataset covers the bundled universe.")
    return 0        # never fail a build over a warm cache


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
