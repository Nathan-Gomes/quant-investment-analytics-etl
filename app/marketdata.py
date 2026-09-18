"""Where prices come from, and what is recorded about them.

Two sources, chosen per request:

``yahoo``    Adjusted daily closes via yfinance, cached on disk per ticker so a
             repeated study does not re-hit the provider. Every cached file
             carries a small provenance record.
``bundled``  The frozen CSV that ships with the research repository. Nine
             Canadian securities, 2018 to 2026, checksum-verified. This is what
             the app uses with no network access.

A download that fails is reported as a failure. Prices are never quietly
replaced with simulated data.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "app" / "cache"
BUNDLED_PRICES = ROOT / "data" / "cached_prices.csv"
BUNDLED_SECURITIES = ROOT / "data" / "securities.csv"
CACHE_MAX_AGE_HOURS = 12


@dataclass
class PriceSet:
    prices: pd.DataFrame          # date, ticker, adjusted_price
    metadata: pd.DataFrame        # ticker, name, sector, currency
    source: str
    note: str


def bundled_universe() -> pd.DataFrame:
    return pd.read_csv(BUNDLED_SECURITIES)


def load_bundled(tickers: list[str] | None = None) -> PriceSet:
    digest = hashlib.sha256(BUNDLED_PRICES.read_bytes()).hexdigest()
    manifest = BUNDLED_PRICES.with_suffix(".json")
    if manifest.exists():
        recorded = json.loads(manifest.read_text()).get("sha256")
        if recorded and recorded != digest:
            raise ValueError("The bundled price file no longer matches its recorded checksum.")
    prices = pd.read_csv(BUNDLED_PRICES, usecols=["date", "ticker", "adjusted_price"])
    metadata = bundled_universe()
    if tickers:
        missing = sorted(set(tickers) - set(metadata.ticker))
        if missing:
            raise ValueError(
                "The bundled dataset only covers " + ", ".join(metadata.ticker) +
                ". Not available offline: " + ", ".join(missing) + "."
            )
        prices = prices[prices.ticker.isin(tickers)]
        metadata = metadata[metadata.ticker.isin(tickers)]
    return PriceSet(prices, metadata, "bundled", "Frozen research dataset shipped with the repository")


def _cache_path(ticker: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-._^=" else "_" for c in ticker.upper())
    return CACHE / f"{safe}.csv"


def _fresh(path: Path) -> bool:
    meta = path.with_suffix(".json")
    if not path.exists() or not meta.exists():
        return False
    age = time.time() - json.loads(meta.read_text()).get("retrieved_at_epoch", 0)
    return age < CACHE_MAX_AGE_HOURS * 3600


def require_provider():
    """Check the price provider once, so a missing package is reported once.

    Without this the import error is raised per ticker and the person sees the
    same Python message repeated across the whole universe.
    """
    try:
        import yfinance as yf
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ValueError(
            "The yfinance package is not installed, so live prices are unavailable. "
            "Run `pip install -r requirements.txt`, or switch the data source to the bundled dataset."
        ) from error
    return yf


def download(ticker: str, start: str, end: str) -> tuple[pd.DataFrame, dict]:
    """One ticker's adjusted closes, from the cache when it is recent enough."""
    yf = require_provider()

    CACHE.mkdir(parents=True, exist_ok=True)
    path = _cache_path(ticker)
    if _fresh(path):
        cached = pd.read_csv(path)
        info = json.loads(path.with_suffix(".json").read_text())
        covered = cached.date.min() <= str(start) and cached.date.max() >= str(min(end, time.strftime("%Y-%m-%d")))
        if covered or info.get("covers_full_history"):
            return cached, info

    handle = yf.Ticker(ticker)
    frame = handle.history(period="max", auto_adjust=True)
    if frame.empty:
        raise ValueError(f"Yahoo Finance returned no price history for {ticker}. Check the symbol.")
    frame = frame[frame.Close > 0]
    prices = pd.DataFrame({
        "date": pd.to_datetime(frame.index).tz_localize(None).strftime("%Y-%m-%d"),
        "ticker": ticker,
        "adjusted_price": frame.Close.to_numpy(),
    })
    profile = {}
    try:
        raw = handle.get_info()
        profile = {
            "name": raw.get("shortName") or raw.get("longName") or ticker,
            "sector": raw.get("sector") or ("ETF" if raw.get("quoteType") == "ETF" else "Unclassified"),
            "currency": raw.get("currency", "").upper() or None,
            "quote_type": raw.get("quoteType"),
        }
    except Exception:  # a missing profile must not fail a valid price series
        profile = {"name": ticker, "sector": "Unclassified", "currency": None, "quote_type": None}
    info = {
        **profile,
        "ticker": ticker,
        "source": "Yahoo Finance via yfinance",
        "price_basis": "Split and dividend adjusted close",
        "retrieved_at_epoch": time.time(),
        "retrieved_at": pd.Timestamp.now("UTC").isoformat(),
        "first_date": prices.date.min(),
        "last_date": prices.date.max(),
        "covers_full_history": True,
    }
    prices.to_csv(path, index=False)
    path.with_suffix(".json").write_text(json.dumps(info, indent=2, default=str))
    return prices, info


def load(tickers: list[str], start: str, end: str, source: str = "auto") -> PriceSet:
    """Prices and profiles for a universe, from the requested source."""
    tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if not tickers:
        raise ValueError("Add at least one ticker.")
    if len(tickers) > 25:
        raise ValueError("Up to 25 tickers can be studied at once.")
    if source == "bundled":
        return load_bundled(tickers)
    require_provider()

    frames, profiles, failures = [], [], []
    for ticker in tickers:
        try:
            frame, info = download(ticker, start, end)
        except Exception as error:  # noqa: BLE001 - reported to the caller verbatim
            failures.append(f"{ticker}: {error}")
            continue
        frames.append(frame)
        profiles.append({
            "ticker": ticker,
            "name": info.get("name", ticker),
            "sector": info.get("sector", "Unclassified"),
            "currency": info.get("currency"),
        })
    if failures:
        raise ValueError("Could not load " + "; ".join(failures))
    if not frames:
        raise ValueError(
            "No prices could be loaded (" + "; ".join(failures) +
            "). Switch the data source to the bundled dataset to work without a network connection."
        )
    prices = pd.concat(frames, ignore_index=True)
    prices = prices[(prices.date >= str(start)) & (prices.date <= str(end))]
    note = "Yahoo Finance adjusted closes via yfinance"
    if failures:
        note += " (skipped " + "; ".join(failures) + ")"
    return PriceSet(prices, pd.DataFrame(profiles), "yahoo", note)
