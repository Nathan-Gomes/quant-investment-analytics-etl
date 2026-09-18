"""A traceable identity for every analysis the app produces.

A number on a screen is worth what you can trace it back to. Every response
carries a manifest: the exact request, a digest of the price data it read, the
version of the engine that ran, the versions of the libraries underneath it, and
a digest of the source files themselves.

The run identifier is derived from those inputs rather than generated at random,
so it is a fingerprint rather than a serial number: the same request against the
same data on the same code produces the same identifier, and any difference in
any of them produces a different one. Two people comparing results can tell in
one line whether they ran the same thing.

This mirrors what ``src/pipeline.py`` already writes for the research pipeline,
so a figure quoted from the app and a figure quoted from the nightly run can be
checked against each other.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = Path(__file__).resolve().parent
TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn", "fastapi", "yfinance")


def digest(payload) -> str:
    """A stable digest of any JSON-shaped value.

    Sorted keys and fixed separators, so the same content digests the same way
    regardless of the order it was assembled in.
    """
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


@lru_cache(maxsize=1)
def code_digest() -> dict:
    """A digest per source file, and one over the whole engine.

    If a result cannot be reproduced, the first question is whether the code is
    the same code. This answers it without needing a git history to hand.
    """
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(SOURCE.glob("*.py"))}
    return {"files": files, "combined": digest(files)}


@lru_cache(maxsize=1)
def environment() -> dict:
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "packages": versions}


def data_digest(prices: pd.DataFrame) -> dict:
    """A digest of the price matrix actually used, not of the file it came from.

    Providers revise adjusted history. Digesting the values that entered the
    calculation means a changed price shows up as a changed run, which is the
    point.
    """
    frame = prices.sort_values(["ticker", "date"])
    values = np.ascontiguousarray(frame["adjusted_price"].to_numpy(dtype=float))
    keys = ("|".join(frame["ticker"].astype(str)) + "|"
            + "|".join(pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")))
    return {
        "rows": int(len(frame)),
        "tickers": sorted(frame["ticker"].unique().tolist()),
        "first_date": str(pd.to_datetime(frame["date"]).min().date()),
        "last_date": str(pd.to_datetime(frame["date"]).max().date()),
        "sha256": hashlib.sha256(values.tobytes() + keys.encode()).hexdigest(),
    }


def manifest(request: dict, prices: pd.DataFrame, engine_version: str, source_note: str) -> dict:
    """Everything needed to say whether two results came from the same run."""
    data = data_digest(prices)
    code = code_digest()
    inputs = {
        "request": request,
        "data_sha256": data["sha256"],
        "code_sha256": code["combined"],
        "engine_version": engine_version,
    }
    return {
        # Derived from the inputs, so it is reproducible rather than merely unique.
        "run_id": digest(inputs)[:16],
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine_version": engine_version,
        "request_sha256": digest(request),
        "request": request,
        "data": {**data, "source": source_note},
        "code_sha256": code["combined"],
        "code_files": code["files"],
        "environment": environment(),
    }
