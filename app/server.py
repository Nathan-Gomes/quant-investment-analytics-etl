"""Strata's web service: the analysis endpoints, plus the static interface.

Run it with::

    uvicorn app.server:api --reload

Then open http://127.0.0.1:8000. With no network access, set
``STRATA_SOURCE=bundled`` to work from the frozen dataset.
"""

from __future__ import annotations

import json
import logging
import copy
from collections import OrderedDict
import os
import queue
import threading
import time
import uuid
from logging.handlers import RotatingFileHandler
from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import conformance, marketdata, observability, provenance, strategies
from .analysis import ENGINE_VERSION, analyze
from .engine import SCHEDULES, SCHEMES, Portfolio, Settings
from .optimize import OBJECTIVES

STATIC = Path(__file__).resolve().parent / "static"
# STRATA_* is the current spelling; the older PORTFOLIO_LAB_* names still work so
# that a deployment configured before the rename keeps running untouched.
def _setting(name: str, default: str) -> str:
    return os.environ.get(f"STRATA_{name}", os.environ.get(f"PORTFOLIO_LAB_{name}", default))


DEFAULT_SOURCE = _setting("SOURCE", "auto")
# Set when the interface is served from another origin, such as a static site
# calling this API: PORTFOLIO_LAB_ALLOWED_ORIGINS="https://example.com"
ALLOWED_ORIGINS = [o.strip() for o in _setting("ALLOWED_ORIGINS", "").split(",") if o.strip()]
logger = logging.getLogger("strata")

api = FastAPI(title="Strata", version=ENGINE_VERSION,
              description="Backtest and scenario analysis for user-defined portfolios.")


if ALLOWED_ORIGINS:
    api.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


class PortfolioRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    weights: dict[str, float] = Field(default_factory=dict)
    scheme: str = "custom"
    # Used only when scheme is "optimized".
    objective: str = "minimum_variance"
    estimation_days: int = Field(default=252, ge=60, le=2520)
    max_weight: float = Field(default=1.0, gt=0, le=1)
    estimator: str = "ledoit_wolf"
    volatility_target: float | None = None
    parameters: dict = Field(default_factory=dict)


class AnalysisRequest(BaseModel):
    portfolios: list[PortfolioRequest] = Field(min_length=1, max_length=6)
    start: str | None = None
    end: str | None = None
    benchmark: str | None = None
    initial_capital: float = 100_000
    transaction_cost_bps: float = 10
    risk_free_rate: float = 0.03
    rebalance: str = "monthly"
    horizon_years: float = 5
    paths: int = 5000
    block_days: int = 20
    seed: int = 42
    scenario_basis: str = "equal"
    source: str | None = None


@api.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        "default_source": DEFAULT_SOURCE,
        "cached_results": len(_RESULT_CACHE),
        "schedules": list(SCHEDULES),
        "schemes": list(SCHEMES),
        "objectives": sorted(strategies.REGISTRY),
    }


@api.get("/api/strategies")
def strategy_catalogue() -> dict:
    """The methodologies that are registered, straight from the registry.

    The interface builds its menu from this, so a researcher's new methodology
    appears in the app by registering it, with nothing else to edit.
    """
    return {"strategies": strategies.catalogue()}


_CONFORMANCE_CACHE: dict[str, dict] = {}
# Identical request, identical prices, identical code means identical output, so
# the second caller gets the first caller's answer. Bounded, because a payload is
# a couple of hundred kilobytes and this runs on a small instance.
_RESULT_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_RESULT_CACHE_LOCK = threading.Lock()
RESULT_CACHE_SIZE = max(0, int(_setting("RESULT_CACHE", "24")))


@api.get("/api/conformance")
def conformance_report(strategy: str | None = None) -> dict:
    """Run the conformance battery and report it, check by check.

    Cached per strategy: the battery is deterministic, so within one process the
    answer cannot change unless the code does.
    """
    names = [strategy] if strategy else sorted(strategies.REGISTRY)
    unknown = [n for n in names if n not in strategies.REGISTRY]
    if unknown:
        raise HTTPException(404, f"No strategy named '{unknown[0]}'.")
    for name in names:
        if name not in _CONFORMANCE_CACHE:
            _CONFORMANCE_CACHE[name] = conformance.evaluate(strategies.get(name)).to_dict()
    return {"reports": [_CONFORMANCE_CACHE[name] for name in names]}


@api.get("/api/universe")
def universe() -> dict:
    """The tickers available with no network connection."""
    frame = marketdata.bundled_universe()
    return {"bundled": frame.to_dict(orient="records")}


def _prepare(request: AnalysisRequest):
    """Validate the request and resolve it to prices, settings and portfolios."""
    tickers: list[str] = []
    for portfolio in request.portfolios:
        if portfolio.scheme not in SCHEMES:
            raise ValueError(f"Unknown weighting scheme '{portfolio.scheme}'.")
        if portfolio.scheme == "optimized" and portfolio.objective not in strategies.REGISTRY:
            raise ValueError(
                f"Unknown methodology '{portfolio.objective}'. "
                f"Registered: {', '.join(sorted(strategies.REGISTRY))}.")
        portfolio.weights = {
            t.strip().upper(): w for t, w in portfolio.weights.items()
            if t.strip() and (portfolio.scheme != "custom" or w > 0)
        }
        if not portfolio.weights:
            raise ValueError(f"{portfolio.name} needs at least one holding with a weight above zero.")
        tickers.extend(portfolio.weights)
    if request.benchmark:
        tickers.append(request.benchmark.strip().upper())
    tickers = list(dict.fromkeys(t for t in tickers if t))
    if not tickers:
        raise ValueError("Add at least one holding.")

    end = request.end or date.today().isoformat()
    start = request.start or (date.fromisoformat(end) - timedelta(days=365 * 10 + 3)).isoformat()
    if start >= end:
        raise ValueError("The start date must come before the end date.")

    settings = Settings(
        initial_capital=request.initial_capital,
        transaction_cost_bps=request.transaction_cost_bps,
        risk_free_rate=request.risk_free_rate,
        rebalance=request.rebalance,
        benchmark=request.benchmark.strip().upper() if request.benchmark else None,
        horizon_years=request.horizon_years,
        paths=request.paths,
        block_days=request.block_days,
        seed=request.seed,
        scenario_basis=request.scenario_basis,
    )
    portfolios = [
        Portfolio(name=p.name, weights=dict(p.weights), scheme=p.scheme,
                  objective=p.objective, estimation_days=p.estimation_days,
                  max_weight=p.max_weight, estimator=p.estimator,
                  volatility_target=p.volatility_target, parameters=dict(p.parameters))
        for p in request.portfolios
    ]
    return tickers, start, end, settings, portfolios


def _analyze(request: AnalysisRequest, progress=None) -> dict:
    # Begin before the provider is touched: a substituted profile is a
    # degradation of this request and belongs on this request's payload.
    observability.start()
    tickers, start, end, settings, portfolios = _prepare(request)
    source = request.source or DEFAULT_SOURCE
    if progress:
        progress({"phase": "prices", "detail": f"{len(tickers)} securities from "
                  + ("the bundled dataset" if source == "bundled" else "Yahoo Finance")})
    priceset = marketdata.load(tickers, start, end, source)

    payload_request = request.model_dump(exclude_none=False)
    key = provenance.digest({
        "request": payload_request,
        "window": [start, end],
        "source": [priceset.source, priceset.note],
        "metadata": priceset.metadata.to_json(orient="split", date_format="iso"),
        "data": provenance.data_digest(priceset.prices)["sha256"],
        "code": provenance.code_digest()["combined"],
    })
    with _RESULT_CACHE_LOCK:
        cached = _RESULT_CACHE.get(key) if RESULT_CACHE_SIZE else None
        if cached is not None:
            _RESULT_CACHE.move_to_end(key)
            cached = copy.deepcopy(cached)
    if cached is not None:
        cached["meta"]["result_cache_hit"] = True
        if progress:
            progress({"phase": "diagnostics", "detail": "identical to an earlier run; reusing it"})
        logger.info("run %s served from cache", cached["meta"].get("run_id"))
        return cached

    payload = analyze(
        priceset.prices, portfolios, settings,
        metadata=priceset.metadata, start=start, end=end,
        source_note=priceset.note, request=payload_request,
        progress=progress,
        source_warning=priceset.note if getattr(priceset, "degraded", False) else None,
    )
    payload["meta"]["result_cache_hit"] = False
    with _RESULT_CACHE_LOCK:
        if RESULT_CACHE_SIZE:
            _RESULT_CACHE[key] = copy.deepcopy(payload)
            while len(_RESULT_CACHE) > RESULT_CACHE_SIZE:
                _RESULT_CACHE.popitem(last=False)
    return payload


@api.post("/api/analyze/stream")
def run_analysis_stream(request: AnalysisRequest) -> StreamingResponse:
    """The same analysis, reporting each stage as it finishes.

    A request that takes twenty seconds and says nothing is indistinguishable
    from one that has hung. The work runs on a worker thread and pushes progress
    onto a queue that this generator drains, so the client sees the stage it is
    in rather than a spinner.

    Newline-delimited JSON rather than server-sent events, because the request
    is a POST and EventSource cannot make one.
    """
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        started = time.perf_counter()
        # A failed run has no run_id to quote, so the reference is minted up
        # front: whatever the user reports back, it points at a log line.
        reference = uuid.uuid4().hex[:12]
        try:
            payload = _analyze(request, progress=events.put)
            logger.info("run %s (ref %s) finished in %.0f ms: %s", payload["meta"].get("run_id"),
                        reference, (time.perf_counter() - started) * 1000, payload["meta"]["timings_ms"])
            events.put({"phase": "done", "payload": payload})
        except ValueError as error:
            logger.info("run ref %s rejected: %s", reference, error)
            events.put({"phase": "error", "message": str(error), "reference": reference})
        except Exception as error:  # noqa: BLE001
            logger.exception("Analysis failed (ref %s)", reference)
            events.put({"phase": "error", "reference": reference,
                        "message": f"The analysis did not finish (reference {reference}): {error}"})
        finally:
            events.put(None)

    threading.Thread(target=worker, name="analysis", daemon=True).start()

    def stream():
        yield json.dumps({"phase": "accepted"}) + "\n"
        while True:
            event = events.get()
            if event is None:
                break
            yield json.dumps(event, default=str) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@api.post("/api/analyze")
def run_analysis(request: AnalysisRequest) -> JSONResponse:
    reference = uuid.uuid4().hex[:12]
    try:
        payload = _analyze(request)
    except ValueError as error:
        logger.info("run ref %s rejected: %s", reference, error)
        raise HTTPException(400, str(error)) from error
    except Exception as error:  # noqa: BLE001
        logger.exception("Analysis failed (ref %s)", reference)
        raise HTTPException(500, f"The analysis did not finish (reference {reference}): {error}") from error
    return JSONResponse(payload)


@api.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


api.mount("/", StaticFiles(directory=STATIC), name="static")


def main() -> None:
    import uvicorn

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    # Containers want stdout, so a file is opt-in; when it is asked for it
    # rotates, because an unbounded log is one disk-full away from losing the
    # incident it was kept for.
    log_file = _setting("LOG_FILE", "")
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(RotatingFileHandler(path, maxBytes=5_000_000, backupCount=5, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, handlers=handlers,
                        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    uvicorn.run("app.server:api", host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", 8000)), reload=False)


if __name__ == "__main__":
    main()
