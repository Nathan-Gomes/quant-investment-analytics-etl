"""The web service: one analysis endpoint, plus the static interface.

Run it with::

    uvicorn app.server:api --reload

Then open http://127.0.0.1:8000. With no network access, set
``PORTFOLIO_LAB_SOURCE=bundled`` to work from the frozen dataset.
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import marketdata
from .analysis import ENGINE_VERSION, analyze
from .engine import SCHEDULES, SCHEMES, Portfolio, Settings

STATIC = Path(__file__).resolve().parent / "static"
DEFAULT_SOURCE = os.environ.get("PORTFOLIO_LAB_SOURCE", "auto")
# Set when the interface is served from another origin, such as a static site
# calling this API: PORTFOLIO_LAB_ALLOWED_ORIGINS="https://example.com"
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("PORTFOLIO_LAB_ALLOWED_ORIGINS", "").split(",") if o.strip()]
logger = logging.getLogger("portfolio_lab")

api = FastAPI(title="Portfolio Lab", version=ENGINE_VERSION,
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
        "schedules": list(SCHEDULES),
        "schemes": list(SCHEMES),
    }


@api.get("/api/universe")
def universe() -> dict:
    """The tickers available with no network connection."""
    frame = marketdata.bundled_universe()
    return {"bundled": frame.to_dict(orient="records")}


@api.post("/api/analyze")
def run_analysis(request: AnalysisRequest) -> JSONResponse:
    tickers: list[str] = []
    for portfolio in request.portfolios:
        if portfolio.scheme not in SCHEMES:
            raise HTTPException(400, f"Unknown weighting scheme '{portfolio.scheme}'.")
        # A zero weight is not a holding, so its ticker never enters the universe
        # and cannot fail a run over a symbol that carries nothing.
        portfolio.weights = {
            t.strip().upper(): w for t, w in portfolio.weights.items()
            if t.strip() and (portfolio.scheme != "custom" or w > 0)
        }
        if not portfolio.weights:
            raise HTTPException(400, f"{portfolio.name} needs at least one holding with a weight above zero.")
        tickers.extend(portfolio.weights)
    if request.benchmark:
        tickers.append(request.benchmark.strip().upper())
    tickers = list(dict.fromkeys(t for t in tickers if t))
    if not tickers:
        raise HTTPException(400, "Add at least one holding.")

    end = request.end or date.today().isoformat()
    start = request.start or (date.fromisoformat(end) - timedelta(days=365 * 10 + 3)).isoformat()
    if start >= end:
        raise HTTPException(400, "The start date must come before the end date.")

    source = request.source or DEFAULT_SOURCE
    try:
        priceset = marketdata.load(tickers, start, end, source)
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
        payload = analyze(
            priceset.prices,
            [Portfolio(name=p.name, weights=dict(p.weights), scheme=p.scheme)
             for p in request.portfolios],
            settings,
            metadata=priceset.metadata,
            start=start,
            end=end,
            source_note=priceset.note,
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:  # noqa: BLE001
        logger.exception("Analysis failed")
        raise HTTPException(500, f"The analysis did not finish: {error}") from error
    return JSONResponse(payload)


@api.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


api.mount("/", StaticFiles(directory=STATIC), name="static")


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run("app.server:api", host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", 8000)), reload=False)


if __name__ == "__main__":
    main()
