"""FastAPI service: live 24-hour forecasts of UK national grid carbon intensity."""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from . import __version__, config
from .data import UpstreamError, clean_series, fetch_official_forecast, fetch_recent
from .model import CarbonForecaster
from .synthetic import synthetic_intensity

from fastapi.responses import RedirectResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("carbon_forecast.api")


# ---------- response schemas ----------

class ForecastPoint(BaseModel):
    period_start: datetime = Field(description="Start of the half-hour period (UTC)")
    period_end: datetime
    horizon_hours: float
    predicted: float = Field(description="Predicted carbon intensity, gCO2/kWh")
    lower: float | None = Field(None, description="Lower bound of the prediction interval")
    upper: float | None = Field(None, description="Upper bound of the prediction interval")
    neso_forecast: float | None = Field(None, description="NESO's official forecast for comparison")


class GreenestWindow(BaseModel):
    start: datetime
    end: datetime
    mean_predicted: float


class ForecastResponse(BaseModel):
    generated_at: datetime
    latest_observation: datetime = Field(description="Forecast origin: last period with an actual value")
    latest_actual: float
    units: str = "gCO2/kWh"
    interval: str | None
    model_trained_until: str
    greenest_window: GreenestWindow | None
    forecast: list[ForecastPoint]


class Health(BaseModel):
    status: str
    model_loaded: bool
    model_trained_until: str | None
    version: str = __version__


# ---------- app wiring ----------

class _State:
    def __init__(self) -> None:
        self.model: CarbonForecaster | None = None
        self.history_fetcher: Callable[[], pd.DataFrame] = lambda: fetch_recent(config.HISTORY_DAYS_FOR_INFERENCE)
        self.official_fetcher: Callable[[datetime], pd.Series] = fetch_official_forecast
        if os.getenv("CARBON_SYNTHETIC") == "1":  # offline local development only
            demo = synthetic_intensity(days=config.HISTORY_DAYS_FOR_INFERENCE)
            self.history_fetcher = lambda: demo
            self.official_fetcher = lambda start: pd.Series(dtype="float64")
        self.cache: dict[str, tuple[float, ForecastResponse]] = {}
        self.lock = threading.Lock()


state = _State()


def load_model() -> None:
    try:
        state.model = CarbonForecaster.load(config.MODEL_PATH)
        log.info("Loaded model trained until %s", state.model.metadata.get("trained_until"))
    except FileNotFoundError:
        state.model = None
        log.error("No model at %s; run `python -m carbon_forecast.train` first", config.MODEL_PATH)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if state.model is None:
        load_model()
    yield


app = FastAPI(
    title="UK Grid Carbon Intensity Forecaster",
    description=(
        "24-hour-ahead forecasts of UK national electricity carbon intensity (gCO2/kWh) from a gradient "
        "boosting model with split-conformal prediction intervals. Data: National Grid ESO / NESO Carbon "
        "Intensity API (CC BY 4.0)."
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    log.info("%s %s -> %d in %.0f ms", request.method, request.url.path, response.status_code,
             (time.perf_counter() - start) * 1000)
    return response


def _require_model() -> CarbonForecaster:
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return state.model


def _greenest(df: pd.DataFrame, window_steps: int) -> GreenestWindow | None:
    if len(df) < window_steps:
        return None
    rolling = df["point"].rolling(window_steps).mean().to_numpy()
    end_i = int(np.nanargmin(rolling))
    start_i = end_i - window_steps + 1
    # target_time is the start of each predicted half-hour.
    start = pd.Timestamp(df["target_time"].iloc[start_i])
    end = pd.Timestamp(df["target_time"].iloc[end_i]) + pd.Timedelta(config.FREQ)
    return GreenestWindow(start=start, end=end, mean_predicted=round(float(rolling[end_i]), 1))


def _build_forecast(model: CarbonForecaster, hours: int, window_hours: int) -> ForecastResponse:
    try:
        raw = state.history_fetcher()
    except UpstreamError as exc:
        raise HTTPException(status_code=503, detail="Carbon Intensity API unavailable") from exc
    y = clean_series(raw)
    try:
        fc = model.forecast(y, horizon_steps=hours * 2)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=f"Cannot build forecast: {exc}") from exc

    origin = pd.Timestamp(fc["origin"].iloc[0])
    try:
        # The API labels periods by their start, so the first forecast period starts one step after the origin.
        official = state.official_fetcher(origin.to_pydatetime() + timedelta(minutes=30))
    except UpstreamError:
        log.warning("Official forecast unavailable; continuing without it")
        official = pd.Series(dtype="float64")

    # Our rows are keyed by target_time = start of the predicted period.
    step = pd.Timedelta(config.FREQ)
    points = []
    for row in fc.itertuples(index=False):
        start = pd.Timestamp(row.target_time)
        neso = official.get(start, np.nan) if not official.empty else np.nan
        points.append(
            ForecastPoint(
                period_start=start,
                period_end=start + step,
                horizon_hours=row.horizon / 2,
                predicted=round(float(row.point), 1),
                lower=round(float(row.lower), 1) if hasattr(row, "lower") else None,
                upper=round(float(row.upper), 1) if hasattr(row, "upper") else None,
                neso_forecast=None if pd.isna(neso) else float(neso),
            )
        )
    return ForecastResponse(
        generated_at=datetime.now(UTC),
        latest_observation=origin,
        latest_actual=round(float(y.loc[origin]), 1),
        interval=model.metadata.get("interval"),
        model_trained_until=model.metadata.get("trained_until", "unknown"),
        greenest_window=_greenest(fc, window_hours * 2),
        forecast=points,
    )


# ---------- endpoints ----------

# @app.get("/", include_in_schema=False)
# def root() -> dict:
#     return {
#         "service": app.title,
#         "docs": "/docs",
#         "endpoints": {"/forecast": "live forecast", "/model": "model card and backtest", "/health": "health"},
#     }

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")

@app.get("/health", response_model=Health)
def health() -> Health:
    m = state.model
    return Health(
        status="ok" if m else "degraded",
        model_loaded=m is not None,
        model_trained_until=m.metadata.get("trained_until") if m else None,
    )


@app.get("/model")
def model_card() -> dict:
    """Training metadata and rolling-origin backtest results for the deployed model."""
    return _require_model().metadata


@app.get("/forecast", response_model=ForecastResponse)
def forecast(
    hours: int = Query(24, ge=1, le=24, description="How many hours ahead to forecast"),
    window_hours: int = Query(2, ge=1, le=12, description="Length of the 'greenest window' to find"),
) -> ForecastResponse:
    """Forecast carbon intensity for the next `hours`, with prediction intervals and the lowest-carbon window."""
    model = _require_model()
    key = f"{hours}:{window_hours}"
    with state.lock:
        cached = state.cache.get(key)
        if cached and time.time() - cached[0] < config.FORECAST_CACHE_SECONDS:
            return cached[1]
        resp = _build_forecast(model, hours, window_hours)
        state.cache[key] = (time.time(), resp)
        return resp
