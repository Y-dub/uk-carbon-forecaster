"""Central configuration. Values can be overridden with environment variables."""

from __future__ import annotations

import os
from pathlib import Path

API_BASE_URL = os.getenv("CARBON_API_BASE_URL", "https://api.carbonintensity.org.uk")

# The /intensity/{from}/{to} endpoint accepts at most 14 days per request.
MAX_DAYS_PER_REQUEST = 14

# Half-hourly settlement periods.
FREQ = "30min"
STEPS_PER_DAY = 48
STEPS_PER_WEEK = 7 * STEPS_PER_DAY

# Forecast up to 24 hours ahead (48 half-hour steps).
MAX_HORIZON = STEPS_PER_DAY

# History needed to build features for one forecast origin (one week of lags + buffer).
HISTORY_DAYS_FOR_INFERENCE = 9

# Prediction interval: 10th to 90th percentile (an 80% interval).
QUANTILES = (0.1, 0.9)

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parents[2] / "models"))
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# How long the API caches a forecast before refetching live data.
FORECAST_CACHE_SECONDS = int(os.getenv("FORECAST_CACHE_SECONDS", "900"))

RANDOM_STATE = 42
