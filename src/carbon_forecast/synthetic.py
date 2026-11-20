"""Synthetic carbon-intensity-like data for tests and offline demos (not used for reported results)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_intensity(days: int = 120, seed: int = 0, end: str = "2026-06-01T00:00Z") -> pd.DataFrame:
    """Half-hourly series with daily and weekly cycles, slow weather-like swings and noise.

    Returns a frame shaped like ``data.parse_intensity`` output (``actual``, ``forecast``).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp(end), periods=days * 48, freq="30min", name="from")
    t = np.arange(len(idx))
    daily = 40 * np.sin(2 * np.pi * (t % 48) / 48 - np.pi / 2)
    weekly = -15 * (idx.dayofweek >= 5)
    # AR(1) "wind" component that moves intensity over several days.
    wind = np.zeros(len(idx))
    for i in range(1, len(idx)):
        wind[i] = 0.995 * wind[i - 1] + rng.normal(0, 3)
    actual = np.clip(180 + daily + weekly + wind + rng.normal(0, 5, len(idx)), 10, None)
    forecast = actual + rng.normal(0, 8, len(idx))
    return pd.DataFrame({"actual": actual, "forecast": forecast}, index=idx)
