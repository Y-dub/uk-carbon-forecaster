"""Turn a half-hourly intensity series into a supervised, direct multi-horizon dataset.

One row = (forecast origin t, horizon h). Every feature uses only data observed at or before t,
so the same function serves training, backtesting and live inference without leakage.
"""

from __future__ import annotations

from functools import lru_cache

import holidays
import numpy as np
import pandas as pd

from . import config

STEP = pd.Timedelta(config.FREQ)

# Level-type inputs enter only relative to the latest value (lag_0). With the absolute level available,
# the model learns to pull forecasts towards the training-period average, which biases them whenever the
# grid is unusually clean or dirty (e.g. a cold, still November). Day-of-year terms are left out because
# one or two years of history cannot separate seasonality from that year's weather.
RELATIVE_TO_LATEST = [
    "lag_1",
    "lag_2",
    "lag_4",
    "same_slot_yesterday",
    "same_slot_last_week",
    "mean_3h",
    "mean_24h",
    "mean_7d",
]

FEATURES = [
    "horizon",
    *(f"{c}_rel" for c in RELATIVE_TO_LATEST),
    "std_24h",
    "change_24h",
    "slot",
    "slot_sin",
    "slot_cos",
    "day_of_week",
    "is_weekend",
    "is_holiday",
]


@lru_cache(maxsize=1)
def _uk_holidays() -> holidays.HolidayBase:
    return holidays.UK(subdiv="ENG", years=range(2017, 2041))


def _lookup(series: pd.Series, times: pd.DatetimeIndex) -> np.ndarray:
    return series.reindex(times).to_numpy(dtype="float64")


def build_dataset(
    y: pd.Series,
    origins: pd.DatetimeIndex | None = None,
    horizons: range | list[int] | None = None,
) -> pd.DataFrame:
    """Build feature rows for every (origin, horizon) pair.

    Parameters
    ----------
    y : regular half-hourly series of actual intensity (UTC index, NaN allowed).
    origins : forecast origins; defaults to every timestamp in ``y``.
    horizons : steps ahead (1 = next half-hour); defaults to 1..MAX_HORIZON.

    Returns a DataFrame with FEATURES plus ``origin``, ``target_time`` and ``target``
    (``target`` is NaN when the future value is unknown, e.g. at inference).
    """
    horizons = list(horizons or range(1, config.MAX_HORIZON + 1))
    if max(horizons) > config.STEPS_PER_DAY:
        raise ValueError("Horizons beyond 24h would make 'same_slot_yesterday' leak future data.")
    origins = y.index if origins is None else origins

    mean_3h = y.rolling(6, min_periods=6).mean()
    mean_24h = y.rolling(config.STEPS_PER_DAY, min_periods=config.STEPS_PER_DAY).mean()
    std_24h = y.rolling(config.STEPS_PER_DAY, min_periods=config.STEPS_PER_DAY).std()
    mean_7d = y.rolling(config.STEPS_PER_WEEK, min_periods=config.STEPS_PER_WEEK).mean()

    n_o, n_h = len(origins), len(horizons)
    origin_rep = origins.repeat(n_h)
    h_rep = np.tile(np.asarray(horizons), n_o)
    target_time = origin_rep + pd.to_timedelta(h_rep * 30, unit="min")

    lag_0 = _lookup(y, origin_rep)
    df = pd.DataFrame(
        {
            "origin": origin_rep,
            "target_time": target_time,
            "horizon": h_rep,
            "lag_0": lag_0,
            "lag_1": _lookup(y, origin_rep - STEP),
            "lag_2": _lookup(y, origin_rep - 2 * STEP),
            "lag_4": _lookup(y, origin_rep - 4 * STEP),
            "same_slot_yesterday": _lookup(y, target_time - pd.Timedelta(days=1)),
            "same_slot_last_week": _lookup(y, target_time - pd.Timedelta(days=7)),
            "mean_3h": _lookup(mean_3h, origin_rep),
            "mean_24h": _lookup(mean_24h, origin_rep),
            "mean_7d": _lookup(mean_7d, origin_rep),
            "std_24h": _lookup(std_24h, origin_rep),
            "change_24h": lag_0 - _lookup(y, origin_rep - pd.Timedelta(days=1)),
            "target": _lookup(y, target_time),
        }
    )

    for c in RELATIVE_TO_LATEST:
        df[f"{c}_rel"] = df[c] - df["lag_0"]

    tt = pd.DatetimeIndex(target_time)
    # Calendar features use UK local time, because demand follows local clocks (GMT/BST).
    local = tt.tz_convert("Europe/London")
    slot = (local.hour * 2 + local.minute // 30).to_numpy()
    df["slot"] = slot
    df["slot_sin"] = np.sin(2 * np.pi * slot / config.STEPS_PER_DAY)
    df["slot_cos"] = np.cos(2 * np.pi * slot / config.STEPS_PER_DAY)
    df["day_of_week"] = local.dayofweek.to_numpy()
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    hol = _uk_holidays()
    df["is_holiday"] = np.fromiter((d in hol for d in local.date), dtype=int, count=len(local))
    return df


def training_rows(y: pd.Series, origin_stride: int = 2) -> pd.DataFrame:
    """Complete rows (all features and target known), using every ``origin_stride``-th origin."""
    origins = y.index[:: max(1, origin_stride)]
    df = build_dataset(y, origins=origins)
    return df.dropna(subset=FEATURES + ["lag_0", "target"]).reset_index(drop=True)


def latest_origin(y: pd.Series) -> pd.Timestamp:
    observed = y.dropna()
    if observed.empty:
        raise ValueError("No observed values in series")
    return observed.index[-1]
