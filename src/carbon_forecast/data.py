"""Client for the National Grid ESO Carbon Intensity API (national, half-hourly).

Docs: https://carbon-intensity.github.io/api-definitions/
Data licence: CC BY 4.0.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta

import pandas as pd
import requests

from . import config

log = logging.getLogger(__name__)

_TS_FORMAT = "%Y-%m-%dT%H:%MZ"


class UpstreamError(RuntimeError):
    """Raised when the Carbon Intensity API cannot be reached or returns bad data."""


def _fmt(ts: datetime) -> str:
    return ts.astimezone(UTC).strftime(_TS_FORMAT)


def _get_json(url: str, session: requests.Session, retries: int = 3, timeout: float = 20.0) -> dict:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=timeout, headers={"Accept": "application/json"})
            resp.raise_for_status()
            payload = resp.json()
            if "data" not in payload:
                raise UpstreamError(f"Unexpected response shape from {url}: {payload!r:.200}")
            return payload
        except (requests.RequestException, ValueError, UpstreamError) as exc:
            last_exc = exc
            log.warning("Request failed (attempt %d/%d): %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(2 ** (attempt - 1))
    raise UpstreamError(f"Failed to fetch {url}") from last_exc


def parse_intensity(payload: dict) -> pd.DataFrame:
    """Turn an /intensity response into a DataFrame indexed by period start (UTC).

    Columns: ``actual`` and ``forecast`` in gCO2/kWh. Missing values are NaN.
    """
    rows = [
        {
            "from": item["from"],
            "actual": (item.get("intensity") or {}).get("actual"),
            "forecast": (item.get("intensity") or {}).get("forecast"),
        }
        for item in payload.get("data", [])
    ]
    if not rows:
        return pd.DataFrame(columns=["actual", "forecast"], index=pd.DatetimeIndex([], tz="UTC", name="from"))
    df = pd.DataFrame(rows)
    df["from"] = pd.to_datetime(df["from"], utc=True)
    df = df.set_index("from").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df[["actual", "forecast"]].astype("float64")


def fetch_intensity(start: datetime, end: datetime, session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch half-hourly national intensity between ``start`` and ``end``, chunked to the 14-day API limit."""
    if end <= start:
        raise ValueError("end must be after start")
    session = session or requests.Session()
    frames = []
    chunk_start = start
    step = timedelta(days=config.MAX_DAYS_PER_REQUEST)
    while chunk_start < end:
        chunk_end = min(chunk_start + step, end)
        url = f"{config.API_BASE_URL}/intensity/{_fmt(chunk_start)}/{_fmt(chunk_end)}"
        frames.append(parse_intensity(_get_json(url, session)))
        chunk_start = chunk_end
    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def fetch_recent(days: int, now: datetime | None = None, session: requests.Session | None = None) -> pd.DataFrame:
    now = now or datetime.now(UTC)
    return fetch_intensity(now - timedelta(days=days), now, session=session)


def fetch_official_forecast(start: datetime, session: requests.Session | None = None) -> pd.Series:
    """NESO's own forecast for the 48 hours after ``start`` (used for comparison in the API)."""
    session = session or requests.Session()
    url = f"{config.API_BASE_URL}/intensity/{_fmt(start)}/fw48h"
    return parse_intensity(_get_json(url, session))["forecast"]


def clean_series(df: pd.DataFrame, max_gap_steps: int = 4) -> pd.Series:
    """Regular half-hourly series of actual intensity.

    Short gaps (up to ``max_gap_steps`` half-hours) are linearly interpolated; longer gaps stay NaN so
    the feature builder can drop affected rows instead of training on invented data.
    """
    s = df["actual"].copy()
    if s.empty:
        return s
    full_index = pd.date_range(s.index.min(), s.index.max(), freq=config.FREQ, tz="UTC", name="from")
    s = s.reindex(full_index)
    missing = s.isna()
    # Length of the run of consecutive NaNs each missing point belongs to.
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    filled = s.interpolate(method="time", limit_area="inside")
    fill_mask = missing & (run_len <= max_gap_steps)
    s[fill_mask] = filled[fill_mask]
    return s
