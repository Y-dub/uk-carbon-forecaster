from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
import requests

from carbon_forecast import data

SAMPLE = {
    "data": [
        {
            "from": "2026-01-01T00:00Z",
            "to": "2026-01-01T00:30Z",
            "intensity": {"forecast": 150, "actual": 148, "index": "moderate"},
        },
        {
            "from": "2026-01-01T00:30Z",
            "to": "2026-01-01T01:00Z",
            "intensity": {"forecast": 140, "actual": None, "index": "moderate"},
        },
    ]
}


def test_parse_intensity_shape_and_nulls():
    df = data.parse_intensity(SAMPLE)
    assert list(df.columns) == ["actual", "forecast"]
    assert str(df.index.tz) == "UTC"
    assert df["actual"].iloc[0] == 148
    assert np.isnan(df["actual"].iloc[1])


def test_parse_empty_payload():
    assert data.parse_intensity({"data": []}).empty


def test_clean_series_fills_short_gaps_only():
    idx = pd.date_range("2026-01-01", periods=20, freq="30min", tz="UTC")
    s = pd.Series(np.arange(20, dtype=float), index=idx)
    s.iloc[3:5] = np.nan     # 2-step gap: filled
    s.iloc[10:16] = np.nan   # 6-step gap: left missing
    out = data.clean_series(pd.DataFrame({"actual": s}), max_gap_steps=4)
    assert out.iloc[3:5].notna().all()
    assert out.iloc[3] == pytest.approx(3.0)
    assert out.iloc[10:16].isna().all()


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self):
        self.urls = []

    def get(self, url, timeout, headers):
        self.urls.append(url)
        return FakeResponse(SAMPLE)


def test_fetch_chunks_to_14_day_limit():
    session = FakeSession()
    start = datetime(2026, 1, 1, tzinfo=UTC)
    end = datetime(2026, 1, 31, tzinfo=UTC)
    data.fetch_intensity(start, end, session=session)
    assert len(session.urls) == 3  # 14 + 14 + 2 days
    assert session.urls[0].endswith("/intensity/2026-01-01T00:00Z/2026-01-15T00:00Z")


def test_fetch_raises_upstream_error_after_retries(monkeypatch):
    monkeypatch.setattr(data.time, "sleep", lambda s: None)

    class Broken:
        def get(self, *a, **k):
            raise requests.ConnectionError("down")

    with pytest.raises(data.UpstreamError):
        data.fetch_intensity(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2, tzinfo=UTC),
                             session=Broken())
