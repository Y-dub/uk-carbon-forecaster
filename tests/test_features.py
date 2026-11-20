import numpy as np
import pandas as pd
import pytest

from carbon_forecast.features import FEATURES, build_dataset, training_rows


def test_no_feature_uses_data_after_origin(series):
    """Changing every value after the origin must not change any feature for that origin."""
    origin = series.index[-200]
    before = build_dataset(series, origins=pd.DatetimeIndex([origin]))
    perturbed = series.copy()
    perturbed[perturbed.index > origin] += 1000
    after = build_dataset(perturbed, origins=pd.DatetimeIndex([origin]))
    pd.testing.assert_frame_equal(before[FEATURES], after[FEATURES])
    assert not np.allclose(before["target"], after["target"])  # sanity: the target did change


def test_horizons_beyond_24h_rejected(series):
    with pytest.raises(ValueError):
        build_dataset(series, origins=series.index[-5:], horizons=range(1, 50))


def test_target_alignment(series):
    origin = series.index[-100]
    rows = build_dataset(series, origins=pd.DatetimeIndex([origin]), horizons=[1, 48])
    assert rows["target"].iloc[0] == series.loc[origin + pd.Timedelta("30min")]
    assert rows["target"].iloc[1] == series.loc[origin + pd.Timedelta("24h")]
    assert rows["lag_0"].iloc[0] == series.loc[origin]


def test_calendar_uses_uk_local_time():
    # 2026-07-01 12:00 UTC is 13:00 BST -> slot 26
    idx = pd.date_range("2026-06-20", "2026-07-02", freq="30min", tz="UTC")
    y = pd.Series(100.0, index=idx)
    rows = build_dataset(y, origins=pd.DatetimeIndex([pd.Timestamp("2026-07-01T11:30Z")]), horizons=[1])
    assert rows["slot"].iloc[0] == 26


def test_training_rows_complete(series):
    rows = training_rows(series, origin_stride=4)
    assert len(rows) > 1000
    assert rows[FEATURES + ["target"]].notna().all().all()
