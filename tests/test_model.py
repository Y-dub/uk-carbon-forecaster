import numpy as np

from carbon_forecast.evaluate import backtest
from carbon_forecast.model import CarbonForecaster


def test_forecast_shape_and_bounds(fitted_model, series):
    fc = fitted_model.forecast(series)
    assert len(fc) == 48
    assert (fc["lower"] <= fc["point"]).all() and (fc["point"] <= fc["upper"]).all()
    assert (fc[["point", "lower", "upper"]] >= 0).all().all()
    assert fc["horizon"].tolist() == list(range(1, 49))


def test_intervals_calibrated_for_every_horizon_hour(fitted_model):
    assert sorted(fitted_model.interval_offsets) == list(range(1, 25))
    for lo, hi in fitted_model.interval_offsets.values():
        assert lo < hi


def test_save_load_roundtrip(fitted_model, series, tmp_path):
    path = fitted_model.save(tmp_path / "m.joblib")
    loaded = CarbonForecaster.load(path)
    np.testing.assert_allclose(loaded.forecast(series)["point"], fitted_model.forecast(series)["point"])


def test_backtest_beats_persistence(raw):
    preds, summary = backtest(raw, n_folds=2, test_days=5, origin_stride=4, model_kwargs={"origin_stride": 4})
    assert summary["n_folds"] == 2
    assert summary["skill_vs_persistence_pct"] > 0
    assert 0 <= summary["interval"]["empirical_coverage_pct"] <= 100
    # Every fold's test origins come after its training cutoff.
    assert (preds["origin"] >= preds["fold_cutoff"]).all()
