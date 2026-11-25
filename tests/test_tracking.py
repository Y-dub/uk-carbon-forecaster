import numpy as np
import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")


from carbon_forecast.model import CarbonForecaster  # noqa: E402
from carbon_forecast.tracking import (  # noqa: E402
    ABLATIONS,
    history_example,
    log_backtest_run,
    log_forecaster,
    model_params,
    run_grid,
)


@pytest.fixture()
def tracking(tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_registry_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    # Artifacts default to ./mlruns; keep them inside the test's temporary folder.
    mlflow.create_experiment("test", artifact_location=(tmp_path / "artifacts").as_uri())
    mlflow.set_experiment("test")
    yield
    mlflow.set_tracking_uri(None)


@pytest.mark.parametrize("kwargs", list(ABLATIONS.values()))
def test_ablation_configs_fit_and_forecast(series, kwargs):
    m = CarbonForecaster(origin_stride=8, **kwargs).fit(series)
    fc = m.forecast(series)
    assert len(fc) == 48 and fc["point"].notna().all()
    expected_models = 24 if m.per_hour else 1
    assert len(m.models) == expected_models
    if not m.use_shrinkage:
        assert set(m.shrinkage.values()) == {1.0}


def test_absolute_level_adds_feature():
    assert "lag_0" in CarbonForecaster(include_absolute_level=True).features
    assert "lag_0" not in CarbonForecaster().features


def test_model_params_merge_gbm_overrides():
    p = model_params({"gbm_params": {"learning_rate": 0.1}})
    assert p["gbm_learning_rate"] == 0.1 and p["gbm_max_iter"] == 300


def test_backtest_run_logs_metrics_and_horizon_curve(raw, tracking):
    summary = log_backtest_run(raw, "unit", {"origin_stride": 8}, n_folds=1, test_days=3)
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(summary["run_id"])
    assert run.data.metrics["mae"] == pytest.approx(summary["overall"]["model"]["mae"])
    assert run.data.params["per_hour"] == "True"
    assert run.data.tags["config_json"] == '{"origin_stride": 8}'
    history = client.get_metric_history(summary["run_id"], "mae_by_horizon_hour")
    assert sorted(m.step for m in history) == list(range(1, 25))


def test_grid_ranks_configs_by_mae(raw, tracking):
    configs = {"a": {"origin_stride": 8}, "b": {"origin_stride": 8, "use_shrinkage": False}}
    table = run_grid(raw, configs, "grid", n_folds=1, test_days=3)
    assert list(table["mae"]) == sorted(table["mae"])
    assert set(table["config"]) == {"a", "b"}


def test_pyfunc_roundtrip_matches_model(fitted_model, series, tracking):
    with mlflow.start_run():
        info = log_forecaster(fitted_model, series, registered_model_name="carbon_test")
    loaded = mlflow.pyfunc.load_model(info.model_uri)
    example = history_example(series)
    out = loaded.predict(example)
    assert list(out.columns) == ["target_time", "horizon", "point", "lower", "upper"]
    assert len(out) == 48
    np.testing.assert_allclose(out["point"], fitted_model.forecast(series)["point"])
    # Registered and loadable by name.
    reg = mlflow.pyfunc.load_model("models:/carbon_test/1")
    assert isinstance(reg.predict(example), pd.DataFrame)
