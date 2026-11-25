"""MLflow experiment tracking and model packaging.

Works against any MLflow tracking server: a local folder for testing, or Databricks (where the
notebooks in ``databricks/`` call these functions). MLflow is an optional dependency, imported here only.

* ``log_backtest_run`` runs one rolling-origin backtest for a model configuration and logs it as an
  MLflow run: parameters, headline metrics, MAE at every horizon hour (as a stepped metric, so the
  MLflow UI draws error against horizon), the full summary as JSON and a horizon chart.
* ``CarbonForecastModel`` wraps a fitted forecaster as an MLflow pyfunc model: input is recent
  half-hourly history, output is the next 24 hours with an 80% interval.
* ``log_forecaster`` packages a fitted forecaster with a signature, input example and its source code,
  optionally registering it (e.g. to Unity Catalog).
"""

from __future__ import annotations

import json
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from . import config
from .data import clean_series
from .evaluate import backtest
from .model import DEFAULT_GBM_PARAMS, CarbonForecaster

PACKAGE_DIR = Path(__file__).resolve().parent

# Ablations that reproduce the README's model-development story on any data window.
ABLATIONS: dict[str, dict] = {
    "pooled_no_shrinkage": {"per_hour": False, "use_shrinkage": False},
    "pooled_with_absolute_level": {"per_hour": False, "use_shrinkage": False, "include_absolute_level": True},
    "per_hour_no_shrinkage": {"per_hour": True, "use_shrinkage": False},
    "per_hour_with_shrinkage": {},  # the production model
}

# A small hyperparameter sweep around the production settings.
GBM_SWEEP: dict[str, dict] = {
    "gbm_default": {},
    "gbm_faster_learning": {"learning_rate": 0.1, "max_iter": 200},
    "gbm_slower_learning": {"learning_rate": 0.03, "max_iter": 500},
    "gbm_smaller_trees": {"max_leaf_nodes": 15, "min_samples_leaf": 200},
    "gbm_larger_trees": {"max_leaf_nodes": 63, "min_samples_leaf": 50},
}


def model_params(model_kwargs: dict) -> dict:
    """All settings that define a model configuration, flattened for MLflow params."""
    m = CarbonForecaster(**model_kwargs)
    params = {
        "per_hour": m.per_hour,
        "use_shrinkage": m.use_shrinkage,
        "include_absolute_level": m.include_absolute_level,
        "origin_stride": m.origin_stride,
        "calib_days": m.calib_days,
        "interval_quantiles": f"{m.quantiles[0]}-{m.quantiles[1]}",
        "n_features": len(m.features),
    }
    params.update({f"gbm_{k}": v for k, v in {**DEFAULT_GBM_PARAMS, **m.gbm_params}.items()})
    return params


def summary_metrics(summary: dict) -> dict[str, float]:
    """Headline metrics from a backtest summary, flattened and named for the MLflow UI."""
    o = summary["overall"]
    out = {
        "mae": o["model"]["mae"],
        "rmse": o["model"]["rmse"],
        "bias": o["model"]["bias"],
        "mape_pct": o["model"]["mape_pct"],
        "skill_vs_persistence_pct": summary["skill_vs_persistence_pct"],
        "skill_vs_best_naive_pct": summary["skill_vs_best_naive_pct"],
        "baseline_persistence_mae": o["persistence"]["mae"],
        "baseline_yesterday_mae": o["seasonal_naive_24h"]["mae"],
        "baseline_last_week_mae": o["seasonal_naive_7d"]["mae"],
    }
    by_h = summary["mae_by_horizon_hour"]["model"]
    out["mae_1h"] = by_h[1]
    out["mae_24h"] = by_h[max(by_h)]
    if "interval" in summary:
        out["interval_coverage_pct"] = summary["interval"]["empirical_coverage_pct"]
        out["interval_mean_width"] = summary["interval"]["mean_width"]
    return {k: float(v) for k, v in out.items() if v is not None and np.isfinite(v)}


def _horizon_figure(summary: dict, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"model": "This model", "persistence": "Persistence", "seasonal_naive_24h": "Same time yesterday"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, label in labels.items():
        series = {int(h): v for h, v in summary["mae_by_horizon_hour"][key].items()}
        hours = sorted(series)
        ax.plot(hours, [series[h] for h in hours], label=label, linewidth=2.5 if key == "model" else 1.5)
    ax.set(xlabel="Forecast horizon (hours ahead)", ylabel="MAE (gCO2/kWh)", title=title)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def log_backtest_run(
    raw: pd.DataFrame,
    run_name: str,
    model_kwargs: dict | None = None,
    n_folds: int = 6,
    test_days: int = 7,
    tags: dict | None = None,
) -> dict:
    """Backtest one configuration and log it as an MLflow run (nested if a run is already active)."""
    model_kwargs = model_kwargs or {}
    nested = mlflow.active_run() is not None
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.set_tags({"stage": "backtest", "config_json": json.dumps(model_kwargs, sort_keys=True), **(tags or {})})
        y = raw["actual"].dropna()
        mlflow.log_params(
            {
                **model_params(model_kwargs),
                "n_folds": n_folds,
                "test_days": test_days,
                "data_from": str(y.index.min()),
                "data_to": str(y.index.max()),
                "n_observations": int(len(y)),
            }
        )
        _, summary = backtest(raw, n_folds=n_folds, test_days=test_days, model_kwargs=model_kwargs)
        mlflow.log_metrics(summary_metrics(summary))
        for h, v in summary["mae_by_horizon_hour"]["model"].items():
            mlflow.log_metric("mae_by_horizon_hour", float(v), step=int(h))
        for h, v in summary["mae_by_horizon_hour"]["persistence"].items():
            mlflow.log_metric("persistence_mae_by_horizon_hour", float(v), step=int(h))
        mlflow.log_dict(json.loads(json.dumps(summary, default=str)), "backtest_summary.json")
        try:
            mlflow.log_figure(_horizon_figure(summary, run_name), "mae_by_horizon.png")
        except ImportError:
            pass
        summary["run_id"] = run.info.run_id
    return summary


def run_grid(
    raw: pd.DataFrame,
    configs: dict[str, dict],
    parent_run_name: str,
    n_folds: int = 6,
    test_days: int = 7,
) -> pd.DataFrame:
    """Log every configuration as a child run of one parent run and return a comparison table."""
    records = []
    with mlflow.start_run(run_name=parent_run_name) as parent:
        mlflow.set_tag("stage", "grid")
        for name, kwargs in configs.items():
            s = log_backtest_run(
                raw, name, kwargs, n_folds=n_folds, test_days=test_days, tags={"grid": parent_run_name}
            )
            records.append({"config": name, "run_id": s["run_id"], **summary_metrics(s)})
        table = pd.DataFrame(records).sort_values("mae").reset_index(drop=True)
        best = table.iloc[0]
        mlflow.log_metrics({"best_mae": float(best["mae"]), "best_bias": float(best["bias"])})
        mlflow.set_tag("best_config", best["config"])
        mlflow.log_table(table, "comparison.json")
        table.attrs["parent_run_id"] = parent.info.run_id
    return table


class CarbonForecastModel(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc wrapper.

    Input: a DataFrame of recent half-hourly history with columns ``from`` (period start, UTC) and
    ``actual`` (gCO2/kWh). At least 9 days are needed to build the weekly features.
    Output: one row per half-hour for the next 24 hours: ``target_time``, ``horizon``, ``point``,
    ``lower``, ``upper``.
    """

    def load_context(self, context):
        self.model = CarbonForecaster.load(Path(context.artifacts["model"]))

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        df = model_input.copy()
        times = pd.to_datetime(df["from"], utc=True)
        history = pd.DataFrame({"actual": df["actual"].astype(float).to_numpy()}, index=pd.DatetimeIndex(times))
        history = history[~history.index.duplicated(keep="last")].sort_index()
        fc = self.model.forecast(clean_series(history))
        out = fc[["target_time", "horizon", "point", "lower", "upper"]].reset_index(drop=True)
        out["target_time"] = pd.to_datetime(out["target_time"], utc=True).dt.tz_localize(None)
        out["horizon"] = out["horizon"].astype("int64")
        return out


def _requirements() -> list[str]:
    reqs = []
    for pkg in ("pandas", "numpy", "scikit-learn", "joblib", "holidays"):
        try:
            reqs.append(f"{pkg}=={version(pkg)}")
        except PackageNotFoundError:
            reqs.append(pkg)
    return reqs


def history_example(y: pd.Series, days: int = config.HISTORY_DAYS_FOR_INFERENCE) -> pd.DataFrame:
    """The last ``days`` of history in the pyfunc input format."""
    recent = y.dropna().loc[y.dropna().index.max() - pd.Timedelta(days=days) :]
    return pd.DataFrame({"from": recent.index.tz_convert("UTC").tz_localize(None), "actual": recent.to_numpy()})


def log_forecaster(
    model: CarbonForecaster,
    y: pd.Series,
    registered_model_name: str | None = None,
    artifact_name: str = "carbon_forecaster",
):
    """Log a fitted forecaster as a pyfunc model, with signature and input example; optionally register it."""
    example = history_example(y)
    output = CarbonForecastModel()
    output.model = model
    signature = mlflow.models.infer_signature(example, output.predict(None, example))

    with tempfile.TemporaryDirectory() as tmp:
        path = model.save(Path(tmp) / "model.joblib")
        info = mlflow.pyfunc.log_model(
            artifact_name,
            python_model=CarbonForecastModel(),
            artifacts={"model": str(path)},
            code_paths=[str(PACKAGE_DIR)],
            signature=signature,
            input_example=example,
            pip_requirements=_requirements(),
            registered_model_name=registered_model_name,
        )
    return info
