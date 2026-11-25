# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Register the best configuration in Unity Catalog
# MAGIC
# MAGIC Finds the backtest run with the lowest MAE, refits that configuration on all the data, logs it as an MLflow
# MAGIC pyfunc model (with signature, input example, pinned requirements and the package source) and registers it
# MAGIC as `workspace.carbon.carbon_forecaster` with the alias **champion**. A new version only takes the alias if
# MAGIC it backtested at least as well as the current champion.

# COMMAND ----------

# MAGIC %pip install holidays

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

import json

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from carbon_forecast.data import clean_series
from carbon_forecast.model import CarbonForecaster
from carbon_forecast.tracking import log_forecaster, model_params

mlflow.set_experiment(EXPERIMENT)
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

runs = mlflow.search_runs(
    experiment_names=[EXPERIMENT],
    filter_string="tags.stage = 'backtest' AND params.n_folds = '6'",  # compare like with like
    order_by=["metrics.mae ASC"],
)
if runs.empty:
    raise RuntimeError("No backtest runs found: run notebook 01 first")
best = runs.dropna(subset=["tags.config_json"]).iloc[0]
config = json.loads(best["tags.config_json"])
print(f"Best run: {best['tags.mlflow.runName']}  MAE {best['metrics.mae']:.2f}  config {config}")

# COMMAND ----------

raw = load_raw()
y = clean_series(raw)
with mlflow.start_run(run_name="champion_fit") as run:
    mlflow.set_tags({"stage": "final_fit", "source_backtest_run": best["run_id"], "config_json": json.dumps(config)})
    mlflow.log_params(model_params(config))
    mlflow.log_metrics({"backtest_mae": best["metrics.mae"], "backtest_bias": best["metrics.bias"]})
    model = CarbonForecaster(**config).fit(y)
    mlflow.log_dict(json.loads(json.dumps(model.metadata, default=str)), "model_metadata.json")
    info = log_forecaster(model, y, registered_model_name=MODEL_NAME)

version = info.registered_model_version
print(f"Registered {MODEL_NAME} version {version}")

# COMMAND ----------

def backtest_mae(model_version) -> float:
    """Backtest MAE recorded on the run that produced a registered version."""
    return client.get_run(model_version.run_id).data.metrics["backtest_mae"]


new = client.get_model_version(MODEL_NAME, version)
try:
    current = client.get_model_version_by_alias(MODEL_NAME, "champion")
except MlflowException:
    current = None

if current is None or backtest_mae(new) <= backtest_mae(current):
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    print(f"Version {version} is now champion")
else:
    client.set_registered_model_alias(MODEL_NAME, "challenger", version)
    print(f"Kept version {current.version} as champion; version {version} is challenger")

client.update_model_version(
    MODEL_NAME, version,
    description=f"Config {config}. Backtest MAE {best['metrics.mae']:.2f} gCO2/kWh "
                f"({best['metrics.skill_vs_best_naive_pct']:.1f}% better than the best naive baseline).",
)

# COMMAND ----------

# Smoke test: load by alias and forecast from the last 9 days.
from carbon_forecast.tracking import history_example

champion = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
display(champion.predict(history_example(y)))
