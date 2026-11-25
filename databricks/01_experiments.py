# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Experiments tracked in MLflow
# MAGIC
# MAGIC Two grids, each a parent run with one child run per configuration. Every child run is a 6-fold
# MAGIC rolling-origin backtest (one week per fold) and logs:
# MAGIC * **params**: model switches and gradient-boosting settings, data window
# MAGIC * **metrics**: MAE, RMSE, bias, skill vs persistence and the best naive baseline, interval coverage and width,
# MAGIC   and `mae_by_horizon_hour` with one step per forecast hour (chart it in the run page)
# MAGIC * **artifacts**: the full backtest summary (JSON) and an error-by-horizon chart
# MAGIC
# MAGIC 1. **Ablations** reproduce the model-development story: pooled model → absolute level added (control) →
# MAGIC    one model per horizon hour → plus learned shrinkage.
# MAGIC 2. **Gradient-boosting sweep** checks the production hyperparameters against nearby alternatives.
# MAGIC
# MAGIC Timing: each per-hour configuration fits 24 models per fold, about 8–10 minutes for a 6-fold backtest.
# MAGIC The ablations use 6 folds (about 35 minutes); the sweep uses 3 folds by default to keep it near 25 minutes.
# MAGIC Free Edition has a daily compute quota: if you hit it, rerun the remaining cell the next day.

# COMMAND ----------

# MAGIC %pip install holidays

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

import mlflow

from carbon_forecast.tracking import ABLATIONS, GBM_SWEEP, run_grid

dbutils.widgets.text("sweep_folds", "3")
SWEEP_FOLDS = int(dbutils.widgets.get("sweep_folds"))

mlflow.set_experiment(EXPERIMENT)
raw = load_raw()
print(f"{len(raw):,} half-hours from {raw.index.min()} to {raw.index.max()}")

# COMMAND ----------

ablations = run_grid(raw, ABLATIONS, parent_run_name="ablations")
display(ablations)

# COMMAND ----------

sweep = run_grid(
    raw,
    {name: {"gbm_params": p} for name, p in GBM_SWEEP.items()},
    parent_run_name="gbm_sweep",
    n_folds=SWEEP_FOLDS,
)
display(sweep)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare in the MLflow UI
# MAGIC Open **Experiments → carbon-forecaster**, tick the child runs, click **Compare**, and plot
# MAGIC `mae_by_horizon_hour` against step to see where each change helps. Sort by `mae` to find the winner,
# MAGIC which notebook 02 registers.
# MAGIC
# MAGIC Only compare runs with the same `n_folds`: a 3-fold backtest covers different weeks from a 6-fold one.
# MAGIC Notebook 02 therefore picks the champion among 6-fold runs.
