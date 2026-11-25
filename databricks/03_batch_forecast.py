# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Batch forecast with the champion model and monitor accuracy
# MAGIC
# MAGIC Loads `carbon_forecaster@champion`, forecasts the next 24 hours from the latest 9 days in the Delta table,
# MAGIC appends the forecast to `forecasts`, then scores every earlier forecast whose actuals have since arrived.
# MAGIC Schedule it as a Job (Workflows → Create job → this notebook) after each data refresh.

# COMMAND ----------

# MAGIC %pip install holidays

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow import MlflowClient
from pyspark.sql import functions as F

from carbon_forecast.data import clean_series
from carbon_forecast.tracking import history_example

mlflow.set_registry_uri("databricks-uc")
version = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")

y = clean_series(load_raw())
fc = model.predict(history_example(y))
fc["issued_from"] = y.dropna().index.max().tz_localize(None)
fc["model_version"] = int(version)
fc["created_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)

(
    spark.createDataFrame(fc)
    .withColumnRenamed("target_time", "period_start")
    .write.mode("append").option("mergeSchema", True)
    .saveAsTable(FORECAST_TABLE)
)
display(spark.table(FORECAST_TABLE).orderBy(F.desc("created_at"), "horizon").limit(48))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Accuracy of past forecasts
# MAGIC Joins stored forecasts to actuals that have arrived since. On the first run this is empty; it fills in as
# MAGIC new data is ingested (rerun notebook 00 with a newer CSV).

# COMMAND ----------

display(spark.sql(f"""
    SELECT f.model_version,
           CEIL(f.horizon / 2)                              AS horizon_hour,
           COUNT(*)                                         AS n,
           ROUND(AVG(ABS(f.point - a.actual)), 2)           AS mae,
           ROUND(AVG(f.point - a.actual), 2)                AS bias,
           ROUND(AVG(CAST(a.actual BETWEEN f.lower AND f.upper AS INT)) * 100, 1) AS interval_coverage_pct
    FROM {FORECAST_TABLE} f
    JOIN {TABLE} a USING (period_start)
    GROUP BY 1, 2
    ORDER BY 1, 2
"""))
