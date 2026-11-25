# Databricks notebook source
# MAGIC %md
# MAGIC Shared setup, run from the other notebooks with `%run ./_common`. Puts the project's `src/` folder on the
# MAGIC Python path (the repo is a Databricks Git folder) and defines names used everywhere.

# COMMAND ----------

import os
import sys
from pathlib import Path

REPO_ROOT = Path(os.getcwd()).parent
sys.path.insert(0, str(REPO_ROOT / "src"))

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "carbon")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
TABLE = f"{CATALOG}.{SCHEMA}.intensity_halfhourly"
FORECAST_TABLE = f"{CATALOG}.{SCHEMA}.forecasts"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.carbon_forecaster"

spark.conf.set("spark.sql.session.timeZone", "UTC")  # all timestamps are UTC
USER = spark.sql("SELECT current_user()").first()[0]
EXPERIMENT = f"/Users/{USER}/carbon-forecaster"


def load_raw():
    """The Delta table as the pandas frame the forecasting package expects (UTC index named 'from')."""
    import pandas as pd

    pdf = spark.table(TABLE).select("period_start", "actual", "neso_forecast").toPandas()
    pdf["from"] = pd.to_datetime(pdf["period_start"], utc=True)
    return pdf.set_index("from").sort_index()[["actual", "neso_forecast"]].rename(columns={"neso_forecast": "forecast"})
