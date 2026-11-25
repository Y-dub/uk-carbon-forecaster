# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Set up storage and ingest half-hourly carbon intensity into Delta
# MAGIC
# MAGIC Creates a schema and a volume in Unity Catalog, reads the raw CSV you uploaded, checks it, and writes a
# MAGIC Delta table that the other notebooks read.
# MAGIC
# MAGIC **Before running:** run this notebook once to create the volume, upload `data/raw.csv` to it
# MAGIC (Catalog → `workspace` → `carbon` → Volumes → `raw` → **Upload to this volume**), then run it again.
# MAGIC Free Edition restricts outbound internet, so we load a file instead of calling the Carbon Intensity API.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "carbon")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
spark.conf.set("spark.sql.session.timeZone", "UTC")  # all timestamps are UTC

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw")
RAW_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/raw/raw.csv"
TABLE = f"{CATALOG}.{SCHEMA}.intensity_halfhourly"
print("Upload raw.csv to:", RAW_PATH)

# COMMAND ----------

from pyspark.sql import functions as F

raw = (
    spark.read.option("header", True).option("inferSchema", True).csv(RAW_PATH)
    .select(
        F.to_timestamp("from").alias("period_start"),  # UTC; the CSV carries a +00:00 offset
        F.col("actual").cast("double").alias("actual"),
        F.col("forecast").cast("double").alias("neso_forecast"),
    )
)
display(raw.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data checks
# MAGIC The model needs a regular half-hourly series. Count duplicates, nulls and missing half-hours before writing.

# COMMAND ----------

bounds = raw.agg(F.min("period_start").alias("start"), F.max("period_start").alias("end")).first()
expected = spark.sql(
    f"SELECT explode(sequence(timestamp'{bounds.start}', timestamp'{bounds.end}', interval 30 minutes)) AS period_start"
)
checks = {
    "rows": raw.count(),
    "first_period": str(bounds.start),
    "last_period": str(bounds.end),
    "duplicate_periods": raw.groupBy("period_start").count().filter("count > 1").count(),
    "null_actuals": raw.filter(F.col("actual").isNull()).count(),
    "missing_half_hours": expected.join(raw, "period_start", "left_anti").count(),
    "negative_actuals": raw.filter(F.col("actual") < 0).count(),
}
for k, v in checks.items():
    print(f"{k:>20}: {v}")
assert checks["duplicate_periods"] == 0, "Duplicate half-hours: deduplicate before training"
assert checks["negative_actuals"] == 0, "Carbon intensity cannot be negative"

# COMMAND ----------

(
    raw.dropDuplicates(["period_start"])
    .withColumn("ingested_at", F.current_timestamp())
    .write.mode("overwrite").option("overwriteSchema", True)
    .saveAsTable(TABLE)
)
spark.sql(
    f"COMMENT ON TABLE {TABLE} IS 'UK national carbon intensity (gCO2/kWh), half-hourly, from carbonintensity.org.uk'"
)
display(spark.table(TABLE).orderBy(F.desc("period_start")).limit(5))
