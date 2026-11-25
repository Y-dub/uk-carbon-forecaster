# Running the experiments on Databricks with MLflow

The `databricks/` notebooks run the same forecasting package on Databricks: data in a Delta table,
every backtest tracked as an MLflow run, the best configuration registered in Unity Catalog with a
`champion` alias, and a batch job that forecasts with the champion and monitors its accuracy.

```
raw.csv ─► 00 ingest ─► Delta: intensity_halfhourly ─► 01 experiments ─► MLflow runs (params, metrics, charts)
                                         │                                         │
                                         │                                 02 register best
                                         │                                         ▼
                                         └──────────────► 03 batch forecast ◄─ UC model carbon_forecaster@champion
                                                                 │
                                                          Delta: forecasts ─► accuracy by horizon
```

Everything works on the **free** Databricks Free Edition. Its serverless compute restricts outbound
internet, so the data is uploaded as a file instead of fetched from the Carbon Intensity API.

## 1. Create a workspace (5 minutes)

1. Sign up at **databricks.com/learn/free-edition** with Google, Microsoft or email.
2. You get one workspace with a `workspace` catalog. No cluster to configure: notebooks run on serverless compute.

## 2. Add the repo as a Git folder

1. In the left sidebar choose **Workspace → Home → Create → Git folder**.
2. Git repository URL: `https://github.com/Y-dub/uk-carbon-forecaster`, provider GitHub, then **Create Git folder**.
   A public repo needs no credentials.
3. Open `databricks/` inside the new folder. The `.py` files show as notebooks.

After you push changes to GitHub, click the branch name at the top of a notebook and **Pull** to update.

## 3. Ingest the data (`00_setup_and_ingest`)

1. Open `00_setup_and_ingest` and click **Run all**. The first run creates the schema `workspace.carbon` and the
   volume `raw`, then stops at the CSV read because the file is not there yet. That is expected.
2. Go to **Catalog → workspace → carbon → Volumes → raw → Upload to this volume** and upload `data/raw.csv`
   from your laptop (the file the forecaster's training wrote).
3. **Run all** again. The data checks print row counts, duplicate and missing half-hours, and the notebook writes
   the Delta table `workspace.carbon.intensity_halfhourly`.

## 4. Run the experiments (`01_experiments`)

Click **Run all**. It installs `holidays`, then runs two grids, each as a parent MLflow run with child runs:

| Grid | Configurations | Folds | Time (approx.) |
| --- | --- | --- | --- |
| Ablations | pooled, pooled + absolute level (control), per hour, per hour + shrinkage | 6 | 35 min |
| Gradient-boosting sweep | default, faster/slower learning, smaller/larger trees | 3 | 25 min |

Free Edition has a daily compute quota. If a cell stops, rerun it the next day; finished runs stay logged.

**Look at the results:** left sidebar **Experiments → carbon-forecaster**.
- The table lists every run with `mae`, `bias`, `mae_1h`, `skill_vs_best_naive_pct`, `interval_coverage_pct`.
- Tick the four ablation runs → **Compare** → add a line chart of `mae_by_horizon_hour` against step. That chart
  shows why each change was kept: the per-hour model fixes the short horizons, shrinkage fixes the long ones.
- Each run's **Artifacts** tab has `mae_by_horizon.png` and the full `backtest_summary.json`.

Take screenshots of the runs table and the comparison chart for the README.

## 5. Register the champion (`02_register_champion`)

**Run all.** It picks the 6-fold run with the lowest MAE, refits that configuration on all the data, logs it
as an MLflow pyfunc model (signature, input example, pinned requirements, package source) and registers it as
`workspace.carbon.carbon_forecaster`. The new version takes the `champion` alias only if its backtest MAE is no
worse than the current champion's; otherwise it becomes `challenger`. The last cell loads the model by alias and
forecasts the next 24 hours.

See it under **Catalog → workspace → carbon → Models → carbon_forecaster**: versions, aliases, lineage to the run.

## 6. Batch forecast and monitor (`03_batch_forecast`)

**Run all.** It loads `carbon_forecaster@champion`, forecasts the next 24 hours from the latest data, appends the
result to `workspace.carbon.forecasts`, and scores earlier forecasts against actuals that have arrived since.

To make it a scheduled job: **Workflows → Create job**, task type Notebook, path `databricks/03_batch_forecast`,
serverless compute, then add a schedule. (The accuracy table only fills once newer actuals are ingested: download a
fresh CSV with `python -m carbon_forecast.train --data-csv data/raw.csv --days 365 --skip-backtest`, upload it and
rerun notebook 00.)

## What to say about it

- **Experiment tracking:** every model change is a logged, reproducible run with its config, data window and
  error at each horizon, so the README's development table can be regenerated rather than trusted.
- **Model registry and promotion:** Unity Catalog versions with `champion` / `challenger` aliases and a simple
  promotion rule based on backtest error.
- **Batch inference and monitoring:** forecasts stored in Delta and scored against actuals by horizon and model
  version, which is how drift would show up.
