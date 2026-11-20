# UK Grid Carbon Intensity Forecaster

Forecasts the carbon intensity of Great Britain's electricity (gCO2/kWh) up to 24 hours ahead, with
calibrated prediction intervals, and serves it as a production API.

**Live API:** `https://<your-service>.onrender.com/docs` &nbsp;·&nbsp; ![CI](https://github.com/Y-dub/uk-carbon-forecaster/actions/workflows/ci.yml/badge.svg)

Knowing when the grid will be cleanest lets flexible demand (EV charging, heat pumps, batteries, data
centre workloads) shift to low-carbon periods. The `/forecast` endpoint returns a half-hourly forecast
and the greenest upcoming window.

## Results

Rolling-origin backtest, <!-- FILL: n --> weekly folds, forecasts issued hourly for 1–24 hours ahead,
model retrained before each fold using only past data.

| Model | MAE (gCO2/kWh) | RMSE | MAPE |
| --- | --- | --- | --- |
| **Gradient boosting (this project)** | <!-- FILL --> | | |
| Same time yesterday | | | |
| Same time last week | | | |
| Persistence (last value) | | | |

- Skill vs best naive baseline: <!-- FILL -->%
- 80% prediction interval, empirical coverage: <!-- FILL -->% (mean width <!-- FILL --> gCO2/kWh)

![MAE by horizon](docs/backtest_mae_by_horizon.png)

> These numbers come from `python -m carbon_forecast.train` (see below) and are also served live at `/model`.

## Approach

**Data.** Half-hourly national actual intensity from the
[NESO Carbon Intensity API](https://carbon-intensity.github.io/api-definitions/) (CC BY 4.0), fetched in
14-day chunks (the API limit). Gaps of up to 2 hours are interpolated; longer gaps are left missing and
the affected rows are dropped rather than trained on invented values.

**Problem framing.** Direct multi-horizon forecasting: one row per (forecast origin, horizon), with the
horizon as a feature, so one model covers all 48 half-hour steps without compounding errors the way a
recursive forecaster would.

**Features** (all observable at the forecast origin; a unit test perturbs every future value and checks
no feature changes): recent lags, same half-hour yesterday and last week, 3h/24h rolling mean and
volatility, 24h change, and calendar features in UK local time (GMT/BST) including bank holidays.

**Target.** The change from the latest observed value, not the raw level. Tree ensembles cannot
extrapolate beyond levels seen in training, and the grid keeps decarbonising year on year.

**Model.** scikit-learn `HistGradientBoostingRegressor`. Built-in early stopping is deliberately off: its
validation split is random, and neighbouring (origin, horizon) rows overlap so heavily that it would be
validating on near-copies of training rows.

**Uncertainty.** Split-conformal intervals: a copy of the model is trained without the most recent 14
days, its errors on those days are measured for each horizon hour, and the 10th/90th percentiles of
those errors set the interval around the final model's forecast. An earlier version used separate
quantile-loss models; they were well calibrated in-sample but badly under-covered out of sample, which
is what motivated the switch.

**Evaluation.** Expanding-window backtest against persistence, same-time-yesterday and
same-time-last-week baselines, reporting MAE, RMSE, MAPE, bias, error by horizon and interval coverage.
The API's historical `forecast` field (NESO's own forecast) is also compared, with a caveat: it is the
latest forecast for each period, likely issued much closer to delivery than our horizons, so that
comparison favours NESO.

## API

| Endpoint | Description |
| --- | --- |
| `GET /forecast?hours=24&window_hours=2` | Half-hourly forecast with 80% intervals, NESO's official forecast alongside, and the lowest-carbon window |
| `GET /model` | Model card: training window, features, backtest metrics |
| `GET /health` | Liveness and model status |
| `GET /docs` | Interactive OpenAPI docs |

Forecasts are cached for 15 minutes. If the upstream API is down the service returns `503` rather than
serving stale or invented data.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                                                 # 20 tests, offline
python -m carbon_forecast.train --days 365 --folds 6 --data-csv data/raw.csv   # real data, ~5 min
uvicorn carbon_forecast.api:app --reload                               # http://127.0.0.1:8000/docs
```

Offline: `python -m carbon_forecast.train --synthetic --days 90` then `CARBON_SYNTHETIC=1 uvicorn ...`.

## Deployment and MLOps

- **Docker:** the image trains on the latest 365 days at build time, so every deploy ships a fresh model
  with its backtest baked into `/model`. Runs as a non-root user with a container health check.
- **CI (GitHub Actions):** lint (ruff), tests, Docker build with synthetic training, and a smoke test
  that starts the container and calls `/health` and `/forecast`.
- **Hosting:** Render, from `render.yaml`. Auto-deploys on push to `main`.
- **Retraining:** a scheduled workflow triggers a Render redeploy every Monday, which retrains the model
  on the latest data.

## Project structure

```
src/carbon_forecast/
  data.py       API client (chunking, retries), cleaning
  features.py   leakage-safe direct multi-horizon features
  model.py      gradient boosting + split-conformal intervals
  evaluate.py   rolling-origin backtest and baselines
  train.py      CLI: fetch, backtest, train, save model + metrics
  api.py        FastAPI service
tests/          data, features (incl. leakage test), model, API
```

## Limitations and next steps

- No weather inputs. Wind and solar output drive most short-term variation, so adding NWP wind speed
  and solar irradiance forecasts is the obvious next gain, and would be needed to compete with NESO.
- Generation-mix data (`/generation` endpoint) as additional features and regional forecasts (`/regional`).
- Intervals assume recent errors resemble future errors; adaptive conformal methods would handle
  regime changes (e.g. a sudden wind drought) better.
- The free Render instance sleeps when idle, so the first request after a pause takes ~30–60 seconds.

## Data attribution

Carbon intensity data © National Energy System Operator (NESO), Carbon Intensity API, licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
