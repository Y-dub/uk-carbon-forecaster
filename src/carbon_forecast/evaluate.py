"""Rolling-origin (expanding window) backtest against simple and official baselines."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import config
from .data import clean_series
from .features import build_dataset
from .model import CarbonForecaster

log = logging.getLogger(__name__)

MODELS = ["model", "persistence", "seasonal_naive_24h", "seasonal_naive_7d", "neso_official"]


def _metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    err = pred - actual
    nonzero = np.abs(actual) > 1e-9
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape_pct": float(np.mean(np.abs(err[nonzero] / actual[nonzero])) * 100) if nonzero.any() else float("nan"),
        "bias": float(np.mean(err)),
    }


def backtest(
    raw: pd.DataFrame,
    n_folds: int = 6,
    test_days: int = 7,
    origin_stride: int = 2,
    model_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run an expanding-window backtest.

    ``raw`` has ``actual`` (and optionally ``forecast``, NESO's official forecast) indexed by UTC period start.
    For each fold the model is trained only on data before the fold's cutoff, then forecasts every
    ``origin_stride``-th origin in the following ``test_days`` days, 1-48 steps ahead.

    Returns (per-row predictions, summary dict).
    """
    y = clean_series(raw)
    official = raw["forecast"].reindex(y.index) if "forecast" in raw else pd.Series(np.nan, index=y.index)
    end = y.dropna().index.max()
    window = pd.Timedelta(days=test_days)

    results = []
    for k in range(n_folds, 0, -1):
        cutoff = end - k * window
        y_train = y.loc[: cutoff - pd.Timedelta(config.FREQ)]
        test_origins = y.loc[cutoff : cutoff + window - pd.Timedelta(days=1)].index[::origin_stride]
        if len(test_origins) == 0:
            continue
        log.info(
            "Fold cutoff %s: training on %d points, %d test origins",
            cutoff, y_train.notna().sum(), len(test_origins),
        )
        model = CarbonForecaster(**(model_kwargs or {})).fit(y_train)
        rows = build_dataset(y, origins=test_origins)
        rows = rows.dropna(subset=["target", "lag_0", "same_slot_yesterday", "same_slot_last_week"])
        if rows.empty:
            continue
        pred = model.predict_rows(rows)
        pred["persistence"] = pred["lag_0"]
        pred["seasonal_naive_24h"] = pred["same_slot_yesterday"]
        pred["seasonal_naive_7d"] = pred["same_slot_last_week"]
        pred["neso_official"] = official.reindex(pd.DatetimeIndex(pred["target_time"])).to_numpy()
        pred["fold_cutoff"] = cutoff
        results.append(pred)

    if not results:
        raise ValueError("Backtest produced no rows; fetch more history or reduce folds.")
    preds = pd.concat(results, ignore_index=True)
    return preds, summarise(preds)


def summarise(preds: pd.DataFrame) -> dict:
    out: dict = {"n_rows": int(len(preds)), "n_folds": int(preds["fold_cutoff"].nunique())}

    actual = preds["target"].to_numpy()
    out["overall"] = {"model": _metrics(actual, preds["point"].to_numpy())}
    for name in MODELS[1:4]:
        out["overall"][name] = _metrics(actual, preds[name].to_numpy())

    has_official = preds["neso_official"].notna()
    out["official_forecast_coverage_pct"] = float(has_official.mean() * 100)
    if has_official.any():
        sub = preds[has_official]
        out["vs_official_on_shared_rows"] = {
            "n_rows": int(len(sub)),
            "model": _metrics(sub["target"].to_numpy(), sub["point"].to_numpy()),
            "neso_official": _metrics(sub["target"].to_numpy(), sub["neso_official"].to_numpy()),
            "note": (
                "Historical 'forecast' values from the API are the latest NESO forecast for each period, "
                "which may have been issued much closer to delivery than our 1-24h horizons. "
                "Treat this comparison as favouring NESO."
            ),
        }

    persistence_mae = out["overall"]["persistence"]["mae"]
    out["skill_vs_persistence_pct"] = float((1 - out["overall"]["model"]["mae"] / persistence_mae) * 100)
    best_naive = min(out["overall"][m]["mae"] for m in MODELS[1:4])
    out["skill_vs_best_naive_pct"] = float((1 - out["overall"]["model"]["mae"] / best_naive) * 100)

    if {"lower", "upper"} <= set(preds.columns):
        inside = (preds["target"] >= preds["lower"]) & (preds["target"] <= preds["upper"])
        out["interval"] = {
            "nominal_coverage_pct": round((config.QUANTILES[1] - config.QUANTILES[0]) * 100, 1),
            "empirical_coverage_pct": float(inside.mean() * 100),
            "mean_width": float((preds["upper"] - preds["lower"]).mean()),
        }

    hours = np.ceil(preds["horizon"] / 2).astype(int)
    by_h = {}
    for name, col in [("model", "point")] + [(m, m) for m in MODELS[1:4]]:
        by_h[name] = (
            (preds[col] - preds["target"]).abs().groupby(hours).mean().round(2).to_dict()
        )
    out["mae_by_horizon_hour"] = {k: {int(h): v for h, v in d.items()} for k, d in by_h.items()}
    return out
