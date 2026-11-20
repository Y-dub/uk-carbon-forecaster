"""Direct multi-horizon gradient boosting with learned shrinkage and split-conformal intervals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config
from .features import FEATURES, build_dataset, latest_origin, training_rows


def _regressor() -> HistGradientBoostingRegressor:
    # Early stopping is off on purpose: its validation split is random, and neighbouring
    # (origin, horizon) rows overlap heavily, so it would validate on near-copies of training rows.
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=config.RANDOM_STATE,
    )


def horizon_hour(horizon: pd.Series | np.ndarray) -> np.ndarray:
    """Map half-hour steps 1..48 to forecast hours 1..24."""
    return np.ceil(np.asarray(horizon) / 2).astype(int)


def _fit_per_hour(rows: pd.DataFrame) -> dict[int, HistGradientBoostingRegressor]:
    """One regressor per horizon hour, each predicting the change from the latest value.

    A single pooled model shares leaves across horizons, so short horizons (small changes) inherit
    the large changes typical of long horizons. Separate models remove that interference.
    """
    hh = horizon_hour(rows["horizon"])
    return {
        int(h): _regressor().fit(g[FEATURES], g["target"] - g["lag_0"])
        for h, g in rows.groupby(hh)
    }


def _predict_change(models: dict[int, HistGradientBoostingRegressor], rows: pd.DataFrame) -> np.ndarray:
    hh = horizon_hour(rows["horizon"])
    out = np.full(len(rows), np.nan)
    for h in np.unique(hh):
        mask = hh == h
        out[mask] = models[int(h)].predict(rows.loc[mask, FEATURES])
    return out


@dataclass
class CarbonForecaster:
    """Forecasts carbon intensity as ``latest value + shrinkage[h] * predicted change[h]``.

    Fitting holds out the most recent ``calib_days``. A model trained without them is scored on them to
    learn, for each horizon hour,
      * a shrinkage weight in [0, 1] (least squares, clipped) that pulls the predicted change towards
        "no change" where the model's signal is weak, typically at long horizons without weather inputs;
      * split-conformal interval offsets: empirical quantiles of the shrunk forecast's errors.
    The final models are then refit on all data and reuse those weights and offsets.
    """

    quantiles: tuple[float, float] = config.QUANTILES
    origin_stride: int = 2
    calib_days: int = 14
    models: dict[int, HistGradientBoostingRegressor] = field(default_factory=dict)
    shrinkage: dict[int, float] = field(default_factory=dict)
    interval_offsets: dict[int, tuple[float, float]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def fit(self, y: pd.Series, calibrate: bool = True) -> CarbonForecaster:
        rows = training_rows(y, origin_stride=self.origin_stride)
        if len(rows) < 1000:
            raise ValueError(f"Not enough complete training rows ({len(rows)}); fetch more history.")

        self.shrinkage, self.interval_offsets = {}, {}
        n_calib = 0
        if calibrate:
            cutoff = rows["target_time"].max() - pd.Timedelta(days=self.calib_days)
            proper = rows[rows["target_time"] < cutoff]  # no target overlaps the calibration window
            calib = rows[rows["origin"] >= cutoff]
            if len(proper) >= 1000 and len(calib) >= 200:
                change_pred = _predict_change(_fit_per_hour(proper), calib)
                change_true = (calib["target"] - calib["lag_0"]).to_numpy()
                hh = horizon_hour(calib["horizon"])
                lo_q, hi_q = self.quantiles
                for h in np.unique(hh):
                    m = hh == h
                    p, t = change_pred[m], change_true[m]
                    w = float(np.clip(np.dot(p, t) / max(np.dot(p, p), 1e-9), 0.0, 1.0))
                    resid = t - w * p
                    self.shrinkage[int(h)] = w
                    self.interval_offsets[int(h)] = (float(np.quantile(resid, lo_q)), float(np.quantile(resid, hi_q)))
                n_calib = int(len(calib))

        self.models = _fit_per_hour(rows)
        self.metadata = {
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "trained_from": y.dropna().index.min().isoformat(),
            "trained_until": y.dropna().index.max().isoformat(),
            "n_training_rows": int(len(rows)),
            "n_calibration_rows": n_calib,
            "target": "change from latest observed value, one model per horizon hour",
            "features": FEATURES,
            "shrinkage_by_horizon_hour": {h: round(w, 3) for h, w in self.shrinkage.items()},
            "max_horizon_hours": config.MAX_HORIZON / 2,
            "interval": f"{round((self.quantiles[1] - self.quantiles[0]) * 100)}% split-conformal"
            if self.interval_offsets
            else None,
        }
        return self

    def predict_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Predict for prepared feature rows. Adds ``point`` and, if calibrated, ``lower``/``upper``."""
        if not self.models:
            raise RuntimeError("Model is not fitted")
        out = rows.copy()
        hh = horizon_hour(rows["horizon"])
        weight = np.array([self.shrinkage.get(int(h), 1.0) for h in hh])
        out["point"] = rows["lag_0"].to_numpy() + weight * _predict_change(self.models, rows)
        if self.interval_offsets:
            lo = np.array([self.interval_offsets[int(h)][0] for h in hh])
            hi = np.array([self.interval_offsets[int(h)][1] for h in hh])
            out["lower"] = out["point"] + np.minimum(lo, 0)
            out["upper"] = out["point"] + np.maximum(hi, 0)
        # Carbon intensity cannot be negative.
        for col in ("point", "lower", "upper"):
            if col in out:
                out[col] = out[col].clip(lower=0)
        return out

    def forecast(self, y: pd.Series, horizon_steps: int = config.MAX_HORIZON) -> pd.DataFrame:
        """Forecast the next ``horizon_steps`` half-hours from the latest observed value in ``y``."""
        origin = latest_origin(y)
        rows = build_dataset(y, origins=pd.DatetimeIndex([origin]), horizons=range(1, horizon_steps + 1))
        needed = FEATURES + ["lag_0"]
        missing = rows[needed].columns[rows[needed].isna().any()].tolist()
        if missing:
            raise ValueError(f"Insufficient recent history to build features: {missing}")
        return self.predict_rows(rows)

    def save(self, path: Path = config.MODEL_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path = config.MODEL_PATH) -> CarbonForecaster:
        obj = joblib.load(path)
        if not isinstance(obj, CarbonForecaster):
            raise TypeError(f"{path} does not contain a CarbonForecaster")
        return obj
