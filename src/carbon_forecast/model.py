"""Gradient-boosted direct multi-horizon forecaster with split-conformal prediction intervals."""

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


def _horizon_hour(horizon: pd.Series | np.ndarray) -> np.ndarray:
    return np.ceil(np.asarray(horizon) / 2).astype(int)


@dataclass
class CarbonForecaster:
    """One gradient-boosting model for every horizon (horizon is a feature).

    The target is the change from the latest observed value (``lag_0``), not the raw level: tree models
    cannot extrapolate beyond levels seen in training, and the grid keeps decarbonising.

    Prediction intervals are split-conformal: a copy of the model is trained without the most recent
    ``calib_days``, its errors on those days are measured per horizon hour, and the empirical quantiles
    of those errors become the interval offsets around the final model's forecast.
    """

    quantiles: tuple[float, float] = config.QUANTILES
    origin_stride: int = 2
    calib_days: int = 14
    model: HistGradientBoostingRegressor | None = None
    interval_offsets: dict[int, tuple[float, float]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def _fit_model(rows: pd.DataFrame) -> HistGradientBoostingRegressor:
        return _regressor().fit(rows[FEATURES], rows["target"] - rows["lag_0"])

    def fit(self, y: pd.Series, with_intervals: bool = True) -> CarbonForecaster:
        rows = training_rows(y, origin_stride=self.origin_stride)
        if len(rows) < 1000:
            raise ValueError(f"Not enough complete training rows ({len(rows)}); fetch more history.")

        self.interval_offsets = {}
        n_calib = 0
        if with_intervals:
            cutoff = rows["target_time"].max() - pd.Timedelta(days=self.calib_days)
            proper = rows[rows["target_time"] < cutoff]  # no target overlaps the calibration window
            calib = rows[rows["origin"] >= cutoff]
            if len(proper) >= 1000 and len(calib) >= 200:
                m0 = self._fit_model(proper)
                resid = calib["target"] - (calib["lag_0"] + m0.predict(calib[FEATURES]))
                lo_q, hi_q = self.quantiles
                grouped = resid.groupby(_horizon_hour(calib["horizon"]))
                self.interval_offsets = {
                    int(h): (float(g.quantile(lo_q)), float(g.quantile(hi_q))) for h, g in grouped
                }
                n_calib = int(len(calib))

        self.model = self._fit_model(rows)
        self.metadata = {
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "trained_from": y.dropna().index.min().isoformat(),
            "trained_until": y.dropna().index.max().isoformat(),
            "n_training_rows": int(len(rows)),
            "n_calibration_rows": n_calib,
            "target": "change from latest observed value (lag_0)",
            "features": FEATURES,
            "max_horizon_hours": config.MAX_HORIZON / 2,
            "interval": f"{round((self.quantiles[1] - self.quantiles[0]) * 100)}% split-conformal"
            if self.interval_offsets
            else None,
        }
        return self

    def predict_rows(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Predict for prepared feature rows. Adds ``point`` and, if calibrated, ``lower``/``upper``."""
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        out = rows.copy()
        out["point"] = rows["lag_0"].to_numpy() + self.model.predict(rows[FEATURES])
        if self.interval_offsets:
            hh = _horizon_hour(rows["horizon"])
            lo = np.array([self.interval_offsets.get(h, (np.nan, np.nan))[0] for h in hh])
            hi = np.array([self.interval_offsets.get(h, (np.nan, np.nan))[1] for h in hh])
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
        missing = rows[FEATURES].columns[rows[FEATURES].isna().any()].tolist()
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
