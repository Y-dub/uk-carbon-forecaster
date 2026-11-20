"""Fetch history, backtest, train the final model and save it with its metrics.

Usage:
    python -m carbon_forecast.train --days 365            # real data from the Carbon Intensity API
    python -m carbon_forecast.train --synthetic --days 90 # offline smoke run (no network)
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from . import config
from .data import clean_series, fetch_recent
from .evaluate import backtest
from .model import CarbonForecaster
from .synthetic import synthetic_intensity

log = logging.getLogger("carbon_forecast.train")


def load_raw(days: int, synthetic: bool, data_csv: Path | None) -> pd.DataFrame:
    if synthetic:
        log.info("Using SYNTHETIC data (%d days); results are not meaningful", days)
        return synthetic_intensity(days=days)
    if data_csv and data_csv.exists():
        log.info("Loading cached data from %s", data_csv)
        return pd.read_csv(data_csv, index_col="from", parse_dates=["from"])
    log.info("Fetching %d days from the Carbon Intensity API", days)
    raw = fetch_recent(days)
    if data_csv:
        data_csv.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(data_csv)
        log.info("Cached raw data to %s", data_csv)
    return raw


def plot_by_horizon(summary: dict, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib not installed; skipping plot")
        return
    labels = {
        "model": "Gradient boosting (this project)",
        "persistence": "Persistence (last value)",
        "seasonal_naive_24h": "Same time yesterday",
    }  # "same time last week" stays in the table; on the plot its much larger error flattens the rest
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, label in labels.items():
        series = {int(h): v for h, v in summary["mae_by_horizon_hour"][key].items()}
        hours = sorted(series)
        ax.plot(hours, [series[h] for h in hours], label=label, linewidth=2.5 if key == "model" else 1.5)
    ax.set_xlabel("Forecast horizon (hours ahead)")
    ax.set_ylabel("MAE (gCO2/kWh)")
    ax.set_ylim(bottom=0)
    ax.set_title("Backtest error by horizon, UK national carbon intensity")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=365, help="days of history to use")
    p.add_argument("--folds", type=int, default=6, help="backtest folds (one week each)")
    p.add_argument("--skip-backtest", action="store_true", help="train only (faster; used in Docker builds)")
    p.add_argument("--synthetic", action="store_true", help="use synthetic data, no network")
    p.add_argument("--data-csv", type=Path, default=None, help="cache raw API data here and reuse it")
    p.add_argument("--plot", type=Path, default=Path("docs/backtest_mae_by_horizon.png"))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raw = load_raw(args.days, args.synthetic, args.data_csv)

    metrics: dict = {"data_source": "synthetic" if args.synthetic else "carbonintensity.org.uk", "days": args.days}
    if not args.skip_backtest:
        _, summary = backtest(raw, n_folds=args.folds)
        metrics["backtest"] = summary
        o = summary["overall"]
        log.info(
            "Backtest MAE: model %.1f | persistence %.1f | yesterday %.1f | last week %.1f (gCO2/kWh)",
            o["model"]["mae"], o["persistence"]["mae"], o["seasonal_naive_24h"]["mae"], o["seasonal_naive_7d"]["mae"],
        )
        if not args.synthetic:
            plot_by_horizon(summary, args.plot)

    model = CarbonForecaster().fit(clean_series(raw))
    model.metadata["backtest"] = metrics.get("backtest")
    model.metadata["data_source"] = metrics["data_source"]
    model.save()
    metrics["model"] = {k: v for k, v in model.metadata.items() if k != "backtest"}
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str))
    log.info("Saved model to %s and metrics to %s", config.MODEL_PATH, config.METRICS_PATH)
    return metrics


if __name__ == "__main__":
    main()
