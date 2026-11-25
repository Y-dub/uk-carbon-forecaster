"""Rewrite the README's Results section from models/metrics.json.

Usage:
    python scripts/update_readme.py                     # uses models/metrics.json and README.md
    python scripts/update_readme.py --metrics path.json --readme README.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LABELS = {
    "model": "**Gradient boosting (this project)**",
    "persistence": "Persistence (last value)",
    "seasonal_naive_24h": "Same time yesterday",
    "seasonal_naive_7d": "Same time last week",
}
NAIVE = ["persistence", "seasonal_naive_24h", "seasonal_naive_7d"]


def _skill(model: float, baseline: float) -> str:
    pct = (1 - model / baseline) * 100
    return f"{abs(pct):.1f}% {'better' if pct >= 0 else 'worse'}"


def render(metrics: dict) -> str:
    bt = metrics["backtest"]
    o = bt["overall"]
    trained_until = metrics.get("model", {}).get("trained_until", "")[:10]

    lines = [
        "## Results",
        "",
        f"Rolling-origin backtest: {bt['n_folds']} weekly folds ({bt['n_rows']:,} forecasts) covering the "
        f"{bt['n_folds']} weeks to {trained_until}, using {metrics['days']} days of history. Forecasts are issued "
        "hourly for 1–24 hours ahead, and the model is retrained before each fold on past data only.",
        "",
        "| Model | MAE (gCO2/kWh) | RMSE | MAPE | Bias |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key, label in LABELS.items():
        m = o[key]
        lines.append(f"| {label} | {m['mae']:.1f} | {m['rmse']:.1f} | {m['mape_pct']:.1f}% | {m['bias']:+.1f} |")

    best_mae = min(NAIVE, key=lambda k: o[k]["mae"])
    best_rmse = min(NAIVE, key=lambda k: o[k]["rmse"])
    lines += [
        "",
        f"- MAE: {_skill(o['model']['mae'], o[best_mae]['mae'])} than the best naive baseline, "
        f"{LABELS[best_mae].lower()}.",
        f"- RMSE: {_skill(o['model']['rmse'], o[best_rmse]['rmse'])} than the best naive baseline, "
        f"{LABELS[best_rmse].lower()}.",
    ]
    if "interval" in bt:
        iv = bt["interval"]
        lines.append(
            f"- {iv['nominal_coverage_pct']:.0f}% prediction interval: empirical coverage "
            f"{iv['empirical_coverage_pct']:.1f}%, mean width {iv['mean_width']:.0f} gCO2/kWh."
        )
    if "vs_official_on_shared_rows" in bt:
        neso = bt["vs_official_on_shared_rows"]["neso_official"]
        lines.append(
            f"- NESO's official forecast: MAE {neso['mae']:.1f} on the same periods. This comparison favours "
            "NESO, because the API keeps its latest forecast for each period, often issued much closer to "
            "delivery than our horizons, and NESO uses weather forecasts that this model does not."
        )
    lines += [
        "",
        "![MAE by horizon](docs/backtest_mae_by_horizon.png)",
        "",
        "> Generated from `models/metrics.json` by `scripts/update_readme.py`; the same numbers are served live "
        "at `/model`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=Path, default=Path("models/metrics.json"))
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    args = p.parse_args()

    metrics = json.loads(args.metrics.read_text())
    readme = args.readme.read_text()
    section = re.compile(r"## Results\n.*?(?=\n## )", re.DOTALL)
    if not section.search(readme):
        raise SystemExit("Could not find a '## Results' section followed by another '## ' heading")
    args.readme.write_text(section.sub(lambda _: render(metrics).rstrip("\n") + "\n", readme, count=1))
    print(render(metrics))


if __name__ == "__main__":
    main()
