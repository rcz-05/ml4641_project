"""Aggregate metrics across models and generate comparison plots/tables."""

from __future__ import annotations

import json
import os
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR.parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import pandas as pd

from src.config import load_config
from src.paths import FIGURES_DIR, REPORTS_DIR
from src.visualization import plot_metric_bars


def _load_metrics():
    metrics_path = REPORTS_DIR / "metrics.json"
    with metrics_path.open() as f:
        return json.load(f)


def _load_lstm_metrics():
    lstm_path = REPORTS_DIR / "lstm_metrics.json"
    if lstm_path.exists():
        with lstm_path.open() as f:
            return json.load(f)
    return None


def main():
    cfg = load_config()
    metrics = _load_metrics()
    lstm_metrics = _load_lstm_metrics()

    model_rows = []
    for name, payload in [
        ("rf_sentiment", metrics.get("direction_model")),
        ("rf_technical_only", metrics.get("direction_model_baseline")),
        ("gradient_boosting", metrics.get("direction_model_gbdt")),
        ("extra_trees", metrics.get("direction_model_extratrees")),
    ]:
        if payload:
            row = {"model": name}
            row.update(payload.get("metrics", payload))
            model_rows.append(row)

    # LSTM removed from pipeline; keep optional load guard for older runs.
    if lstm_metrics:
        row = {"model": "lstm_sequence"}
        row.update(lstm_metrics.get("direction_metrics", {}))
        model_rows.append(row)

    comparison_df = pd.DataFrame(model_rows)
    comparison_csv = REPORTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    for metric in ["accuracy", "f1", "roc_auc"]:
        plot_metric_bars(
            comparison_df,
            metric,
            FIGURES_DIR / f"comparison_{metric}.png",
            title=f"Model Comparison: {metric}",
        )

    hyperparams_rows = [
        {
            "model": "rf_sentiment",
            "type": "classification",
            "features": "technical + sentiment",
            "notes": "n_estimators=400, max_depth=8, min_samples_split=4, class_weight=balanced",
        },
        {
            "model": "rf_technical_only",
            "type": "classification",
            "features": "technical only",
            "notes": "n_estimators=400, max_depth=8, min_samples_split=4, class_weight=balanced",
        },
        {
            "model": "gradient_boosting",
            "type": "classification",
            "features": "technical + sentiment",
            "notes": "default GradientBoostingClassifier, random_state from cfg",
        },
        {
            "model": "extra_trees",
            "type": "classification",
            "features": "technical + sentiment",
            "notes": "n_estimators=400, class_weight=balanced, randomized thresholds",
        },
    ]
    hyperparams_df = pd.DataFrame(hyperparams_rows)
    hyperparams_df.to_csv(REPORTS_DIR / "model_hyperparameters.csv", index=False)

    print(f"Comparison artifacts written to {comparison_csv} and figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
