"""Rolling / expanding evaluation to get more robust test metrics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config, get_active_window
from src.data_utils import load_phrasebank
from src.pipeline_utils import ensure_modeling_frame, prepare_ticker_frame
from src.modeling import train_sentiment_model, train_direction_model


def rolling_splits(frame: pd.DataFrame, min_train: int = 30, test_size: int = 10, step: int = 5):
    frame = frame.sort_values("date").reset_index(drop=True)
    splits = []
    start = min_train
    while start + test_size <= len(frame):
        train_idx = start
        test_idx = start + test_size
        splits.append((train_idx, test_idx))
        start += step
    return splits


def evaluate_model(frame: pd.DataFrame, test_size: int = 10, step: int = 5):
    splits = rolling_splits(frame, min_train=max(30, len(frame) // 3), test_size=test_size, step=step)
    metrics = []
    for train_idx, test_idx in splits:
        train_df = frame.iloc[:train_idx]
        test_df = frame.iloc[train_idx:test_idx]
        model_result = train_direction_model(
            pd.concat([train_df, test_df], ignore_index=True),
            train_frac=train_idx / (train_idx + len(test_df)),
            val_frac=0.0,
            random_state=42,
        )
        metrics.append(model_result.metrics)
    return metrics


def aggregate_metrics(metric_list):
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    aggregated = {}
    for k in keys:
        values = [m[k] for m in metric_list if m.get(k) is not None]
        if values:
            aggregated[k] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "n": len(values),
            }
    return aggregated


def main():
    cfg = load_config()
    window = get_active_window(cfg)
    frame = ensure_modeling_frame(cfg, use_local_processed=True)
    tech_only = frame.drop(columns=[c for c in frame.columns if "sentiment" in c or c.startswith("prob_") or c == "article_count"], errors="ignore")

    # Sentiment training only needed if you rebuild news; here we assume processed files exist.
    # If not, uncomment below:
    # phrasebank_df = load_phrasebank(cfg.sentiment_subset)
    # sentiment_result = train_sentiment_model(phrasebank_df, random_state=cfg.global_seed)
    # frame, _, _ = prepare_ticker_frame(cfg, sentiment_result, cfg.primary_ticker, window, use_local_processed=True)

    sent_metrics = evaluate_model(frame, test_size=10, step=5)
    tech_metrics = evaluate_model(tech_only, test_size=10, step=5)

    payload = {
        "rolling_rf_with_sentiment": aggregate_metrics(sent_metrics),
        "rolling_rf_technical_only": aggregate_metrics(tech_metrics),
    }
    out_path = Path("reports/metrics_rolling.json")
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Rolling evaluation saved to {out_path}")


if __name__ == "__main__":
    main()
