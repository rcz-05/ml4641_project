"""Orchestrate the sentiment + stock-direction modeling workflow."""

from __future__ import annotations

import json
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import get_active_window, load_config
from src.data_utils import load_phrasebank
from src.modeling import (
    train_direction_model,
    train_direction_gradient_boosting,
    train_direction_extratrees,
    train_sentiment_model,
    tune_random_forest,
)
from src.paths import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from src.pipeline_utils import prepare_ticker_frame
from src.visualization import (
    plot_confusion_matrix,
    plot_feature_importances,
    plot_roc_curve,
    plot_sentiment_distribution,
    plot_sentiment_vs_price,
)

SENTIMENT_FEATURES = [
    "model_sentiment",
    "prob_positive",
    "prob_negative",
    "prob_neutral",
    "ticker_sentiment_score",
    "overall_sentiment_score",
    "sentiment_rolling_3",
    "sentiment_rolling_7",
    "article_count",
]


def _serialize(data: Any):
    if isinstance(data, dict):
        return {k: _serialize(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_serialize(v) for v in data]
    if isinstance(data, (np.floating,)):
        return float(data)
    if isinstance(data, (np.integer,)):
        return int(data)
    if hasattr(data, "tolist"):
        return data.tolist()
    return data


def main():
    cfg = load_config()
    window = get_active_window(cfg)
    ticker = cfg.primary_ticker
    sentiment_result = None
    if cfg.use_sentiment_features:
        print("Loading Financial PhraseBank subset...")
        phrasebank_df = load_phrasebank(cfg.sentiment_subset)
        print("Training sentiment classifier...")
        sentiment_result = train_sentiment_model(phrasebank_df, random_state=cfg.global_seed)
        joblib.dump(sentiment_result.pipeline, MODELS_DIR / "sentiment_model.joblib")

    frames: list[pd.DataFrame] = []
    sentiment_daily_frames: list[pd.DataFrame] = []
    price_frames: list[pd.DataFrame] = []

    print(f"Fetching market data for {ticker}...")
    primary_frame, primary_sentiment, primary_price = prepare_ticker_frame(
        cfg,
        sentiment_result,
        ticker,
        window,
        use_local_processed=cfg.use_local_processed,
        require_sentiment=cfg.use_sentiment_features,
    )
    frames.append(primary_frame)
    if not primary_sentiment.empty:
        sentiment_daily_frames.append(primary_sentiment.assign(ticker=ticker))
    price_frames.append(primary_price.assign(ticker=ticker))

    if cfg.include_secondary:
        print(f"Fetching market data and news for {cfg.secondary_ticker}...")
        secondary_frame, secondary_sentiment, secondary_price = prepare_ticker_frame(cfg, sentiment_result, cfg.secondary_ticker, window)
        frames.append(secondary_frame)
        sentiment_daily_frames.append(secondary_sentiment.assign(ticker=cfg.secondary_ticker))
        price_frames.append(secondary_price.assign(ticker=cfg.secondary_ticker))

    modeling_frame = pd.concat(frames, ignore_index=True)

    print("Training stock direction model (sentiment + technical)...")
    direction_result = train_direction_model(
        modeling_frame,
        train_frac=cfg.chrono_train_frac,
        val_frac=cfg.chrono_val_frac,
        random_state=cfg.global_seed,
    )

    print("Training technical-only baseline...")
    technical_only_frame = modeling_frame.drop(columns=[col for col in SENTIMENT_FEATURES if col in modeling_frame], errors="ignore")
    baseline_result = train_direction_model(
        technical_only_frame,
        train_frac=cfg.chrono_train_frac,
        val_frac=cfg.chrono_val_frac,
        random_state=cfg.global_seed,
    )

    print("Training Gradient Boosting direction model...")
    gbdt_result = train_direction_gradient_boosting(
        modeling_frame,
        train_frac=cfg.chrono_train_frac,
        val_frac=cfg.chrono_val_frac,
        random_state=cfg.global_seed,
    )

    print("Training ExtraTrees direction model...")
    extratrees_result = train_direction_extratrees(
        modeling_frame,
        train_frac=cfg.chrono_train_frac,
        val_frac=cfg.chrono_val_frac,
        random_state=cfg.global_seed,
    )

    rf_tuning = None
    if cfg.tune_random_forest:
        print("Running small Random Forest hyperparameter search...")
        rf_tuning = tune_random_forest(
            modeling_frame,
            train_frac=cfg.chrono_train_frac,
            val_frac=cfg.chrono_val_frac,
            random_state=cfg.global_seed,
        )

    metrics_payload = {
        "direction_model": {
            "metrics": _serialize(direction_result.metrics),
            "confusion_matrix": _serialize(direction_result.confusion),
        },
        "direction_model_baseline": {
            "metrics": _serialize(baseline_result.metrics),
            "confusion_matrix": _serialize(baseline_result.confusion),
        },
        "direction_model_gbdt": {
            "metrics": _serialize(gbdt_result.metrics),
            "confusion_matrix": _serialize(gbdt_result.confusion),
        },
        "direction_model_extratrees": {
            "metrics": _serialize(extratrees_result.metrics),
            "confusion_matrix": _serialize(extratrees_result.confusion),
        },
    }
    if sentiment_result:
        metrics_payload["sentiment_model"] = {
            "accuracy": sentiment_result.metrics["accuracy"],
            "f1_macro": sentiment_result.metrics["f1_macro"],
            "confusion_matrix": sentiment_result.confusion.tolist(),
        }
    if rf_tuning:
        metrics_payload["direction_model"]["tuning"] = _serialize(rf_tuning)

    metrics_path = REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))

    if sentiment_result:
        plot_sentiment_distribution(phrasebank_df, FIGURES_DIR / "sentiment_distribution.png")
        plot_confusion_matrix(
            sentiment_result.confusion,
            sorted(phrasebank_df["label"].unique()),
            FIGURES_DIR / "sentiment_confusion.png",
            title="Sentiment Classifier Confusion Matrix",
        )
    primary_price_df = price_frames[0]
    if sentiment_daily_frames:
        primary_sentiment_df = sentiment_daily_frames[0].drop(columns="ticker", errors="ignore")
        if not primary_sentiment_df.empty:
            plot_sentiment_vs_price(primary_price_df, primary_sentiment_df, FIGURES_DIR / "sentiment_vs_price.png")
    plot_feature_importances(direction_result.feature_importances, FIGURES_DIR / "feature_importance.png")
    plot_roc_curve(direction_result.roc_curve_data, FIGURES_DIR / "roc_curve_rf.png", label="Random Forest")
    plot_roc_curve(gbdt_result.roc_curve_data, FIGURES_DIR / "roc_curve_gbdt.png", label="Gradient Boosting")
    plot_roc_curve(extratrees_result.roc_curve_data, FIGURES_DIR / "roc_curve_extratrees.png", label="Extra Trees")
    plot_confusion_matrix(direction_result.confusion, direction_result.labels, FIGURES_DIR / "direction_confusion_rf.png", title="Random Forest Direction Confusion")
    plot_confusion_matrix(baseline_result.confusion, baseline_result.labels, FIGURES_DIR / "direction_confusion_baseline.png", title="Technical-Only RF Confusion")
    plot_confusion_matrix(gbdt_result.confusion, gbdt_result.labels, FIGURES_DIR / "direction_confusion_gbdt.png", title="Gradient Boosting Direction Confusion")
    plot_confusion_matrix(extratrees_result.confusion, extratrees_result.labels, FIGURES_DIR / "direction_confusion_extratrees.png", title="Extra Trees Direction Confusion")

    print(f"Pipeline finished. Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
