"""Plotting utilities for the project."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
palette = sns.color_palette("tab10")


def plot_sentiment_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart of sentiment label counts."""

    plt.figure(figsize=(6, 4))
    sns.countplot(x="label", data=df, order=sorted(df["label"].unique()))
    plt.title("Financial PhraseBank Sentiment Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_matrix(matrix, labels: Sequence[str], output_path: Path, title: str = "Confusion Matrix") -> None:
    """Heatmap confusion matrix."""

    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_sentiment_vs_price(price_df: pd.DataFrame, sentiment_df: pd.DataFrame, output_path: Path) -> None:
    """Line chart comparing price and aggregated sentiment."""

    merged = price_df.merge(sentiment_df, on="date", how="left")
    merged = merged.dropna(subset=["model_sentiment"])

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(merged["date"], merged["adj_close"], color="tab:blue", label="Adj Close")
    ax1.set_ylabel("Adj Close", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(merged["date"], merged["model_sentiment"], color="tab:orange", label="Sentiment")
    ax2.set_ylabel("Avg Sentiment (Model)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    fig.autofmt_xdate()
    fig.tight_layout()
    plt.title("Daily Sentiment vs. Price")
    fig.savefig(output_path)
    plt.close(fig)


def plot_feature_importances(importances: pd.DataFrame, output_path: Path, top_n: int = 10) -> None:
    """Horizontal bar plot of top feature importances."""

    top_features = importances.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, 5))
    sns.barplot(x="importance", y="feature", data=top_features)
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(roc_data, output_path: Path, label: str = "Model", title: str = "ROC Curve") -> None:
    """ROC curve plot."""

    if roc_data is None:
        return
    fpr = roc_data["fpr"]
    tpr = roc_data["tpr"]
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_loss_curves(history: dict, output_path: Path, title: str = "Training vs Validation Loss") -> None:
    """Plot training/validation loss curves."""

    if not history:
        return
    plt.figure(figsize=(6, 4))
    for key, values in history.items():
        plt.plot(values, label=key.capitalize())
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_price_predictions(dates, actual, predicted, output_path: Path, title: str = "Predicted vs Actual Price") -> None:
    """Line plot of predicted vs actual prices."""

    plt.figure(figsize=(8, 4))
    plt.plot(dates, actual, label="Actual", color="tab:blue")
    plt.plot(dates, predicted, label="Predicted", color="tab:orange")
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Adj Close")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_metric_bars(data: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    """Bar plot comparing a metric across models."""

    if metric not in data.columns:
        return
    plt.figure(figsize=(7, 4))
    sns.barplot(data=data, x="model", y=metric, palette=palette)
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(metric.capitalize())
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
