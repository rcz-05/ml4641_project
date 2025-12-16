"""Feature engineering helpers."""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator


def clean_text(text: pd.Series) -> pd.Series:
    """Apply lightweight normalization needed for classical models."""

    cleaned = text.str.lower()
    cleaned = cleaned.str.replace(r"http\S+", " ", regex=True)
    cleaned = cleaned.str.replace(r"www\.\S+", " ", regex=True)
    cleaned = cleaned.str.replace(r"[^a-z0-9%\s]", " ", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
    return cleaned.str.strip()


def compute_technical_indicators(price_df: pd.DataFrame) -> pd.DataFrame:
    """Add simple returns and RSI features."""

    df = price_df.copy()
    df["return_1d"] = df["adj_close"].pct_change()
    df["return_5d"] = df["adj_close"].pct_change(5)
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["sma_3"] = df["adj_close"].rolling(3).mean()
    df["sma_7"] = df["adj_close"].rolling(7).mean()
    df["sma_ratio"] = df["sma_3"] / df["sma_7"] - 1
    df["rsi_14"] = RSIIndicator(close=df["adj_close"], window=14).rsi()
    return df


def aggregate_daily_sentiment(news_df: pd.DataFrame, sentiment_model, class_labels: Sequence[str]) -> pd.DataFrame:
    """Score news headlines with the trained sentiment model and aggregate by day."""

    if news_df.empty:
        columns = [
            "date",
            "article_count",
            "model_sentiment",
            "prob_positive",
            "prob_negative",
            "prob_neutral",
            "ticker_sentiment_score",
            "overall_sentiment_score",
            "sentiment_rolling_3",
            "sentiment_rolling_7",
        ]
        return pd.DataFrame(columns=columns)

    scored = news_df.copy()
    cleaned_text = clean_text(scored["text"])
    proba = sentiment_model.predict_proba(cleaned_text)
    classes = list(class_labels)

    def _prob(label: str) -> np.ndarray:
        if label not in classes:
            return np.zeros(len(scored))
        idx = classes.index(label)
        return proba[:, idx]

    scored["prob_positive"] = _prob("positive")
    scored["prob_negative"] = _prob("negative")
    scored["prob_neutral"] = _prob("neutral")
    scored["model_sentiment"] = scored["prob_positive"] - scored["prob_negative"]

    daily = (
        scored.groupby("date")
        .agg(
            article_count=("text", "count"),
            model_sentiment=("model_sentiment", "mean"),
            prob_positive=("prob_positive", "mean"),
            prob_negative=("prob_negative", "mean"),
            prob_neutral=("prob_neutral", "mean"),
            ticker_sentiment_score=("ticker_sentiment_score", "mean"),
            overall_sentiment_score=("overall_sentiment_score", "mean"),
        )
        .reset_index()
    )
    daily["sentiment_rolling_3"] = daily["model_sentiment"].rolling(3).mean()
    daily["sentiment_rolling_7"] = daily["model_sentiment"].rolling(7).mean()
    return daily


def _add_lagged_columns(df: pd.DataFrame, columns: Sequence[str], lags: Sequence[int]) -> pd.DataFrame:
    for lag in lags:
        for col in columns:
            if col in df:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def build_model_frame(
    price_features: pd.DataFrame,
    sentiment_daily: pd.DataFrame,
    use_lagged_features: bool = True,
    lags: Sequence[int] = (1,),
) -> pd.DataFrame:
    """Combine price indicators with sentiment aggregates and create a prediction target."""

    merged = price_features.merge(sentiment_daily, on="date", how="left")
    merged = merged.sort_values("date")
    sentiment_min = sentiment_daily["date"].min() if "date" in sentiment_daily and not sentiment_daily.empty else None
    sentiment_max = sentiment_daily["date"].max() if "date" in sentiment_daily and not sentiment_daily.empty else None
    sentiment_cols = ["model_sentiment", "prob_positive", "prob_negative", "prob_neutral", "ticker_sentiment_score", "overall_sentiment_score", "sentiment_rolling_3", "sentiment_rolling_7", "article_count"]
    for col in sentiment_cols:
        if col in merged:
            merged[col] = merged[col].ffill().fillna(0.0)
            if sentiment_min is not None and sentiment_max is not None:
                outside_range = (merged["date"] < sentiment_min) | (merged["date"] > sentiment_max)
                merged.loc[outside_range, col] = 0.0

    if use_lagged_features:
        technical_cols = [col for col in merged.columns if col not in {"date", "target_up"} and col.startswith(("return_", "volatility_", "sma_", "rsi_"))]
        merged = _add_lagged_columns(merged, sentiment_cols + technical_cols, lags)

    merged["target_up"] = (merged["adj_close"].shift(-1) > merged["adj_close"]).astype(int)
    merged = merged.dropna()
    return merged
