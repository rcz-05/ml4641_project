"""Reusable helpers for preparing modeling frames."""

from __future__ import annotations

import pandas as pd

from .config import ProjectConfig, get_active_window
from .data_utils import fetch_price_data, load_phrasebank, load_stock_news
from .features import aggregate_daily_sentiment, build_model_frame, compute_technical_indicators
from .modeling import train_sentiment_model
from .paths import MODELS_DIR, PROCESSED_DATA_DIR
import joblib
import pandas as pd
from pathlib import Path


def prepare_ticker_frame(
    cfg: ProjectConfig,
    sentiment_result,
    ticker: str | None = None,
    window=None,
    use_local_processed: bool = True,
    require_sentiment: bool = True,
):
    """Load price/news, aggregate sentiment, engineer features, and persist processed CSVs."""

    if ticker is None:
        ticker = cfg.primary_ticker
    if window is None:
        window = get_active_window(cfg)

    processed_news_path = PROCESSED_DATA_DIR / f"{ticker.lower()}_news_daily.csv"
    processed_frame_path = PROCESSED_DATA_DIR / f"{ticker.lower()}_model_frame.csv"

    price_df = fetch_price_data(ticker=ticker, start_date=window.start, end_date=window.end, source="csv")

    if use_local_processed and processed_news_path.exists():
        news_df = pd.read_csv(processed_news_path)
        news_df["date"] = pd.to_datetime(news_df["date"]).dt.date
        sentiment_daily = news_df
    else:
        news_df = load_stock_news(ticker=ticker, max_rows=cfg.max_news_rows, start_date=window.start, end_date=window.end)
        sentiment_daily = aggregate_daily_sentiment(news_df, sentiment_result.pipeline, sentiment_result.classes)

    if sentiment_daily.empty and require_sentiment:
        raise RuntimeError(f"No news available in the specified window for {ticker}; cannot proceed with sentiment analysis.")

    price_features = compute_technical_indicators(price_df)
    modeling_frame = build_model_frame(
        price_features,
        sentiment_daily if not sentiment_daily.empty else pd.DataFrame(columns=["date"]),
        use_lagged_features=cfg.use_lagged_features,
        lags=cfg.lag_days,
    )
    modeling_frame["ticker"] = ticker

    sentiment_daily.to_csv(processed_news_path, index=False)
    modeling_frame.to_csv(processed_frame_path, index=False)
    return modeling_frame, sentiment_daily, price_df


def ensure_modeling_frame(cfg: ProjectConfig, use_local_processed: bool = True):
    """Load processed modeling frame for the primary ticker or rebuild if missing."""

    window = get_active_window(cfg)
    frame_path = PROCESSED_DATA_DIR / f"{cfg.primary_ticker.lower()}_model_frame.csv"
    if use_local_processed and frame_path.exists():
        frame = pd.read_csv(frame_path)
        frame["date"] = pd.to_datetime(frame["date"]).dt.date
        return frame

    phrasebank_df = load_phrasebank(cfg.sentiment_subset)
    sentiment_result = train_sentiment_model(phrasebank_df, random_state=cfg.global_seed)
    joblib.dump(sentiment_result.pipeline, MODELS_DIR / "sentiment_model.joblib")
    frame, _, _ = prepare_ticker_frame(cfg, sentiment_result, cfg.primary_ticker, window, use_local_processed=use_local_processed)
    return frame
