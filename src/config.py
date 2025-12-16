"""Project-wide configuration for data windows, seeds, and feature toggles."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass


@dataclass(frozen=True)
class DataWindow:
    start: dt.date
    end: dt.date


@dataclass(frozen=True)
class ProjectConfig:
    primary_ticker: str = "AAPL"
    secondary_ticker: str = "MSFT"
    include_secondary: bool = False
    use_extended_window: bool = True
    base_window: DataWindow = DataWindow(dt.date(2020, 1, 1), dt.date(2020, 12, 31)) # Changed to larger 2020 window
    extended_window: DataWindow = DataWindow(dt.date(2020, 1, 1), dt.date(2020, 12, 31))
    sentiment_subset: str = "sentences_75agree"
    use_sentiment_features: bool = True
    chrono_train_frac: float = 0.6
    chrono_val_frac: float = 0.2
    # If true, prefer local processed news/frames; if absent, skip sentiment gracefully
    use_local_processed: bool = True
    global_seed: int = 42
    max_news_rows: int = 800
    use_lagged_features: bool = True
    lag_days: tuple[int, ...] = (1,)
    tune_random_forest: bool = False
    enable_lstm: bool = False
    lstm_lookback: int = 10
    lstm_hidden_size: int = 64
    lstm_epochs: int = 30
    lstm_batch_size: int = 16
    lstm_learning_rate: float = 1e-3


def get_active_window(cfg: ProjectConfig) -> DataWindow:
    """Return the active evaluation window based on toggle."""
    return cfg.extended_window if cfg.use_extended_window else cfg.base_window


def load_config() -> ProjectConfig:
    """Provide a single configuration instance for reuse across modules."""
    return ProjectConfig()
