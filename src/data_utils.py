"""Data ingestion helpers for the midterm prototype."""

from __future__ import annotations

import datetime as dt
import shutil
import zipfile
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
import yfinance as yf
from huggingface_hub import snapshot_download

from .paths import RAW_DATA_DIR

PHRASEBANK_SUBSETS: dict[str, str] = {
    "sentences_allagree": "Sentences_AllAgree.txt",
    "sentences_75agree": "Sentences_75Agree.txt",
    "sentences_66agree": "Sentences_66Agree.txt",
    "sentences_50agree": "Sentences_50Agree.txt",
}

PHRASEBANK_ZIP = RAW_DATA_DIR / "FinancialPhraseBank-v1.0.zip"


def _download_phrasebank_zip() -> Path:
    """Ensure the Financial PhraseBank archive is available locally."""
    if PHRASEBANK_ZIP.exists():
        return PHRASEBANK_ZIP

    snapshot_path = snapshot_download("takala/financial_phrasebank", repo_type="dataset")
    src_zip = Path(snapshot_path) / "data" / "FinancialPhraseBank-v1.0.zip"
    if not src_zip.exists():
        raise FileNotFoundError("Could not locate FinancialPhraseBank archive inside the dataset snapshot.")

    shutil.copy(src_zip, PHRASEBANK_ZIP)
    return PHRASEBANK_ZIP


def load_phrasebank(subset: Literal["sentences_allagree", "sentences_75agree", "sentences_66agree", "sentences_50agree"] = "sentences_allagree") -> pd.DataFrame:
    """Load the requested subset of the Financial PhraseBank dataset."""

    subset_key = subset.lower()
    if subset_key not in PHRASEBANK_SUBSETS:
        raise ValueError(f"Unknown subset '{subset}'. Options: {list(PHRASEBANK_SUBSETS)}")

    zip_path = _download_phrasebank_zip()
    file_name = f"FinancialPhraseBank-v1.0/{PHRASEBANK_SUBSETS[subset_key]}"

    records: list[dict[str, str]] = []
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(file_name) as handle:
            for raw_line in handle:
                line = raw_line.decode("latin-1").strip()
                if not line or "@" not in line:
                    continue
                sentence, label = line.rsplit("@", 1)
                records.append({"sentence": sentence.strip(), "label": label.strip()})

    df = pd.DataFrame(records)
    return df


def fetch_price_data(
    ticker: str,
    start_date: dt.date,
    end_date: dt.date,
    source: Literal["csv", "yfinance"] = "csv",
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load OHLCV data for the supplied ticker within a date window."""

    if csv_path is None:
        candidate_names = [
            "AAPL.csv",
            "apple_stock_history.csv",
            "apple_stock_history_1990-2024.csv",
            "apple_stock_data.csv",
        ]
        for name in candidate_names:
            potential_path = RAW_DATA_DIR / name
            if potential_path.exists():
                csv_path = potential_path
                break
        else:
            csv_path = RAW_DATA_DIR / "apple_stock_history.csv"
    data: pd.DataFrame | None = None

    if source == "csv" and csv_path.exists():
        data = pd.read_csv(csv_path)
        data.columns = [col.lower() for col in data.columns]
        data = data.rename(
            columns={
                "adj close": "adj_close",
                "adjclose": "adj_close",
            }
        )
        if "adj_close" not in data.columns and "close" in data.columns:
            data["adj_close"] = data["close"]
        data["date"] = pd.to_datetime(data["date"]).dt.date
    else:
        data = yf.download(
            ticker,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            auto_adjust=False,
            progress=False,
        )
        if data.empty:
            raise RuntimeError(f"No price data returned for {ticker}.")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [level0.lower() for level0, _ in data.columns.to_flat_index()]
        else:
            data.columns = [col.lower() for col in data.columns]
        data = data.reset_index().rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
                "adj close": "adj_close",
            }
        )
        data["date"] = pd.to_datetime(data["date"]).dt.date

    mask = (data["date"] >= start_date) & (data["date"] <= end_date)
    windowed = data.loc[mask].copy()
    if windowed.empty:
        raise ValueError(f"No price rows for {ticker} between {start_date} and {end_date}.")
    return windowed


def load_stock_news(ticker: str, max_rows: int = 300, start_date: Optional[dt.date] = None, end_date: Optional[dt.date] = None) -> pd.DataFrame:
    """Load dated news for the given ticker with union of local datasets and fallback to HF."""

    def _label(score: float) -> str:
        if score > 0.05:
            return "positive"
        if score < -0.05:
            return "negative"
        return "neutral"

    frames: list[pd.DataFrame] = []

    # Frankossai dataset: apple_news_data.csv
    apple_news_path = RAW_DATA_DIR / "apple_news_data.csv"
    if apple_news_path.exists():
        df = pd.read_csv(apple_news_path)
        if "date" not in df.columns:
            raise ValueError("apple_news_data.csv must include a 'date' column.")
        df["published_at"] = pd.to_datetime(df["date"], utc=True)
        df["date"] = df["published_at"].dt.date
        title_col = "title" if "title" in df.columns else "content"
        text_col = "content" if "content" in df.columns else title_col
        df["title"] = df[title_col].fillna("")
        df["summary"] = df[text_col].fillna("")
        df["text"] = df[text_col].fillna("")
        df["overall_sentiment_score"] = df.get("sentiment_polarity", 0.0).fillna(0.0)
        df["prob_positive"] = df.get("sentiment_pos", 0.0).fillna(0.0)
        df["prob_neutral"] = df.get("sentiment_neu", 0.0).fillna(0.0)
        df["prob_negative"] = df.get("sentiment_neg", 0.0).fillna(0.0)
        df["overall_sentiment_label"] = df["overall_sentiment_score"].apply(_label)
        df["ticker_sentiment_score"] = df["overall_sentiment_score"]
        df["ticker_sentiment_label"] = df["overall_sentiment_label"]
        df["relevance_score"] = 1.0
        df["source"] = "apple_news_data"
        if "symbols" in df.columns:
            df = df[df["symbols"].fillna("").str.upper().str.contains(ticker.upper())]
        frames.append(df)

    # Miguelaenlle dataset: analyst_ratings_processed.csv (firehose, neutral placeholders)
    ratings_path = RAW_DATA_DIR / "analyst_ratings_processed.csv"
    if ratings_path.exists():
        ratings = pd.read_csv(ratings_path)
        if {"date", "stock"}.issubset(ratings.columns):
            ratings = ratings[ratings["stock"].str.upper() == ticker.upper()]
            if not ratings.empty:
                ratings["published_at"] = pd.to_datetime(ratings["date"], utc=True)
                ratings["date"] = ratings["published_at"].dt.date
                ratings["title"] = ratings["title"].fillna("")
                ratings["summary"] = ratings["title"]
                ratings["text"] = ratings["title"]
                ratings["overall_sentiment_score"] = 0.0
                ratings["prob_positive"] = 0.0
                ratings["prob_neutral"] = 0.0
                ratings["prob_negative"] = 0.0
                ratings["overall_sentiment_label"] = "neutral"
                ratings["ticker_sentiment_score"] = 0.0
                ratings["ticker_sentiment_label"] = "neutral"
                ratings["relevance_score"] = 1.0
                ratings["source"] = "analyst_ratings"
                frames.append(ratings)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined["published_at"] = pd.to_datetime(combined["published_at"], utc=True)
        combined = combined.sort_values("published_at")
        if start_date:
            combined = combined[combined["published_at"].dt.date >= start_date]
        if end_date:
            combined = combined[combined["published_at"].dt.date <= end_date]
        # Sparsity check for H1 2020 in the frankossai subset
        h1_mask = (
            (combined["source"] == "apple_news_data")
            & (combined["published_at"].dt.date >= dt.date(2020, 1, 1))
            & (combined["published_at"].dt.date <= dt.date(2020, 6, 30))
        )
        h1_count = h1_mask.sum()
        if h1_count < 100:
            print(
                f"[WARN] apple_news_data coverage Jan-Jun 2020 is sparse ({h1_count} rows); "
                "H1 analysis will rely primarily on analyst_ratings."
            )
        if max_rows:
            combined = combined.tail(max_rows)
        columns = [
            "published_at",
            "date",
            "title",
            "summary",
            "text",
            "source",
            "overall_sentiment_score",
            "overall_sentiment_label",
            "ticker_sentiment_score",
            "ticker_sentiment_label",
            "relevance_score",
            "prob_positive",
            "prob_neutral",
            "prob_negative",
        ]
        return combined[columns]

    # Fallback: Hugging Face stock_news_sentiment dataset.
    snapshot_path = Path(snapshot_download("ic-fspml/stock_news_sentiment", repo_type="dataset"))
    data_dir = snapshot_path / "data"
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("Could not locate parquet files in the stock_news_sentiment dataset.")

    frames = [pd.read_parquet(file) for file in parquet_files]
    combined = pd.concat(frames, ignore_index=True)
    filtered = combined[combined["ticker"].str.upper() == ticker.upper()]
    if filtered.empty:
        raise ValueError(f"No news rows found for ticker {ticker} in the stock_news_sentiment dataset.")

    filtered = filtered.copy()
    filtered["published_at"] = pd.to_datetime(filtered["article_date"])
    filtered["date"] = filtered["published_at"].dt.date
    filtered["title"] = filtered["article_headline"].fillna("")
    filtered["summary"] = filtered["article_headline"].fillna("")
    filtered["text"] = filtered["article_headline"].fillna("")
    sentiment_map = {
        "strongly bearish": -1.0,
        "mildly bearish": -0.5,
        "neutral": 0.0,
        "mildly bullish": 0.5,
        "strongly bullish": 1.0,
    }
    filtered["overall_sentiment_score"] = filtered["label"].map(sentiment_map).fillna(0.0)
    filtered["overall_sentiment_label"] = filtered["label"]
    filtered["ticker_sentiment_score"] = filtered["overall_sentiment_score"]
    filtered["ticker_sentiment_label"] = filtered["label"]
    filtered["relevance_score"] = 1.0
    filtered["prob_positive"] = (filtered["label"] == "strongly bullish").astype(float)
    filtered["prob_neutral"] = (filtered["label"] == "neutral").astype(float)
    filtered["prob_negative"] = (filtered["label"] == "strongly bearish").astype(float)
    filtered["source"] = filtered["name"].fillna("")

    filtered = filtered.sort_values("published_at")
    if start_date:
        filtered = filtered[filtered["published_at"].dt.date >= start_date]
    if end_date:
        filtered = filtered[filtered["published_at"].dt.date <= end_date]
    if max_rows:
        filtered = filtered.tail(max_rows)
    columns = [
        "published_at",
        "date",
        "title",
        "summary",
        "text",
        "source",
        "overall_sentiment_score",
        "overall_sentiment_label",
        "ticker_sentiment_score",
        "ticker_sentiment_label",
        "relevance_score",
        "prob_positive",
        "prob_neutral",
        "prob_negative",
    ]
    return filtered[columns]
