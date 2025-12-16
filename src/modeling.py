"""Model training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import clean_text
from .evaluation import compute_classification_metrics


@dataclass
class SentimentModelResult:
    pipeline: Pipeline
    metrics: Dict[str, Any]
    confusion: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    classes: List[str]


@dataclass
class DirectionModelResult:
    model_name: str
    model: Any
    metrics: Dict[str, Any]
    feature_importances: pd.DataFrame | None
    roc_curve_data: Dict[str, np.ndarray] | None
    confusion: np.ndarray
    labels: List[int]


def train_sentiment_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> SentimentModelResult:
    """Train a TF-IDF + Logistic Regression classifier on the Financial PhraseBank dataset."""

    texts = clean_text(df["sentence"])
    labels = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")),
            # multi_class is auto by default; omit explicit arg for sklearn >=1.8 compatibility
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs")),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "support": dict(zip(*np.unique(y_test, return_counts=True))),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    confusion = confusion_matrix(y_test, y_pred, labels=sorted(df["label"].unique()))
    classes = list(pipeline.named_steps["clf"].classes_)
    return SentimentModelResult(
        pipeline=pipeline,
        metrics=metrics,
        confusion=confusion,
        y_test=y_test,
        y_pred=y_pred,
        classes=classes,
    )


def _select_feature_columns(frame: pd.DataFrame) -> List[str]:
    exclude = {"target_up", "date"}
    numeric_cols = [
        col for col in frame.columns if col not in exclude and pd.api.types.is_numeric_dtype(frame[col])
    ]
    return numeric_cols


def _chrono_split_indices(n_rows: int, train_frac: float = 0.6, val_frac: float = 0.2) -> tuple[int, int]:
    train_end = int(n_rows * train_frac)
    val_end = int(n_rows * (train_frac + val_frac))
    train_end = max(train_end, 1)
    val_end = max(val_end, train_end + 1)
    return train_end, val_end


def train_direction_model(
    frame: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    random_state: int = 42,
) -> DirectionModelResult:
    """Train a Random Forest classifier to predict next-day direction."""

    frame = frame.sort_values("date").reset_index(drop=True)
    if feature_cols is None:
        feature_cols = _select_feature_columns(frame)

    train_end, val_end = _chrono_split_indices(len(frame), train_frac, val_frac)
    train_df = frame.iloc[:train_end]
    test_df = frame.iloc[val_end:]
    if len(test_df) < 5:
        raise ValueError("Not enough samples in the test split. Increase window or adjust split fractions.")

    X_train = train_df[feature_cols]
    y_train = train_df["target_up"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_up"]

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_split=4,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics, confusion, roc_data, labels = compute_classification_metrics(y_test, y_pred, y_proba)

    feature_importances = (
        pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
    )

    return DirectionModelResult(
        model_name="random_forest",
        model=model,
        metrics=metrics,
        feature_importances=feature_importances,
        roc_curve_data=roc_data,
        confusion=confusion,
        labels=labels,
    )


def train_direction_logreg(
    frame: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    random_state: int = 42,
) -> DirectionModelResult:
    """Train a Logistic Regression classifier with standardized features."""

    frame = frame.sort_values("date").reset_index(drop=True)
    if feature_cols is None:
        feature_cols = _select_feature_columns(frame)

    train_end, val_end = _chrono_split_indices(len(frame), train_frac=train_frac, val_frac=val_frac)
    train_df = frame.iloc[:train_end]
    test_df = frame.iloc[val_end:]
    if len(test_df) < 5:
        raise ValueError("Not enough samples in the test split. Increase window or adjust split fractions.")

    X_train = train_df[feature_cols]
    y_train = train_df["target_up"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_up"]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    n_jobs=1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    metrics, confusion, roc_data, labels = compute_classification_metrics(y_test, y_pred, y_proba)

    clf = model.named_steps["clf"]
    coef = np.abs(clf.coef_[0])
    feature_importances = pd.DataFrame({"feature": feature_cols, "importance": coef}).sort_values(
        "importance", ascending=False
    )

    return DirectionModelResult(
        model_name="logistic_regression",
        model=model,
        metrics=metrics,
        feature_importances=feature_importances,
        roc_curve_data=roc_data,
        confusion=confusion,
        labels=labels,
    )


def train_direction_gradient_boosting(
    frame: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    random_state: int = 42,
) -> DirectionModelResult:
    """Train a Gradient Boosting classifier to compare against RF/LogReg baselines."""

    frame = frame.sort_values("date").reset_index(drop=True)
    if feature_cols is None:
        feature_cols = _select_feature_columns(frame)

    train_end, val_end = _chrono_split_indices(len(frame), train_frac=train_frac, val_frac=val_frac)
    train_df = frame.iloc[:train_end]
    test_df = frame.iloc[val_end:]
    if len(test_df) < 5:
        raise ValueError("Not enough samples in the test split. Increase window or adjust split fractions.")

    X_train = train_df[feature_cols]
    y_train = train_df["target_up"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_up"]

    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    metrics, confusion, roc_data, labels = compute_classification_metrics(y_test, y_pred, y_proba)

    feature_importances = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return DirectionModelResult(
        model_name="gradient_boosting",
        model=model,
        metrics=metrics,
        feature_importances=feature_importances,
        roc_curve_data=roc_data,
        confusion=confusion,
        labels=labels,
    )


def train_direction_extratrees(
    frame: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    random_state: int = 42,
) -> DirectionModelResult:
    """Train an ExtraTrees classifier for another non-linear baseline."""

    frame = frame.sort_values("date").reset_index(drop=True)
    if feature_cols is None:
        feature_cols = _select_feature_columns(frame)

    train_end, val_end = _chrono_split_indices(len(frame), train_frac=train_frac, val_frac=val_frac)
    train_df = frame.iloc[:train_end]
    test_df = frame.iloc[val_end:]
    if len(test_df) < 5:
        raise ValueError("Not enough samples in the test split. Increase window or adjust split fractions.")

    X_train = train_df[feature_cols]
    y_train = train_df["target_up"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_up"]

    model = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    metrics, confusion, roc_data, labels = compute_classification_metrics(y_test, y_pred, y_proba)

    feature_importances = pd.DataFrame(
        {"feature": feature_cols, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return DirectionModelResult(
        model_name="extra_trees",
        model=model,
        metrics=metrics,
        feature_importances=feature_importances,
        roc_curve_data=roc_data,
        confusion=confusion,
        labels=labels,
    )


def tune_random_forest(
    frame: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    random_state: int = 42,
    candidate_params: Sequence[dict] | None = None,
) -> dict:
    """Simple grid search over a few RF hyperparameters to log best settings."""

    frame = frame.sort_values("date").reset_index(drop=True)
    if feature_cols is None:
        feature_cols = _select_feature_columns(frame)
    train_end, val_end = _chrono_split_indices(len(frame), train_frac, val_frac)
    train_df = frame.iloc[:train_end]
    test_df = frame.iloc[val_end:]
    if len(test_df) < 15:
        raise ValueError("Not enough samples in the test split. Increase window or adjust split fractions.")

    X_train = train_df[feature_cols]
    y_train = train_df["target_up"]
    X_test = test_df[feature_cols]
    y_test = test_df["target_up"]

    if candidate_params is None:
        candidate_params = [
            {"n_estimators": 300, "max_depth": 6, "min_samples_split": 2},
            {"n_estimators": 400, "max_depth": 8, "min_samples_split": 4},
        ]

    best_result: dict[str, Any] = {"roc_auc": -np.inf}
    for params in candidate_params:
        model = RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            **params,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics, _, _, _ = compute_classification_metrics(y_test, y_pred, y_proba)
        if metrics["roc_auc"] is not None and metrics["roc_auc"] > best_result.get("roc_auc", -np.inf):
            best_result = {
                "params": params,
                "metrics": metrics,
            }
    return best_result
