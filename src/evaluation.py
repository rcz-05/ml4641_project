"""Shared evaluation helpers for classification tasks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(y_true, y_pred, y_proba):
    """Return metrics dict, confusion matrix, and ROC data."""

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "test_samples": len(y_true),
    }

    roc_data = None
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_data = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
    except ValueError:
        metrics["roc_auc"] = None

    labels = sorted(np.unique(y_true).tolist())
    conf = confusion_matrix(y_true, y_pred, labels=labels)
    return metrics, conf, roc_data, labels
