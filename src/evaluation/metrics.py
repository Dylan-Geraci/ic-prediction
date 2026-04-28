"""
Per-fold metric calculations for comparing Bayesian networks.

Each function takes the held-out test fold's true labels and the model's
predicted positive-class probabilities, then returns a single float. Keeping
each metric self-contained makes it easy to add or remove metrics from the
comparison without touching the cross-validation runner.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

from src.models.base import TARGET_POSITIVE


def _to_binary(labels: np.ndarray) -> np.ndarray:
    """Convert string labels (``Claim``/``NoClaim``) to 1/0 for sklearn."""
    return (labels == TARGET_POSITIVE).astype(int)


def accuracy(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> float:
    """Hard-label accuracy at the given probability threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    return float(accuracy_score(_to_binary(y_true), y_pred))


def f1(y_true: np.ndarray, y_proba: np.ndarray, threshold: float | None = None) -> float:
    """F1 of the positive (Claim) class.

    With ~6% positive prevalence, the standard 0.5 threshold causes every
    model to predict only "NoClaim" (and thus F1=0), which hides genuine
    differences between models. We default the threshold to the *empirical
    base rate* in the test fold instead. This is a defensible choice for
    imbalanced binary problems: it asks "does the model rank true positives
    above the marginal probability?" rather than the trivially-failing
    "above 50%?".
    """
    y_bin = _to_binary(y_true)
    if threshold is None:
        threshold = float(y_bin.mean()) if len(y_bin) else 0.5
    y_pred = (y_proba >= threshold).astype(int)
    return float(f1_score(y_bin, y_pred, zero_division=0))


def average_precision(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Area under the precision-recall curve. Threshold-free and the standard
    metric for ranking quality on imbalanced binary problems."""
    y_bin = _to_binary(y_true)
    if len(np.unique(y_bin)) < 2:
        return float("nan")
    return float(average_precision_score(y_bin, y_proba))


def brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Brier score = mean squared error between predicted probability and
    actual outcome. Lower is better. Sensitive to *calibration*, which matters
    because our UI displays the probability number directly to the user."""
    return float(brier_score_loss(_to_binary(y_true), y_proba))


def auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """ROC-AUC. Threshold-independent ranking quality."""
    y_bin = _to_binary(y_true)
    # If a fold happens to contain only one class, AUC is undefined.
    if len(np.unique(y_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(y_bin, y_proba))


def cross_entropy(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Log-loss / cross-entropy. Lower is better. Equivalent to negative
    average log-likelihood of the labels under the model's predictions, which
    is what the Bayesian-network literature calls *predictive log-likelihood*
    (computed only over the target variable rather than the full joint)."""
    # Clip to keep the log finite when predictions hit exactly 0 or 1.
    eps = 1e-12
    p = np.clip(y_proba, eps, 1 - eps)
    return float(log_loss(_to_binary(y_true), p, labels=[0, 1]))


# Registry consumed by compare.py. Each entry: (display name, function,
# "lower-is-better" flag for the summary table).
METRICS: dict[str, tuple[Callable, bool]] = {
    "accuracy": (accuracy, False),
    "f1": (f1, False),
    "auc": (auc, False),
    "average_precision": (average_precision, False),
    "brier": (brier, True),
    "cross_entropy": (cross_entropy, True),
}
