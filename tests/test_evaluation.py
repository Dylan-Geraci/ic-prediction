"""Smoke test for the evaluation harness.

We run a deliberately tiny K-fold sweep (2 folds, ~500 rows) so the test is
fast, and just check that the comparison produces one row per (model, fold)
with finite metric values. The full sweep is an offline script."""

from __future__ import annotations

import math

from src.evaluation.compare import MODEL_CLASSES, run_comparison, summarize


def test_run_comparison_smoke(small_sample):
    long_df = run_comparison(df=small_sample, n_splits=2, sample_size=None)
    assert len(long_df) == len(MODEL_CLASSES) * 2
    # Every numeric metric must be finite (no NaNs, no infs) for the rows we
    # do generate.
    finite_cols = ["accuracy", "f1", "brier", "cross_entropy"]
    for col in finite_cols:
        assert long_df[col].apply(math.isfinite).all(), col


def test_summarize_returns_one_row_per_model(small_sample):
    long_df = run_comparison(df=small_sample, n_splits=2, sample_size=None)
    summary = summarize(long_df)
    assert len(summary) == len(MODEL_CLASSES)
    assert {"model", "auc_mean", "brier_mean"}.issubset(summary.columns)
