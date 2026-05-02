"""
Cross-validated comparison of the three candidate Bayesian networks.

This is the script that answers the professor's added requirement: *"Make sure
there is some kind of evaluation to decide which specific Bayesian network
works best."* It performs stratified K-fold cross-validation on the cleaned
dataset, fits each candidate on the training fold, scores it on the held-out
fold, and aggregates the metrics into a single side-by-side table.

Usage::

    python -m src.evaluation.compare

Outputs:
    results/comparison.csv  -- one row per (model, metric) with mean and std
    results/comparison.png  -- grouped bar chart of the headline metrics
    results/best_model.txt  -- name of the winning model (used by the UI)
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; we save PNGs not show windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data.load import load_clean
from src.data.preprocess import TARGET
from src.evaluation.metrics import METRICS
from src.models.base import BNModel, TARGET_POSITIVE
from src.models.expert_dag import ExpertDAG
from src.models.hill_climb import HillClimbModel
from src.models.tan import TANModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

# The three candidates. Order is preserved in the comparison table and chart.
MODEL_CLASSES: list[type[BNModel]] = [ExpertDAG, HillClimbModel, TANModel]

# K-fold splitter config.
N_SPLITS = 5
RANDOM_STATE = 42

# Evaluation sample size. ``None`` means "use the full dataset" (~58k rows).
# Set to an int to subsample for faster iteration.
EVAL_SAMPLE_SIZE: int | None = None


def _evaluate_fold(
    model: BNModel, train_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict[str, float]:
    """Fit ``model`` on ``train_df`` and score on ``test_df``.

    Returns a dict of metric name -> value, plus the wall-clock fit time and
    mean per-row inference time (both useful for the comparison table).
    """
    fit_start = time.perf_counter()
    model.fit(train_df)
    fit_time = time.perf_counter() - fit_start

    inf_start = time.perf_counter()
    y_proba = model.predict_proba_batch(test_df)
    inf_time = (time.perf_counter() - inf_start) / max(len(test_df), 1)

    y_true = test_df[TARGET].to_numpy()
    out = {name: fn(y_true, y_proba) for name, (fn, _) in METRICS.items()}
    out["fit_time_s"] = fit_time
    out["inference_ms_per_row"] = inf_time * 1000.0
    return out


def run_comparison(
    df: pd.DataFrame | None = None,
    n_splits: int = N_SPLITS,
    sample_size: int | None = EVAL_SAMPLE_SIZE,
) -> pd.DataFrame:
    """Run the K-fold comparison and return the long-form results DataFrame.

    Each row is one (model, fold) measurement. The caller (or :func:`main`)
    can then aggregate into a summary table.
    """
    if df is None:
        df = load_clean()
    if sample_size is not None and sample_size < len(df):
        # Stratify the subsample on the target so each fold sees realistic
        # claim prevalence even when ``sample_size`` is small.
        _, df = train_test_split(
            df, test_size=sample_size, stratify=df[TARGET], random_state=RANDOM_STATE
        )
        df = df.reset_index(drop=True)

    y = df[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, y)):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        for cls in MODEL_CLASSES:
            print(f"  fold {fold_idx + 1}/{n_splits}: {cls.name} ...", flush=True)
            model = cls()
            metrics = _evaluate_fold(model, train_df, test_df)
            metrics.update({"model": cls.name, "fold": fold_idx})
            rows.append(metrics)

    return pd.DataFrame(rows)


def summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-fold results into a mean ± std table per model."""
    # Columns to summarize: every metric plus timing.
    metric_cols = [c for c in long_df.columns if c not in ("model", "fold")]
    grouped = long_df.groupby("model")[metric_cols]
    summary = grouped.agg(["mean", "std"])
    # Flatten the multi-index columns: ("auc","mean") -> "auc_mean".
    summary.columns = [f"{m}_{stat}" for m, stat in summary.columns]
    return summary.reset_index()


def _pick_winner(summary: pd.DataFrame) -> str:
    """Choose the best model by AUC, breaking ties on (lowest) Brier score.

    AUC measures ranking quality and is robust to class imbalance; Brier
    captures probability calibration, which matters for the UI's gauge.
    """
    ranked = summary.sort_values(
        by=["auc_mean", "brier_mean"],
        ascending=[False, True],
    )
    return str(ranked.iloc[0]["model"])


def _plot_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart of the headline classification metrics."""
    headline = ["f1", "auc", "average_precision", "brier"]
    means = summary.set_index("model")[[f"{m}_mean" for m in headline]]
    means.columns = headline

    fig, ax = plt.subplots(figsize=(9, 5))
    means.plot(kind="bar", ax=ax)
    ax.set_title("Bayesian Network Comparison (5-fold stratified CV)")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_xticklabels(means.index, rotation=0)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore")  # silence pgmpy's deprecation chatter
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows_label = "all available" if EVAL_SAMPLE_SIZE is None else f"{EVAL_SAMPLE_SIZE}"
    print(f"Loading data and running {N_SPLITS}-fold CV on {rows_label} rows ...")
    long_df = run_comparison()
    summary = summarize(long_df)

    long_df.to_csv(RESULTS_DIR / "comparison_long.csv", index=False)
    summary.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    _plot_comparison(summary, RESULTS_DIR / "comparison.png")

    winner = _pick_winner(summary)
    (RESULTS_DIR / "best_model.txt").write_text(winner)

    # Pretty-print a short summary so the user sees the result without
    # opening the CSV.
    print("\n=== Summary (mean over folds) ===")
    headline_cols = ["model", "accuracy_mean", "f1_mean", "auc_mean",
                     "average_precision_mean", "brier_mean",
                     "cross_entropy_mean", "fit_time_s_mean"]
    print(summary[headline_cols].to_string(index=False))
    print(f"\nWinner (by AUC, then Brier): {winner}")
    print(f"Wrote: {RESULTS_DIR / 'comparison.csv'}")
    print(f"Wrote: {RESULTS_DIR / 'comparison.png'}")


if __name__ == "__main__":
    main()
