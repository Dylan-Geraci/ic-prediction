"""
Convenience loader for the preprocessed dataset.

Other modules (models, evaluation, the Streamlit app) call ``load_clean()``
instead of touching the CSV directly, which keeps the file path and any
caching logic in one place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.preprocess import PROCESSED_CSV, preprocess


def load_clean(force_rebuild: bool = False) -> pd.DataFrame:
    """Return the cleaned, discretized DataFrame.

    If ``data/processed/clean.csv`` does not yet exist (first run), this
    transparently runs the preprocessor. Pass ``force_rebuild=True`` to
    regenerate it even if it already exists.
    """
    if force_rebuild or not PROCESSED_CSV.exists():
        PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
        df = preprocess()
        df.to_csv(PROCESSED_CSV, index=False)
        return df

    # Read everything as string so categorical states match exactly what the
    # preprocessor wrote (no pandas dtype inference surprises).
    return pd.read_csv(PROCESSED_CSV, dtype=str)


def feature_states(df: pd.DataFrame) -> dict[str, list[str]]:
    """Return ``{column: sorted unique states}``. Used by the Streamlit app to
    populate dropdowns and by models to declare CPT state lists."""
    return {col: sorted(df[col].unique().tolist()) for col in df.columns}
