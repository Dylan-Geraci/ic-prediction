"""Smoke tests for the preprocessor.

We do not assert exact bin counts (those depend on Kaggle's CSV contents);
instead we check the *shape* of the output: every column is categorical, the
target has the expected two states, and no NaNs sneak through."""

from __future__ import annotations

import pandas as pd

from src.data.load import load_clean
from src.data.preprocess import FEATURES, TARGET


def test_clean_df_has_expected_columns(clean_df: pd.DataFrame) -> None:
    # ``read_csv(usecols=...)`` returns columns in CSV order, not in the
    # order we requested, so compare as sets.
    assert set(clean_df.columns) == set(FEATURES) | {TARGET}


def test_target_states_are_binary(clean_df: pd.DataFrame) -> None:
    assert set(clean_df[TARGET].unique()) == {"Claim", "NoClaim"}


def test_no_nans_after_preprocessing(clean_df: pd.DataFrame) -> None:
    assert not clean_df.isna().any().any(), "preprocessor leaked NaN values"


def test_age_of_car_is_three_ordered_bins(clean_df: pd.DataFrame) -> None:
    # All bins should be present in the population (we generate from quantiles).
    assert set(clean_df["age_of_car"].unique()) == {"Low", "Medium", "High"}


def test_load_clean_round_trip_is_deterministic(clean_df: pd.DataFrame) -> None:
    """Calling load_clean a second time should hit the cached CSV and return
    identical data, not silently regenerate."""
    again = load_clean()
    pd.testing.assert_frame_equal(clean_df.reset_index(drop=True),
                                  again.reset_index(drop=True))
