"""Shared pytest fixtures.

Loading the cleaned dataset is the slowest part of any test, so we cache it
once at session scope and hand small subsamples to the individual tests."""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

# Silence pgmpy's deprecation noise during the test run; tests assert on
# behaviour, not on stderr cleanliness.
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def clean_df() -> pd.DataFrame:
    """Full preprocessed DataFrame, loaded once per test session."""
    from src.data.load import load_clean

    return load_clean()


@pytest.fixture()
def small_sample(clean_df: pd.DataFrame) -> pd.DataFrame:
    """A 2000-row stratified subsample, ample for smoke tests but fast."""
    return (
        clean_df.groupby("is_claim", group_keys=False)
        .apply(lambda g: g.sample(
            n=min(len(g), int(round(len(g) / len(clean_df) * 2000))),
            random_state=0,
        ))
        .reset_index(drop=True)
    )
