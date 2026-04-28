"""
Preprocess the raw Kaggle Car Insurance Claim CSV into a small, fully
categorical dataset suitable for a Bayesian network.

Bayesian networks in pgmpy require *discrete* (categorical) variables so that
each node has a finite Conditional Probability Table (CPT). This module:

  1. Loads the raw train.csv.
  2. Selects a small set of vehicle-centric features (we deliberately keep the
     feature set small so the learned DAG and the CPTs stay readable in the
     report and the Streamlit demo).
  3. Discretizes continuous features into 2-3 ordered bins.
  4. Writes the result to data/processed/clean.csv.

The feature set was chosen to match the *Second-hand Car Assistant* framing:
all features are things a buyer can read off a vehicle listing.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Project root: this file lives at <root>/src/data/preprocess.py, so go up 3.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = PROJECT_ROOT / "data" / "train.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_CSV = PROCESSED_DIR / "clean.csv"

# Vehicle-centric features kept for the network. Everything else (policy_id,
# policy_tenure, area_cluster, model, ...) is dropped because it is either an
# identifier, a policy/buyer attribute (not a vehicle attribute), or has too
# many categories to fit cleanly into a Bayesian network's CPTs.
FEATURES = [
    "age_of_car",
    "segment",
    "fuel_type",
    "transmission_type",
    "displacement",
    "airbags",
    "ncap_rating",
    "rear_brakes_type",
]
TARGET = "is_claim"


def _bin_age_of_car(series: pd.Series) -> pd.Series:
    """Bin the (already 0-1 normalized) age_of_car into Low / Medium / High."""
    # Quantile-based cuts give roughly balanced bin sizes regardless of the
    # underlying distribution shape, which matters for CPT estimation stability.
    return pd.qcut(series, q=3, labels=["Low", "Medium", "High"])


def _bin_displacement(series: pd.Series) -> pd.Series:
    """Bin engine displacement (cc) into Small / Medium / Large."""
    return pd.qcut(series, q=3, labels=["Small", "Medium", "Large"])


def _bin_airbags(series: pd.Series) -> pd.Series:
    """Two bins: Few (<=2 airbags) vs. Many (>2). The dataset effectively has
    only two clusters of values around 2 and 6, so a simple threshold suffices.
    """
    return series.apply(lambda x: "Few" if x <= 2 else "Many").astype("category")


def _bin_ncap(series: pd.Series) -> pd.Series:
    """NCAP safety rating (0-5) collapsed into Low / Medium / High."""
    def _label(x: int) -> str:
        if x <= 1:
            return "Low"
        if x <= 3:
            return "Medium"
        return "High"

    return series.apply(_label).astype("category")


def _bin_target(series: pd.Series) -> pd.Series:
    """is_claim is already 0/1; relabel to NoClaim/Claim for readability in
    CPTs and the Streamlit gauge."""
    return series.map({0: "NoClaim", 1: "Claim"}).astype("category")


def preprocess(raw_csv: Path = RAW_CSV) -> pd.DataFrame:
    """Load raw CSV, discretize, return the cleaned DataFrame.

    The DataFrame contains only the columns in ``FEATURES + [TARGET]`` and all
    columns are categorical (``pandas.CategoricalDtype``).
    """
    df = pd.read_csv(raw_csv, usecols=FEATURES + [TARGET])

    # Apply discretization to numerics; leave categoricals as-is but cast to
    # the categorical dtype so pgmpy treats them consistently.
    df["age_of_car"] = _bin_age_of_car(df["age_of_car"])
    df["displacement"] = _bin_displacement(df["displacement"])
    df["airbags"] = _bin_airbags(df["airbags"])
    df["ncap_rating"] = _bin_ncap(df["ncap_rating"])
    df[TARGET] = _bin_target(df[TARGET])

    # Cast the remaining string features to category dtype so every column has
    # a known finite state space.
    for col in ("segment", "fuel_type", "transmission_type", "rear_brakes_type"):
        df[col] = df[col].astype("category")

    # pgmpy works most reliably with plain string values rather than pandas
    # Categorical objects when serialized through CSV, so coerce to str at the
    # end. (We keep the categorical dtype for in-memory speed up to this point.)
    return df.astype(str)


def main() -> None:
    """CLI entry point: write data/processed/clean.csv."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = preprocess()
    df.to_csv(PROCESSED_CSV, index=False)
    print(f"Wrote {PROCESSED_CSV} ({len(df)} rows, {len(df.columns)} cols).")
    # Quick sanity print so the user can eyeball the bin distributions.
    for col in df.columns:
        counts = df[col].value_counts().to_dict()
        print(f"  {col}: {counts}")


if __name__ == "__main__":
    main()
