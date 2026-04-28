"""
Hand-designed Bayesian network based on domain knowledge about used cars.

Structural intuition (each arrow encodes a causal-ish "influences" claim):

    age_of_car ──┐
    segment ─────┤
    fuel_type ───┼─► is_claim
    displacement ┤
    airbags ─────┤
    ncap_rating ─┘
    transmission_type ──► is_claim
    rear_brakes_type ───► is_claim

We also encode a few structural relationships *between* features that a
mechanic or auto-journalist would consider obvious:

  * ``segment`` (vehicle class) influences ``displacement`` — small hatchbacks
    have small engines, SUVs have larger ones.
  * ``segment`` influences ``airbags`` and ``ncap_rating`` — premium segments
    ship with more safety equipment and tend to score higher in crash tests.
  * ``displacement`` influences ``transmission_type`` — bigger engines are more
    often paired with automatics.

These inter-feature edges matter: without them the network behaves like a
naive Bayes classifier (all features independent given the target). With them
the joint distribution captures real correlations and the held-out
log-likelihood improves.

This is the *Expert DAG* baseline. It will be compared against two
data-driven structures (Hill-Climb and TAN).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.models.base import TARGET, BNModel


class ExpertDAG(BNModel):
    """Domain-knowledge DAG. No data is used to choose its structure; only the
    CPTs are learned from data."""

    name = "Expert DAG"

    def _build_structure(self, df: pd.DataFrame) -> Iterable[tuple[str, str]]:
        # We list edges explicitly rather than building them programmatically
        # so the structure is easy to read and easy to defend in the report.
        #
        # Design choice: keep the *direct* parents of the target small
        # (3 parents: 18 parent-state combinations). With ~6% positive class
        # rate and ~58k rows, that yields ~200 claim observations per cell on
        # average, which is enough for stable conditional probabilities.
        # A wider parent set would shatter the CPT into thousands of mostly
        # empty cells and predictions would collapse to the prior on rare
        # evidence combinations.
        return [
            # Inter-feature dependencies (vehicle class drives spec & safety).
            ("segment", "displacement"),
            ("segment", "airbags"),
            ("segment", "ncap_rating"),
            ("displacement", "transmission_type"),
            ("displacement", "fuel_type"),
            ("ncap_rating", "rear_brakes_type"),

            # Direct parents of the target: the three signals a buyer would
            # most reasonably tie to long-run claim risk.
            #   * age_of_car: wear-and-tear is the dominant story for used-car
            #     reliability.
            #   * ncap_rating: crash safety directly affects claim severity.
            #   * airbags: occupant-protection signal that complements ncap.
            # The other features (displacement, fuel_type, transmission, brakes,
            # segment) influence the target *indirectly* through these three
            # and through each other.
            ("age_of_car", TARGET),
            ("ncap_rating", TARGET),
            ("airbags", TARGET),
        ]
