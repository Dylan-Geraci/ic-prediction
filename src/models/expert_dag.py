"""
Hand-designed Bayesian network based on domain knowledge about used cars
and insurance-claim risk.

Structural intuition:

    age_of_car ─────────┐
    policy_tenure ──────┼─► is_claim
    area_cluster ───────┘

We also encode inter-feature relationships that a domain expert would
consider obvious:

  * ``segment`` (vehicle class) influences ``displacement``, ``airbags``,
    and ``ncap_rating`` — premium segments ship larger engines and more
    safety equipment.
  * ``displacement`` influences ``fuel_type`` — bigger engines lean diesel,
    smaller engines lean petrol.
  * ``age_of_policyholder`` influences ``policy_tenure`` — older drivers
    have generally been insured for longer.
  * ``population_density`` influences ``area_cluster`` — urban areas tend
    to share denser, more claim-prone geographies.

These inter-feature edges matter: without them the network behaves like a
naive Bayes classifier (all features independent given the target). With
them the joint distribution captures real correlations.

This is the *Expert DAG* baseline. It will be compared against two
data-driven structures (Hill-Climb and TAN).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.models.base import TARGET, BNModel


class ExpertDAG(BNModel):
    """Domain-knowledge DAG. No data is used to choose its structure; only
    the CPTs are learned from data."""

    name = "Expert DAG"

    def _build_structure(self, df: pd.DataFrame) -> Iterable[tuple[str, str]]:
        # Direct parents of the target are the three strongest signals in
        # the data: how long the driver has been insured, where the car is
        # registered, and how old the car is. With three discrete parents
        # (3 x 4 x 3 = 36 parent-state combinations) and ~3,700 claim rows
        # in the full dataset, the CPT averages ~100 claim observations per
        # cell — enough for stable conditional probabilities.
        return [
            # Inter-feature dependencies.
            ("segment", "displacement"),
            ("segment", "airbags"),
            ("segment", "ncap_rating"),
            ("displacement", "fuel_type"),
            ("age_of_policyholder", "policy_tenure"),
            ("population_density", "area_cluster"),

            # Direct parents of the target (strongest predictors).
            ("policy_tenure", TARGET),
            ("area_cluster", TARGET),
            ("age_of_car", TARGET),
        ]
