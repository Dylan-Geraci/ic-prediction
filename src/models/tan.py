"""
Tree-Augmented Naive Bayes (TAN) Bayesian network.

A naive Bayes classifier assumes every feature is conditionally independent
of every other feature given the class. That assumption is convenient but
usually wrong. TAN relaxes it: in addition to the class-to-feature edges of
naive Bayes, TAN allows each feature to have *one* extra parent that is
another feature, with the augmenting structure forming a tree (built from
mutual information between features).

TAN is a strong, well-known baseline for classification because:

  * It is provably optimal among Bayesian networks of a certain restricted
    family (Friedman, Geiger, Goldszmidt, 1997).
  * It strikes a good balance between expressive power and the small CPT
    sizes you need when training data is limited or imbalanced.

We use ``pgmpy.estimators.TreeSearch`` with ``estimator_type="tan"`` and
``class_node=TARGET`` to build the structure.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from pgmpy.estimators import TreeSearch

from src.models.base import TARGET, BNModel


class TANModel(BNModel):
    """Tree-Augmented Naive Bayes using mutual-information edge weights."""

    name = "Tree-Augmented Naive Bayes"

    def _build_structure(self, df: pd.DataFrame) -> Iterable[tuple[str, str]]:
        # TreeSearch requires ``root_node`` to be distinct from ``class_node``;
        # the root is the feature that anchors the augmenting *feature tree*,
        # while the class node is the prediction target. Pick the first
        # non-target column as the root.
        first_feature = next(c for c in df.columns if c != TARGET)
        searcher = TreeSearch(df, root_node=first_feature)
        learned_dag = searcher.estimate(
            estimator_type="tan",
            class_node=TARGET,
            show_progress=False,
        )
        return list(learned_dag.edges())
