"""
Data-driven Bayesian network learned by hill-climbing the BIC score.

Hill-climbing is a greedy local search over DAGs. Starting from an empty
graph it repeatedly adds, removes, or reverses single edges, each time
keeping the change that most improves the score. We use the BIC
(Bayesian Information Criterion) as the score, which trades off:

    log-likelihood of the data    vs.    a complexity penalty

so the search is naturally biased against over-fit dense graphs.

This is the model's chance to *disagree* with the expert DAG: if the data
strongly suggests a structure the human did not draw, hill-climb will find it
(within the limits of greedy search). It is therefore a meaningful comparison
point for the report.

Note on runtime: hill-climb on the full dataset (58k rows × 9 vars) runs in
a few seconds because pgmpy caches sufficient statistics. We also pin a small
``max_indegree`` so no node ends up with an explosively large CPT.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from pgmpy.estimators import HillClimbSearch

from src.models.base import TARGET, BNModel


class HillClimbModel(BNModel):
    """DAG structure learned via :class:`HillClimbSearch` with BIC scoring."""

    name = "Hill-Climb (BIC)"

    # Cap each node's number of parents. Without a cap, hill-climb can find
    # high-fan-in structures that produce huge CPTs and slow inference, which
    # would also penalize this model unfairly on the inference-time metric.
    MAX_INDEGREE = 3

    def _build_structure(self, df: pd.DataFrame) -> Iterable[tuple[str, str]]:
        searcher = HillClimbSearch(df)
        # ``estimate`` returns a DAG (pgmpy DAG object). We pull its edges.
        # ``scoring_method='bic-d'`` selects the discrete-variable BIC score.
        # pgmpy 1.x dispatches this to the canonical BIC implementation.
        learned_dag = searcher.estimate(
            scoring_method="bic-d",
            max_indegree=self.MAX_INDEGREE,
            show_progress=False,
        )
        edges = list(learned_dag.edges())

        # Safety net: hill-climb may converge to a DAG where the target has
        # no parents and no children connected to other features. That would
        # make the prediction degenerate to the marginal P(is_claim).
        # We force at least one edge from a known-informative feature
        # (age_of_car) into the target if the learned DAG isolated it.
        target_connected = any(TARGET in edge for edge in edges)
        if not target_connected:
            edges.append(("age_of_car", TARGET))
        return edges
