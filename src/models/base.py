"""
Shared base class for all candidate Bayesian networks.

The course requirement is to compare *several* Bayesian networks and pick the
best one. To do that fairly, every candidate must expose the same interface so
the evaluation harness can treat them identically. That common interface lives
here.

A ``BNModel`` subclass only needs to implement ``_build_structure(df)``, which
returns the list of directed edges defining the DAG. The base class handles
parameter learning (CPT fitting), batch inference, and metric-friendly outputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork

# The classification target across all models. Defined in one place so a future
# change to the dataset only touches this constant.
TARGET = "is_claim"
TARGET_POSITIVE = "Claim"  # the "positive" class for AUC / F1 / Brier


class BNModel(ABC):
    """Abstract base for a Bayesian network candidate.

    Subclasses must implement :meth:`_build_structure`. Everything else
    (parameter fitting, prediction, scoring) is shared so that the three
    candidate networks differ *only* in their structure-learning step. That is
    exactly what we want for the comparison: we hold the data, the parameter
    estimator, and the inference algorithm constant, and only vary the DAG.
    """

    # Human-readable name used in the comparison table and the Streamlit footer.
    name: str = "BNModel"

    def __init__(self) -> None:
        self.network: DiscreteBayesianNetwork | None = None
        self._inference: VariableElimination | None = None
        # Cached during fit() so predict_proba can reorder columns to match.
        self._feature_columns: list[str] = []

    # ---- structure (subclass responsibility) ---------------------------------

    @abstractmethod
    def _build_structure(self, df: pd.DataFrame) -> Iterable[tuple[str, str]]:
        """Return the directed edges of the DAG as ``(parent, child)`` tuples.

        The DAG must include ``TARGET`` as a node; otherwise the network has
        nothing to predict.
        """

    # ---- fitting --------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "BNModel":
        """Learn structure (delegated to the subclass) and CPTs (Bayesian
        estimator with a BDeu prior).

        We use ``BayesianEstimator`` rather than plain MLE because the dataset
        is class-imbalanced (only ~6% claims). MLE would assign zero
        probability to feature combinations never seen with the rare class,
        which then makes the model assign zero likelihood to perfectly
        plausible test rows. A Dirichlet/BDeu prior smooths those zeros.
        """
        edges = list(self._build_structure(df))
        self.network = DiscreteBayesianNetwork(edges)

        # If a feature column never appears in the structure (an isolated
        # node), pgmpy will not estimate a CPT for it. We add such columns as
        # parentless nodes so the network covers every observed variable.
        for col in df.columns:
            if col not in self.network.nodes():
                self.network.add_node(col)

        self.network.fit(
            df,
            estimator=BayesianEstimator,
            prior_type="BDeu",
            equivalent_sample_size=10,
        )
        self._inference = VariableElimination(self.network)
        self._feature_columns = [c for c in df.columns if c != TARGET]
        return self

    # ---- prediction -----------------------------------------------------------

    def predict_proba(self, evidence: dict[str, str]) -> dict[str, float]:
        """Return the posterior over ``TARGET`` given a single evidence dict.

        Used by the Streamlit app, which passes one set of slider values at a
        time. For batch evaluation we use :meth:`predict_proba_batch` instead,
        which is much faster.
        """
        self._require_fit()
        # Drop any evidence keys the network does not know about (defensive
        # against typos from the UI layer).
        clean_evidence = {k: v for k, v in evidence.items() if k in self.network.nodes()}
        result = self._inference.query(
            variables=[TARGET],
            evidence=clean_evidence,
            show_progress=False,
        )
        # ``result.state_names[TARGET]`` is the ordered list of states
        # corresponding to result.values.
        states = result.state_names[TARGET]
        return dict(zip(states, result.values.tolist()))

    def predict_proba_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Return ``P(TARGET = TARGET_POSITIVE | evidence)`` for every row.

        Returns a 1-D float array of length ``len(df)`` aligned with the input
        DataFrame's row order. Uses pgmpy's vectorized
        :meth:`DiscreteBayesianNetwork.predict_probability`, which is
        substantially faster than per-row ``query`` calls.
        """
        self._require_fit()
        # predict_probability needs the input *without* the target column.
        evidence_df = df.drop(columns=[TARGET], errors="ignore")
        # Keep only columns the network actually has. (Some structure-learning
        # algorithms occasionally drop nodes; this guards against KeyErrors.)
        known = [c for c in evidence_df.columns if c in self.network.nodes()]
        evidence_df = evidence_df[known]

        proba_df = self.network.predict_probability(evidence_df)
        # pgmpy returns columns named like ``is_claim_Claim`` /
        # ``is_claim_NoClaim``. We pluck the positive-class column.
        positive_col = f"{TARGET}_{TARGET_POSITIVE}"
        return proba_df[positive_col].to_numpy()

    def predict_batch(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Hard label predictions thresholded on the positive-class probability."""
        proba = self.predict_proba_batch(df)
        return np.where(proba >= threshold, TARGET_POSITIVE, "NoClaim")

    # ---- scoring helpers ------------------------------------------------------

    def log_likelihood(self, df: pd.DataFrame) -> float:
        """Average log-likelihood of the rows in ``df`` under the network.

        Used as one of the comparison metrics: a model that fits the joint
        distribution well will assign higher likelihood to held-out data.
        We sum log-probabilities across rows and divide by row count so the
        number is comparable across folds of different sizes.
        """
        self._require_fit()
        # pgmpy's BayesianNetwork has ``log_likelihood`` in some versions but
        # for portability we reconstruct it from the CPDs directly.
        log_probs: list[float] = []
        cpds = {cpd.variable: cpd for cpd in self.network.get_cpds()}
        for _, row in df.iterrows():
            total = 0.0
            for var, cpd in cpds.items():
                # Build the (var, parents) state tuple.
                state_value = row[var]
                parents = cpd.get_evidence()  # list of parent variable names
                parent_states = [row[p] for p in parents]
                try:
                    p = cpd.get_value(**{var: state_value, **dict(zip(parents, parent_states))})
                except Exception:
                    # Unknown state combination: treat as a tiny floor to avoid
                    # log(0). With BDeu smoothing this should be rare.
                    p = 1e-12
                total += np.log(max(p, 1e-12))
            log_probs.append(total)
        return float(np.mean(log_probs))

    # ---- internals ------------------------------------------------------------

    def _require_fit(self) -> None:
        if self.network is None or self._inference is None:
            raise RuntimeError(f"{self.name} has not been fit yet; call .fit(df) first.")
