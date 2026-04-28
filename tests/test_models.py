"""Smoke tests for the three Bayesian network candidates.

We assert each model:

  * fits without raising,
  * produces a DAG that includes the target node,
  * returns a normalized probability distribution from ``predict_proba``,
  * returns one probability per row from ``predict_proba_batch``."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.models.base import TARGET
from src.models.expert_dag import ExpertDAG
from src.models.hill_climb import HillClimbModel
from src.models.tan import TANModel

ALL_MODELS = [ExpertDAG, HillClimbModel, TANModel]


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_model_fits_and_includes_target(model_cls, small_sample: pd.DataFrame):
    model = model_cls().fit(small_sample)
    assert TARGET in model.network.nodes()


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_predict_proba_is_normalized(model_cls, small_sample: pd.DataFrame):
    model = model_cls().fit(small_sample)
    # Build evidence from the first row of the sample so we know all states
    # are valid for whichever DAG the model produced.
    row = small_sample.iloc[0].to_dict()
    row.pop(TARGET, None)
    proba = model.predict_proba(row)
    total = sum(proba.values())
    assert math.isclose(total, 1.0, abs_tol=1e-6), proba


@pytest.mark.parametrize("model_cls", ALL_MODELS)
def test_predict_proba_batch_shape_and_range(model_cls, small_sample: pd.DataFrame):
    model = model_cls().fit(small_sample)
    test_rows = small_sample.head(50)
    probs = model.predict_proba_batch(test_rows)
    assert probs.shape == (50,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
