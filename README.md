# Second-hand Car Assistant — Bayesian Network Comparison

CPSC 481 final project. Predicts the probability that a used vehicle will generate
an insurance claim (a proxy for failure / costly repair) using a Bayesian network,
and compares **three candidate network structures** to choose the best one.

## Why three networks?

The course requirement is not just to ship a Bayesian network, but to *evaluate
which specific Bayesian network works best*. We train and compare:

1. **Expert DAG** — hand-designed structure based on domain knowledge.
2. **Hill-Climb learned DAG** — structure learned from data via `HillClimbSearch`
   with the BIC score.
3. **Tree-Augmented Naive Bayes (TAN)** — a strong classification baseline that
   augments naive Bayes with a tree of feature-to-feature dependencies.

All three are evaluated under identical 5-fold stratified cross-validation using
log-likelihood, BIC, accuracy, F1, Brier score, ROC-AUC, and inference latency.

## Dataset

`data/train.csv` is the Kaggle *Car Insurance Claim Prediction* dataset
(features: vehicle age, make, fuel type, engine type, etc.; binary target
`is_claim`). `data/test.csv` from the same source is unlabeled (Kaggle competition
split) and is **not** used here — our K-fold CV builds its own held-out folds.

## Layout

```
src/
  data/           # load + discretize CSV
  models/         # three Bayesian network candidates + shared base class
  evaluation/     # metrics + K-fold comparison harness
  app/            # Streamlit UI (uses the winning model)
tests/            # smoke tests
results/          # generated comparison.csv + comparison.png
data/             # raw CSVs (train.csv, test.csv) + processed/
```

## Usage

```bash
pip install -r requirements.txt

# 1. Preprocess the raw CSV into discretized categorical features.
python -m src.data.preprocess

# 2. Run the comparison: trains all 3 networks under 5-fold CV and
#    writes results/comparison.csv + results/comparison.png.
python -m src.evaluation.compare

# 3. Launch the demo UI (uses the winning model).
streamlit run src/app/streamlit_app.py

# Tests
pytest
```
