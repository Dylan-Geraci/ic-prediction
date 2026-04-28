"""
Streamlit demo for the Second-hand Car Assistant.

The app loads the *winning* Bayesian network (chosen by
``src/evaluation/compare.py`` and recorded in ``results/best_model.txt``),
fits it once on the full cleaned dataset, and exposes a small form so the
user can describe a candidate vehicle and read back the model's estimate of
``P(Claim | vehicle attributes)``.

Run with::

    streamlit run src/app/streamlit_app.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Streamlit launches the script with only its own directory (src/app) on
# ``sys.path``, so absolute imports from the ``src`` package fail unless we
# prepend the project root ourselves. This file lives at
# <root>/src/app/streamlit_app.py, hence ``parents[2]``.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

from src.data.load import feature_states, load_clean
from src.evaluation.compare import RESULTS_DIR
from src.models.base import BNModel
from src.models.expert_dag import ExpertDAG
from src.models.hill_climb import HillClimbModel
from src.models.tan import TANModel

warnings.filterwarnings("ignore")

# Map the winner-name string written by compare.py back to the model class.
_MODEL_REGISTRY: dict[str, type[BNModel]] = {
    cls.name: cls for cls in (ExpertDAG, HillClimbModel, TANModel)
}

# Buyer-friendly column labels. Keys must match column names in the cleaned
# data; values are what the user sees as the dropdown title.
FEATURE_LABELS = {
    "age_of_car": "How old is the car?",
    "segment": "What kind of car is it?",
    "fuel_type": "Fuel",
    "transmission_type": "Transmission",
    "displacement": "Engine size",
    "airbags": "Number of airbags",
    "ncap_rating": "Crash-safety rating",
    "rear_brakes_type": "Rear brakes",
}

# Short tooltip shown next to each label. Keeps the demo self-explanatory for
# someone who doesn't know car spec jargon.
FEATURE_HELP = {
    "age_of_car": "Roughly how old this listing is compared to other used cars on the market.",
    "segment": "The body style / class of the vehicle. Smaller classes are typically cheaper to buy and own.",
    "fuel_type": "What the car burns. CNG = compressed natural gas (cheaper to fuel, less common).",
    "transmission_type": "Manual = stick shift. Automatic = no clutch pedal, generally more expensive.",
    "displacement": "How big the engine is, in cc. Bigger engines usually mean more power but worse fuel economy.",
    "airbags": "More airbags = more occupant protection in a crash.",
    "ncap_rating": "Independent crash-test rating. Higher is safer.",
    "rear_brakes_type": "Disc brakes stop better and handle heat well; drum brakes are older/cheaper.",
}

# Buyer-friendly *value* labels per feature. The keys are the raw category
# values stored in the cleaned dataset (and required by the Bayesian network);
# the values are the human-readable strings shown in the dropdowns. The
# selectbox's ``format_func`` handles the translation, so the model still
# receives the raw value when the user makes a selection.
VALUE_LABELS: dict[str, dict[str, str]] = {
    "age_of_car": {
        "Low": "Newer (0-3 yrs)",
        "Medium": "Moderate (3-7 yrs)",
        "High": "Older (7+ yrs)",
    },
    "segment": {
        # The Kaggle dataset uses Indian-market segment codes. We translate
        # them into body-style descriptions a buyer can recognise.
        "A": "Mini hatchback (city car)",
        "B1": "Entry hatchback",
        "B2": "Compact hatchback",
        "C1": "Compact sedan",
        "C2": "Mid-size sedan",
        "Utility": "SUV / utility vehicle",
    },
    "fuel_type": {
        "Petrol": "Petrol (gasoline)",
        "Diesel": "Diesel",
        "CNG": "CNG (compressed natural gas)",
    },
    "transmission_type": {
        "Manual": "Manual (stick shift)",
        "Automatic": "Automatic",
    },
    "displacement": {
        "Small": "Small (~800 cc, light city car)",
        "Medium": "Medium (~1,200 cc, typical compact)",
        "Large": "Large (~1,500 cc, bigger compact / small SUV)",
    },
    "airbags": {
        "Few": "Few (2 or fewer)",
        "Many": "Many (4 or more)",
    },
    "ncap_rating": {
        "Low": "Low (0-1 stars)",
        "Medium": "Medium (2-3 stars)",
        "High": "High (4-5 stars)",
    },
    "rear_brakes_type": {
        "Drum": "Drum (older)",
        "Disc": "Disc (modern)",
    },
}

# Order options sensibly per feature (e.g. ordinal categories should ascend
# rather than appear alphabetised). Any state not listed here falls back to
# the dataset's natural sort order.
PREFERRED_ORDER: dict[str, list[str]] = {
    "age_of_car": ["Low", "Medium", "High"],
    "displacement": ["Small", "Medium", "Large"],
    "airbags": ["Few", "Many"],
    "ncap_rating": ["Low", "Medium", "High"],
    "segment": ["A", "B1", "B2", "C1", "C2", "Utility"],
    "fuel_type": ["Petrol", "Diesel", "CNG"],
    "transmission_type": ["Manual", "Automatic"],
    "rear_brakes_type": ["Drum", "Disc"],
}


def _ordered_options(col: str, states: list[str]) -> list[str]:
    """Return ``states`` in our preferred order for ``col``.

    Any extra state present in the data but missing from the preferred order
    is appended at the end so the form never silently hides a category."""
    preferred = PREFERRED_ORDER.get(col, [])
    ordered = [s for s in preferred if s in states]
    extras = [s for s in states if s not in ordered]
    return ordered + extras


def _read_winner() -> str:
    """Return the winning-model name from the evaluation step.

    If the comparison has not been run yet (no ``best_model.txt``), fall back
    to TAN, which has been the winner across our test runs.
    """
    winner_file = RESULTS_DIR / "best_model.txt"
    if winner_file.exists():
        return winner_file.read_text().strip()
    return TANModel.name


@st.cache_resource(show_spinner="Training the winning Bayesian network ...")
def _load_model(winner_name: str) -> tuple[BNModel, dict[str, list[str]]]:
    """Fit the winning model on the full cleaned dataset.

    Cached by Streamlit so the model is trained once per session, not on every
    slider change. ``feature_states`` is returned alongside so the form can
    populate its dropdowns from the actual category levels in the data.
    """
    df = load_clean()
    cls = _MODEL_REGISTRY.get(winner_name, TANModel)
    model = cls().fit(df)
    return model, feature_states(df)


def _render_inputs(states: dict[str, list[str]]) -> dict[str, str]:
    """Render the sidebar form and return the user's selections as evidence.

    We use ordered dropdowns rather than free-text inputs so the user cannot
    submit an unknown category that the network would reject. Each dropdown
    shows the buyer-friendly label (via ``format_func``) but returns the raw
    category value the Bayesian network expects.
    """
    st.sidebar.header("Tell us about the car")
    evidence: dict[str, str] = {}
    for col, label in FEATURE_LABELS.items():
        # ``states`` includes ``is_claim`` (the target); skip it.
        if col not in states:
            continue
        options = _ordered_options(col, states[col])
        # Translate raw category -> friendly string for display only.
        labels_for_col = VALUE_LABELS.get(col, {})
        evidence[col] = st.sidebar.selectbox(
            label,
            options=options,
            key=col,
            help=FEATURE_HELP.get(col),
            format_func=lambda v, _labels=labels_for_col: _labels.get(v, v),
        )
    return evidence


def _gauge(prob: float) -> None:
    """Render the predicted probability as a metric and a horizontal bar.

    Streamlit doesn't ship a true gauge widget, so we approximate one with
    its built-in ``progress`` bar plus a coloured caption that switches
    between low/medium/high risk thresholds.
    """
    pct = prob * 100
    st.metric(
        "Estimated chance of a costly insurance claim",
        f"{pct:.1f}%",
        help="Probability that a car with these attributes generates an "
             "insurance claim. We use claim frequency as a proxy for "
             "real-world breakdowns and accident-related repair bills.",
    )
    st.progress(min(max(prob, 0.0), 1.0))

    if prob < 0.05:
        st.success(
            "Looks like a relatively safe bet. Predicted claim rate is "
            "below the typical used-car average."
        )
    elif prob < 0.10:
        st.info(
            "Average risk. This car is roughly in line with the typical "
            "used-car claim rate (~6%)."
        )
    else:
        st.warning(
            "Higher than average risk. Worth budgeting for a thorough "
            "inspection and possible repairs before buying."
        )


def _what_if(model: BNModel, evidence: dict[str, str]) -> None:
    """Show how the predicted probability would change if the car were one
    bin older. Helps the user understand which inputs move the needle."""
    if "age_of_car" not in evidence:
        return
    bumped = dict(evidence)
    age_bins = ["Low", "Medium", "High"]
    current = bumped["age_of_car"]
    if current in age_bins and current != "High":
        bumped["age_of_car"] = age_bins[age_bins.index(current) + 1]
        new_prob = model.predict_proba(bumped).get("Claim", float("nan"))
        # Show the friendly version of both the current and bumped age values.
        age_labels = VALUE_LABELS.get("age_of_car", {})
        st.caption(
            f"What if this car were a step older "
            f"({age_labels.get(bumped['age_of_car'], bumped['age_of_car'])} "
            f"instead of {age_labels.get(current, current)})? "
            f"Estimated risk would be **{new_prob * 100:.1f}%**."
        )


def main() -> None:
    st.set_page_config(page_title="Second-hand Car Assistant", page_icon=None)
    st.title("Second-hand Car Assistant")
    st.caption(
        "Thinking about buying a used car? Describe it on the left and we'll "
        "estimate how likely it is to need an expensive repair or insurance "
        "claim, based on patterns from real used-vehicle data."
    )

    winner_name = _read_winner()
    model, states = _load_model(winner_name)

    evidence = _render_inputs(states)

    if st.sidebar.button("Estimate risk", use_container_width=True):
        result = model.predict_proba(evidence)
        prob = result.get("Claim", float("nan"))
        _gauge(prob)
        _what_if(model, evidence)
    else:
        st.info(
            "Fill in the car's details on the left, then click "
            "**Estimate risk**."
        )

    # Footer: be transparent about which network is powering the demo and
    # where the comparison numbers came from. Important for the class demo.
    st.divider()
    st.caption(
        f"Powered by *{winner_name}* — chosen by 5-fold stratified "
        f"cross-validation against two other Bayesian networks. "
        f"See `results/comparison.csv` for the full evaluation."
    )


if __name__ == "__main__":
    main()
