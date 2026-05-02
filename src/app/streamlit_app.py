"""
Streamlit demo for the Second-hand Car Assistant.

Loads the *winning* Bayesian network (chosen by ``src/evaluation/compare.py``
and recorded in ``results/best_model.txt``), fits it once on the full cleaned
dataset, and exposes a form so the user can describe a candidate vehicle and
their own context, then read back the model's estimate of
``P(Claim | evidence)``.

Run with::

    streamlit run src/app/streamlit_app.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

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

# ---- Static configuration ----------------------------------------------------

_MODEL_REGISTRY: dict[str, type[BNModel]] = {
    cls.name: cls for cls in (ExpertDAG, HillClimbModel, TANModel)
}

SECTIONS: dict[str, list[str]] = {
    "Vehicle details": [
        "age_of_car",
        "segment",
        "fuel_type",
        "displacement",
        "airbags",
        "ncap_rating",
    ],
    "Driver profile": [
        "age_of_policyholder",
        "policy_tenure",
    ],
    "Location": [
        "area_cluster",
        "population_density",
    ],
}

FEATURE_LABELS = {
    "age_of_car": "Vehicle age",
    "segment": "Body type",
    "fuel_type": "Fuel",
    "displacement": "Engine size",
    "airbags": "Airbags",
    "ncap_rating": "Crash-safety rating",
    "policy_tenure": "Insurance tenure",
    "age_of_policyholder": "Driver age",
    "area_cluster": "Area risk tier",
    "population_density": "Population density",
}

FEATURE_HELP = {
    "age_of_car": "Roughly how old this listing is compared to other used cars on the market.",
    "segment": "The body style or class of the vehicle.",
    "fuel_type": "What the car burns. CNG = compressed natural gas.",
    "displacement": "Engine displacement in cc.",
    "airbags": "More airbags means more occupant protection.",
    "ncap_rating": "Independent crash-test rating. Higher is safer.",
    "policy_tenure": "How long you have held continuous auto insurance.",
    "age_of_policyholder": "Age band of the primary driver.",
    "area_cluster": "Risk tier of the registration area, derived from regional claim history.",
    "population_density": "Whether you live in a rural, suburban, or dense urban area.",
}

VALUE_LABELS: dict[str, dict[str, str]] = {
    "age_of_car": {
        "Low": "Newer (0–3 yrs)",
        "Medium": "Moderate (3–7 yrs)",
        "High": "Older (7+ yrs)",
    },
    "segment": {
        "A": "Mini hatchback",
        "B1": "Entry hatchback",
        "B2": "Compact hatchback",
        "C1": "Compact sedan",
        "C2": "Mid-size sedan",
        "Utility": "SUV / utility",
    },
    "fuel_type": {
        "Petrol": "Petrol",
        "Diesel": "Diesel",
        "CNG": "CNG",
    },
    "displacement": {
        "Small": "Small (~800 cc)",
        "Medium": "Medium (~1,200 cc)",
        "Large": "Large (~1,500 cc)",
    },
    "airbags": {
        "Few": "2 or fewer",
        "Many": "4 or more",
    },
    "ncap_rating": {
        "Low": "0–1 stars",
        "Medium": "2–3 stars",
        "High": "4–5 stars",
    },
    "policy_tenure": {
        "Short": "Short",
        "Medium": "Medium",
        "Long": "Long",
    },
    "age_of_policyholder": {
        "Young": "Younger driver",
        "Middle": "Middle-aged driver",
        "Senior": "Older driver",
    },
    "area_cluster": {
        "AreaRisk1": "Tier 1 (lowest risk)",
        "AreaRisk2": "Tier 2",
        "AreaRisk3": "Tier 3",
        "AreaRisk4": "Tier 4 (highest risk)",
    },
    "population_density": {
        "LowDensity": "Rural",
        "MediumDensity": "Suburban",
        "HighDensity": "Urban",
    },
}

PREFERRED_ORDER: dict[str, list[str]] = {
    "age_of_car": ["Low", "Medium", "High"],
    "displacement": ["Small", "Medium", "Large"],
    "airbags": ["Few", "Many"],
    "ncap_rating": ["Low", "Medium", "High"],
    "segment": ["A", "B1", "B2", "C1", "C2", "Utility"],
    "fuel_type": ["Petrol", "Diesel", "CNG"],
    "policy_tenure": ["Short", "Medium", "Long"],
    "age_of_policyholder": ["Young", "Middle", "Senior"],
    "area_cluster": ["AreaRisk1", "AreaRisk2", "AreaRisk3", "AreaRisk4"],
    "population_density": ["LowDensity", "MediumDensity", "HighDensity"],
}


# ---- Custom CSS --------------------------------------------------------------

CUSTOM_CSS = """
<style>
  /* Reset Streamlit's default container padding so we can control layout */
  .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1180px;
  }

  /* Page background and base typography */
  html, body, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif, "Apple Color Emoji",
                 "Segoe UI Emoji";
    color: #050505;
  }
  .stApp {
    background: #F0F2F5;
  }

  /* Top header bar */
  .topbar {
    background: #FFFFFF;
    border-bottom: 1px solid #DADDE1;
    padding: 14px 24px;
    margin: -2rem -2rem 24px -2rem;
    display: flex;
    align-items: center;
    gap: 14px;
  }
  .topbar-mark {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: #1877F2;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    color: #FFFFFF;
    font-weight: 800;
    font-size: 16px;
    letter-spacing: -0.5px;
  }
  .topbar-title {
    font-size: 17px;
    font-weight: 700;
    color: #050505;
    line-height: 1;
  }
  .topbar-sub {
    font-size: 12px;
    color: #65676B;
    margin-top: 3px;
  }

  /* Generic card */
  .card {
    background: #FFFFFF;
    border: 1px solid #DADDE1;
    border-radius: 8px;
    padding: 24px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
  }
  .card + .card {
    margin-top: 16px;
  }

  /* Section title within main content */
  .section-eyebrow {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #65676B;
    margin-bottom: 6px;
  }
  .section-heading {
    font-size: 18px;
    font-weight: 700;
    color: #050505;
    margin: 0 0 12px 0;
  }

  /* Sidebar styling */
  div[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #DADDE1;
  }
  div[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
  }
  .sidebar-group-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: #65676B;
    margin: 18px 0 6px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #E4E6EB;
  }
  div[data-testid="stSidebar"] label {
    font-weight: 600;
    color: #050505;
    font-size: 13px;
  }

  /* Sidebar primary action button */
  div[data-testid="stSidebar"] .stButton > button {
    background: #1877F2;
    color: #FFFFFF;
    border: 1px solid #1877F2;
    border-radius: 6px;
    font-weight: 600;
    padding: 8px 16px;
    transition: background 0.15s ease;
  }
  div[data-testid="stSidebar"] .stButton > button:hover {
    background: #166FE5;
    border-color: #166FE5;
  }
  div[data-testid="stSidebar"] .stButton > button:active {
    background: #1463CC;
  }

  /* Result block */
  .result-eyebrow {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: #65676B;
  }
  .result-value {
    font-size: 44px;
    font-weight: 700;
    color: #050505;
    line-height: 1.1;
    margin: 6px 0 4px 0;
    letter-spacing: -1px;
  }
  .result-sub {
    font-size: 14px;
    color: #65676B;
    margin: 0 0 14px 0;
  }

  /* Risk badge */
  .risk-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 12px;
    letter-spacing: 0.2px;
    border: 1px solid transparent;
  }
  .risk-low  { background: #E7F3E8; color: #1B5E20; border-color: #C5E1C7; }
  .risk-mid  { background: #FFF7E0; color: #6B4F00; border-color: #F4E6B0; }
  .risk-high { background: #FCE7E9; color: #9A1B27; border-color: #F4C7CC; }

  /* Gauge */
  .gauge-track {
    height: 8px;
    border-radius: 999px;
    background: #E4E6EB;
    overflow: hidden;
    margin: 14px 0 6px 0;
  }
  .gauge-fill {
    height: 100%;
    border-radius: 999px;
    background: #1877F2;
    transition: width 0.4s ease;
  }
  .gauge-scale {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #8A8D91;
  }

  /* What-if note */
  .whatif {
    margin-top: 20px;
    padding: 12px 14px;
    background: #F7F8FA;
    border: 1px solid #E4E6EB;
    border-radius: 6px;
    font-size: 13px;
    color: #050505;
  }
  .whatif strong { color: #050505; }

  /* Empty state */
  .empty {
    text-align: center;
    padding: 40px 24px;
    color: #65676B;
  }
  .empty .empty-title {
    font-size: 16px;
    font-weight: 600;
    color: #050505;
    margin-bottom: 6px;
  }
  .empty p {
    font-size: 14px;
    margin: 0;
  }

  /* Info card on the right */
  .info-list {
    margin: 0;
    padding-left: 18px;
    color: #050505;
    font-size: 13.5px;
    line-height: 1.55;
  }
  .info-list li { margin-bottom: 6px; }
  .info-list li:last-child { margin-bottom: 0; }

  /* Footer */
  .app-footer {
    margin-top: 28px;
    padding-top: 16px;
    border-top: 1px solid #DADDE1;
    font-size: 12px;
    color: #65676B;
    text-align: center;
  }
  .app-footer code {
    background: #F0F2F5;
    color: #050505;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 12px;
  }

  /* Hide Streamlit's own header/menu (cleaner top edge) */
  header[data-testid="stHeader"] { display: none; }
</style>
"""


# ---- Helpers -----------------------------------------------------------------

def _ordered_options(col: str, states: list[str]) -> list[str]:
    preferred = PREFERRED_ORDER.get(col, [])
    ordered = [s for s in preferred if s in states]
    extras = [s for s in states if s not in ordered]
    return ordered + extras


def _read_winner() -> str:
    winner_file = RESULTS_DIR / "best_model.txt"
    if winner_file.exists():
        return winner_file.read_text().strip()
    return TANModel.name


@st.cache_resource(show_spinner="Training the winning Bayesian network ...")
def _load_model(winner_name: str) -> tuple[BNModel, dict[str, list[str]]]:
    df = load_clean()
    cls = _MODEL_REGISTRY.get(winner_name, TANModel)
    model = cls().fit(df)
    return model, feature_states(df)


def _render_topbar() -> None:
    st.markdown(
        """
        <div class='topbar'>
          <span class='topbar-mark'>S</span>
          <div>
            <div class='topbar-title'>Second-hand Car Assistant</div>
            <div class='topbar-sub'>Insurance-claim risk estimator</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_sidebar(states: dict[str, list[str]]) -> tuple[dict[str, str], bool]:
    st.sidebar.markdown(
        "<div style='font-size:14px;font-weight:600;color:#050505;"
        "margin:4px 0 4px 0;'>Build your scenario</div>"
        "<div style='font-size:12.5px;color:#65676B;margin-bottom:8px;'>"
        "Describe the car, the driver, and where it is based.</div>",
        unsafe_allow_html=True,
    )

    evidence: dict[str, str] = {}
    for section_name, cols in SECTIONS.items():
        st.sidebar.markdown(
            f"<div class='sidebar-group-label'>{section_name}</div>",
            unsafe_allow_html=True,
        )
        for col in cols:
            if col not in states:
                continue
            options = _ordered_options(col, states[col])
            labels_for_col = VALUE_LABELS.get(col, {})
            evidence[col] = st.sidebar.selectbox(
                FEATURE_LABELS.get(col, col),
                options=options,
                key=col,
                help=FEATURE_HELP.get(col),
                format_func=lambda v, _labels=labels_for_col: _labels.get(v, v),
            )

    st.sidebar.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    submit = st.sidebar.button("Estimate risk", use_container_width=True)
    return evidence, submit


def _risk_badge(prob: float) -> str:
    if prob < 0.05:
        return "<span class='risk-badge risk-low'>Below average</span>"
    if prob < 0.10:
        return "<span class='risk-badge risk-mid'>Average</span>"
    return "<span class='risk-badge risk-high'>Above average</span>"


def _risk_message(prob: float) -> str:
    if prob < 0.05:
        return ("Below the typical used-car claim rate (~6%). Looks like a "
                "relatively safe bet on the data we have.")
    if prob < 0.10:
        return ("Roughly in line with the typical used-car claim rate (~6%).")
    return ("Above the typical used-car claim rate. Worth budgeting for a "
            "thorough inspection before buying.")


def _render_result(prob: float) -> None:
    pct = prob * 100
    fill_pct = min(max(prob * 100 / 0.20 * 100, 4), 100)  # cap visual at 20%
    badge = _risk_badge(prob)
    message = _risk_message(prob)

    st.markdown(
        f"""
        <div class='card'>
          <div class='result-eyebrow'>Estimated probability of a claim</div>
          <div class='result-value'>{pct:.1f}%</div>
          <div class='result-sub'>{message}</div>
          <div>{badge}</div>
          <div class='gauge-track'>
            <div class='gauge-fill' style='width:{fill_pct:.1f}%;'></div>
          </div>
          <div class='gauge-scale'><span>0%</span><span>10%</span><span>20%+</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_what_if(model: BNModel, evidence: dict[str, str]) -> None:
    if "age_of_car" not in evidence:
        return
    bumped = dict(evidence)
    age_bins = ["Low", "Medium", "High"]
    current = bumped["age_of_car"]
    if current not in age_bins or current == "High":
        return
    bumped["age_of_car"] = age_bins[age_bins.index(current) + 1]
    new_prob = model.predict_proba(bumped).get("Claim", float("nan"))
    age_labels = VALUE_LABELS.get("age_of_car", {})
    st.markdown(
        f"<div class='whatif'><strong>What if it were a step older?</strong> "
        f"{age_labels.get(bumped['age_of_car'], bumped['age_of_car'])} "
        f"instead of {age_labels.get(current, current)} → estimated risk "
        f"<strong>{new_prob * 100:.1f}%</strong>.</div>",
        unsafe_allow_html=True,
    )


def _render_empty() -> None:
    st.markdown(
        """
        <div class='card'>
          <div class='empty'>
            <div class='empty-title'>Ready when you are</div>
            <p>Pick the car, driver, and location details on the left,
            then choose <strong>Estimate risk</strong>.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_info_card(winner_name: str) -> None:
    st.markdown(
        f"""
        <div class='card'>
          <div class='section-eyebrow'>About this estimate</div>
          <div class='section-heading'>How it works</div>
          <ul class='info-list'>
            <li>Trained on ~58,000 real used-car insurance records.</li>
            <li>Combines vehicle, driver, and location signals into a single
                probability of a claim.</li>
            <li>Three Bayesian network structures were compared with 5-fold
                cross-validation. <strong>{winner_name}</strong> is the
                strongest and is loaded automatically.</li>
            <li>Results are statistical estimates, not guarantees.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_footer(winner_name: str) -> None:
    st.markdown(
        f"""
        <div class='app-footer'>
          Powered by <strong>{winner_name}</strong>. Selected by 5-fold
          stratified cross-validation. Full evaluation in
          <code>results/comparison.csv</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---- Main --------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Second-hand Car Assistant",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    winner_name = _read_winner()
    model, states = _load_model(winner_name)

    _render_topbar()
    evidence, submit = _render_sidebar(states)

    left, right = st.columns([1.2, 0.8], gap="large")
    with left:
        if submit:
            result = model.predict_proba(evidence)
            prob = result.get("Claim", float("nan"))
            _render_result(prob)
            _render_what_if(model, evidence)
        else:
            _render_empty()

    with right:
        _render_info_card(winner_name)

    _render_footer(winner_name)


if __name__ == "__main__":
    main()
