
import pickle
from datetime import date as date_cls
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Family Health Guardian",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .main { background-color: #0e1117; }

        .hero {
            padding: 1.8rem 2rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #0f2027 0%, #203a43 45%, #2c5364 100%);
            margin-bottom: 1.5rem;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .hero h1 {
            color: #ffffff;
            font-size: 2.1rem;
            margin-bottom: 0.2rem;
        }
        .hero p {
            color: #c9d6df;
            font-size: 1rem;
            margin: 0;
        }

        .metric-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 1rem 1.2rem;
            text-align: center;
        }

        .risk-card-high {
            background: linear-gradient(135deg, #3a0d0d, #5c1010);
            border: 1px solid #ff4b4b55;
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
        }
        .risk-card-low {
            background: linear-gradient(135deg, #0d3a1a, #103a20);
            border: 1px solid #2ecc7155;
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
        }
        .risk-title { font-size: 1.4rem; font-weight: 700; margin-bottom: .3rem; }
        .risk-sub { color: #c9d6df; font-size: .95rem; }

        .factor-chip {
            display: inline-block;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            margin: 0.2rem;
            font-size: 0.85rem;
            font-weight: 600;
        }
        .chip-bad { background: #ff4b4b22; color: #ff8a8a; border: 1px solid #ff4b4b55; }
        .chip-good { background: #2ecc7122; color: #7be3ab; border: 1px solid #2ecc7155; }

        footer, #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# LOAD MODEL ARTIFACTS
# ----------------------------------------------------------------------
FILES_DIR_CANDIDATES = [
    Path("Files"),
    Path("."),
    Path("../Files"),
    Path(__file__).resolve().parent / "Files",
]


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    model, scaler, feature_columns = None, None, None
    errors = []

    for base in FILES_DIR_CANDIDATES:
        try:
            with open(base / "random_forest_model.pkl", "rb") as f:
                model = pickle.load(f)
            with open(base / "scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            with open(base / "feature_columns.pkl", "rb") as f:
                feature_columns = pickle.load(f)
            return model, scaler, feature_columns, None
        except FileNotFoundError as e:
            errors.append(str(e))
            continue

    return None, None, None, errors


model, scaler, feature_columns, load_errors = load_artifacts()

if model is None:
    st.error(
        "Couldn't find `random_forest_model.pkl`, `scaler.pkl`, and "
        "`feature_columns.pkl`. Place them in a `Files/` folder next to "
        "`app.py` (or in the same folder as `app.py`) and rerun."
    )
    with st.expander("Details"):
        st.write(load_errors)
    st.stop()

st.markdown(
    """
    <div class="hero">
        <h1>🩺 Family Health Guardian</h1>
        <p>Daily vitals in, a 2-day health-risk forecast out — built to help you
        keep an eye on the people who matter most.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header(" Family Member Profile")

    member_name = st.text_input("Name", value="Aunt", help="Just a label — doesn't affect the prediction.")
    age = st.number_input("Age", min_value=1, max_value=110, value=42, step=1)
    gender = st.radio("Gender", options=["Female", "Male"], horizontal=True)
    observation_date = st.date_input("Observation date", value=date_cls.today())
    day_of_week = observation_date.strftime("%A")
    st.caption(f"Day of week: **{day_of_week}**")

    st.divider()
    st.header(" Today's Vitals & Lifestyle")

    resting_hr = st.slider("Resting Heart Rate (bpm)", 40.0, 140.0, 68.0, 0.1)
    spo2 = st.slider("SpO₂ (%)", 85.0, 100.0, 97.0, 0.1)
    stress_score = st.slider("Stress Score (0-100)", 0.0, 100.0, 35.0, 0.1)
    steps = st.slider("Steps", 0, 30000, 8500, 50)
    screen_time = st.slider("Screen Time (hours)", 0.0, 16.0, 5.0, 0.1)
    sleep_hours = st.slider("Sleep (hours)", 0.0, 14.0, 7.0, 0.1)

    st.divider()
    predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")

def get_expected_columns():
    if hasattr(scaler, "feature_names_in_"):
        return list(scaler.feature_names_in_)
    # Fallback: exact order captured from the training notebook (X_test.columns)
    return [
        "age", "resting_heart_rate_bpm", "spo2_percent", "stress_score",
        "steps", "screen_time_hours", "sleep_hours", "year", "month", "day",
        "gender_M", "day_of_week_Monday", "day_of_week_Saturday",
        "day_of_week_Sunday", "day_of_week_Thursday", "day_of_week_Tuesday",
        "day_of_week_Wednesday",
    ]


def build_input_row():
    expected_cols = get_expected_columns()
    row = {col: 0 for col in expected_cols}

    values = {
        "age": age,
        "resting_heart_rate_bpm": resting_hr,
        "spo2_percent": spo2,
        "stress_score": stress_score,
        "steps": steps,
        "screen_time_hours": screen_time,
        "sleep_hours": sleep_hours,
        "year": observation_date.year,
        "month": observation_date.month,
        "day": observation_date.day,
    }
    for col, val in values.items():
        if col in row:
            row[col] = val

    # gender was one-hot encoded with drop_first=True -> only gender_M kept
    if "gender_M" in row:
        row["gender_M"] = 1 if gender == "Male" else 0

    # day_of_week was one-hot encoded with drop_first=True -> Friday dropped (baseline)
    dow_col = f"day_of_week_{day_of_week}"
    if dow_col in row:
        row[dow_col] = 1

    return pd.DataFrame([row])[expected_cols]


def get_risk_factors():
    factors = []
    if resting_hr > 90:
        factors.append(("High resting heart rate", True))
    else:
        factors.append(("Resting heart rate normal", False))

    if spo2 < 95:
        factors.append(("Low SpO₂", True))
    else:
        factors.append(("SpO₂ healthy", False))

    if stress_score > 60:
        factors.append(("High stress", True))
    else:
        factors.append(("Stress under control", False))

    if sleep_hours < 6:
        factors.append(("Sleep deprived", True))
    else:
        factors.append(("Sleep adequate", False))

    if steps < 3000:
        factors.append(("Very low activity", True))
    else:
        factors.append(("Activity level okay", False))

    if screen_time > 8:
        factors.append(("Excessive screen time", True))
    else:
        factors.append(("Screen time reasonable", False))

    return factors


def gauge_chart(probability):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 42, "color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white", "tickfont": {"color": "white"}},
                "bar": {"color": "#ff4b4b" if probability >= 0.5 else "#2ecc71"},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "#1e5631"},
                    {"range": [30, 60], "color": "#7a5c00"},
                    {"range": [60, 100], "color": "#5c1010"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
    )
    return fig


# ----------------------------------------------------------------------
# MAIN AREA
# ----------------------------------------------------------------------
tab1, tab2 = st.tabs(["Prediction", " About this model"])

with tab1:
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader(f"Snapshot for {member_name or 'this family member'}")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="metric-card"><b>❤️ Resting HR</b><br><span style="font-size:1.4rem">{resting_hr:.0f} bpm</span></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="metric-card"><b>🫁 SpO₂</b><br><span style="font-size:1.4rem">{spo2:.1f}%</span></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="metric-card"><b>😴 Sleep</b><br><span style="font-size:1.4rem">{sleep_hours:.1f} hrs</span></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        c4, c5, c6 = st.columns(3)
        with c4:
            st.markdown(
                f'<div class="metric-card"><b>🧠 Stress</b><br><span style="font-size:1.4rem">{stress_score:.0f}/100</span></div>',
                unsafe_allow_html=True,
            )
        with c5:
            st.markdown(
                f'<div class="metric-card"><b>🚶 Steps</b><br><span style="font-size:1.4rem">{steps:,}</span></div>',
                unsafe_allow_html=True,
            )
        with c6:
            st.markdown(
                f'<div class="metric-card"><b>📱 Screen Time</b><br><span style="font-size:1.4rem">{screen_time:.1f} hrs</span></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        st.markdown("**Contributing factors observed today:**")
        chips_html = ""
        for label, is_bad in get_risk_factors():
            css_class = "chip-bad" if is_bad else "chip-good"
            icon = "⚠️" if is_bad else "✅"
            chips_html += f'<span class="factor-chip {css_class}">{icon} {label}</span>'
        st.markdown(chips_html, unsafe_allow_html=True)

    with right:
        if predict_btn:
            X_input = build_input_row()
            X_scaled = scaler.transform(X_input)

            proba = model.predict_proba(X_scaled)[0][1]
            prediction = model.predict(X_scaled)[0]

            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="risk-card-high">
                        <div class="risk-title">⚠️ Elevated Risk Detected</div>
                        <div class="risk-sub">
                            {member_name or 'This family member'} shows a
                            <b>{proba*100:.1f}%</b> likelihood of a medical issue
                            within the next 2 days.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="risk-card-low">
                        <div class="risk-title">✅ Low Risk</div>
                        <div class="risk-sub">
                            {member_name or 'This family member'} shows a
                            <b>{proba*100:.1f}%</b> likelihood of a medical issue
                            within the next 2 days.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.plotly_chart(gauge_chart(proba), use_container_width=True)

            st.caption(
                "This is a statistical estimate from a Random Forest model trained "
                "on historical wearable/lifestyle data — not a medical diagnosis. "
                "If someone feels unwell, please consult a doctor."
            )
        else:
            st.info("👈 Fill in the vitals in the sidebar and click **Run Prediction** to see the risk forecast.")

with tab2:
    st.subheader("How this works")
    st.markdown(
        """
        - **Model:** Random Forest classifier (`n_estimators=200`, `max_depth=5`,
          `min_samples_leaf=5`, `class_weight='balanced'`), tuned via `GridSearchCV`
          on ROC-AUC.
        - **Target:** `medical_issue_flag_next_2_days` — whether the family member
          reported/was flagged with a medical issue within the following 2 days.
        - **Features used:** age, gender, day/month/year (from the observation date),
          day of week, resting heart rate, SpO₂, stress score, steps, screen time,
          and sleep hours — scaled with the same `StandardScaler` fitted during
          training. The exact 17 columns and their order are read directly from
          `scaler.feature_names_in_` to guarantee they match what the model was
          trained on.
        - **Note:** `family_member` and `day_number` were excluded from training
          features (used only for the time-based train/test split), so the name
          field above is for your own tracking and doesn't influence the prediction.
        """
    )
    st.markdown("Made with considerable care by **ALOK GUPTA**. [GitHub](https://github.com/Alok-0601) | [LinkedIn](https://www.linkedin.com/in/alok-gupta-0601/)")
