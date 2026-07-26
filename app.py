%%writefile app.py
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Optima Life Renewal Rate Predictor",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #fafafa; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1000px; }
    h1 { color: #1a1a2e; font-weight: 700; }
    h3 { color: #16213e; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.4rem; margin-top: 1.5rem; }
    .stButton>button {
        background-color: #0f4c81;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #0a3860; }
    div[data-testid="stMetricValue"] { font-size: 2.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Load model bundle ──────────────────────────────────────────────────────────
with open("churn_model_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

model               = bundle['model']
encoder             = bundle['encoder']
feature_columns     = bundle['feature_columns']
categorical_columns = bundle['categorical_columns']

income_choices    = list(encoder.categories_[0])
education_choices = list(encoder.categories_[1])
device_choices    = list(encoder.categories_[2])

# ── Header ──────────────────────────────────────────────────────────────────────
st.title("🔄 Optima Life Renewal Rate Predictor")
st.markdown(
    "Estimate the likelihood that a **Healthy Meals** subscriber will renew, "
    "based on demographics and prior-year engagement."
)
st.divider()

# ── Input form ──────────────────────────────────────────────────────────────────
with st.form("prediction_form"):

    st.markdown("### 👤 Customer Demographics")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", min_value=18, max_value=90, value=40, step=1)
        income_level = st.selectbox("Income Level", income_choices)
        education = st.selectbox("Education", education_choices)

    with col2:
        tech_comfort_score = st.slider("Tech Comfort Score", min_value=0, max_value=10, value=5, step=1)
        device_type = st.selectbox("Device Type", device_choices)

    st.markdown("### 📊 2022 Activity")
    col3, col4 = st.columns(2)

    with col3:
        total_num_sessions = st.slider("Total Number of Sessions (2022)", min_value=0, max_value=200, value=30, step=1)
        gross_total_session_length = st.slider("Gross Total Session Length (minutes, 2022)", min_value=0, max_value=5000, value=600, step=10)
        active_days = st.slider("Active Days (2022)", min_value=0, max_value=4, value=2, step=1,
                                  help="Activity is recorded quarterly, so this reflects up to 4 recorded periods.")
        active_quarters = st.slider("Active Quarters (2022)", min_value=0, max_value=4, value=2, step=1)

    with col4:
        avg_sessions_per_active_quarter = st.slider("Avg Sessions per Active Quarter", min_value=0.0, max_value=100.0, value=15.0, step=0.5)
        avg_session_length_per_session = st.slider("Avg Session Length per Session (minutes)", min_value=0.0, max_value=500.0, value=20.0, step=1.0)
        recency_weighted_sessions = st.slider("Recency-Weighted Sessions", min_value=0.0, max_value=100.0, value=15.0, step=0.5,
                                                help="Weights recent quarters more heavily than earlier ones — higher values indicate stronger recent engagement.")
        subscription_tenure_days = st.slider("Subscription Tenure (days)", min_value=0, max_value=2000, value=365, step=5)

    st.markdown("")
    submitted = st.form_submit_button("Predict Renewal Probability")

# ── Prediction ──────────────────────────────────────────────────────────────────
if submitted:
    raw_categorical = pd.DataFrame([{
        'INCOME_LEVEL': income_level,
        'EDUCATION':    education,
        'DEVICE_TYPE':  device_type,
    }])
    encoded = encoder.transform(raw_categorical)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    numeric_df = pd.DataFrame([{
        'TOTAL_NUM_SESSIONS': total_num_sessions,
        'GROSS_TOTAL_SESSION_LENGTH': gross_total_session_length,
        'ACTIVE_DAYS': active_days,
        'ACTIVE_QUARTERS': active_quarters,
        'AVG_SESSIONS_PER_ACTIVE_QUARTER': avg_sessions_per_active_quarter,
        'AVG_SESSION_LENGTH_PER_SESSION': avg_session_length_per_session,
        'RECENCY_WEIGHTED_SESSIONS': recency_weighted_sessions,
        'SUBSCRIPTION_TENURE_DAYS': subscription_tenure_days,
        'AGE': age,
        'TECH_COMFORT_SCORE': tech_comfort_score,
    }])

    input_df = pd.concat([numeric_df, encoded_df], axis=1)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    probability = model.predict_proba(input_df)[0][1]

    if probability >= 0.6:
        risk, color = "Low", "🟢"
    elif probability >= 0.4:
        risk, color = "Medium", "🟡"
    else:
        risk, color = "High", "🔴"

    st.divider()
    st.markdown("### 📈 Prediction Result")

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric("Renewal Probability", f"{probability:.0%}")
    with res_col2:
        st.metric("Churn Risk", f"{color} {risk}")

    st.progress(float(probability))
