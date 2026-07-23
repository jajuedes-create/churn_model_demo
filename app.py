import pickle
import pandas as pd
import streamlit as st

# ── Load model bundle ──────────────────────────────────────────────────────
# churn_model_bundle.pkl is tracked in this repo alongside app.py.
with open("churn_model_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

model               = bundle['model']
encoder             = bundle['encoder']
feature_columns     = bundle['feature_columns']
categorical_columns = bundle['categorical_columns']

income_choices    = list(encoder.categories_[0])
education_choices = list(encoder.categories_[1])
device_choices    = list(encoder.categories_[2])

st.title("Customer Renewal Probability Predictor")
st.write("Enter customer attributes to predict the likelihood of subscription renewal.")

with st.form("prediction_form"):
    st.subheader("Demographics")
    age = st.number_input("Age", min_value=0, max_value=120, value=35)
    tech_comfort_score = st.number_input("Tech Comfort Score", min_value=0, max_value=10, value=5)
    income_level = st.radio("Income Level", income_choices)
    education = st.radio("Education", education_choices)
    device_type = st.radio("Device Type", device_choices)

    st.subheader("2022 Activity")
    gross_total_session_length = st.number_input("Gross Total Session Length (minutes, 2022)", min_value=0, value=0)
    active_days = st.number_input("Active Days (2022)", min_value=0, max_value=365, value=0)
    active_quarters = st.number_input("Active Quarters (2022)", min_value=0, max_value=4, value=0)
    avg_sessions_per_active_quarter = st.number_input("Avg Sessions per Active Quarter", min_value=0.0, value=0.0)
    avg_session_length_per_session = st.number_input("Avg Session Length per Session (minutes)", min_value=0.0, value=0.0)
    days_since_last_activity_2022 = st.number_input("Days Since Last Activity (2022)", min_value=0, max_value=365, value=365)

    submitted = st.form_submit_button("Predict Renewal Probability")

if submitted:
    raw_categorical = pd.DataFrame([{
        'INCOME_LEVEL': income_level,
        'EDUCATION':    education,
        'DEVICE_TYPE':  device_type,
    }])
    encoded = encoder.transform(raw_categorical)
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())

    numeric_df = pd.DataFrame([{
        'AGE': age,
        'TECH_COMFORT_SCORE': tech_comfort_score,
        'GROSS_TOTAL_SESSION_LENGTH': gross_total_session_length,
        'ACTIVE_DAYS': active_days,
        'ACTIVE_QUARTERS': active_quarters,
        'AVG_SESSIONS_PER_ACTIVE_QUARTER': avg_sessions_per_active_quarter,
        'AVG_SESSION_LENGTH_PER_SESSION': avg_session_length_per_session,
        'DAYS_SINCE_LAST_ACTIVITY_2022': days_since_last_activity_2022,
    }])

    input_df = pd.concat([numeric_df, encoded_df], axis=1)
    # Reindex to the exact training column order — guards against silent
    # column misalignment, which would otherwise produce wrong predictions.
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    probability = model.predict_proba(input_df)[0][1]
    risk = "Low" if probability >= 0.6 else "Medium" if probability >= 0.4 else "High"

    st.metric("Renewal Probability", f"{probability:.2f}")
    st.write(f"**Churn Risk:** {risk}")
