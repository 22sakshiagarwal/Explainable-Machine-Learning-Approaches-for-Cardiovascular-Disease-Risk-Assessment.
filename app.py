# --- Streamlit Cardio Risk Predictor ---
# To run locally: `streamlit run app.py`

import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# Load model and training column info
@st.cache_resource
def load_model():
    try:
        model = joblib.load("cardio_model.pkl")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_columns():
    try:
        training_columns = joblib.load("training_columns.pkl")
        return training_columns
    except Exception as e:
        st.error(f"Error loading columns: {e}")
        return None

model = load_model()
training_columns = load_columns()

# Check if model and columns loaded successfully
if model is None or training_columns is None:
    st.stop()  # Stop the app if there are issues with loading model/columns

# Set page config and title
st.set_page_config(page_title="Cardiovascular Risk Predictor", layout="centered")
st.title(" Cardiovascular Disease Risk Predictor")
st.markdown("Enter the patient details below:")

# Input fields
age = st.number_input("Age (years)", 18, 100)
gender = st.selectbox("Gender", ["Male", "Female"])
ap_hi = st.number_input("Resting Blood Pressure (mmHg)", 80, 200)
ap_lo = st.number_input("Diastolic Blood Pressure (mmHg)", 40, 150)
cholesterol = st.selectbox("Cholesterol Level", [1, 2, 3])
glucose = st.selectbox("Glucose Level", [1, 2, 3])
smoke = st.selectbox("Smoking (0=No, 1=Yes)", [0, 1])
alco = st.selectbox("Alcohol Intake (0=No, 1=Yes)", [0, 1])
active = st.selectbox("Physical Activity (0=No, 1=Yes)", [0, 1])

# Prepare input data as DataFrame
input_data = pd.DataFrame([{
    'age': age,
    'gender': 1 if gender == "Male" else 0,
    'ap_hi': ap_hi,
    'ap_lo': ap_lo,
    'cholesterol': cholesterol,
    'gluc': glucose,
    'smoke': smoke,
    'alco': alco,
    'active': active
}])

# Apply one-hot encoding like training
input_encoded = pd.get_dummies(input_data)

# Reindex to match training columns
input_encoded = input_encoded.reindex(columns=training_columns, fill_value=0)

# Prediction button
if st.button("Predict Risk"):
    prediction = model.predict(input_encoded)[0]
    pred_prob = model.predict_proba(input_encoded)[0][prediction]

    if prediction == 1:
        st.error(f" High Risk of Cardiovascular Disease (Confidence: {pred_prob:.2f})")
    else:
        st.success(f" Low Risk of Cardiovascular Disease (Confidence: {pred_prob:.2f})")

  
   
