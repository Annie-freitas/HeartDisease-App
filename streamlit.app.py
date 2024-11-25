import streamlit as st
import pandas as pd
import numpy as np
import pickle  # or joblib
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model = pickle.load(open('Heart_Disease_model.pkl', 'rb'))


st.title("Heart Disease Indicator")
st.write("Predict the likelihood of heart disease based on health indicators.")

# Input features
age = st.number_input("Age", min_value=0, max_value=120, value=30)
sex = st.selectbox("Sex", options=["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=0, max_value=300, value=120)
chol = st.number_input("Cholesterol Level (chol)", min_value=0, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1])
restecg = st.selectbox("Resting ECG Results (restecg)", options=[0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", min_value=0, max_value=250, value=150)
exang = st.selectbox("Exercise-Induced Angina (exang)", options=[0, 1])
oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise (slope)", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])


# Map sex to binary
sex = 1 if sex == "Male" else 0

# Predict heart disease
if st.button("Predict"):
    features = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    prediction = model.predict(features)
    st.write("Heart Disease Likelihood: ", "Yes" if prediction[0] == 1 else "No")
