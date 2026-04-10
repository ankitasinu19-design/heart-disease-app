import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Heart Disease Prediction")

st.title("❤️ Heart Disease Prediction App")

# INPUTS
age = st.slider("Age", 20, 80, 40)

sex_text = st.selectbox("Gender", ["Male", "Female"])
sex = 1 if sex_text == "Male" else 0

bp = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Cholesterol", 100, 400, 200)

fbs_text = st.selectbox("Fasting Blood Sugar > 120", ["No", "Yes"])
fbs = 1 if fbs_text == "Yes" else 0

thalach = st.slider("Maximum Heart Rate", 60, 220, 150)

# LOAD MODEL
model = pickle.load(open("model.pkl", "rb"))

# PREDICT
if st.button("Predict"):
    input_data = np.array([[age, sex, bp, chol, fbs, thalach]])
    result = model.predict(input_data)

    if result[0] == 1:
        st.error("⚠️ High risk of Heart Disease")
    else:
        st.success("✅ Low risk of Heart Disease")
