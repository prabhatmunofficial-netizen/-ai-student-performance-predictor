
import streamlit as st
import pandas as pd
import joblib

# Load machine learning model
model = joblib.load("student_performance_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📊",
    layout="centered"
)

# Heading
st.title("AI Student Performance Predictor")

st.write(
    "Estimate your academic performance using "
    "machine learning and your study habits."
)

st.divider()

# Student inputs
st.subheader("Enter Your Details")

study_hours = st.number_input(
    "Daily Study Hours",
    min_value=0.0,
    max_value=12.0,
    value=6.0,
    step=0.5
)

attendance = st.number_input(
    "Attendance Percentage",
    min_value=0,
    max_value=100,
    value=85
)

previous_marks = st.number_input(
    "Previous Examination Marks",
    min_value=0,
    max_value=100,
    value=70
)

sleep_hours = st.number_input(
    "Daily Sleep Hours",
    min_value=0.0,
    max_value=12.0,
    value=8.0,
    step=0.5
)

# Prediction
if st.button("Predict My Performance"):

    student = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "sleep_hours": sleep_hours
    }])

    prediction = model.predict(student)[0]

    prediction = max(0, min(100, prediction))

    st.success(
        f"Estimated Final Marks: {prediction:.2f}/100"
    )

    st.info(
        "This is an experimental estimate based on "
        "synthetic training data, not a reliable "
        "prediction of actual academic results."
    )
