import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("data/models/best_model.joblib")

st.title("Career Path Recommendation System")
st.write("Fill the details below to get a career recommendation.")


# Input fields
gpa = st.number_input(
    "GPA (out of 10)", 
    min_value=0.0, 
    max_value=10.0, 
    step=0.1
)

interestarea = st.number_input(
    "Interest Area (encoded value)", 
    min_value=0, 
    step=1
)

skills = st.number_input(
    "Skills (encoded value)", 
    min_value=0, 
    step=1
)

extracurricular = st.number_input(
    "Extracurricular Activities (encoded value)", 
    min_value=0, 
    step=1
)


# Predict button
if st.button("Recommend Career"):
    input_df = pd.DataFrame([{
        "gpa": gpa,
        "interestarea": interestarea,
        "skills": skills,
        "extracurricularactivities": extracurricular
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"Recommended Career: {prediction}")

