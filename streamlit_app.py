import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import pickle

# ======================================
# LOAD MODEL
# ======================================
model = pickle.load(open("heart_model.pkl", "rb"))

# ======================================
# DATABASE CONNECTION
# ======================================
conn = sqlite3.connect("heart_data.db")
cursor = conn.cursor()

# ======================================
# CREATE TABLE
# ======================================
cursor.execute("""
CREATE TABLE IF NOT EXISTS patients (

    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    sex INTEGER,
    chest_pain_type INTEGER,
    resting_bp INTEGER,
    cholesterol INTEGER,
    fasting_bs INTEGER,
    rest_ecg INTEGER,
    max_hr INTEGER,
    exercise_angina INTEGER,
    oldpeak REAL,
    slope INTEGER,
    prediction TEXT

)
""")

conn.commit()

# ======================================
# PAGE TITLE
# ======================================
st.title("Heart Disease Prediction System")

# ======================================
# TABS
# ======================================
tab1, tab2, tab3 = st.tabs(
    ["Predict", "Bulk Predict", "Model Information"]
)

# ======================================
# TAB 1 - PREDICT
# ======================================
with tab1:

    st.header("Patient Prediction")

    age = st.number_input("Age", 1, 120, 45)

    sex = st.number_input(
        "Sex (1=Male, 0=Female)",
        0,
        1,
        1
    )

    cp = st.number_input(
        "Chest Pain Type",
        0,
        3,
        1
    )

    trestbps = st.number_input(
        "Resting Blood Pressure",
        80,
        250,
        120
    )

    chol = st.number_input(
        "Cholesterol",
        100,
        600,
        200
    )

    fbs = st.number_input(
        "Fasting Blood Sugar",
        0,
        1,
        0
    )

    restecg = st.number_input(
        "Rest ECG",
        0,
        2,
        1
    )

    thalach = st.number_input(
        "Max Heart Rate",
        60,
        250,
        150
    )

    exang = st.number_input(
        "Exercise Angina",
        0,
        1,
        0
    )

    oldpeak = st.number_input(
        "Old Peak",
        0.0,
        10.0,
        1.0
    )

    slope = st.number_input(
        "Slope",
        0,
        2,
        1
    )

    # ======================================
    # PREDICT BUTTON
    # ======================================
    if st.button("Predict"):

        input_data = np.array([[
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope
        ]])

        prediction = model.predict(input_data)

        # ======================================
        # RESULT
        # ======================================
        if prediction[0] == 1:
            result = "Heart Disease Present"
            
            st.error(result)

        st.subheader("Conclusion")

        st.write("""
        The patient shows a higher possibility of heart disease.

        Recommended actions:
        - Consult a cardiologist
        - Maintain healthy diet
        - Exercise regularly
        - Reduce stress
        - Monitor blood pressure regularly
        - Avoid smoking and alcohol
        """)

    else:
        result = "No Heart Disease"

        st.success(result)

        st.subheader("Conclusion")

        st.write("""
        The patient shows low risk of heart disease.

        Recommended actions:
        - Continue healthy lifestyle
        - Regular exercise
        - Balanced diet
        - Regular health checkups
        """)

        # ======================================
        # STORE DATA
        # ======================================
        cursor.execute("""
        INSERT INTO patients (

            age,
            sex,
            chest_pain_type,
            resting_bp,
            cholesterol,
            fasting_bs,
            rest_ecg,
            max_hr,
            exercise_angina,
            oldpeak,
            slope,
            prediction

        )

        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)

        """, (
            age,
            sex,
            cp,
            trestbps,
            chol,
            fbs,
            restecg,
            thalach,
            exang,
            oldpeak,
            slope,
            result
        ))

        conn.commit()

        st.success("Data Stored Successfully")

# ======================================
# TAB 2 - BULK PREDICT
# ======================================
with tab2:

    st.header("Bulk Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(uploaded_file)

        st.write("Uploaded Data")
        st.dataframe(data)

        predictions = model.predict(data)

        data["Prediction"] = predictions

        data["Prediction"] = data["Prediction"].replace({
            1: "Heart Disease Present",
            0: "No Heart Disease"
        })

        st.write("Prediction Result")
        st.dataframe(data)

# ======================================
# TAB 3 - MODEL INFORMATION
# ======================================
with tab3:

    st.header("Model Information")
    
    st.write("Model used : Decision tree")

    st.write("Model Used : Logistic Regression")
    
    st.write("Model Used : Random Forest")
    
    st.write("Model Used : Support Vector Machine"
    )

    st.write("Features Used :")

    st.write("""
    1. Age
    2. Sex
    3. Chest Pain Type
    4. Resting Blood Pressure
    5. Cholesterol
    6. Fasting Blood Sugar
    7. Rest ECG
    8. Max Heart Rate
    9. Exercise Angina
    10. Old Peak
    11. Slope
    """)

# ======================================
# SHOW DATABASE DATA
# ======================================
st.header("Stored Patient Data")

df = pd.read_sql_query(
    "SELECT * FROM patients",
    conn
)

st.dataframe(df)