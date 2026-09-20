
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺"
)

st.title("🩺 Diabetes Prediction")

st.write("Enter the patient information below.")

st.warning(
    "This application is for educational purposes only "
    "and is not a medical diagnosis."
)

st.subheader("Patient Information")

pregnancies = st.text_input("Pregnancies", "1")
glucose = st.text_input("Glucose", "120")
blood_pressure = st.text_input("Blood Pressure", "70")
skin_thickness = st.text_input("Skin Thickness", "20")
bmi = st.text_input("BMI", "25")
diabetes_pedigree = st.text_input(
    "Diabetes Pedigree Function", "0.47"
)
age = st.text_input("Age", "30")

if st.button("🔍 Predict Diabetes"):

    try:
        pregnancies = float(pregnancies)
        glucose = float(glucose)
        blood_pressure = float(blood_pressure)
        skin_thickness = float(skin_thickness)
        bmi = float(bmi)
        diabetes_pedigree = float(diabetes_pedigree)
        age = float(age)

        model = joblib.load("diabetes_model.pkl")

        input_data = pd.DataFrame({
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [blood_pressure],
            "SkinThickness": [skin_thickness],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [diabetes_pedigree],
            "Age": [age]
        })

        # Same zero-handling approach used during training
        original_data = pd.read_csv("diabetes.csv")

        zero_columns = [
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "BMI"
        ]

        for column in zero_columns:
            median_value = original_data[column].median()

            if input_data.loc[0, column] == 0:
                input_data.loc[0, column] = median_value

        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Model prediction: Class 1")
        else:
            st.success("Model prediction: Class 0")

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]
            st.write(
                f"Estimated probability of Class 1: "
                f"**{probability * 100:.2f}%**"
            )

    except ValueError:
        st.error("Please enter valid numeric values.")
    except Exception as e:
        st.error(f"Prediction error: {e}")
