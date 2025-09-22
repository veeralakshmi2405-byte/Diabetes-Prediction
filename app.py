import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

# ------------------------------
# Load Model
# ------------------------------
model = joblib.load("best_model.joblib")

st.set_page_config(
    page_title="🩺 Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# ------------------------------
# App Title
# ------------------------------
st.title("🩺 Diabetes Prediction System")
st.markdown("### Predict the likelihood of diabetes based on health parameters")
st.write("---")

# ------------------------------
# Sidebar - User Input
# ------------------------------
st.sidebar.header("📋 Enter Patient Details")

def user_input():
    Pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
    Glucose = st.sidebar.slider("Glucose Level", 0, 200, 100)
    BloodPressure = st.sidebar.slider("Blood Pressure (mm Hg)", 0, 140, 70)
    SkinThickness = st.sidebar.slider("Skin Thickness (mm)", 0, 100, 20)
    Insulin = st.sidebar.slider("Insulin Level", 0, 900, 80)
    BMI = st.sidebar.slider("BMI", 0.0, 70.0, 25.0)
    DiabetesPedigreeFunction = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    Age = st.sidebar.slider("Age", 18, 100, 30)

    data = {
        "Pregnancies": Pregnancies,
        "Glucose": Glucose,
        "BloodPressure": BloodPressure,
        "SkinThickness": SkinThickness,
        "Insulin": Insulin,
        "BMI": BMI,
        "DiabetesPedigreeFunction": DiabetesPedigreeFunction,
        "Age": Age
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# ------------------------------
# Prediction
# ------------------------------
if st.sidebar.button("🔍 Predict Diabetes"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]  # probability of diabetes

    # Display result card
    st.subheader("📊 Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ The patient is **likely Diabetic** with probability {proba:.2f}")
    else:
        st.success(f"✅ The patient is **likely Non-Diabetic** with probability {1-proba:.2f}")

    # ------------------------------
    # SHAP Explanation
    # ------------------------------
    st.subheader("🔎 Model Explainability (SHAP Values)")

    explainer = shap.TreeExplainer(model) if "XGB" in str(type(model)) or "Forest" in str(type(model)) else shap.Explainer(model, input_df)
    shap_values = explainer(input_df)

    # Force Plot (per-sample)
    st.write("Feature contributions for this prediction:")

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

    # Bar plot
    st.write("Overall feature importance for this prediction:")
    shap.plots.bar(shap_values, show=False)
    st.pyplot(bbox_inches='tight', dpi=120)

else:
    st.info("👈 Enter patient details in the sidebar and click **Predict Diabetes** to see results.")
