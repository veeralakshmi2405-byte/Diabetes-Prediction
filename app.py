import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Load Model
# ------------------------------
model = joblib.load("best_model.joblib")

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(
    page_title="🩺 Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# ------------------------------
# Title Section
# ------------------------------
st.title("🩺 Diabetes Prediction Dashboard")
st.markdown(
    """
    Welcome to the **Diabetes Prediction System**.  
    Enter patient health details on the left panel to check the likelihood of diabetes.  
    """
)
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
# Main Layout
# ------------------------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🧾 Patient Summary")
    st.dataframe(input_df.style.set_properties(**{'background-color': '#f0f2f6'}))

with col2:
    st.subheader("📊 Prediction Result")
    if st.sidebar.button("🔍 Predict Diabetes"):
        prediction = model.predict(input_df)[0]

        # Probability handling
        try:
            proba = model.predict_proba(input_df)[0][1]
        except:
            proba = 0.5  # fallback if model has no predict_proba

        if prediction == 1:
            st.error(f"⚠️ The patient is **likely Diabetic** with probability {proba:.2f}")
        else:
            st.success(f"✅ The patient is **likely Non-Diabetic** with probability {1-proba:.2f}")
    else:
        st.info("👈 Enter details in the sidebar and click **Predict Diabetes**.")

st.write("---")

# ------------------------------
# Model Performance Section
# ------------------------------
st.subheader("📈 Model Performance Comparison")

try:
    results_df = pd.read_csv("model_results.csv")
    # Convert column names to lowercase to avoid mismatch
    results_df.columns = [col.lower() for col in results_df.columns]

    col3, col4 = st.columns(2)

    with col3:
        st.write("### Accuracy by Model")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="model", y="accuracy", data=results_df, ax=ax, palette="Blues_d")
        plt.xticks(rotation=30)
        st.pyplot(fig)

    with col4:
        st.write("### F1 Score by Model")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x="model", y="f1", data=results_df, ax=ax, palette="Greens_d")
        plt.xticks(rotation=30)
        st.pyplot(fig)

    # ROC AUC full-width chart
    st.write("### ROC AUC by Model")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x="model", y="roc_auc", data=results_df, ax=ax, palette="Oranges_d")
    plt.xticks(rotation=30)
    st.pyplot(fig)

except FileNotFoundError:
    st.warning("⚠️ `model_results.csv` not found. Upload it to see model performance charts.")
except Exception as e:
    st.error(f"Error in plotting model results: {e}")
