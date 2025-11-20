
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import shap
import matplotlib.pyplot as plt

st.set_option("deprecation.showPyplotGlobalUse", False)

@st.cache_data
def load_clean_data():
    data_path = os.path.join("data", "cleaned_telco.csv")
    return pd.read_csv(data_path)

@st.cache_resource
def load_model_and_shap():
    models_dir = "models"

    model_path = os.path.join(models_dir, "logreg_telco.pkl")
    shap_values_path = os.path.join(models_dir, "shap_values_telco.npy")
    background_path = os.path.join(models_dir, "shap_background_telco.csv")

    if not (os.path.exists(model_path) and
            os.path.exists(shap_values_path) and
            os.path.exists(background_path)):
        return None, None, None

    model = joblib.load(model_path)
    shap_values = np.load(shap_values_path)
    X_sample = pd.read_csv(background_path)

    return model, shap_values, X_sample

def main():
    st.title("Model Interpretability – Telco Customer Churn")

    st.write(
        """
        This page uses **SHAP** values to explain which features are driving 
        the churn predictions of our model.
        """
    )

    df_clean = load_clean_data()

    st.subheader("1. Cleaned Dataset Overview")
    st.write(f"Shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
    st.dataframe(df_clean.head())

    st.subheader("2. Load Precomputed SHAP Values")

    model, shap_values, X_sample = load_model_and_shap()

    if model is None:
        st.warning(
            "SHAP files not found yet. Once the modeling person computes SHAP "
            "values in Colab and uploads them to the `models/` folder, "
            "this page will show the explanations automatically."
        )
        st.info(
            "Expected files: `models/logreg_telco.pkl`, "
            "`models/shap_values_telco.npy`, "
            "`models/shap_background_telco.csv`."
        )
        return

    st.success("Precomputed SHAP values loaded successfully.")

    st.subheader("3. Global Feature Importance (SHAP Summary – Bar Plot)")
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    st.pyplot(bbox_inches="tight")

    st.subheader("4. Feature Impact Distribution (SHAP Summary – Beeswarm)")
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(bbox_inches="tight")

    st.markdown(
        """
        **How to read this:**
        - Features at the top are the most important for predicting churn.  
        - Red dots = higher feature values, blue dots = lower feature values.  
        - Points to the right push predictions *towards churn*, 
          points to the left push predictions *away from churn*.
        """
    )

if __name__ == "__main__":
    main()
