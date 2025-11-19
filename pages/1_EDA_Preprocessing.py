import streamlit as st
import pandas as pd
import os

# ---------- Helper function to load data ----------
@st.cache_data
def load_data():
    # Adjust path if needed depending on where Streamlit runs from
    data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(data_path)
    
    # Clean TotalCharges (same logic you used in Colab)
    df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
    
    # Convert SeniorCitizen to Yes/No
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    
    return df

# ---------- Page layout ----------
def main():
    st.title("EDA & Preprocessing – Telco Customer Churn")
    st.write(
        """
        This page lets you explore the Telco Customer Churn dataset and shows 
        the preprocessing steps that prepare it for modeling.
        """
    )

    # Load data
    df = load_data()

    # --- Section 1: Raw data preview ---
    st.header("1. Raw Data Preview")
    st.write(f"Dataset shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
    st.dataframe(df.head())

    # --- Section 2: Dataset info ---
    st.header("2. Column Types & Basic Info")
    st.write("**Column data types:**")
    st.write(df.dtypes)

    # --- Section 3: Missing values ---
    st.header("3. Missing Values Check")
    missing_counts = df.isnull().sum()
    st.write(missing_counts)

    # Placeholder sections for later (we'll fill them in next steps)
    st.header("4. Numeric Distributions (Coming Next)")
    st.info("Here we will add histograms and boxplots for numeric columns.")

    st.header("5. Categorical Distributions (Coming Next)")
    st.info("Here we will add bar charts for categorical columns.")

    st.header("6. Correlation & Preprocessing (Coming Next)")
    st.info("Here we will show correlations and the final cleaned dataset export.")

if __name__ == "__main__":
    main()

