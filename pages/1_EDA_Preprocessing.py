import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns


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
###################################################################
    # Placeholder sections for later (DONE)
    st.header("4. Numeric Distributions")

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    selected_num_col = st.selectbox(
        "Select a numeric column to explore:",
        numeric_cols,
        index=1
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Histogram")
        fig, ax = plt.subplots()
        ax.hist(df[selected_num_col], bins=30)
        ax.set_xlabel(selected_num_col)
        ax.set_ylabel("Count")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Boxplot")
        fig, ax = plt.subplots()
        ax.boxplot(df[selected_num_col].dropna(), vert=True)
        ax.set_xlabel(selected_num_col)
        st.pyplot(fig)
#########################################################

###################################################### Categorical Distribution 
    
    st.header("5. Categorical Distributions")
    
    # Treat these as categorical (except ID)
    categorical_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod", "Churn"
    ]
    
    selected_cat_col = st.selectbox(
        "Select a categorical column to explore:",
        categorical_cols,
        index=categorical_cols.index("Churn")
    )
    
    cat_counts = df[selected_cat_col].value_counts().sort_values(ascending=False)
    
    fig, ax = plt.subplots()
    sns.barplot(x=cat_counts.index, y=cat_counts.values, ax=ax)
    ax.set_xlabel(selected_cat_col)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
    
    st.write("Value counts:")
    st.write(cat_counts)


    ###################################################################################
    
    st.header("6. Correlation & Preprocessing (Coming Next)")
    st.info("Here we will show correlations and the final cleaned dataset export.")

if __name__ == "__main__":
    main()

