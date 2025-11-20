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

    ############################################################## Heatmap
    
    st.header("6. Correlation Heatmap (Numeric Features)")

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    corr_matrix = df[numeric_cols].corr()
    
    st.write("Correlation matrix:")
    st.dataframe(corr_matrix)
    
    fig, ax = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

#################################################################################

##################################################### Preprocessing 
    st.header("7. Preprocessing Pipeline & Cleaned Dataset")
    
    # --- Drop ID column ---
    df_prep = df.drop(["customerID"], axis=1)
    
    # --- Binary Encode Yes/No columns ---
    binary_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "Churn"
    ]
    
    # Encode Yes/No or Male/Female as 1/0
    for col in binary_cols:
        df_prep[col] = df_prep[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})
    
    # --- One-Hot Encode multi-category columns ---
    multi_cat_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod"
    ]
    
    df_prep = pd.get_dummies(df_prep, columns=multi_cat_cols, drop_first=True)
    
    # --- Scale numeric features ---
    from sklearn.preprocessing import StandardScaler
    
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    df_prep[numeric_cols] = scaler.fit_transform(df_prep[numeric_cols])

    #Inserting the before and after comparisoin

    # --- Before vs After Preprocessing ---
    st.subheader("Before vs After Preprocessing")
    
    tab1, tab2 = st.tabs(["Before (Raw Data)", "After (Cleaned Data)"])
    
    with tab1:
        st.write("**Raw dataset (original features):**")
        st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        st.dataframe(df.head())
    
    with tab2:
        st.write("**Cleaned, encoded, and scaled dataset:**")
        st.write(f"Shape: {df_prep.shape[0]} rows × {df_prep.shape[1]} columns")
        st.dataframe(df_prep.head())

    # End of before/after block



    
    # --- Show cleaned dataset preview ---
    st.subheader("Cleaned Dataset Preview")
    st.dataframe(df_prep.head())
    
    # --- Download cleaned dataset ---
    csv = df_prep.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Cleaned Dataset (CSV)",
        data=csv,
        file_name="cleaned_telco.csv",
        mime="text/csv"
    )
    
    # --- Save it into /data so teammates can use it ---
    output_path = os.path.join("data", "cleaned_telco.csv")
    df_prep.to_csv(output_path, index=False)
    st.success(f"Cleaned dataset saved to: {output_path}")


if __name__ == "__main__":
    main()

