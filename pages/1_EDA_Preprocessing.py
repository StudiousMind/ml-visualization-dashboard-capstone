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
    This page is all about understanding the Telco churn dataset and getting it ready
    for machine learning. We explore the data, clean it up, fix any issues, and 
    transform everything into a format that the model can actually use.
    """
)


    # Load data
    df = load_data()

    # --- Section 1: Raw data preview ---
    st.header("1. Raw Data Preview")
    st.write(
    """
    Here’s the dataset we’re working with. It’s basically a list of Telco customers, 
    their services, how long they’ve been with the company, how much they pay, 
    and whether they cancelled. This gives us a quick feel for what we’re dealing 
    with before doing anything fancy.
    """
)
    st.write(f"Dataset shape: **{df.shape[0]} rows × {df.shape[1]} columns**")
    st.dataframe(df.head())

    # --- Section 2: Dataset info ---
    st.header("2. Column Types & Basic Info")
    st.write(
    """
    This shows the type of each column — numbers vs categories. 
    It helps us decide what needs encoding, what needs scaling, 
    and what should probably be dropped before modeling.
    """
)
    st.write("**Column data types:**")
    st.write(df.dtypes)

    # --- Section 3: Missing values ---
    st.header("3. Missing Values Check")
    st.write(
    """
    Just making sure the dataset isn’t hiding anything weird. 
    The only issue was TotalCharges having blank values for customers 
    with zero tenure — which totally makes sense. 
    We cleaned that up during preprocessing.
    """
)
    missing_counts = df.isnull().sum()
    st.write(missing_counts)
###################################################################

    
    # Placeholder sections for later (DONE)
    st.header("4. Numeric Distributions")
    st.write(
    """
    These charts show how the numeric features are spread out — things like tenure 
    and monthly charges. This helps us spot skewed distributions or customers who 
    behave differently from the rest.
    """
)


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
    st.write(
    """
    Here we look at the categories — contract type, payment method, internet service, 
    and so on. This tells us how many people fall into each group and gives us a feel 
    for the overall customer mix.
    """
)

    
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
    st.write(
    """
    This heatmap shows which numeric features move together. For example, TotalCharges 
    is closely tied to tenure and MonthlyCharges — which makes sense, since it’s basically 
    the total money a customer has paid over time.
    """
)


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
    st.write(
    """
    Here’s where the real cleaning happens. We fix the TotalCharges issue, 
    convert categorical columns into numbers, scale the numeric features, 
    and basically get the dataset into a format that a machine learning model 
    can actually understand.
    """
)

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
    st.write(
    """
    This lets us compare the raw data with the final processed version. 
    The raw data is human-friendly but messy for ML. 
    The cleaned version is all numbers, scaled values, and encoded categories — 
    this is what we actually feed to the model.
    """
)

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
    st.write(
    """
    If anyone in the group wants to work offline or rerun the model, 
    this is the cleaned dataset ready to go. You can download it here 
    or just load it directly from the repo.
    """
)
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

