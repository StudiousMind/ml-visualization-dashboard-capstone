import streamlit as st
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ---------- Helper: load cleaned data ----------
@st.cache_data
def load_clean_data():
    data_path = os.path.join("data", "cleaned_telco.csv")
    df = pd.read_csv(data_path)
    return df

def main():
    st.title("Modeling & Evaluation – Telco Customer Churn")

    st.write(
        """
        This page trains a simple baseline model on the **cleaned dataset**
        and visualizes basic evaluation metrics.
        """
    )

    # --- Load data ---
    df = load_clean_data()

    st.subheader("1. Data Preview (Model Input)")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head())

    # --- Split features/target ---
    st.subheader("2. Train/Test Split")

    target_col = "Churn"
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found in cleaned data.")
        return

    X = df.drop(columns=[target_col])
    y = df[target_col]

    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    st.write(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # --- Baseline model: Logistic Regression ---
    st.subheader("3. Train Baseline Model (Logistic Regression)")

    if st.button("Train model"):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        filename = os.path.join("models", "trained_model.joblib")
        joblib.dump(model, filename)

        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {acc:.3f}")

        # Confusion matrix
        st.subheader("4. Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Classification report
        st.subheader("5. Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.info(
            "This is just a baseline model. "
            "Your teammates can extend this page with more models "
            "(Random Forest, XGBoost, etc.) and add ROC / PR curves."
        )

if __name__ == "__main__":
    main()

