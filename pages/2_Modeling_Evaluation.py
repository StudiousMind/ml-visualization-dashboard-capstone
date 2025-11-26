import streamlit as st
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

    # Training a baseline model
    st.subheader("3. Train Baseline Model")

    model_dict = {
        "Logistic Regression" : LogisticRegression(max_iter=1000),
        "Decision Tree" : DecisionTreeClassifier(),
        "Random Forest" : RandomForestClassifier(),
        "K-Nearest Neighbors" : KNeighborsClassifier()
    }
    
    selected_model = st.selectbox("Select a model to train:", model_dict.keys(), None)

    if st.button("Train model"):

        model = model_dict[selected_model]
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

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
        st.code(classification_report(y_test, y_pred))
        
        # Precision recall curve
        st.subheader("6. Precision Recall Curve")
        baseline_precision = y_test.mean()
        precision, recall, _ =  precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)

        fig2, ax2 = plt.subplots()
        ax2.plot(recall, precision, label=f"Average Precision = {ap:.2f}")
        ax2.axhline(baseline_precision, linestyle = "--", label=f"Baseline = {baseline_precision:.2f}", color = "orange")
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve")
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

        # Feature importance plot
        
        st.subheader("7. Feature Importance Plot")
        feat_names = X.columns

        if selected_model == "Logistic Regression":
            importance = model.coef_[0]
        elif selected_model == "Decision Tree" or selected_model == "Random Forest":
            importance = model.feature_importances_

        feat_importance_df = pd.DataFrame({
            "Feature" : feat_names,
            "Importance" : importance
        }).sort_values(by='Importance', ascending=False)

        fig3, ax3 = plt.subplots(figsize = (12,8))
        sns.barplot(data=feat_importance_df, x="Importance", y="Feature", ax=ax3, palette="coolwarm")
        ax3.set_title("Feature Importance")
        st.pyplot(fig3)

if __name__ == "__main__":
    main()