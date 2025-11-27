
# import streamlit as st
# import pandas as pd
# import numpy as np
# import os
# import joblib
# import shap
# import matplotlib.pyplot as plt

# #st.set_option("deprecation.showPyplotGlobalUse", False)

# @st.cache_data
# def load_clean_data():
#     data_path = os.path.join("data", "cleaned_telco.csv")
#     return pd.read_csv(data_path)

# @st.cache_resource
# def load_model_and_shap():
#     models_dir = "models"

#     model_path = os.path.join(models_dir, "logreg_telco.pkl")
#     shap_values_path = os.path.join(models_dir, "shap_values_telco.npy")
#     background_path = os.path.join(models_dir, "shap_background_telco.csv")

#     if not (os.path.exists(model_path) and
#             os.path.exists(shap_values_path) and
#             os.path.exists(background_path)):
#         return None, None, None

#     model = joblib.load(model_path)
#     shap_values = np.load(shap_values_path)
#     X_sample = pd.read_csv(background_path)

#     return model, shap_values, X_sample

# def main():
#     st.title("Model Interpretability – Telco Customer Churn")

#     st.write(
#         """
#         This page uses **SHAP** values to explain which features are driving 
#         the churn predictions of our model.
#         """
#     )

#     df_clean = load_clean_data()

#     st.subheader("1. Cleaned Dataset Overview")
#     st.write(f"Shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
#     st.dataframe(df_clean.head())

#     st.subheader("2. Load Precomputed SHAP Values")

#     model, shap_values, X_sample = load_model_and_shap()

#     if model is None:
#         st.warning(
#             "SHAP files not found yet. Once the modeling person computes SHAP "
#             "values in Colab and uploads them to the `models/` folder, "
#             "this page will show the explanations automatically."
#         )
#         st.info(
#             "Expected files: `models/logreg_telco.pkl`, "
#             "`models/shap_values_telco.npy`, "
#             "`models/shap_background_telco.csv`."
#         )
#         return

#     st.success("Precomputed SHAP values loaded successfully.")

#     st.subheader("3. Global Feature Importance (SHAP Summary – Bar Plot)")
#     shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
#     st.pyplot(bbox_inches="tight")

#     st.subheader("4. Feature Impact Distribution (SHAP Summary – Beeswarm)")
#     shap.summary_plot(shap_values, X_sample, show=False)
#     st.pyplot(bbox_inches="tight")

#     st.markdown(
#         """
#         **How to read this:**
#         - Features at the top are the most important for predicting churn.  
#         - Red dots = higher feature values, blue dots = lower feature values.  
#         - Points to the right push predictions *towards churn*, 
#           points to the left push predictions *away from churn*.
#         """
#     )

# if __name__ == "__main__":
#     main()

import streamlit as st
import pandas as pd
import numpy as np
import os
import shap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# -----------------------------
# 1. Data loading
# -----------------------------
@st.cache_data
def load_clean_data():
    """Load the preprocessed Telco churn dataset created on Page 1."""
    data_path = os.path.join("data", "cleaned_telco.csv")
    df = pd.read_csv(data_path)
    return df


# -----------------------------
# 2. Model training + SHAP
# -----------------------------
@st.cache_resource
def train_model_and_compute_shap():
    """
    Train a Logistic Regression model on the cleaned Telco dataset
    and compute SHAP values using shap.Explainer.

    Returns
    -------
    model : fitted LogisticRegression
    shap_explanation : shap.Explanation object
    shap_values_array : np.ndarray of shape (n_samples, n_features)
    X_for_shap : pd.DataFrame (features used for SHAP)
    """
    df = load_clean_data()

    target_col = "Churn"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in cleaned data.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Train a simple Logistic Regression model (same features as cleaned data)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # For SHAP, use the full dataset or a sample if it is very large
    if len(X) > 3000:
        X_for_shap = X.sample(n=3000, random_state=42)
    else:
        X_for_shap = X.copy()

    # Use the general Explainer interface
    explainer = shap.Explainer(model, X_for_shap)
    shap_explanation = explainer(X_for_shap)

    # Convert SHAP values to a clean float array
    shap_values_array = np.array(shap_explanation.values, dtype=float)

    # Ensure X_for_shap aligns with shap_values_array
    if isinstance(shap_explanation.data, np.ndarray):
        X_for_shap = pd.DataFrame(
            shap_explanation.data,
            columns=X.columns
        )

    return model, shap_explanation, shap_values_array, X_for_shap


# -----------------------------
# 3. Streamlit page
# -----------------------------
def main():
    st.title("Model Interpretability – Telco Customer Churn")

    st.write(
        """
        This page focuses on **model interpretability** using **SHAP**
        (SHapley Additive exPlanations).

        - A Logistic Regression model is trained on the cleaned Telco churn data.  
        - SHAP values are computed to understand **which features drive churn predictions**.  
        - Both **global importance** and **local, per-customer explanations** are shown.
        """
    )

    # -------------------------
    # 1. Show cleaned dataset
    # -------------------------
    df_clean = load_clean_data()
    st.subheader("1. Cleaned Dataset Overview")
    st.write(f"Shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
    st.dataframe(df_clean.head())

    # -------------------------
    # 2. Train / load model and SHAP values
    # -------------------------
    st.subheader("2. Train Model & Compute SHAP Values")
    with st.spinner(
        "Training Logistic Regression model and computing SHAP values "
        "(only runs once thanks to caching)..."
    ):
        try:
            model, shap_explanation, shap_values, X_shap = train_model_and_compute_shap()
        except Exception as e:
            st.error(f"Could not compute SHAP values: {e}")
            return

    st.success("Model and SHAP values are ready.")

    # -------------------------
    # 3. Global importance (mean |SHAP|)
    # -------------------------
    st.subheader("3. Global Feature Importance (Mean |SHAP|)")
    st.write(
        """
        Here the **mean absolute SHAP value** is computed for each feature.  
        This shows, on average, how strongly each feature influences the model's
        churn prediction across all customers.
        """
    )

    # Compute mean absolute SHAP value per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    feature_names = X_shap.columns

    # Sort features by importance (descending)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    sorted_features = feature_names[sorted_idx]
    sorted_importance = mean_abs_shap[sorted_idx]

    # Show top N features to keep chart readable
    top_n = min(15, len(sorted_features))
    top_features = sorted_features[:top_n]
    top_importance = sorted_importance[:top_n]

    fig_imp, ax_imp = plt.subplots()
    y_pos = np.arange(len(top_features))
    ax_imp.barh(y_pos, top_importance)
    ax_imp.set_yticks(y_pos)
    ax_imp.set_yticklabels(top_features)
    ax_imp.invert_yaxis()  # most important at the top
    ax_imp.set_xlabel("Mean |SHAP value|")
    ax_imp.set_title(f"Top {top_n} Most Important Features (Global)")
    fig_imp.tight_layout()
    st.pyplot(fig_imp, bbox_inches="tight")
    plt.close(fig_imp)

    # -------------------------
    # 4. Beeswarm plot (global distribution)
    # -------------------------
    st.subheader("4. Feature Impact Distribution (SHAP Beeswarm Plot)")
    st.write(
        """
        The beeswarm plot shows:

        - Each point = one customer.  
        - X-axis = how much that feature pushed the prediction towards
          **churn** (right) or **no churn** (left).  
        - Colour = actual feature value (red = high, blue = low).

        This combines **importance** and **direction of impact** for each feature.
        """
    )

    shap.summary_plot(
        shap_values,
        X_shap,
        show=False
    )
    fig_beeswarm = plt.gcf()
    st.pyplot(fig_beeswarm, bbox_inches="tight")
    plt.clf()

    # -------------------------
    # 5. Local explanation for a single customer
    # -------------------------
    st.subheader("5. Individual Prediction Explanation (Local SHAP)")

    st.write(
        """
        Select a specific row from the dataset used for SHAP to see **why**
        the model predicts churn or no-churn for that customer.
        """
    )

    n_samples = X_shap.shape[0]
    selected_index = st.number_input(
        "Select row index for explanation (0-based index within SHAP sample)",
        min_value=0,
        max_value=int(n_samples - 1),
        value=0,
        step=1,
    )

    # Show the selected customer's feature values
    x_row = X_shap.iloc[int(selected_index)]
    st.markdown("**Selected customer's feature values:**")
    st.dataframe(x_row.to_frame(name="value"))

    # Model prediction for the selected row
    proba = model.predict_proba(x_row.to_frame().T)[0, 1]
    st.write(f"**Predicted churn probability for this customer:** {proba:.3f}")

    # Local SHAP values for this row
    row_shap_values = shap_values[int(selected_index)]

    # Plot top features for this single prediction
    abs_row = np.abs(row_shap_values)
    row_sorted_idx = np.argsort(abs_row)[::-1]
    top_n_local = min(10, len(row_sorted_idx))
    top_idx_local = row_sorted_idx[:top_n_local]

    local_features = feature_names[top_idx_local]
    local_shap_vals = row_shap_values[top_idx_local]

    fig_local, ax_local = plt.subplots()
    y_pos_local = np.arange(len(local_features))
    ax_local.barh(y_pos_local, local_shap_vals)
    ax_local.set_yticks(y_pos_local)
    ax_local.set_yticklabels(local_features)
    ax_local.axvline(0, color="black", linewidth=1)
    ax_local.invert_yaxis()
    ax_local.set_xlabel("SHAP value (impact on model output)")
    ax_local.set_title(f"Top {top_n_local} feature impacts for selected customer")
    fig_local.tight_layout()
    st.pyplot(fig_local, bbox_inches="tight")
    plt.close(fig_local)

    st.markdown(
        """
        **Interpretation:**

        - Features with **positive SHAP values** (bars to the right) push the model
          towards predicting **churn** for this customer.  
        - Features with **negative SHAP values** (bars to the left) push the model
          towards **no churn**.  
        - The longer the bar, the stronger the impact of that feature for this
          specific prediction.
        """
    )


if __name__ == "__main__":
    main()
