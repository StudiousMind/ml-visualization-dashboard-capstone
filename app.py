import streamlit as st
import os

# ---------- Page config ----------
st.set_page_config(
    page_title="ML Workflow Visualization Dashboard",
    page_icon="📊",
    layout="wide",
)


# # Load and show logo
# logo_path = load_logo()
# if logo_path:
#     st.image(logo_path, width=180)

# ---------- Optional: load logo ----------
def load_logo():
    # Put your logo at: assets/logo.png
    logo_path = os.path.join("assets", "logo.png")
    if os.path.exists(logo_path):
        return logo_path
    return None

def main():
    # ===== HERO SECTION =====
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## 📊 Machine Learning Workflow Visualization Dashboard")
        st.markdown(
            """
            This app walks through a **full ML workflow** – from exploring the data, 
            to preprocessing, modeling, and finally explaining predictions.

            It’s built as our capstone project for the **Data Visualization for ML** course.
            """
        )

        st.markdown("### 👨‍💻 Team")
        st.markdown(
            """
            - **Kai Balharith [1250916]** – EDA & Preprocessing  
            - **Brian [Student ID]** – Modeling & Evaluation  
            - **Rabil [Student ID]** – Interpretability & Explainability  
            """
        )




    with col2:
        logo_path = load_logo()
        if logo_path:
            st.image(logo_path, use_column_width=True)
        else:
            st.markdown("### 🧠 Project Logo")
            st.markdown(
                """
                You can add a custom logo by placing a file at  
                `assets/logo.png` in the repo.
                """
            )

    st.markdown("---")

    # ===== QUICK LINKS (FAKE CARDS) =====
    st.markdown("### 🚀 Jump into the workflow")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("#### 🔍 EDA & Preprocessing")
        st.markdown(
            """
            Explore the Telco churn dataset, fix issues like `TotalCharges`,
            and generate a cleaned, model-ready dataset.
            """
        )
        st.markdown("➡️ Open from the sidebar: **EDA_Preprocessing**")

    with col_b:
        st.markdown("#### 🤖 Modeling & Evaluation")
        st.markdown(
            """
            Train baseline models, check metrics like accuracy,
            and visualize confusion matrices and reports.
            """
        )
        st.markdown("➡️ Open from the sidebar: **Modeling_Evaluation**")

    with col_c:
        st.markdown("#### 🧠 Interpretability")
        st.markdown(
            """
            Use SHAP-based visualizations (once uploaded) 
            to understand which features drive churn predictions.
            """
        )
        st.markdown("➡️ Open from the sidebar: **Interpretability**")

    st.markdown("---")
    st.caption("Built with ❤️ by Group 5 – Data Visualization for ML Capstone")

if __name__ == "__main__":
    main()

