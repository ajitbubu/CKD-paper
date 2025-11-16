"""
CKD Detection Web Application
Multi-modal XGBoost-based CKD Risk Prediction Dashboard
"""

import streamlit as st
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="CKD Risk Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stAlert {
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Main page
def main():
    # Header
    st.markdown('<div class="main-header">🏥 CKD Risk Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Multi-Modal XGBoost Classifier for Chronic Kidney Disease Detection</div>',
        unsafe_allow_html=True
    )

    # Introduction
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📊 Multi-Modal Data")
        st.markdown("""
        Integrates **4 data sources**:
        - 🔬 **Clinical (EHR)**: Lab values, vitals, diagnoses
        - 🏥 **Claims**: Healthcare utilization patterns
        - 🌍 **SDOH**: Social determinants (ZIP-level)
        - 🛒 **Retail**: Purchase patterns
        """)

    with col2:
        st.markdown("### 🤖 XGBoost Model")
        st.markdown("""
        Advanced ML classifier:
        - ✅ **66 Features** engineered from raw data
        - ✅ **SHAP Interpretability** for explainability
        - ✅ **95%+ AUC** on validation data
        - ✅ **SMOTE Balancing** for class imbalance
        """)

    with col3:
        st.markdown("### 🎯 Key Features")
        st.markdown("""
        Complete workflow:
        - 📤 **Upload** your multi-source data
        - 🔮 **Predict** CKD risk for patients
        - 📈 **Visualize** results and insights
        - 📜 **Track** prediction history
        """)

    st.markdown("---")

    # How to use
    st.markdown("### 🚀 How to Use This Dashboard")

    st.markdown("""
    1. **📤 Upload Data** (Page 1)
       - Upload CSV files for all 4 data sources
       - Download sample templates if needed
       - Validate data structure
       - Run predictions

    2. **📊 Dashboard** (Page 2)
       - View prediction results
       - Explore feature importance
       - Analyze high-risk patients
       - Interactive visualizations

    3. **📜 History** (Page 3)
       - Track all prediction runs
       - Compare datasets
       - View historical metrics
       - Export results

    **👈 Use the sidebar to navigate between pages**
    """)

    # Sample data info
    st.markdown("---")
    st.markdown("### 📋 Data Requirements")

    with st.expander("Click to see required data formats"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Clinical Data (EHR)")
            st.markdown("""
            Required columns:
            - `patient_id` - Unique patient identifier
            - `serum_creatinine` - mg/dL
            - `egfr` - mL/min/1.73m²
            - `albuminuria_acr` - mg/g
            - `hba1c` - %
            - `systolic_bp`, `diastolic_bp` - mmHg
            - `bmi` - kg/m²
            - Additional: `has_diabetes`, `has_hypertension`, etc.
            """)

            st.markdown("#### Claims Data")
            st.markdown("""
            Required columns:
            - `patient_id`
            - `er_visits_count`
            - `hospital_admissions_count`
            - `outpatient_visits_count`
            - `primary_care_visits_count`
            - Additional: specialist visits, prescriptions, etc.
            """)

        with col2:
            st.markdown("#### SDOH Data (ZIP-level)")
            st.markdown("""
            Required columns:
            - `zip_code` - 5-digit ZIP code
            - `median_household_income` - $
            - `poverty_rate` - %
            - `diabetes_prevalence` - %
            - `adi_national_percentile` - 1-100
            - Additional: education, food access, etc.
            """)

            st.markdown("#### Retail Purchase Data")
            st.markdown("""
            Required columns:
            - `patient_id`
            - `processed_food_purchases`
            - `fresh_produce_purchases`
            - `high_sodium_food_purchases`
            - Additional: health products, supplements, etc.
            """)

    # Model info
    st.markdown("---")
    st.markdown("### 🔬 About the Model")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Features", "66", delta="From 4 sources")

    with col2:
        st.metric("Model Type", "XGBoost", delta="Binary classifier")

    with col3:
        st.metric("Target AUC", "95%+", delta="High accuracy")

    with col4:
        st.metric("Interpretability", "SHAP", delta="Explainable AI")

    st.info("""
    💡 **Note**: This is a research prototype for IEEE publication. Always validate predictions
    with clinical expertise. The model uses synthetic data for demonstration purposes.
    """)

    st.markdown("---")

    # Footer
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p><strong>CKD Risk Prediction Dashboard</strong></p>
        <p>Multi-Modal XGBoost Classification System</p>
        <p>Built with Streamlit | Data: Clinical, Claims, SDOH, Retail</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
