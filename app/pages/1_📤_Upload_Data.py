"""
Data Upload Page
Upload multi-modal data and run predictions
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import io

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.utils.database import CKDDatabase
from app.utils.helpers import (
    validate_dataframe, get_required_columns, make_predictions,
    get_risk_distribution, create_sample_data_templates
)

st.set_page_config(page_title="Upload Data", page_icon="📤", layout="wide")

# Initialize database
@st.cache_resource
def get_database():
    return CKDDatabase()

db = get_database()

# Page header
st.title("📤 Upload Multi-Modal Data")
st.markdown("Upload your data files and run CKD risk predictions")

st.markdown("---")

# Sample templates section
with st.expander("📥 Download Sample Data Templates"):
    st.markdown("Download sample CSV templates to understand the required format:")

    templates = create_sample_data_templates()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        csv_buffer = io.StringIO()
        templates['clinical'].to_csv(csv_buffer, index=False)
        st.download_button(
            label="📋 Clinical Template",
            data=csv_buffer.getvalue(),
            file_name="clinical_template.csv",
            mime="text/csv"
        )

    with col2:
        csv_buffer = io.StringIO()
        templates['claims'].to_csv(csv_buffer, index=False)
        st.download_button(
            label="🏥 Claims Template",
            data=csv_buffer.getvalue(),
            file_name="claims_template.csv",
            mime="text/csv"
        )

    with col3:
        csv_buffer = io.StringIO()
        templates['sdoh'].to_csv(csv_buffer, index=False)
        st.download_button(
            label="🌍 SDOH Template",
            data=csv_buffer.getvalue(),
            file_name="sdoh_template.csv",
            mime="text/csv"
        )

    with col4:
        csv_buffer = io.StringIO()
        templates['retail'].to_csv(csv_buffer, index=False)
        st.download_button(
            label="🛒 Retail Template",
            data=csv_buffer.getvalue(),
            file_name="retail_template.csv",
            mime="text/csv"
        )

st.markdown("---")

# Dataset name
dataset_name = st.text_input(
    "📝 Dataset Name",
    placeholder="e.g., Q4_2024_Patient_Cohort",
    help="Give your dataset a descriptive name for tracking"
)

st.markdown("---")

# File upload section
st.markdown("### 📁 Upload Data Files")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Clinical & Claims Data")

    clinical_file = st.file_uploader(
        "🔬 Clinical Data (EHR)",
        type=['csv'],
        help="Upload clinical/EHR data with lab values, vitals, diagnoses"
    )

    claims_file = st.file_uploader(
        "🏥 Claims Data",
        type=['csv'],
        help="Upload healthcare claims and utilization data"
    )

with col2:
    st.markdown("#### SDOH & Retail Data")

    sdoh_file = st.file_uploader(
        "🌍 SDOH Data",
        type=['csv'],
        help="Upload social determinants of health data (ZIP-code level)"
    )

    retail_file = st.file_uploader(
        "🛒 Retail Purchase Data",
        type=['csv'],
        help="Upload retail purchase pattern data"
    )

# Optional labels file
labels_file = st.file_uploader(
    "🏷️ Labels (Optional)",
    type=['csv'],
    help="Upload known CKD labels for validation (optional)"
)

st.markdown("---")

# Process uploads
if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
    # Validation
    if not dataset_name:
        st.error("❌ Please provide a dataset name")
        st.stop()

    if not all([clinical_file, claims_file, sdoh_file, retail_file]):
        st.error("❌ Please upload all required data files (Clinical, Claims, SDOH, Retail)")
        st.stop()

    try:
        with st.spinner("📊 Loading and validating data..."):
            # Load dataframes
            clinical_df = pd.read_csv(clinical_file)
            claims_df = pd.read_csv(claims_file)
            sdoh_df = pd.read_csv(sdoh_file)
            retail_df = pd.read_csv(retail_file)
            labels_df = pd.read_csv(labels_file) if labels_file else None

            # Get required columns
            required_cols = get_required_columns()

            # Validate each dataframe
            validations = [
                validate_dataframe(clinical_df, required_cols['clinical'], "Clinical data"),
                validate_dataframe(claims_df, required_cols['claims'], "Claims data"),
                validate_dataframe(sdoh_df, required_cols['sdoh'], "SDOH data"),
                validate_dataframe(retail_df, required_cols['retail'], "Retail data")
            ]

            # Check if all valid
            all_valid = all(v[0] for v in validations)

            if not all_valid:
                st.error("❌ Data validation failed:")
                for is_valid, error_msg in validations:
                    if not is_valid:
                        st.error(f"  • {error_msg}")
                st.stop()

            st.success("✅ Data validation passed!")

        # Display data summary
        st.markdown("### 📊 Data Summary")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Clinical Records", len(clinical_df))

        with col2:
            st.metric("Claims Records", len(claims_df))

        with col3:
            st.metric("SDOH Locations", len(sdoh_df))

        with col4:
            st.metric("Retail Records", len(retail_df))

        # Save dataset to database
        with st.spinner("💾 Saving dataset..."):
            dataset_id = db.save_dataset(
                dataset_name=dataset_name,
                clinical_df=clinical_df,
                claims_df=claims_df,
                sdoh_df=sdoh_df,
                retail_df=retail_df,
                file_paths={
                    'clinical': clinical_file.name,
                    'claims': claims_file.name,
                    'sdoh': sdoh_file.name,
                    'retail': retail_file.name
                }
            )

            st.success(f"✅ Dataset saved (ID: {dataset_id})")

        # Run predictions
        with st.spinner("🔮 Running CKD risk predictions..."):
            predictions_df, metrics, feature_importance, X, y = make_predictions(
                clinical_df, claims_df, sdoh_df, retail_df, labels_df
            )

            st.success("✅ Predictions complete!")

        # Save predictions to database
        with st.spinner("💾 Saving prediction results..."):
            prediction_id = db.save_prediction(
                dataset_id=dataset_id,
                predictions=predictions_df,
                metrics=metrics,
                feature_importance=feature_importance
            )

            st.success(f"✅ Results saved (Prediction ID: {prediction_id})")

        # Display results
        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")

        # Risk distribution
        risk_dist = get_risk_distribution(predictions_df)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Patients",
                len(predictions_df),
                delta=None
            )

        with col2:
            st.metric(
                "High Risk",
                risk_dist['high'],
                delta=f"{risk_dist['high_pct']:.1f}%",
                delta_color="inverse"
            )

        with col3:
            st.metric(
                "Moderate Risk",
                risk_dist['moderate'],
                delta=f"{risk_dist['moderate_pct']:.1f}%",
                delta_color="off"
            )

        with col4:
            st.metric(
                "Low Risk",
                risk_dist['low'],
                delta=f"{risk_dist['low_pct']:.1f}%",
                delta_color="normal"
            )

        # Model metrics (if labels available)
        if metrics.get('accuracy', 0) > 0:
            st.markdown("#### 📊 Model Performance")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.3f}")

            with col2:
                st.metric("Precision", f"{metrics['precision']:.3f}")

            with col3:
                st.metric("Recall", f"{metrics['recall']:.3f}")

            with col4:
                st.metric("F1-Score", f"{metrics['f1_score']:.3f}")

            with col5:
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

        # Top 10 features
        st.markdown("#### 🎯 Top 10 Most Important Features")

        top_features = feature_importance.head(10)

        import plotly.express as px

        fig = px.bar(
            top_features,
            x='importance',
            y='feature',
            orientation='h',
            title='Feature Importance',
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # High risk patients preview
        st.markdown("#### ⚠️ Top 10 High-Risk Patients")

        high_risk_patients = predictions_df.nlargest(10, 'probability')

        st.dataframe(
            high_risk_patients[['patient_id', 'probability', 'risk_level']].style.format({
                'probability': '{:.2%}'
            }),
            use_container_width=True,
            hide_index=True
        )

        # Download results
        st.markdown("---")
        st.markdown("### 📥 Download Results")

        csv_buffer = io.StringIO()
        predictions_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="📊 Download Predictions (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"{dataset_name}_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.success("✅ All done! Check the Dashboard page for detailed visualizations.")

        # Store in session state for dashboard
        st.session_state['latest_predictions'] = predictions_df
        st.session_state['latest_metrics'] = metrics
        st.session_state['latest_feature_importance'] = feature_importance
        st.session_state['latest_dataset_name'] = dataset_name

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

# Footer
st.markdown("---")
st.info("""
💡 **Tips**:
- Ensure all files have the required columns
- Patient IDs must match across Clinical, Claims, and Retail files
- ZIP codes must match between SDOH data and patient records
- Download sample templates above if you need guidance
""")
