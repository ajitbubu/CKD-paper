"""
Dashboard Page
Interactive visualizations and analytics for CKD predictions
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.utils.database import CKDDatabase
from app.utils.helpers import get_risk_distribution

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

# Initialize database
@st.cache_resource
def get_database():
    return CKDDatabase()

db = get_database()

# Page header
st.title("📊 CKD Risk Prediction Dashboard")
st.markdown("Interactive visualizations and detailed analytics")

st.markdown("---")

# Check if there are any predictions
predictions_history = db.get_all_predictions()

if len(predictions_history) == 0:
    st.warning("⚠️ No prediction data available. Please upload data first!")
    st.info("👉 Go to the **Upload Data** page to upload your data and run predictions")
    st.stop()

# Selection controls
col1, col2 = st.columns([2, 1])

with col1:
    # Select prediction run
    selected_idx = st.selectbox(
        "📋 Select Prediction Run",
        range(len(predictions_history)),
        format_func=lambda i: f"{predictions_history.iloc[i]['dataset_name']} - "
                             f"{predictions_history.iloc[i]['prediction_date']} "
                             f"({predictions_history.iloc[i]['total_patients']} patients)"
    )

    selected_prediction_id = predictions_history.iloc[selected_idx]['id']

with col2:
    st.markdown(f"""
    **Dataset**: {predictions_history.iloc[selected_idx]['dataset_name']}
    **Date**: {predictions_history.iloc[selected_idx]['prediction_date']}
    **Patients**: {predictions_history.iloc[selected_idx]['total_patients']}
    """)

# Get prediction details
metrics, patients_df = db.get_prediction_details(selected_prediction_id)

st.markdown("---")

# Overview metrics
st.markdown("### 📈 Overview Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Total Patients",
        metrics['total_patients'],
        delta=None
    )

with col2:
    ckd_rate = metrics['ckd_positive_rate'] * 100
    st.metric(
        "CKD Detection Rate",
        f"{ckd_rate:.1f}%",
        delta=f"{metrics['ckd_positive']} patients",
        delta_color="inverse"
    )

with col3:
    if metrics['accuracy'] > 0:
        st.metric(
            "Model Accuracy",
            f"{metrics['accuracy']:.1%}",
            delta="Validated"
        )
    else:
        st.metric(
            "Model Accuracy",
            "N/A",
            delta="No labels"
        )

with col4:
    if metrics['roc_auc'] > 0:
        st.metric(
            "ROC-AUC Score",
            f"{metrics['roc_auc']:.3f}",
            delta="High"
        )
    else:
        st.metric(
            "ROC-AUC",
            "N/A",
            delta="No labels"
        )

with col5:
    high_risk_count = len(patients_df[patients_df['risk_category'] == 'High Risk'])
    st.metric(
        "High Risk Patients",
        high_risk_count,
        delta=f"{high_risk_count/len(patients_df)*100:.1f}%",
        delta_color="inverse"
    )

st.markdown("---")

# Risk distribution visualization
st.markdown("### 🎯 Risk Distribution")

col1, col2 = st.columns([1, 1])

with col1:
    # Pie chart
    risk_counts = patients_df['risk_category'].value_counts()

    fig = go.Figure(data=[go.Pie(
        labels=risk_counts.index,
        values=risk_counts.values,
        hole=0.4,
        marker=dict(colors=['#2ecc71', '#f39c12', '#e74c3c']),
        textinfo='label+percent+value',
        textfont=dict(size=14)
    )])

    fig.update_layout(
        title="Risk Category Distribution",
        height=400,
        showlegend=True
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Probability distribution histogram
    fig = px.histogram(
        patients_df,
        x='probability',
        nbins=30,
        title='CKD Probability Distribution',
        labels={'probability': 'CKD Probability', 'count': 'Number of Patients'},
        color_discrete_sequence=['#3498db']
    )

    fig.add_vline(x=0.3, line_dash="dash", line_color="green", annotation_text="Low/Mod")
    fig.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="Mod/High")

    fig.update_layout(height=400)

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Feature importance
if not metrics['feature_importance'].empty:
    st.markdown("### 🎯 Feature Importance Analysis")

    feature_importance = metrics['feature_importance']

    # Top N selector
    top_n = st.slider("Select number of top features to display", 10, 30, 20)

    top_features = feature_importance.nlargest(top_n, 'importance')

    # Feature importance bar chart
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title=f'Top {top_n} Most Important Features',
        labels={'importance': 'Importance Score', 'feature': 'Feature'},
        color='importance',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        height=max(400, top_n * 20),
        yaxis={'categoryorder': 'total ascending'}
    )

    st.plotly_chart(fig, use_container_width=True)

    # Feature importance by category
    st.markdown("#### Feature Importance by Data Source")

    # Categorize features
    def categorize_feature(feature_name):
        clinical_keywords = ['creatinine', 'egfr', 'albuminuria', 'hba1c', 'bp', 'bmi', 'heart_rate', 'map', 'pulse']
        claims_keywords = ['visits', 'admissions', 'referrals', 'dialysis', 'utilization', 'claim']
        sdoh_keywords = ['income', 'poverty', 'unemployment', 'education', 'adi', 'food_desert', 'prevalence', 'sdoh']
        retail_keywords = ['food', 'produce', 'sodium', 'beverage', 'supplement', 'monitor', 'vitamin', 'dietary', 'health_conscious']

        feature_lower = feature_name.lower()

        if any(keyword in feature_lower for keyword in clinical_keywords):
            return 'Clinical'
        elif any(keyword in feature_lower for keyword in claims_keywords):
            return 'Healthcare Utilization'
        elif any(keyword in feature_lower for keyword in sdoh_keywords):
            return 'SDOH'
        elif any(keyword in feature_lower for keyword in retail_keywords):
            return 'Retail/Dietary'
        elif 'diabetes' in feature_lower or 'hypertension' in feature_lower or 'cvd' in feature_lower or 'comorbidity' in feature_lower:
            return 'Comorbidities'
        elif 'medication' in feature_lower or 'ace' in feature_lower or 'arb' in feature_lower:
            return 'Medications'
        else:
            return 'Other'

    feature_importance['category'] = feature_importance['feature'].apply(categorize_feature)

    category_importance = feature_importance.groupby('category')['importance'].sum().reset_index()
    category_importance = category_importance.sort_values('importance', ascending=False)

    col1, col2 = st.columns([1, 1])

    with col1:
        fig = px.bar(
            category_importance,
            x='importance',
            y='category',
            orientation='h',
            title='Total Importance by Data Source',
            labels={'importance': 'Total Importance', 'category': 'Data Source'},
            color='importance',
            color_continuous_scale='Blues'
        )

        fig.update_layout(height=400, showlegend=False)

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure(data=[go.Pie(
            labels=category_importance['category'],
            values=category_importance['importance'],
            hole=0.4
        )])

        fig.update_layout(
            title="Importance Distribution by Source",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# High risk patients analysis
st.markdown("### ⚠️ High Risk Patients")

col1, col2 = st.columns([3, 1])

with col1:
    # Filter controls
    risk_filter = st.multiselect(
        "Filter by Risk Category",
        options=['Low Risk', 'Moderate Risk', 'High Risk'],
        default=['High Risk']
    )

with col2:
    sort_by = st.selectbox(
        "Sort by",
        options=['Probability (High to Low)', 'Probability (Low to High)', 'Patient ID'],
        index=0
    )

# Apply filters
filtered_patients = patients_df[patients_df['risk_category'].isin(risk_filter)]

# Apply sorting
if sort_by == 'Probability (High to Low)':
    filtered_patients = filtered_patients.sort_values('probability', ascending=False)
elif sort_by == 'Probability (Low to High)':
    filtered_patients = filtered_patients.sort_values('probability', ascending=True)
else:
    filtered_patients = filtered_patients.sort_values('patient_id')

# Display table
st.dataframe(
    filtered_patients[['patient_id', 'probability', 'risk_category']].style.format({
        'probability': '{:.2%}'
    }).background_gradient(subset=['probability'], cmap='RdYlGn_r'),
    use_container_width=True,
    height=400
)

# Download filtered results
import io

csv_buffer = io.StringIO()
filtered_patients.to_csv(csv_buffer, index=False)

st.download_button(
    label="📥 Download Filtered Results",
    data=csv_buffer.getvalue(),
    file_name=f"filtered_patients_{risk_filter}.csv",
    mime="text/csv"
)

st.markdown("---")

# Model performance metrics (if available)
if metrics['accuracy'] > 0:
    st.markdown("### 📊 Model Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Accuracy gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=metrics['accuracy'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Accuracy"},
            delta={'reference': 90},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#2ecc71"},
                'steps': [
                    {'range': [0, 70], 'color': "#e74c3c"},
                    {'range': [70, 85], 'color': "#f39c12"},
                    {'range': [85, 100], 'color': "#2ecc71"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ))

        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Precision gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics['precision'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Precision"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#3498db"},
                'steps': [
                    {'range': [0, 70], 'color': "lightgray"},
                    {'range': [70, 85], 'color': "gray"},
                    {'range': [85, 100], 'color': "darkgray"}
                ]
            }
        ))

        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        # Recall gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics['recall'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Recall"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#9b59b6"}
            }
        ))

        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # F1-Score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metrics['f1_score'] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "F1-Score"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#e67e22"}
            }
        ))

        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.info("""
💡 **Dashboard Tips**:
- Use the dropdown to compare different prediction runs
- Filter high-risk patients for targeted interventions
- Download filtered results for further analysis
- Feature importance shows which factors drive predictions
""")
