"""
History Page
Track and compare all prediction runs
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
from datetime import datetime

# Add paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.utils.database import CKDDatabase

st.set_page_config(page_title="History", page_icon="📜", layout="wide")

# Initialize database
@st.cache_resource
def get_database():
    return CKDDatabase()

db = get_database()

# Page header
st.title("📜 Prediction History")
st.markdown("Track and compare all prediction runs over time")

st.markdown("---")

# Get all data
datasets = db.get_all_datasets()
predictions = db.get_all_predictions()
history_stats = db.get_prediction_history_stats()

# Check if there's any history
if len(predictions) == 0:
    st.warning("⚠️ No prediction history available yet")
    st.info("👉 Go to the **Upload Data** page to run your first prediction")
    st.stop()

# Summary statistics
st.markdown("### 📊 Overall Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Prediction Runs",
        history_stats['total_prediction_runs']
    )

with col2:
    st.metric(
        "Patients Analyzed",
        f"{history_stats['total_patients_analyzed']:,}"
    )

with col3:
    st.metric(
        "Avg CKD Detection Rate",
        f"{history_stats['average_ckd_rate']:.1%}"
    )

with col4:
    st.metric(
        "Avg Model Accuracy",
        f"{history_stats['average_accuracy']:.1%}" if history_stats['average_accuracy'] > 0 else "N/A"
    )

st.markdown("---")

# Timeline visualization
st.markdown("### 📅 Prediction Timeline")

# Prepare timeline data
timeline_data = predictions.copy()
timeline_data['prediction_date'] = pd.to_datetime(timeline_data['prediction_date'])

col1, col2 = st.columns([2, 1])

with col1:
    # Timeline chart
    fig = px.scatter(
        timeline_data,
        x='prediction_date',
        y='total_patients',
        size='ckd_positive',
        color='ckd_positive_rate',
        hover_data=['dataset_name', 'ckd_positive', 'model_auc'],
        title='Prediction Runs Over Time',
        labels={
            'prediction_date': 'Date',
            'total_patients': 'Total Patients',
            'ckd_positive_rate': 'CKD Rate'
        },
        color_continuous_scale='RdYlGn_r'
    )

    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # CKD detection rate trend
    if len(timeline_data) > 1:
        fig = px.line(
            timeline_data.sort_values('prediction_date'),
            x='prediction_date',
            y='ckd_positive_rate',
            markers=True,
            title='CKD Detection Rate Trend',
            labels={
                'prediction_date': 'Date',
                'ckd_positive_rate': 'CKD Rate'
            }
        )

        fig.update_layout(
            height=400,
            yaxis_tickformat='.1%'
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 2 predictions to show trend")

st.markdown("---")

# Datasets overview
st.markdown("### 📂 Uploaded Datasets")

# Display datasets table
st.dataframe(
    datasets[['dataset_name', 'upload_date', 'total_patients',
             'clinical_records', 'claims_records', 'sdoh_records', 'retail_records']].style.format({
        'upload_date': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M')
    }),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# Predictions table
st.markdown("### 🔮 All Predictions")

# Add filters
col1, col2, col3 = st.columns(3)

with col1:
    date_filter = st.date_input(
        "Filter by Date (from)",
        value=None
    )

with col2:
    dataset_filter = st.multiselect(
        "Filter by Dataset",
        options=predictions['dataset_name'].unique(),
        default=None
    )

with col3:
    min_patients = st.number_input(
        "Minimum Patients",
        min_value=0,
        value=0,
        step=10
    )

# Apply filters
filtered_predictions = predictions.copy()

if date_filter:
    filtered_predictions = filtered_predictions[
        pd.to_datetime(filtered_predictions['prediction_date']).dt.date >= date_filter
    ]

if dataset_filter:
    filtered_predictions = filtered_predictions[
        filtered_predictions['dataset_name'].isin(dataset_filter)
    ]

if min_patients > 0:
    filtered_predictions = filtered_predictions[
        filtered_predictions['total_patients'] >= min_patients
    ]

# Display filtered predictions
st.dataframe(
    filtered_predictions[['prediction_date', 'dataset_name', 'total_patients',
                         'ckd_positive', 'ckd_negative', 'ckd_positive_rate',
                         'model_accuracy', 'model_auc']].style.format({
        'prediction_date': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M'),
        'ckd_positive_rate': '{:.1%}',
        'model_accuracy': lambda x: f'{x:.1%}' if x > 0 else 'N/A',
        'model_auc': lambda x: f'{x:.3f}' if x > 0 else 'N/A'
    }),
    use_container_width=True,
    height=400
)

st.markdown(f"**Showing {len(filtered_predictions)} of {len(predictions)} predictions**")

st.markdown("---")

# Comparison section
st.markdown("### ⚖️ Compare Predictions")

if len(predictions) >= 2:
    col1, col2 = st.columns(2)

    with col1:
        prediction_1_idx = st.selectbox(
            "Select First Prediction",
            range(len(predictions)),
            format_func=lambda i: f"{predictions.iloc[i]['dataset_name']} - "
                                 f"{predictions.iloc[i]['prediction_date']}"
        )

    with col2:
        prediction_2_idx = st.selectbox(
            "Select Second Prediction",
            range(len(predictions)),
            index=min(1, len(predictions) - 1),
            format_func=lambda i: f"{predictions.iloc[i]['dataset_name']} - "
                                 f"{predictions.iloc[i]['prediction_date']}"
        )

    if prediction_1_idx != prediction_2_idx:
        pred1 = predictions.iloc[prediction_1_idx]
        pred2 = predictions.iloc[prediction_2_idx]

        # Comparison metrics
        st.markdown("#### Comparison Metrics")

        comparison_data = pd.DataFrame({
            'Metric': ['Total Patients', 'CKD Positive', 'CKD Rate', 'Accuracy', 'AUC'],
            'Prediction 1': [
                pred1['total_patients'],
                pred1['ckd_positive'],
                f"{pred1['ckd_positive_rate']:.1%}",
                f"{pred1['model_accuracy']:.1%}" if pred1['model_accuracy'] > 0 else 'N/A',
                f"{pred1['model_auc']:.3f}" if pred1['model_auc'] > 0 else 'N/A'
            ],
            'Prediction 2': [
                pred2['total_patients'],
                pred2['ckd_positive'],
                f"{pred2['ckd_positive_rate']:.1%}",
                f"{pred2['model_accuracy']:.1%}" if pred2['model_accuracy'] > 0 else 'N/A',
                f"{pred2['model_auc']:.3f}" if pred2['model_auc'] > 0 else 'N/A'
            ]
        })

        st.dataframe(comparison_data, use_container_width=True, hide_index=True)

        # Visualization comparison
        col1, col2 = st.columns(2)

        with col1:
            # Bar chart comparison
            metrics_to_compare = ['Total Patients', 'CKD Positive', 'CKD Negative']
            values_1 = [pred1['total_patients'], pred1['ckd_positive'], pred1['ckd_negative']]
            values_2 = [pred2['total_patients'], pred2['ckd_positive'], pred2['ckd_negative']]

            fig = go.Figure(data=[
                go.Bar(name=pred1['dataset_name'], x=metrics_to_compare, y=values_1),
                go.Bar(name=pred2['dataset_name'], x=metrics_to_compare, y=values_2)
            ])

            fig.update_layout(
                title='Patient Counts Comparison',
                barmode='group',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Performance comparison (if available)
            if pred1['model_accuracy'] > 0 and pred2['model_accuracy'] > 0:
                perf_metrics = ['Accuracy', 'AUC']
                values_1 = [pred1['model_accuracy'], pred1['model_auc']]
                values_2 = [pred2['model_accuracy'], pred2['model_auc']]

                fig = go.Figure(data=[
                    go.Bar(name=pred1['dataset_name'], x=perf_metrics, y=values_1),
                    go.Bar(name=pred2['dataset_name'], x=perf_metrics, y=values_2)
                ])

                fig.update_layout(
                    title='Model Performance Comparison',
                    barmode='group',
                    height=400,
                    yaxis_range=[0, 1]
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Model performance metrics not available for comparison")

    else:
        st.warning("⚠️ Please select two different predictions to compare")

else:
    st.info("Need at least 2 predictions to enable comparison")

st.markdown("---")

# Export section
st.markdown("### 📥 Export History")

col1, col2 = st.columns(2)

with col1:
    # Export datasets
    import io

    csv_buffer = io.StringIO()
    datasets.to_csv(csv_buffer, index=False)

    st.download_button(
        label="📊 Download Datasets History (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"datasets_history_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with col2:
    # Export predictions
    csv_buffer = io.StringIO()
    predictions.to_csv(csv_buffer, index=False)

    st.download_button(
        label="📊 Download Predictions History (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"predictions_history_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.info("""
💡 **History Page Tips**:
- Track model performance across different datasets
- Compare predictions to identify trends
- Filter and export results for reporting
- Monitor CKD detection rates over time
""")
