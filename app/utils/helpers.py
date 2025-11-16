"""
Helper utilities for the CKD prediction web application
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from data_processing.data_integration import DataIntegrationPipeline
from models.xgboost_model import CKDXGBoostClassifier
import joblib


def load_model_and_pipeline():
    """Load trained model and preprocessing pipeline"""
    model_path = Path('models/ckd_xgboost_model.pkl')
    pipeline_path = Path('models/data_pipeline.pkl')

    if not model_path.exists() or not pipeline_path.exists():
        return None, None

    # Load model
    model_obj = CKDXGBoostClassifier()
    model_obj.load_model(str(model_path))

    # Load pipeline
    pipeline = joblib.load(str(pipeline_path))

    return model_obj, pipeline


def validate_dataframe(df: pd.DataFrame, required_columns: list, df_name: str) -> tuple:
    """
    Validate dataframe structure

    Returns:
        (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, f"{df_name} is empty"

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"{df_name} is missing required columns: {', '.join(missing_cols)}"

    return True, ""


def get_required_columns():
    """Get required columns for each data source"""
    return {
        'clinical': ['patient_id', 'serum_creatinine', 'egfr', 'albuminuria_acr',
                    'hba1c', 'systolic_bp', 'diastolic_bp', 'bmi'],
        'claims': ['patient_id', 'er_visits_count', 'hospital_admissions_count',
                  'outpatient_visits_count', 'primary_care_visits_count'],
        'sdoh': ['zip_code', 'median_household_income', 'poverty_rate',
                'diabetes_prevalence', 'adi_national_percentile'],
        'retail': ['patient_id', 'processed_food_purchases', 'fresh_produce_purchases',
                  'high_sodium_food_purchases']
    }


def make_predictions(clinical_df: pd.DataFrame, claims_df: pd.DataFrame,
                    sdoh_df: pd.DataFrame, retail_df: pd.DataFrame,
                    labels_df: pd.DataFrame = None):
    """
    Make predictions on uploaded data

    Returns:
        (predictions_df, metrics_dict, feature_importance_df, X_test, y_test)
    """
    # Load model and pipeline
    model, pipeline = load_model_and_pipeline()
    if model is None or pipeline is None:
        raise ValueError("Model not found. Please train the model first.")

    # If no labels provided, create dummy labels
    if labels_df is None:
        labels_df = pd.DataFrame({
            'patient_id': clinical_df['patient_id'].unique(),
            'has_ckd': 0,  # Dummy label
            'ckd_stage': 0,
            'zip_code': clinical_df.merge(
                pd.DataFrame({'patient_id': clinical_df['patient_id'].unique()}),
                on='patient_id', how='left'
            )['patient_id'].map(lambda x: '00000')  # Will be updated from clinical/retail
        })

        # Try to get zip_code from clinical or retail data
        if 'zip_code' in clinical_df.columns:
            zip_map = clinical_df.groupby('patient_id')['zip_code'].first()
            labels_df['zip_code'] = labels_df['patient_id'].map(zip_map).fillna('00000')

    # Prepare data using pipeline
    X, y = pipeline.prepare_for_modeling(
        clinical_df=clinical_df,
        claims_df=claims_df,
        sdoh_df=sdoh_df,
        retail_df=retail_df,
        labels_df=labels_df,
        fit=False  # Use existing scalers
    )

    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Get patient IDs
    # Merge to get patient IDs back
    integrated_df = pipeline.integrate_data(clinical_df, claims_df, sdoh_df, retail_df, labels_df)
    patient_ids = integrated_df['patient_id'].values[:len(predictions)]

    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'patient_id': patient_ids,
        'prediction': predictions,
        'probability': probabilities,
        'risk_level': pd.cut(probabilities,
                            bins=[0, 0.3, 0.7, 1.0],
                            labels=['Low', 'Moderate', 'High'])
    })

    # Calculate metrics (if labels available)
    metrics = {}
    if y is not None and not (y == 0).all():
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        metrics = {
            'accuracy': accuracy_score(y, predictions),
            'precision': precision_score(y, predictions, zero_division=0),
            'recall': recall_score(y, predictions, zero_division=0),
            'f1_score': f1_score(y, predictions, zero_division=0),
            'roc_auc': roc_auc_score(y, probabilities) if len(np.unique(y)) > 1 else 0
        }
    else:
        # Dummy metrics for unlabeled data
        metrics = {
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1_score': 0,
            'roc_auc': 0
        }

    # Get feature importance
    feature_importance = model.get_feature_importance(top_n=30)

    return predictions_df, metrics, feature_importance, X, y


def get_risk_distribution(predictions_df: pd.DataFrame) -> dict:
    """Get distribution of risk levels"""
    risk_counts = predictions_df['risk_level'].value_counts()
    total = len(predictions_df)

    return {
        'low': int(risk_counts.get('Low', 0)),
        'moderate': int(risk_counts.get('Moderate', 0)),
        'high': int(risk_counts.get('High', 0)),
        'low_pct': float(risk_counts.get('Low', 0) / total * 100) if total > 0 else 0,
        'moderate_pct': float(risk_counts.get('Moderate', 0) / total * 100) if total > 0 else 0,
        'high_pct': float(risk_counts.get('High', 0) / total * 100) if total > 0 else 0
    }


def analyze_high_risk_patients(predictions_df: pd.DataFrame, clinical_df: pd.DataFrame,
                               top_n: int = 10) -> pd.DataFrame:
    """Get detailed information about high-risk patients"""
    # Get high risk patients
    high_risk = predictions_df.nlargest(top_n, 'probability')

    # Merge with clinical data
    result = high_risk.merge(
        clinical_df[['patient_id', 'serum_creatinine', 'egfr', 'albuminuria_acr',
                    'systolic_bp', 'hba1c', 'bmi']],
        on='patient_id',
        how='left'
    )

    return result


def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:.1f}%"


def format_metric(value: float) -> str:
    """Format metric value"""
    return f"{value:.3f}"


def create_sample_data_templates():
    """Create sample CSV templates for users to download"""
    templates = {
        'clinical': pd.DataFrame({
            'patient_id': ['PAT001', 'PAT002'],
            'serum_creatinine': [0.9, 1.2],
            'egfr': [95, 75],
            'albuminuria_acr': [10, 50],
            'hba1c': [5.5, 6.2],
            'systolic_bp': [120, 135],
            'diastolic_bp': [80, 85],
            'bmi': [24.5, 28.3],
            'heart_rate': [72, 78],
            'has_diabetes': [0, 1],
            'has_hypertension': [0, 1],
            'has_cvd': [0, 0],
            'medication_count': [0, 3],
            'on_ace_inhibitor': [0, 1],
            'on_arb': [0, 0],
            'on_diuretic': [0, 1],
            'icd10_codes': ['', 'E11.9,I10'],
            'medications': ['', 'lisinopril,metformin']
        }),
        'claims': pd.DataFrame({
            'patient_id': ['PAT001', 'PAT002'],
            'er_visits_count': [0, 2],
            'hospital_admissions_count': [0, 1],
            'outpatient_visits_count': [2, 8],
            'primary_care_visits_count': [1, 4],
            'specialist_visits_count': [0, 2],
            'nephrology_referrals_count': [0, 1],
            'dialysis_encounters_count': [0, 0],
            'insurance_coverage_gap_days': [0, 15],
            'claim_denials_count': [0, 2],
            'prescription_fills_count': [2, 12],
            'total_claims_amount': [500, 15000],
            'out_of_pocket_amount': [50, 3000],
            'generic_vs_brand_ratio': [1.0, 0.8]
        }),
        'sdoh': pd.DataFrame({
            'zip_code': ['10001', '10002'],
            'median_household_income': [75000, 45000],
            'poverty_rate': [10.5, 25.3],
            'unemployment_rate': [5.2, 12.1],
            'high_school_grad_rate': [92.5, 78.4],
            'bachelor_degree_rate': [45.2, 18.9],
            'diabetes_prevalence': [8.5, 14.2],
            'obesity_prevalence': [22.1, 35.6],
            'physical_inactivity_rate': [18.5, 28.9],
            'smoking_rate': [12.3, 22.1],
            'food_desert_indicator': [0, 1],
            'low_access_to_grocery_store': [0, 1],
            'snap_participation_rate': [8.5, 28.7],
            'adi_national_percentile': [35, 78],
            'adi_state_percentile': [4, 8]
        }),
        'retail': pd.DataFrame({
            'patient_id': ['PAT001', 'PAT002'],
            'processed_food_purchases': [5, 25],
            'fresh_produce_purchases': [20, 8],
            'high_sodium_food_purchases': [3, 18],
            'sugary_beverage_purchases': [2, 15],
            'whole_grain_purchases': [12, 4],
            'red_meat_purchases': [4, 10],
            'bp_monitor_purchases': [0, 1],
            'glucose_monitor_purchases': [0, 1],
            'health_supplements_purchases': [2, 5],
            'otc_pain_medication_purchases': [1, 4],
            'otc_antacid_purchases': [0, 3],
            'vitamin_mineral_purchases': [3, 6],
            'shopping_frequency_per_month': [8, 12],
            'avg_basket_size': [45.5, 62.3],
            'health_conscious_score': [75, 35]
        })
    }

    return templates
