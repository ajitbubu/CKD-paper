"""
Data Integration and Preprocessing Pipeline
Combines multi-source data and prepares for ML modeling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class DataIntegrationPipeline:
    """Integrate and preprocess multi-modal patient data"""

    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.feature_names = []

    def integrate_data(self,
                      clinical_df: pd.DataFrame,
                      claims_df: pd.DataFrame,
                      sdoh_df: pd.DataFrame,
                      retail_df: pd.DataFrame,
                      labels_df: pd.DataFrame) -> pd.DataFrame:
        """
        Integrate all data sources into a single dataframe

        Args:
            clinical_df: EHR clinical data
            claims_df: Healthcare claims data
            sdoh_df: Social determinants of health data (ZIP-level)
            retail_df: Retail purchase pattern data
            labels_df: Target labels

        Returns:
            integrated_df: Combined dataframe with all features
        """
        print("Integrating data sources...")

        # Start with clinical data
        integrated_df = clinical_df.copy()

        # Merge claims data
        integrated_df = integrated_df.merge(
            claims_df,
            on='patient_id',
            how='left',
            suffixes=('', '_claims')
        )

        # Merge labels (includes zip_code)
        integrated_df = integrated_df.merge(
            labels_df,
            on='patient_id',
            how='left'
        )

        # Merge SDOH data (by ZIP code)
        integrated_df = integrated_df.merge(
            sdoh_df,
            on='zip_code',
            how='left',
            suffixes=('', '_sdoh')
        )

        # Merge retail data
        integrated_df = integrated_df.merge(
            retail_df,
            on='patient_id',
            how='left',
            suffixes=('', '_retail')
        )

        print(f"Integrated data shape: {integrated_df.shape}")
        print(f"Missing values per column:\n{integrated_df.isnull().sum()[integrated_df.isnull().sum() > 0]}")

        return integrated_df

    def preprocess_features(self,
                           df: pd.DataFrame,
                           fit: bool = True,
                           feature_groups: Dict[str, List[str]] = None) -> pd.DataFrame:
        """
        Preprocess features with appropriate scaling and imputation

        Args:
            df: Integrated dataframe
            fit: If True, fit scalers/imputers; if False, use existing
            feature_groups: Dictionary of feature groups for different preprocessing

        Returns:
            preprocessed_df: Dataframe with preprocessed features
        """
        print("Preprocessing features...")

        df_processed = df.copy()

        if feature_groups is None:
            feature_groups = self._get_default_feature_groups()

        # Process each feature group
        for group_name, features in feature_groups.items():
            available_features = [f for f in features if f in df_processed.columns]

            if not available_features:
                continue

            print(f"Processing {group_name}: {len(available_features)} features")

            # Handle missing values
            if fit:
                self.imputers[group_name] = SimpleImputer(strategy='median')
                df_processed[available_features] = self.imputers[group_name].fit_transform(
                    df_processed[available_features]
                )
            else:
                if group_name in self.imputers:
                    df_processed[available_features] = self.imputers[group_name].transform(
                        df_processed[available_features]
                    )

            # Scale features (except binary/categorical)
            if group_name not in ['binary_features', 'categorical_features']:
                if fit:
                    # Use RobustScaler for clinical features (outlier-resistant)
                    if group_name == 'clinical_features':
                        self.scalers[group_name] = RobustScaler()
                    else:
                        self.scalers[group_name] = StandardScaler()

                    df_processed[available_features] = self.scalers[group_name].fit_transform(
                        df_processed[available_features]
                    )
                else:
                    if group_name in self.scalers:
                        df_processed[available_features] = self.scalers[group_name].transform(
                            df_processed[available_features]
                        )

        return df_processed

    def _get_default_feature_groups(self) -> Dict[str, List[str]]:
        """Define default feature groups for preprocessing"""
        return {
            'clinical_features': [
                'serum_creatinine', 'egfr', 'albuminuria_acr', 'hba1c',
                'systolic_bp', 'diastolic_bp', 'bmi', 'heart_rate'
            ],
            'binary_features': [
                'has_diabetes', 'has_hypertension', 'has_cvd',
                'on_ace_inhibitor', 'on_arb', 'on_diuretic',
                'food_desert_indicator', 'low_access_to_grocery_store'
            ],
            'claims_features': [
                'er_visits_count', 'hospital_admissions_count', 'outpatient_visits_count',
                'primary_care_visits_count', 'specialist_visits_count',
                'nephrology_referrals_count', 'dialysis_encounters_count',
                'insurance_coverage_gap_days', 'claim_denials_count',
                'prescription_fills_count', 'total_claims_amount', 'out_of_pocket_amount'
            ],
            'sdoh_features': [
                'median_household_income', 'poverty_rate', 'unemployment_rate',
                'high_school_grad_rate', 'bachelor_degree_rate',
                'diabetes_prevalence', 'obesity_prevalence',
                'physical_inactivity_rate', 'smoking_rate',
                'snap_participation_rate', 'adi_national_percentile', 'adi_state_percentile'
            ],
            'retail_features': [
                'processed_food_purchases', 'fresh_produce_purchases',
                'high_sodium_food_purchases', 'sugary_beverage_purchases',
                'whole_grain_purchases', 'red_meat_purchases',
                'bp_monitor_purchases', 'glucose_monitor_purchases',
                'health_supplements_purchases',
                'otc_pain_medication_purchases', 'otc_antacid_purchases',
                'vitamin_mineral_purchases', 'shopping_frequency_per_month',
                'avg_basket_size', 'health_conscious_score'
            ]
        }

    def create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw data

        Args:
            df: Integrated dataframe

        Returns:
            df_engineered: Dataframe with additional engineered features
        """
        print("Creating engineered features...")

        df_eng = df.copy()

        # Clinical feature engineering
        if 'systolic_bp' in df_eng.columns and 'diastolic_bp' in df_eng.columns:
            df_eng['pulse_pressure'] = df_eng['systolic_bp'] - df_eng['diastolic_bp']
            df_eng['map'] = (df_eng['systolic_bp'] + 2 * df_eng['diastolic_bp']) / 3  # Mean arterial pressure

        if 'serum_creatinine' in df_eng.columns and 'egfr' in df_eng.columns:
            df_eng['creatinine_egfr_ratio'] = df_eng['serum_creatinine'] / (df_eng['egfr'] + 1)

        # Comorbidity score
        comorbidity_cols = ['has_diabetes', 'has_hypertension', 'has_cvd']
        available_comorbidity = [c for c in comorbidity_cols if c in df_eng.columns]
        if available_comorbidity:
            df_eng['comorbidity_count'] = df_eng[available_comorbidity].sum(axis=1)

        # Healthcare utilization score
        utilization_cols = ['er_visits_count', 'hospital_admissions_count',
                           'specialist_visits_count', 'nephrology_referrals_count']
        available_utilization = [c for c in utilization_cols if c in df_eng.columns]
        if available_utilization:
            df_eng['healthcare_utilization_score'] = df_eng[available_utilization].sum(axis=1)

        # Medication adherence proxy
        if 'prescription_fills_count' in df_eng.columns and 'medication_count' in df_eng.columns:
            df_eng['medication_adherence_ratio'] = df_eng['prescription_fills_count'] / (
                df_eng['medication_count'] + 1
            )

        # SDOH composite risk score
        sdoh_risk_cols = ['poverty_rate', 'unemployment_rate', 'physical_inactivity_rate',
                         'adi_national_percentile']
        available_sdoh_risk = [c for c in sdoh_risk_cols if c in df_eng.columns]
        if len(available_sdoh_risk) >= 2:
            # Normalize to 0-1 and average
            df_eng['sdoh_risk_score'] = df_eng[available_sdoh_risk].apply(
                lambda x: (x - x.min()) / (x.max() - x.min() + 1e-6)
            ).mean(axis=1) * 100

        # Dietary health score from retail data
        healthy_food_cols = ['fresh_produce_purchases', 'whole_grain_purchases']
        unhealthy_food_cols = ['processed_food_purchases', 'high_sodium_food_purchases',
                               'sugary_beverage_purchases']

        available_healthy = [c for c in healthy_food_cols if c in df_eng.columns]
        available_unhealthy = [c for c in unhealthy_food_cols if c in df_eng.columns]

        if available_healthy and available_unhealthy:
            healthy_sum = df_eng[available_healthy].sum(axis=1)
            unhealthy_sum = df_eng[available_unhealthy].sum(axis=1)
            df_eng['dietary_health_ratio'] = healthy_sum / (unhealthy_sum + 1)

        # Health engagement score (monitors + health products)
        engagement_cols = ['bp_monitor_purchases', 'glucose_monitor_purchases',
                          'health_supplements_purchases']
        available_engagement = [c for c in engagement_cols if c in df_eng.columns]
        if available_engagement:
            df_eng['health_engagement_score'] = df_eng[available_engagement].sum(axis=1)

        print(f"Created {len(df_eng.columns) - len(df.columns)} new engineered features")

        return df_eng

    def prepare_for_modeling(self,
                            clinical_df: pd.DataFrame,
                            claims_df: pd.DataFrame,
                            sdoh_df: pd.DataFrame,
                            retail_df: pd.DataFrame,
                            labels_df: pd.DataFrame,
                            fit: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Complete pipeline: integrate, engineer, and preprocess data

        Args:
            clinical_df, claims_df, sdoh_df, retail_df, labels_df: Input dataframes
            fit: Whether to fit preprocessors

        Returns:
            X: Feature matrix
            y: Target variable
        """
        # Integrate data
        integrated_df = self.integrate_data(
            clinical_df, claims_df, sdoh_df, retail_df, labels_df
        )

        # Create engineered features
        engineered_df = self.create_engineered_features(integrated_df)

        # Separate features and target
        # Exclude non-feature columns
        exclude_cols = ['patient_id', 'record_date', 'time_period', 'zip_code',
                       'geolocation_area', 'icd10_codes', 'medications',
                       'ckd_stage', 'has_ckd']

        feature_cols = [col for col in engineered_df.columns if col not in exclude_cols]
        self.feature_names = feature_cols

        X = engineered_df[feature_cols].copy()
        y = engineered_df['has_ckd'].copy() if 'has_ckd' in engineered_df.columns else None

        # Preprocess features
        X_processed = self.preprocess_features(X, fit=fit)

        print(f"\nFinal feature matrix shape: {X_processed.shape}")
        print(f"Number of features: {len(feature_cols)}")

        if y is not None:
            print(f"Target distribution:\n{y.value_counts(normalize=True)}")

        return X_processed, y

    def get_feature_importance_mapping(self) -> Dict[str, str]:
        """Get mapping of features to their source"""
        feature_groups = self._get_default_feature_groups()

        mapping = {}
        for group, features in feature_groups.items():
            for feature in features:
                mapping[feature] = group

        # Add engineered features
        engineered = [
            'pulse_pressure', 'map', 'creatinine_egfr_ratio', 'comorbidity_count',
            'healthcare_utilization_score', 'medication_adherence_ratio',
            'sdoh_risk_score', 'dietary_health_ratio', 'health_engagement_score'
        ]
        for feat in engineered:
            mapping[feat] = 'engineered_features'

        return mapping
