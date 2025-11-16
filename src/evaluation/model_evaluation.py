"""
Model Evaluation and Interpretability
Includes SHAP analysis, performance visualization, and clinical insights
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from typing import Optional, Dict, List
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluation and interpretability"""

    def __init__(self, model, feature_names: List[str]):
        """
        Initialize evaluator

        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None

    def compute_shap_values(self, X: pd.DataFrame, max_samples: int = 500):
        """
        Compute SHAP values for model interpretability

        Args:
            X: Feature matrix
            max_samples: Maximum samples for SHAP computation (for efficiency)
        """
        print("Computing SHAP values for model interpretability...")

        # Use subset for computational efficiency
        if len(X) > max_samples:
            X_sample = X.sample(n=max_samples, random_state=42)
        else:
            X_sample = X

        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        self.shap_values = self.explainer.shap_values(X_sample)

        print(f"SHAP values computed for {len(X_sample)} samples")

        return self.shap_values

    def plot_shap_summary(self, X: pd.DataFrame, save_path: Optional[str] = None):
        """Plot SHAP summary plot"""
        if self.shap_values is None:
            self.compute_shap_values(X)

        # Use subset for visualization
        X_sample = X.sample(n=min(500, len(X)), random_state=42)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            X_sample,
            plot_type="dot",
            show=False
        )
        plt.title('SHAP Summary Plot - Feature Impact on CKD Prediction', fontsize=14, pad=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP summary plot saved to {save_path}")

        plt.show()

    def plot_shap_bar(self, X: pd.DataFrame, top_n: int = 20, save_path: Optional[str] = None):
        """Plot SHAP bar plot showing mean absolute SHAP values"""
        if self.shap_values is None:
            self.compute_shap_values(X)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            X.sample(n=min(500, len(X)), random_state=42),
            plot_type="bar",
            max_display=top_n,
            show=False
        )
        plt.title(f'Top {top_n} Features by Mean |SHAP Value|', fontsize=14, pad=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP bar plot saved to {save_path}")

        plt.show()

    def plot_shap_waterfall(self, X: pd.DataFrame, sample_idx: int = 0,
                           save_path: Optional[str] = None):
        """Plot SHAP waterfall plot for a single prediction"""
        if self.shap_values is None:
            self.compute_shap_values(X)

        # Create explanation object for the sample
        sample_data = X.iloc[sample_idx:sample_idx+1]
        shap_vals = self.explainer(sample_data)

        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(shap_vals[0], show=False)
        plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}', fontsize=14, pad=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP waterfall plot saved to {save_path}")

        plt.show()

    def plot_shap_dependence(self, X: pd.DataFrame, feature_name: str,
                            interaction_feature: Optional[str] = None,
                            save_path: Optional[str] = None):
        """Plot SHAP dependence plot for a specific feature"""
        if self.shap_values is None:
            self.compute_shap_values(X)

        X_sample = X.sample(n=min(500, len(X)), random_state=42)

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_name,
            self.shap_values,
            X_sample,
            interaction_index=interaction_feature,
            show=False
        )
        plt.title(f'SHAP Dependence Plot - {feature_name}', fontsize=14, pad=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP dependence plot saved to {save_path}")

        plt.show()

    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      save_path: Optional[str] = None):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")

        plt.show()

        return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': roc_auc}

    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                    save_path: Optional[str] = None):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2,
                label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14)
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Precision-Recall curve saved to {save_path}")

        plt.show()

        return {'precision': precision, 'recall': recall, 'thresholds': thresholds, 'auc': pr_auc}

    def analyze_feature_groups(self, feature_importance_df: pd.DataFrame) -> Dict:
        """
        Analyze feature importance by data source groups

        Args:
            feature_importance_df: DataFrame with features and importance scores

        Returns:
            group_analysis: Dictionary with importance by group
        """
        # Define feature groups
        groups = {
            'Clinical': ['serum_creatinine', 'egfr', 'albuminuria_acr', 'hba1c',
                        'systolic_bp', 'diastolic_bp', 'bmi', 'heart_rate',
                        'pulse_pressure', 'map', 'creatinine_egfr_ratio'],
            'Comorbidities': ['has_diabetes', 'has_hypertension', 'has_cvd',
                             'comorbidity_count'],
            'Medications': ['medication_count', 'on_ace_inhibitor', 'on_arb', 'on_diuretic',
                           'medication_adherence_ratio'],
            'Healthcare Utilization': ['er_visits_count', 'hospital_admissions_count',
                                      'outpatient_visits_count', 'primary_care_visits_count',
                                      'specialist_visits_count', 'nephrology_referrals_count',
                                      'dialysis_encounters_count', 'healthcare_utilization_score'],
            'Insurance/Claims': ['insurance_coverage_gap_days', 'claim_denials_count',
                                'prescription_fills_count', 'total_claims_amount'],
            'SDOH': ['median_household_income', 'poverty_rate', 'unemployment_rate',
                    'diabetes_prevalence', 'obesity_prevalence', 'physical_inactivity_rate',
                    'food_desert_indicator', 'adi_national_percentile', 'sdoh_risk_score'],
            'Retail/Dietary': ['processed_food_purchases', 'fresh_produce_purchases',
                              'high_sodium_food_purchases', 'sugary_beverage_purchases',
                              'dietary_health_ratio', 'health_conscious_score'],
            'Health Engagement': ['bp_monitor_purchases', 'glucose_monitor_purchases',
                                 'health_supplements_purchases', 'health_engagement_score']
        }

        # Calculate importance by group
        group_importance = {}
        for group_name, group_features in groups.items():
            group_data = feature_importance_df[
                feature_importance_df['feature'].isin(group_features)
            ]
            if len(group_data) > 0:
                group_importance[group_name] = {
                    'total_importance': group_data['importance'].sum(),
                    'mean_importance': group_data['importance'].mean(),
                    'feature_count': len(group_data),
                    'top_features': group_data.head(3)['feature'].tolist()
                }

        return group_importance

    def plot_group_importance(self, feature_importance_df: pd.DataFrame,
                             save_path: Optional[str] = None):
        """Plot feature importance by data source group"""
        group_analysis = self.analyze_feature_groups(feature_importance_df)

        # Prepare data for plotting
        groups = list(group_analysis.keys())
        total_importance = [group_analysis[g]['total_importance'] for g in groups]

        # Sort by importance
        sorted_data = sorted(zip(groups, total_importance), key=lambda x: x[1], reverse=True)
        groups_sorted, importance_sorted = zip(*sorted_data)

        plt.figure(figsize=(12, 6))
        bars = plt.barh(range(len(groups_sorted)), importance_sorted, color='steelblue')

        # Color code by importance
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        plt.yticks(range(len(groups_sorted)), groups_sorted)
        plt.xlabel('Total Feature Importance', fontsize=12)
        plt.title('Feature Importance by Data Source Category', fontsize=14, pad=20)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Group importance plot saved to {save_path}")

        plt.show()

        return group_analysis

    def generate_clinical_insights(self, feature_importance_df: pd.DataFrame,
                                   top_n: int = 10) -> Dict:
        """
        Generate clinical insights from model analysis

        Args:
            feature_importance_df: DataFrame with feature importance
            top_n: Number of top features to analyze

        Returns:
            insights: Dictionary with clinical insights
        """
        top_features = feature_importance_df.head(top_n)

        insights = {
            'top_predictors': top_features['feature'].tolist(),
            'feature_importance_scores': dict(zip(
                top_features['feature'],
                top_features['importance']
            )),
            'clinical_interpretation': {}
        }

        # Clinical interpretation of top features
        interpretations = {
            'egfr': 'eGFR is a direct measure of kidney function - critical predictor',
            'serum_creatinine': 'Elevated creatinine indicates reduced kidney function',
            'albuminuria_acr': 'Albumin in urine is an early marker of kidney damage',
            'hba1c': 'Diabetes control impacts CKD progression',
            'systolic_bp': 'Hypertension is both cause and consequence of CKD',
            'has_diabetes': 'Diabetes is a leading cause of CKD',
            'has_hypertension': 'Hypertension significantly increases CKD risk',
            'adi_national_percentile': 'Socioeconomic deprivation linked to CKD prevalence',
            'nephrology_referrals_count': 'Specialist care indicates advanced disease',
            'poverty_rate': 'Economic factors influence health outcomes and care access'
        }

        for feature in insights['top_predictors']:
            if feature in interpretations:
                insights['clinical_interpretation'][feature] = interpretations[feature]

        return insights

    def create_comprehensive_report(self, X_test: pd.DataFrame, y_test: np.ndarray,
                                   y_pred: np.ndarray, y_proba: np.ndarray,
                                   save_dir: str = 'results'):
        """
        Create comprehensive evaluation report with all visualizations

        Args:
            X_test: Test features
            y_test: Test labels
            y_pred: Predictions
            y_proba: Prediction probabilities
            save_dir: Directory to save visualizations
        """
        print("Generating comprehensive evaluation report...")

        # 1. Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        # 2. SHAP analysis
        self.compute_shap_values(X_test)
        self.plot_shap_summary(X_test, save_path=f'{save_dir}/figures/shap_summary.png')
        self.plot_shap_bar(X_test, save_path=f'{save_dir}/figures/shap_bar.png')

        # 3. ROC and PR curves
        self.plot_roc_curve(y_test, y_proba, save_path=f'{save_dir}/figures/roc_curve.png')
        self.plot_precision_recall_curve(y_test, y_proba,
                                         save_path=f'{save_dir}/figures/pr_curve.png')

        # 4. Group importance
        group_analysis = self.plot_group_importance(
            feature_importance,
            save_path=f'{save_dir}/figures/group_importance.png'
        )

        # 5. Clinical insights
        insights = self.generate_clinical_insights(feature_importance)

        # Save feature importance
        feature_importance.to_csv(f'{save_dir}/metrics/feature_importance.csv', index=False)

        print(f"\nComprehensive report generated in {save_dir}/")
        print(f"- Visualizations saved to {save_dir}/figures/")
        print(f"- Metrics saved to {save_dir}/metrics/")

        return {
            'feature_importance': feature_importance,
            'group_analysis': group_analysis,
            'clinical_insights': insights
        }
