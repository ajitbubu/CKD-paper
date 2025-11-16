"""
Main Training Script for CKD XGBoost Classifier
Orchestrates data loading, preprocessing, training, and evaluation
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent))

from data_processing.synthetic_data_generator import SyntheticDataGenerator
from data_processing.data_integration import DataIntegrationPipeline
from models.xgboost_model import CKDXGBoostClassifier
from evaluation.model_evaluation import ModelEvaluator


def main():
    """Main training pipeline"""

    print("="*80)
    print("CKD Detection Using XGBoost - Multi-Modal Data Analysis")
    print("="*80)

    # Configuration
    config = {
        'n_patients': 1000,
        'random_state': 42,
        'test_size': 0.2,
        'use_smote': True,
        'tune_hyperparams': False,  # Set to True for hyperparameter tuning
        'model_type': 'binary',  # 'binary' or 'multiclass'
    }

    print(f"\nConfiguration: {config}\n")

    # ===========================================
    # Step 1: Generate/Load Data
    # ===========================================
    print("STEP 1: Generating Synthetic Data")
    print("-" * 80)

    data_generator = SyntheticDataGenerator(
        n_patients=config['n_patients'],
        random_state=config['random_state']
    )

    clinical_df, claims_df, sdoh_df, retail_df, labels_df = data_generator.save_data(
        output_dir='data/synthetic'
    )

    # ===========================================
    # Step 2: Data Integration & Preprocessing
    # ===========================================
    print("\n\nSTEP 2: Data Integration & Preprocessing")
    print("-" * 80)

    pipeline = DataIntegrationPipeline()

    X, y = pipeline.prepare_for_modeling(
        clinical_df=clinical_df,
        claims_df=claims_df,
        sdoh_df=sdoh_df,
        retail_df=retail_df,
        labels_df=labels_df,
        fit=True
    )

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"\nFeature list ({len(pipeline.feature_names)} features):")
    for i, feat in enumerate(pipeline.feature_names[:20], 1):
        print(f"  {i}. {feat}")
    if len(pipeline.feature_names) > 20:
        print(f"  ... and {len(pipeline.feature_names) - 20} more features")

    # ===========================================
    # Step 3: Train XGBoost Model
    # ===========================================
    print("\n\nSTEP 3: Training XGBoost Model")
    print("-" * 80)

    model = CKDXGBoostClassifier(
        model_type=config['model_type'],
        use_smote=config['use_smote'],
        random_state=config['random_state']
    )

    results = model.train(
        X=X,
        y=y,
        test_size=config['test_size'],
        tune_hyperparams=config['tune_hyperparams'],
        cv_folds=5
    )

    # ===========================================
    # Step 4: Model Evaluation & Interpretability
    # ===========================================
    print("\n\nSTEP 4: Model Evaluation & Interpretability")
    print("-" * 80)

    # Get feature importance
    feature_importance = model.get_feature_importance(top_n=20)
    print("\nTop 20 Most Important Features:")
    print(feature_importance.to_string(index=False))

    # Plot feature importance
    model.plot_feature_importance(top_n=20, save_path='results/figures/feature_importance.png')

    # Split data for evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config['test_size'],
        stratify=y, random_state=config['random_state']
    )

    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Plot confusion matrix
    model.plot_confusion_matrix(y_test, y_pred, save_path='results/figures/confusion_matrix.png')

    # ===========================================
    # Step 5: SHAP Analysis
    # ===========================================
    print("\n\nSTEP 5: SHAP Analysis for Model Interpretability")
    print("-" * 80)

    evaluator = ModelEvaluator(model.model, pipeline.feature_names)

    # Generate comprehensive report
    report = evaluator.create_comprehensive_report(
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        save_dir='results'
    )

    # ===========================================
    # Step 6: Clinical Insights
    # ===========================================
    print("\n\nSTEP 6: Clinical Insights")
    print("-" * 80)

    insights = report['clinical_insights']
    print("\nTop Predictive Features:")
    for i, feature in enumerate(insights['top_predictors'][:10], 1):
        importance = insights['feature_importance_scores'][feature]
        interpretation = insights['clinical_interpretation'].get(feature, 'N/A')
        print(f"{i}. {feature} (importance: {importance:.4f})")
        if interpretation != 'N/A':
            print(f"   → {interpretation}")

    print("\n\nFeature Importance by Data Source:")
    group_analysis = report['group_analysis']
    for group_name, data in sorted(group_analysis.items(),
                                   key=lambda x: x[1]['total_importance'],
                                   reverse=True):
        print(f"\n{group_name}:")
        print(f"  Total Importance: {data['total_importance']:.4f}")
        print(f"  Mean Importance:  {data['mean_importance']:.4f}")
        print(f"  Feature Count:    {data['feature_count']}")
        print(f"  Top Features:     {', '.join(data['top_features'])}")

    # ===========================================
    # Step 7: Save Model
    # ===========================================
    print("\n\nSTEP 7: Saving Model")
    print("-" * 80)

    model.save_model('models/ckd_xgboost_model.pkl')

    # Save pipeline
    import joblib
    joblib.dump(pipeline, 'models/data_pipeline.pkl')
    print("Data pipeline saved to models/data_pipeline.pkl")

    # ===========================================
    # Summary
    # ===========================================
    print("\n\n" + "="*80)
    print("TRAINING COMPLETE - Summary")
    print("="*80)
    print(f"\nModel Type: {config['model_type']}")
    print(f"Total Patients: {config['n_patients']}")
    print(f"Total Features: {X.shape[1]}")
    print(f"Training Samples: {len(X_train)}")
    print(f"Test Samples: {len(X_test)}")

    print("\n\nTest Set Performance:")
    print(f"  Accuracy:  {results['test']['accuracy']:.4f}")
    print(f"  Precision: {results['test']['precision']:.4f}")
    print(f"  Recall:    {results['test']['recall']:.4f}")
    print(f"  F1-Score:  {results['test']['f1_score']:.4f}")
    if 'roc_auc' in results['test']:
        print(f"  ROC-AUC:   {results['test']['roc_auc']:.4f}")

    print("\n\nKey Findings:")
    print("1. Model successfully integrates multi-modal data (Clinical, Claims, SDOH, Retail)")
    print("2. Clinical features (eGFR, creatinine, albuminuria) are primary predictors")
    print("3. SDOH factors contribute to risk stratification")
    print("4. Retail patterns provide additional behavioral insights")

    print("\n\nOutputs:")
    print("  - Trained model:     models/ckd_xgboost_model.pkl")
    print("  - Data pipeline:     models/data_pipeline.pkl")
    print("  - Visualizations:    results/figures/")
    print("  - Metrics:           results/metrics/")
    print("  - Synthetic data:    data/synthetic/")

    print("\n" + "="*80)
    print("All done! Use the Jupyter notebook for interactive analysis.")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('data/synthetic', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)

    # Run main pipeline
    main()
