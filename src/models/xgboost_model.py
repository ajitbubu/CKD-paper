"""
XGBoost Classification Model for CKD Detection
Implements binary and multi-class classification with hyperparameter tuning
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from typing import Dict, Tuple, Optional
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns


class CKDXGBoostClassifier:
    """XGBoost classifier for CKD detection from multi-modal data"""

    def __init__(self,
                 model_type: str = 'binary',
                 use_smote: bool = True,
                 random_state: int = 42):
        """
        Initialize CKD classifier

        Args:
            model_type: 'binary' for CKD/No CKD or 'multiclass' for stages 0-5
            use_smote: Whether to use SMOTE for class imbalance
            random_state: Random seed
        """
        self.model_type = model_type
        self.use_smote = use_smote
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.best_params = None
        self.training_history = {}

    def get_default_params(self) -> Dict:
        """Get default XGBoost parameters"""
        if self.model_type == 'binary':
            return {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0,
                'reg_lambda': 1,
                'scale_pos_weight': 1,
                'random_state': self.random_state
            }
        else:  # multiclass
            return {
                'objective': 'multi:softmax',
                'eval_metric': 'mlogloss',
                'num_class': 6,  # 0-5
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state
            }

    def get_param_grid(self) -> Dict:
        """Get parameter grid for hyperparameter tuning"""
        return {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.01, 0.1],
            'reg_lambda': [1, 1.5, 2]
        }

    def train(self,
              X: pd.DataFrame,
              y: pd.Series,
              test_size: float = 0.2,
              tune_hyperparams: bool = False,
              cv_folds: int = 5) -> Dict:
        """
        Train XGBoost model

        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing
            tune_hyperparams: Whether to perform hyperparameter tuning
            cv_folds: Number of CV folds for tuning

        Returns:
            results: Dictionary with training metrics
        """
        print(f"Training {self.model_type} XGBoost classifier...")
        print(f"Dataset shape: {X.shape}")
        print(f"Class distribution:\n{y.value_counts()}")

        self.feature_names = list(X.columns)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=self.random_state
        )

        # Handle class imbalance with SMOTE
        if self.use_smote and self.model_type == 'binary':
            print("Applying SMOTE for class imbalance...")
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE - Class distribution:\n{pd.Series(y_train).value_counts()}")

        # Initialize model with default or tuned parameters
        if tune_hyperparams:
            print("Performing hyperparameter tuning...")
            self.model, self.best_params = self._tune_hyperparameters(
                X_train, y_train, cv_folds
            )
        else:
            params = self.get_default_params()
            self.best_params = params
            self.model = xgb.XGBClassifier(**params)
            self.model.fit(X_train, y_train)

        # Evaluate on test set
        results = self._evaluate(X_train, y_train, X_test, y_test)

        # Store training history
        self.training_history = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'features': self.feature_names,
            'results': results,
            'best_params': self.best_params
        }

        return results

    def _tune_hyperparameters(self,
                             X_train: pd.DataFrame,
                             y_train: pd.Series,
                             cv_folds: int = 5) -> Tuple[xgb.XGBClassifier, Dict]:
        """
        Hyperparameter tuning using GridSearchCV

        Args:
            X_train: Training features
            y_train: Training labels
            cv_folds: Number of CV folds

        Returns:
            best_model: Trained model with best parameters
            best_params: Best parameters found
        """
        base_params = self.get_default_params()
        param_grid = self.get_param_grid()

        # Create base model
        model = xgb.XGBClassifier(**{k: v for k, v in base_params.items()
                                     if k not in param_grid})

        # Setup GridSearch
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
            scoring='roc_auc' if self.model_type == 'binary' else 'accuracy',
            n_jobs=-1,
            verbose=1
        )

        # Fit
        grid_search.fit(X_train, y_train)

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_search.best_params_

    def _evaluate(self,
                  X_train: pd.DataFrame,
                  y_train: pd.Series,
                  X_test: pd.DataFrame,
                  y_test: pd.Series) -> Dict:
        """
        Evaluate model performance

        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data

        Returns:
            results: Dictionary with evaluation metrics
        """
        print("\nEvaluating model performance...")

        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Probabilities (for binary classification)
        if self.model_type == 'binary':
            y_train_proba = self.model.predict_proba(X_train)[:, 1]
            y_test_proba = self.model.predict_proba(X_test)[:, 1]
        else:
            y_train_proba = None
            y_test_proba = None

        # Calculate metrics
        results = {
            'train': self._calculate_metrics(y_train, y_train_pred, y_train_proba),
            'test': self._calculate_metrics(y_test, y_test_pred, y_test_proba)
        }

        # Print results
        print("\n=== Training Set Performance ===")
        self._print_metrics(results['train'])

        print("\n=== Test Set Performance ===")
        self._print_metrics(results['test'])

        return results

    def _calculate_metrics(self,
                          y_true: pd.Series,
                          y_pred: np.ndarray,
                          y_proba: Optional[np.ndarray] = None) -> Dict:
        """Calculate classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, zero_division=0)
        }

        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)

        return metrics

    def _print_metrics(self, metrics: Dict):
        """Print metrics in formatted way"""
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        if 'roc_auc' in metrics:
            print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance scores

        Args:
            top_n: Number of top features to return

        Returns:
            importance_df: DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        importance = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        """Plot feature importance"""
        importance_df = self.get_feature_importance(top_n)

        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importance - XGBoost CKD Classifier')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")

        plt.show()

    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray,
                             save_path: Optional[str] = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
        plt.title('Confusion Matrix - XGBoost CKD Classifier')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.show()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict_proba(X)

    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'best_params': self.best_params,
            'training_history': self.training_history
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.best_params = model_data['best_params']
        self.training_history = model_data['training_history']

        print(f"Model loaded from {filepath}")

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation

        Args:
            X: Feature matrix
            y: Target variable
            cv_folds: Number of CV folds

        Returns:
            cv_results: Cross-validation scores
        """
        print(f"Performing {cv_folds}-fold cross-validation...")

        if self.model is None:
            self.model = xgb.XGBClassifier(**self.get_default_params())

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        if self.model_type == 'binary':
            scoring.append('roc_auc')

        cv_results = {}
        for score in scoring:
            scores = cross_val_score(self.model, X, y, cv=cv, scoring=score)
            cv_results[score] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores.tolist()
            }
            print(f"{score}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return cv_results
