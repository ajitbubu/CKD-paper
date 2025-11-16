"""
Database utilities for tracking prediction history and uploaded datasets
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import hashlib


class CKDDatabase:
    """SQLite database for CKD prediction tracking"""

    def __init__(self, db_path: str = "app/database/ckd_predictions.db"):
        """Initialize database connection"""
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.create_tables()

    def get_connection(self):
        """Get database connection"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        return self.conn

    def create_tables(self):
        """Create database tables if they don't exist"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Table for dataset uploads
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                dataset_name TEXT NOT NULL,
                clinical_records INTEGER,
                claims_records INTEGER,
                sdoh_records INTEGER,
                retail_records INTEGER,
                total_patients INTEGER,
                dataset_hash TEXT UNIQUE,
                file_paths TEXT
            )
        """)

        # Table for predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_patients INTEGER,
                ckd_positive INTEGER,
                ckd_negative INTEGER,
                ckd_positive_rate REAL,
                model_accuracy REAL,
                model_precision REAL,
                model_recall REAL,
                model_f1 REAL,
                model_auc REAL,
                feature_importance TEXT,
                FOREIGN KEY (dataset_id) REFERENCES datasets (id)
            )
        """)

        # Table for individual patient predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER,
                patient_id TEXT,
                prediction INTEGER,
                probability REAL,
                risk_category TEXT,
                top_risk_factors TEXT,
                FOREIGN KEY (prediction_id) REFERENCES predictions (id)
            )
        """)

        conn.commit()

    def compute_dataset_hash(self, clinical_df: pd.DataFrame, claims_df: pd.DataFrame,
                           sdoh_df: pd.DataFrame, retail_df: pd.DataFrame) -> str:
        """Compute unique hash for dataset combination"""
        # Combine basic stats of all dataframes
        hash_string = f"{len(clinical_df)}_{len(claims_df)}_{len(sdoh_df)}_{len(retail_df)}"
        hash_string += f"_{clinical_df.columns.tolist()}"
        return hashlib.md5(hash_string.encode()).hexdigest()

    def save_dataset(self, dataset_name: str, clinical_df: pd.DataFrame,
                    claims_df: pd.DataFrame, sdoh_df: pd.DataFrame,
                    retail_df: pd.DataFrame, file_paths: Dict[str, str]) -> int:
        """
        Save uploaded dataset information

        Returns:
            dataset_id: ID of saved dataset
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        dataset_hash = self.compute_dataset_hash(clinical_df, claims_df, sdoh_df, retail_df)

        # Count unique patients
        patient_ids = set()
        if 'patient_id' in clinical_df.columns:
            patient_ids.update(clinical_df['patient_id'].unique())

        cursor.execute("""
            INSERT OR IGNORE INTO datasets
            (dataset_name, clinical_records, claims_records, sdoh_records,
             retail_records, total_patients, dataset_hash, file_paths)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dataset_name,
            len(clinical_df),
            len(claims_df),
            len(sdoh_df),
            len(retail_df),
            len(patient_ids),
            dataset_hash,
            json.dumps(file_paths)
        ))

        conn.commit()

        # Get dataset ID
        cursor.execute("SELECT id FROM datasets WHERE dataset_hash = ?", (dataset_hash,))
        result = cursor.fetchone()
        return result[0] if result else None

    def save_prediction(self, dataset_id: int, predictions: pd.DataFrame,
                       metrics: Dict, feature_importance: pd.DataFrame) -> int:
        """
        Save prediction results

        Args:
            dataset_id: ID of dataset used
            predictions: DataFrame with patient_id, prediction, probability
            metrics: Dictionary with model metrics
            feature_importance: DataFrame with feature importance

        Returns:
            prediction_id: ID of saved prediction
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        # Count predictions
        ckd_positive = (predictions['prediction'] == 1).sum()
        ckd_negative = (predictions['prediction'] == 0).sum()
        total = len(predictions)

        # Insert prediction run
        cursor.execute("""
            INSERT INTO predictions
            (dataset_id, total_patients, ckd_positive, ckd_negative,
             ckd_positive_rate, model_accuracy, model_precision, model_recall,
             model_f1, model_auc, feature_importance)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            dataset_id,
            total,
            int(ckd_positive),
            int(ckd_negative),
            float(ckd_positive / total) if total > 0 else 0,
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0),
            metrics.get('roc_auc', 0),
            feature_importance.to_json()
        ))

        prediction_id = cursor.lastrowid

        # Save individual patient predictions
        for _, row in predictions.iterrows():
            # Determine risk category
            prob = row['probability']
            if prob >= 0.7:
                risk_category = 'High Risk'
            elif prob >= 0.4:
                risk_category = 'Moderate Risk'
            else:
                risk_category = 'Low Risk'

            cursor.execute("""
                INSERT INTO patient_predictions
                (prediction_id, patient_id, prediction, probability, risk_category)
                VALUES (?, ?, ?, ?, ?)
            """, (
                prediction_id,
                row.get('patient_id', f'Unknown_{_}'),
                int(row['prediction']),
                float(row['probability']),
                risk_category
            ))

        conn.commit()
        return prediction_id

    def get_all_datasets(self) -> pd.DataFrame:
        """Get all uploaded datasets"""
        conn = self.get_connection()
        query = """
            SELECT id, upload_date, dataset_name, total_patients,
                   clinical_records, claims_records, sdoh_records, retail_records
            FROM datasets
            ORDER BY upload_date DESC
        """
        return pd.read_sql_query(query, conn)

    def get_all_predictions(self) -> pd.DataFrame:
        """Get all prediction runs"""
        conn = self.get_connection()
        query = """
            SELECT p.id, p.prediction_date, d.dataset_name, p.total_patients,
                   p.ckd_positive, p.ckd_negative, p.ckd_positive_rate,
                   p.model_accuracy, p.model_auc
            FROM predictions p
            JOIN datasets d ON p.dataset_id = d.id
            ORDER BY p.prediction_date DESC
        """
        return pd.read_sql_query(query, conn)

    def get_prediction_details(self, prediction_id: int) -> Tuple[Dict, pd.DataFrame]:
        """
        Get detailed information about a specific prediction

        Returns:
            (metrics_dict, patient_predictions_df)
        """
        conn = self.get_connection()

        # Get prediction metrics
        cursor = conn.cursor()
        cursor.execute("""
            SELECT total_patients, ckd_positive, ckd_negative, ckd_positive_rate,
                   model_accuracy, model_precision, model_recall, model_f1, model_auc,
                   feature_importance
            FROM predictions
            WHERE id = ?
        """, (prediction_id,))

        result = cursor.fetchone()
        if not result:
            return {}, pd.DataFrame()

        metrics = {
            'total_patients': result[0],
            'ckd_positive': result[1],
            'ckd_negative': result[2],
            'ckd_positive_rate': result[3],
            'accuracy': result[4],
            'precision': result[5],
            'recall': result[6],
            'f1_score': result[7],
            'roc_auc': result[8],
            'feature_importance': pd.read_json(result[9]) if result[9] else pd.DataFrame()
        }

        # Get patient predictions
        patient_query = """
            SELECT patient_id, prediction, probability, risk_category
            FROM patient_predictions
            WHERE prediction_id = ?
            ORDER BY probability DESC
        """
        patients_df = pd.read_sql_query(patient_query, conn, params=(prediction_id,))

        return metrics, patients_df

    def get_prediction_history_stats(self) -> Dict:
        """Get summary statistics across all predictions"""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_runs = cursor.fetchone()[0]

        # Total patients analyzed
        cursor.execute("SELECT SUM(total_patients) FROM predictions")
        total_patients = cursor.fetchone()[0] or 0

        # Average CKD rate
        cursor.execute("SELECT AVG(ckd_positive_rate) FROM predictions")
        avg_ckd_rate = cursor.fetchone()[0] or 0

        # Average model performance
        cursor.execute("""
            SELECT AVG(model_accuracy), AVG(model_precision),
                   AVG(model_recall), AVG(model_auc)
            FROM predictions
        """)
        perf = cursor.fetchone()

        return {
            'total_prediction_runs': total_runs,
            'total_patients_analyzed': int(total_patients),
            'average_ckd_rate': float(avg_ckd_rate),
            'average_accuracy': float(perf[0] or 0),
            'average_precision': float(perf[1] or 0),
            'average_recall': float(perf[2] or 0),
            'average_auc': float(perf[3] or 0)
        }

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None
