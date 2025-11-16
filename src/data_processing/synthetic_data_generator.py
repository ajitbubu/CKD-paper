"""
Synthetic Data Generator for CKD Prediction Model
Generates realistic multi-modal patient data for demonstration
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Dict
import random


class SyntheticDataGenerator:
    """Generate synthetic patient data across all data sources"""

    def __init__(self, n_patients: int = 1000, random_state: int = 42):
        self.n_patients = n_patients
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)

        # Reference data
        self.zip_codes = self._generate_zip_codes()
        self.icd10_codes = {
            'ckd': ['N18.1', 'N18.2', 'N18.3', 'N18.4', 'N18.5', 'N18.6', 'N18.9'],
            'diabetes': ['E11.9', 'E11.65', 'E11.22'],
            'hypertension': ['I10', 'I15.0', 'I15.1'],
            'cvd': ['I25.10', 'I50.9', 'I48.91']
        }

    def _generate_zip_codes(self) -> list:
        """Generate realistic ZIP codes"""
        return [f"{random.randint(10000, 99999):05d}" for _ in range(100)]

    def generate_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate all data sources
        Returns: clinical_df, claims_df, sdoh_df, retail_df, labels_df
        """
        print("Generating synthetic patient data...")

        # Generate patient IDs
        patient_ids = [f"PAT{i:06d}" for i in range(self.n_patients)]

        # Generate labels (CKD stages)
        labels_df = self._generate_labels(patient_ids)

        # Generate each data source
        clinical_df = self._generate_clinical_data(patient_ids, labels_df)
        claims_df = self._generate_claims_data(patient_ids, labels_df)
        sdoh_df = self._generate_sdoh_data()
        retail_df = self._generate_retail_data(patient_ids, labels_df, clinical_df)

        print(f"Generated data for {self.n_patients} patients")
        return clinical_df, claims_df, sdoh_df, retail_df, labels_df

    def _generate_labels(self, patient_ids: list) -> pd.DataFrame:
        """Generate target labels with realistic CKD distribution"""
        # CKD prevalence: ~15% in US adult population
        # Stage distribution among CKD patients:
        # Stage 1-2: 40%, Stage 3: 45%, Stage 4: 10%, Stage 5: 5%

        labels = []
        for pid in patient_ids:
            has_ckd = np.random.random() < 0.15

            if has_ckd:
                stage_prob = np.random.random()
                if stage_prob < 0.40:
                    ckd_stage = np.random.choice([1, 2])
                elif stage_prob < 0.85:
                    ckd_stage = 3
                elif stage_prob < 0.95:
                    ckd_stage = 4
                else:
                    ckd_stage = 5
            else:
                ckd_stage = 0

            labels.append({
                'patient_id': pid,
                'ckd_stage': ckd_stage,
                'has_ckd': has_ckd,
                'zip_code': np.random.choice(self.zip_codes)
            })

        return pd.DataFrame(labels)

    def _generate_clinical_data(self, patient_ids: list, labels_df: pd.DataFrame) -> pd.DataFrame:
        """Generate EHR clinical data with realistic correlations"""
        clinical_records = []

        for _, patient in labels_df.iterrows():
            pid = patient['patient_id']
            ckd_stage = patient['ckd_stage']

            # Generate correlated clinical values based on CKD stage
            if ckd_stage == 0:  # No CKD
                serum_creatinine = np.random.normal(0.9, 0.15)
                egfr = np.random.normal(95, 10)
                albuminuria_acr = np.random.gamma(2, 5)
            elif ckd_stage in [1, 2]:  # Early CKD
                serum_creatinine = np.random.normal(1.1, 0.2)
                egfr = np.random.normal(75, 10)
                albuminuria_acr = np.random.gamma(4, 15)
            elif ckd_stage == 3:  # Moderate CKD
                serum_creatinine = np.random.normal(1.6, 0.3)
                egfr = np.random.normal(45, 8)
                albuminuria_acr = np.random.gamma(6, 25)
            elif ckd_stage == 4:  # Severe CKD
                serum_creatinine = np.random.normal(2.5, 0.4)
                egfr = np.random.normal(22, 5)
                albuminuria_acr = np.random.gamma(8, 40)
            else:  # Stage 5 - Kidney failure
                serum_creatinine = np.random.normal(4.0, 0.6)
                egfr = np.random.normal(12, 3)
                albuminuria_acr = np.random.gamma(10, 50)

            # Comorbidities (more likely with CKD)
            has_diabetes = np.random.random() < (0.15 + ckd_stage * 0.1)
            has_hypertension = np.random.random() < (0.25 + ckd_stage * 0.12)
            has_cvd = np.random.random() < (0.10 + ckd_stage * 0.08)

            # Vital signs
            systolic_bp = int(np.random.normal(120 + ckd_stage * 5, 15))
            diastolic_bp = int(np.random.normal(80 + ckd_stage * 2, 10))
            bmi = np.random.normal(28 + ckd_stage * 0.5, 5)

            # HbA1c (higher if diabetic)
            if has_diabetes:
                hba1c = np.random.normal(7.2, 1.0)
            else:
                hba1c = np.random.normal(5.5, 0.4)

            # ICD-10 codes
            icd_codes = []
            if ckd_stage > 0:
                icd_codes.append(np.random.choice(self.icd10_codes['ckd']))
            if has_diabetes:
                icd_codes.append(np.random.choice(self.icd10_codes['diabetes']))
            if has_hypertension:
                icd_codes.append(np.random.choice(self.icd10_codes['hypertension']))
            if has_cvd:
                icd_codes.append(np.random.choice(self.icd10_codes['cvd']))

            # Medications
            medications = []
            if has_hypertension or ckd_stage >= 2:
                if np.random.random() < 0.6:
                    medications.append(np.random.choice(['lisinopril', 'enalapril']))  # ACE inhibitor
                if np.random.random() < 0.4:
                    medications.append(np.random.choice(['losartan', 'valsartan']))  # ARB
            if ckd_stage >= 3:
                medications.append('furosemide')  # Diuretic
            if has_diabetes:
                medications.append(np.random.choice(['metformin', 'glipizide', 'insulin']))

            clinical_records.append({
                'patient_id': pid,
                'record_date': datetime.now() - timedelta(days=np.random.randint(0, 365)),
                'serum_creatinine': max(0.5, serum_creatinine),
                'egfr': max(5, egfr),
                'albuminuria_acr': max(0, albuminuria_acr),
                'hba1c': max(4.0, min(14.0, hba1c)),
                'systolic_bp': max(90, min(200, systolic_bp)),
                'diastolic_bp': max(60, min(120, diastolic_bp)),
                'bmi': max(15, min(50, bmi)),
                'heart_rate': int(np.random.normal(75, 10)),
                'icd10_codes': ','.join(icd_codes),
                'medications': ','.join(medications),
                'has_diabetes': has_diabetes,
                'has_hypertension': has_hypertension,
                'has_cvd': has_cvd,
                'medication_count': len(medications),
                'on_ace_inhibitor': int(any('pril' in m for m in medications)),
                'on_arb': int(any('sartan' in m for m in medications)),
                'on_diuretic': int('furosemide' in medications)
            })

        return pd.DataFrame(clinical_records)

    def _generate_claims_data(self, patient_ids: list, labels_df: pd.DataFrame) -> pd.DataFrame:
        """Generate healthcare claims data"""
        claims_records = []

        for _, patient in labels_df.iterrows():
            pid = patient['patient_id']
            ckd_stage = patient['ckd_stage']

            # Healthcare utilization increases with CKD severity
            utilization_multiplier = 1 + (ckd_stage * 0.5)

            claims_records.append({
                'patient_id': pid,
                'time_period': '2023-Q4',
                'er_visits_count': np.random.poisson(0.3 * utilization_multiplier),
                'hospital_admissions_count': np.random.poisson(0.1 * utilization_multiplier),
                'outpatient_visits_count': np.random.poisson(4 * utilization_multiplier),
                'primary_care_visits_count': np.random.poisson(2 * utilization_multiplier),
                'specialist_visits_count': np.random.poisson(1 * utilization_multiplier),
                'nephrology_referrals_count': np.random.poisson(0.2 * ckd_stage),
                'dialysis_encounters_count': int(ckd_stage == 5) * np.random.poisson(12),
                'insurance_coverage_gap_days': np.random.poisson(5),
                'claim_denials_count': np.random.poisson(1),
                'prescription_fills_count': np.random.poisson(3 * utilization_multiplier),
                'total_claims_amount': np.random.gamma(3, 2000 * utilization_multiplier),
                'out_of_pocket_amount': np.random.gamma(2, 500 * utilization_multiplier),
                'generic_vs_brand_ratio': np.random.beta(5, 2)
            })

        return pd.DataFrame(claims_records)

    def _generate_sdoh_data(self) -> pd.DataFrame:
        """Generate SDOH data at ZIP code level"""
        sdoh_records = []

        for zip_code in self.zip_codes:
            # Generate correlated SDOH variables
            base_deprivation = np.random.random()

            sdoh_records.append({
                'zip_code': zip_code,
                # Census data
                'median_household_income': np.random.gamma(3, 20000) * (2 - base_deprivation),
                'poverty_rate': np.random.beta(2, 5) * 100 * (0.5 + base_deprivation),
                'unemployment_rate': np.random.beta(2, 10) * 100 * (0.5 + base_deprivation),
                'high_school_grad_rate': np.random.beta(8, 2) * 100 * (1.5 - base_deprivation * 0.5),
                'bachelor_degree_rate': np.random.beta(4, 6) * 100 * (1.5 - base_deprivation * 0.5),
                # CDC PLACES
                'diabetes_prevalence': np.random.beta(3, 15) * 100 * (0.8 + base_deprivation * 0.4),
                'obesity_prevalence': np.random.beta(5, 10) * 100 * (0.8 + base_deprivation * 0.4),
                'physical_inactivity_rate': np.random.beta(4, 8) * 100 * (0.8 + base_deprivation * 0.4),
                'smoking_rate': np.random.beta(2, 8) * 100 * (0.8 + base_deprivation * 0.4),
                # USDA Food Access
                'food_desert_indicator': int(np.random.random() < (0.2 + base_deprivation * 0.3)),
                'low_access_to_grocery_store': int(np.random.random() < (0.15 + base_deprivation * 0.25)),
                'snap_participation_rate': np.random.beta(2, 5) * 100 * (0.5 + base_deprivation * 0.5),
                # ADI
                'adi_national_percentile': int(base_deprivation * 100),
                'adi_state_percentile': int(base_deprivation * 10) + 1
            })

        return pd.DataFrame(sdoh_records)

    def _generate_retail_data(self, patient_ids: list, labels_df: pd.DataFrame,
                              clinical_df: pd.DataFrame) -> pd.DataFrame:
        """Generate retail purchase pattern data"""
        retail_records = []

        clinical_dict = clinical_df.set_index('patient_id').to_dict('index')

        for _, patient in labels_df.iterrows():
            pid = patient['patient_id']
            ckd_stage = patient['ckd_stage']
            zip_code = patient['zip_code']

            # Get clinical info
            has_diabetes = clinical_dict[pid]['has_diabetes']
            has_hypertension = clinical_dict[pid]['has_hypertension']

            # Health consciousness increases with disease severity
            health_conscious = (ckd_stage > 0) or has_diabetes or has_hypertension

            retail_records.append({
                'patient_id': pid,
                'geolocation_area': zip_code,
                'time_period': '2023-Q4',
                # Food purchases (unhealthy decreases if health conscious)
                'processed_food_purchases': np.random.poisson(20 * (0.5 if health_conscious else 1.2)),
                'fresh_produce_purchases': np.random.poisson(15 * (1.5 if health_conscious else 0.8)),
                'high_sodium_food_purchases': np.random.poisson(12 * (0.4 if health_conscious else 1.3)),
                'sugary_beverage_purchases': np.random.poisson(10 * (0.3 if health_conscious else 1.2)),
                'whole_grain_purchases': np.random.poisson(8 * (1.6 if health_conscious else 0.7)),
                'red_meat_purchases': np.random.poisson(6 * (0.6 if health_conscious else 1.1)),
                # Health products (increases with CKD)
                'bp_monitor_purchases': int(has_hypertension or ckd_stage >= 2) * np.random.poisson(0.5),
                'glucose_monitor_purchases': int(has_diabetes) * np.random.poisson(1),
                'health_supplements_purchases': np.random.poisson(2 * (1 + ckd_stage * 0.3)),
                # OTC medications
                'otc_pain_medication_purchases': np.random.poisson(3),
                'otc_antacid_purchases': np.random.poisson(2 * (1 + ckd_stage * 0.2)),
                'vitamin_mineral_purchases': np.random.poisson(4 * (1 + int(health_conscious))),
                # Shopping patterns
                'shopping_frequency_per_month': np.random.poisson(8),
                'avg_basket_size': np.random.gamma(4, 15),
                'health_conscious_score': np.random.beta(5 if health_conscious else 2,
                                                         2 if health_conscious else 5) * 100
            })

        return pd.DataFrame(retail_records)

    def save_data(self, output_dir: str = 'data/synthetic'):
        """Generate and save all synthetic data"""
        clinical_df, claims_df, sdoh_df, retail_df, labels_df = self.generate_all_data()

        # Save to CSV
        clinical_df.to_csv(f'{output_dir}/clinical_data.csv', index=False)
        claims_df.to_csv(f'{output_dir}/claims_data.csv', index=False)
        sdoh_df.to_csv(f'{output_dir}/sdoh_data.csv', index=False)
        retail_df.to_csv(f'{output_dir}/retail_data.csv', index=False)
        labels_df.to_csv(f'{output_dir}/labels.csv', index=False)

        print(f"\nData saved to {output_dir}/")
        print(f"Clinical data: {clinical_df.shape}")
        print(f"Claims data: {claims_df.shape}")
        print(f"SDOH data: {sdoh_df.shape}")
        print(f"Retail data: {retail_df.shape}")
        print(f"Labels: {labels_df.shape}")

        return clinical_df, claims_df, sdoh_df, retail_df, labels_df


if __name__ == "__main__":
    generator = SyntheticDataGenerator(n_patients=1000, random_state=42)
    generator.save_data()
