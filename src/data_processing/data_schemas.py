"""
Data schemas for CKD prediction model
Defines the structure and types for each data source
"""

from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class ClinicalDataSchema:
    """EHR Clinical Data Schema"""
    patient_id: str
    record_date: datetime

    # Laboratory Values
    serum_creatinine: float  # mg/dL
    egfr: float  # mL/min/1.73m²
    albuminuria_acr: float  # mg/g (Albumin-to-Creatinine Ratio)
    hba1c: float  # %

    # Vital Signs
    systolic_bp: int  # mmHg
    diastolic_bp: int  # mmHg
    bmi: float  # kg/m²
    heart_rate: Optional[int]  # bpm

    # Diagnosis Codes (ICD-10)
    icd10_codes: List[str]  # e.g., ['N18.3', 'E11.9', 'I10']

    # Medications (generic names)
    medications: List[str]  # e.g., ['lisinopril', 'metformin']

    # Derived flags
    has_diabetes: bool
    has_hypertension: bool
    has_cvd: bool


@dataclass
class ClaimsDataSchema:
    """Healthcare Claims Data Schema"""
    patient_id: str
    time_period: str  # e.g., '2023-Q4'

    # Healthcare Utilization
    er_visits_count: int
    hospital_admissions_count: int
    outpatient_visits_count: int
    primary_care_visits_count: int
    specialist_visits_count: int

    # Nephrology Specific
    nephrology_referrals_count: int
    dialysis_encounters_count: int

    # Insurance Patterns
    insurance_coverage_gap_days: int
    claim_denials_count: int
    total_claims_amount: float
    out_of_pocket_amount: float

    # Prescription patterns
    prescription_fills_count: int
    generic_vs_brand_ratio: float


@dataclass
class SDOHDataSchema:
    """Social Determinants of Health Data Schema (ZIP-code level)"""
    zip_code: str

    # U.S. Census Bureau
    median_household_income: float
    poverty_rate: float  # %
    unemployment_rate: float  # %
    high_school_grad_rate: float  # %
    bachelor_degree_rate: float  # %

    # CDC PLACES Dataset
    diabetes_prevalence: float  # %
    obesity_prevalence: float  # %
    physical_inactivity_rate: float  # %
    smoking_rate: float  # %

    # USDA Food Access Research Atlas
    food_desert_indicator: bool
    low_access_to_grocery_store: bool  # >1 mile urban, >10 miles rural
    snap_participation_rate: float  # %

    # Area Deprivation Index
    adi_national_percentile: int  # 1-100 (higher = more deprived)
    adi_state_percentile: int  # 1-10


@dataclass
class RetailPurchaseSchema:
    """Retail Purchase Pattern Data Schema (Geolocation-based)"""
    patient_id: str
    geolocation_area: str  # ZIP or census tract
    time_period: str  # e.g., '2023-Q4'

    # Food Categories (counts or spending)
    processed_food_purchases: int
    fresh_produce_purchases: int
    high_sodium_food_purchases: int
    sugary_beverage_purchases: int
    whole_grain_purchases: int
    red_meat_purchases: int

    # Health Products
    bp_monitor_purchases: int
    glucose_monitor_purchases: int
    health_supplements_purchases: int

    # Pharmacy/OTC
    otc_pain_medication_purchases: int
    otc_antacid_purchases: int
    vitamin_mineral_purchases: int

    # Shopping Patterns
    shopping_frequency_per_month: int
    avg_basket_size: float
    health_conscious_score: float  # 0-100 derived metric


@dataclass
class IntegratedPatientRecord:
    """Integrated patient record combining all data sources"""
    patient_id: str
    zip_code: str
    observation_date: datetime

    # Target Variable
    ckd_stage: int  # 0=No CKD, 1-5=CKD Stages
    has_ckd: bool  # Binary classification

    # Clinical features
    clinical_data: ClinicalDataSchema

    # Claims features
    claims_data: ClaimsDataSchema

    # SDOH features (linked by ZIP code)
    sdoh_data: SDOHDataSchema

    # Retail features
    retail_data: RetailPurchaseSchema


# Feature naming conventions for ML pipeline
CLINICAL_FEATURES = [
    'serum_creatinine', 'egfr', 'albuminuria_acr', 'hba1c',
    'systolic_bp', 'diastolic_bp', 'bmi', 'heart_rate',
    'has_diabetes', 'has_hypertension', 'has_cvd',
    'medication_count', 'on_ace_inhibitor', 'on_arb', 'on_diuretic'
]

CLAIMS_FEATURES = [
    'er_visits_count', 'hospital_admissions_count', 'outpatient_visits_count',
    'primary_care_visits_count', 'specialist_visits_count',
    'nephrology_referrals_count', 'dialysis_encounters_count',
    'insurance_coverage_gap_days', 'claim_denials_count',
    'prescription_fills_count'
]

SDOH_FEATURES = [
    'median_household_income', 'poverty_rate', 'unemployment_rate',
    'high_school_grad_rate', 'diabetes_prevalence', 'obesity_prevalence',
    'physical_inactivity_rate', 'food_desert_indicator',
    'low_access_to_grocery_store', 'adi_national_percentile'
]

RETAIL_FEATURES = [
    'processed_food_purchases', 'fresh_produce_purchases',
    'high_sodium_food_purchases', 'sugary_beverage_purchases',
    'bp_monitor_purchases', 'glucose_monitor_purchases',
    'shopping_frequency_per_month', 'health_conscious_score'
]

ALL_FEATURES = CLINICAL_FEATURES + CLAIMS_FEATURES + SDOH_FEATURES + RETAIL_FEATURES
