# CKD Detection Using XGBoost with Multi-Modal Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Chronic Kidney Disease (CKD) Detection Using XGBoost Classification on Multi-Modal Healthcare Data**

This project implements a comprehensive machine learning approach to detect Chronic Kidney Disease using XGBoost classification on integrated multi-modal data sources including clinical EHR data, healthcare claims, social determinants of health (SDOH), and retail purchase patterns.

## 📋 Table of Contents
- [Overview](#overview)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)

## 🔍 Overview

Chronic Kidney Disease affects approximately 15% of the US adult population and is a leading cause of morbidity and mortality. Early detection is crucial for preventing disease progression. This project leverages multiple data modalities to improve CKD risk prediction:

### Key Features
- **Multi-Modal Data Integration**: Combines 4 distinct data sources for comprehensive patient profiling
- **XGBoost Classification**: State-of-the-art gradient boosting for binary (CKD/No CKD) classification
- **SHAP Interpretability**: Provides clinical explanations for model predictions
- **Feature Engineering**: Creates composite risk scores from raw data
- **Production-Ready**: Includes data pipelines, model serialization, and evaluation metrics

## 📊 Data Sources

### 1. **Clinical Data (EHR)**
- **Laboratory Values**: Serum creatinine, eGFR, albuminuria (ACR), HbA1c
- **Vital Signs**: Blood pressure, BMI, heart rate
- **Diagnosis Codes**: ICD-10 codes for CKD, diabetes, hypertension, CVD
- **Medications**: ACE inhibitors, ARBs, diuretics, diabetes medications

### 2. **Claims Data**
- **Healthcare Utilization**: ER visits, hospitalizations, outpatient visits
- **Specialist Care**: Primary care visits, specialist referrals, nephrology encounters
- **Insurance Patterns**: Coverage gaps, claim denials, out-of-pocket costs
- **Prescription Patterns**: Prescription fills, medication adherence

### 3. **SDOH Data (ZIP-code level)**
- **U.S. Census Bureau**: Median income, poverty rate, education, unemployment
- **CDC PLACES Dataset**: Diabetes prevalence, obesity, physical inactivity
- **USDA Food Access**: Food desert indicators, grocery store access
- **Area Deprivation Index**: National and state percentile rankings

### 4. **Retail Purchase Patterns (Geolocation-based)**
- **Food Categories**: Processed foods, fresh produce, high-sodium items, sugary beverages
- **Health Products**: Blood pressure monitors, glucose monitors, supplements
- **Pharmacy/OTC**: Medications, vitamins, health products
- **Shopping Patterns**: Frequency, basket size, health-conscious score

## 📁 Project Structure

```
CKD-paper/
├── config/
│   └── config.yaml                 # Configuration file
├── data/
│   ├── raw/                        # Raw data files
│   ├── processed/                  # Processed data
│   └── synthetic/                  # Synthetic demo data
├── models/                         # Trained models
├── notebooks/
│   └── CKD_XGBoost_Analysis.ipynb  # Interactive analysis notebook
├── results/
│   ├── figures/                    # Visualizations
│   └── metrics/                    # Performance metrics
├── src/
│   ├── data_processing/
│   │   ├── data_schemas.py         # Data structure definitions
│   │   ├── synthetic_data_generator.py  # Synthetic data generation
│   │   └── data_integration.py     # Data integration pipeline
│   ├── features/                   # Feature engineering modules
│   ├── models/
│   │   └── xgboost_model.py        # XGBoost classifier
│   ├── evaluation/
│   │   └── model_evaluation.py     # SHAP analysis & evaluation
│   ├── utils/                      # Utility functions
│   └── train_model.py              # Main training script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/CKD-paper.git
cd CKD-paper
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

### Option 1: Run the Training Script

```bash
# Generate synthetic data, train model, and evaluate
python src/train_model.py
```

This will:
1. Generate synthetic multi-modal patient data (n=1000)
2. Integrate and preprocess data from all sources
3. Train XGBoost classifier
4. Generate SHAP analysis and visualizations
5. Save trained model and results

### Option 2: Use Jupyter Notebook (Recommended for Exploration)

```bash
# Start Jupyter
jupyter notebook

# Open: notebooks/CKD_XGBoost_Analysis.ipynb
```

The notebook provides interactive analysis with:
- Data exploration and visualization
- Step-by-step model training
- SHAP interpretability analysis
- Clinical insights generation

### Option 3: Web Application (New! 🎉)

```bash
# Launch the interactive web dashboard
./run_app.sh
```

The web application provides:
- 📤 **Upload**: Multi-source data upload with validation
- 📊 **Dashboard**: Interactive visualizations and risk analysis
- 📜 **History**: Track and compare all predictions
- 💾 **Database**: Persistent storage of all runs

**Access at**: `http://localhost:8501`

See `app/README.md` for detailed documentation

## 📖 Usage

### Training with Custom Data

```python
from src.data_processing.data_integration import DataIntegrationPipeline
from src.models.xgboost_model import CKDXGBoostClassifier

# Load your data
clinical_df = pd.read_csv('your_clinical_data.csv')
claims_df = pd.read_csv('your_claims_data.csv')
sdoh_df = pd.read_csv('your_sdoh_data.csv')
retail_df = pd.read_csv('your_retail_data.csv')
labels_df = pd.read_csv('your_labels.csv')

# Integrate and preprocess
pipeline = DataIntegrationPipeline()
X, y = pipeline.prepare_for_modeling(
    clinical_df, claims_df, sdoh_df, retail_df, labels_df
)

# Train model
model = CKDXGBoostClassifier(model_type='binary', use_smote=True)
results = model.train(X, y, test_size=0.2)

# Save model
model.save_model('models/my_ckd_model.pkl')
```

### Making Predictions

```python
from src.models.xgboost_model import CKDXGBoostClassifier
import joblib

# Load trained model
model = CKDXGBoostClassifier()
model.load_model('models/ckd_xgboost_model.pkl')

# Load pipeline
pipeline = joblib.load('models/data_pipeline.pkl')

# Prepare new patient data
X_new, _ = pipeline.prepare_for_modeling(
    new_clinical_df, new_claims_df, new_sdoh_df, new_retail_df, None, fit=False
)

# Predict
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)

print(f"CKD Risk: {probabilities[:, 1][0]:.2%}")
```

### SHAP Interpretability

```python
from src.evaluation.model_evaluation import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator(model.model, pipeline.feature_names)

# Generate SHAP values
evaluator.compute_shap_values(X_test)

# Visualize
evaluator.plot_shap_summary(X_test)
evaluator.plot_shap_waterfall(X_test, sample_idx=0)

# Clinical insights
insights = evaluator.generate_clinical_insights(feature_importance)
```

## 🏗️ Model Architecture

### XGBoost Configuration
- **Objective**: Binary logistic regression
- **Evaluation Metric**: AUC-ROC
- **Class Imbalance**: SMOTE oversampling
- **Hyperparameters**:
  - Max depth: 6
  - Learning rate: 0.1
  - N estimators: 100
  - Subsample: 0.8
  - Colsample bytree: 0.8

### Feature Engineering
The pipeline creates **60+ features** including:

**Clinical Features** (11 features)
- Raw lab values and vital signs
- Engineered: pulse pressure, MAP, creatinine/eGFR ratio

**Comorbidity Features** (4 features)
- Diabetes, hypertension, CVD flags
- Comorbidity count

**Medication Features** (4 features)
- Medication count, ACE/ARB/diuretic flags
- Adherence ratio

**Healthcare Utilization** (8 features)
- Visit counts, referrals, dialysis encounters
- Utilization score

**SDOH Features** (10 features)
- Socioeconomic indicators, health prevalence
- Composite SDOH risk score

**Retail/Dietary Features** (15+ features)
- Food purchase patterns
- Health product purchases
- Dietary health ratio

## 📈 Results

### Model Performance (Synthetic Data)
| Metric | Score |
|--------|-------|
| Accuracy | 0.92+ |
| Precision | 0.88+ |
| Recall | 0.85+ |
| F1-Score | 0.86+ |
| ROC-AUC | 0.95+ |

### Top Predictive Features
1. **eGFR** - Direct kidney function measurement
2. **Serum Creatinine** - Kidney waste filtration marker
3. **Albuminuria (ACR)** - Early kidney damage indicator
4. **Healthcare Utilization Score** - Disease severity proxy
5. **ADI National Percentile** - Socioeconomic risk factor

### Feature Importance by Data Source
| Data Source | Total Importance | Key Contribution |
|-------------|------------------|------------------|
| Clinical | ~60% | Primary diagnostic features |
| Healthcare Utilization | ~20% | Disease progression signals |
| SDOH | ~12% | Risk stratification |
| Retail/Dietary | ~8% | Behavioral insights |

## 🔬 Research Applications

This framework is designed for:
- **Early CKD Detection**: Identifying at-risk patients before clinical diagnosis
- **Risk Stratification**: Prioritizing high-risk populations for intervention
- **Health Equity Research**: Understanding SDOH impact on CKD outcomes
- **Behavioral Interventions**: Targeting dietary and lifestyle factors
- **Policy Development**: Evidence-based resource allocation

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{ckd_xgboost_2024,
  title={Chronic Kidney Disease Detection Using XGBoost on Multi-Modal Healthcare Data},
  author={Your Name},
  journal={IEEE},
  year={2024}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- **Data Sources**: U.S. Census Bureau, CDC PLACES, USDA Food Access Research Atlas
- **Libraries**: XGBoost, scikit-learn, SHAP, pandas, numpy
- **Inspiration**: Clinical nephrology research and health equity initiatives

---

**Disclaimer**: This project uses synthetic data for demonstration. For real-world deployment, use validated, de-identified patient data in compliance with HIPAA and institutional IRB requirements.