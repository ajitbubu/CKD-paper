# CKD Risk Prediction Dashboard

Interactive web application for Chronic Kidney Disease risk prediction using multi-modal XGBoost classification.

## 🎯 Features

### 1. **Multi-Modal Data Upload** 📤
- Upload data from 4 different sources:
  - **Clinical (EHR)**: Lab values, vitals, diagnoses, medications
  - **Claims**: Healthcare utilization patterns, insurance data
  - **SDOH**: Social determinants of health (ZIP-code level)
  - **Retail**: Purchase pattern data
- Data validation with helpful error messages
- Sample template downloads
- Automatic prediction execution

### 2. **Interactive Dashboard** 📊
- Real-time risk distribution visualization
- Feature importance analysis
- High-risk patient identification
- Model performance metrics
- Filterable and sortable patient lists
- Export capabilities

### 3. **History Tracking** 📜
- Track all prediction runs
- Compare different datasets
- Trend analysis over time
- Export historical data
- Performance metrics tracking

### 4. **Database Persistence** 💾
- SQLite database for all predictions
- Patient-level prediction storage
- Dataset metadata tracking
- Query and compare past runs

## 🚀 Quick Start

### Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Ensure model is trained**:
```bash
python src/train_model.py
```

### Launch the Application

**Option 1: Using the launch script** (Recommended)
```bash
./run_app.sh
```

**Option 2: Direct Streamlit command**
```bash
streamlit run app/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📖 User Guide

### Page 1: Upload Data

1. **Download Sample Templates** (Optional)
   - Click on template buttons to download CSV examples
   - Use these as reference for your data format

2. **Enter Dataset Name**
   - Provide a descriptive name for tracking
   - Example: "Q4_2024_Patient_Cohort"

3. **Upload Files**
   - Clinical Data (required)
   - Claims Data (required)
   - SDOH Data (required)
   - Retail Data (required)
   - Labels (optional - for validation)

4. **Run Prediction**
   - Click "Run Prediction" button
   - Wait for validation and processing
   - View results immediately

5. **Results Displayed**:
   - Risk distribution (High/Moderate/Low)
   - Model performance metrics
   - Top 10 important features
   - High-risk patient list
   - Download option for predictions

### Page 2: Dashboard

1. **Select Prediction Run**
   - Use dropdown to choose from history
   - See dataset name, date, and patient count

2. **Overview Metrics**
   - Total patients analyzed
   - CKD detection rate
   - Model accuracy and AUC
   - High-risk patient count

3. **Risk Distribution**
   - Pie chart showing risk categories
   - Probability distribution histogram
   - Risk thresholds visualized

4. **Feature Importance**
   - Adjust number of features (10-30)
   - View by individual features
   - Group by data source
   - Interactive charts

5. **High-Risk Patients**
   - Filter by risk category
   - Sort by probability or patient ID
   - Color-coded probability scores
   - Download filtered results

6. **Model Performance** (if labels available)
   - Accuracy, Precision, Recall, F1-Score
   - Gauge visualizations
   - Performance targets highlighted

### Page 3: History

1. **Overall Statistics**
   - Total prediction runs
   - Patients analyzed (cumulative)
   - Average CKD detection rate
   - Average model accuracy

2. **Timeline Visualization**
   - Scatter plot of predictions over time
   - Size represents CKD positive count
   - Color represents CKD rate
   - Trend line for detection rate

3. **Datasets Table**
   - All uploaded datasets
   - Upload dates and record counts
   - Sortable and filterable

4. **Predictions Table**
   - All prediction runs
   - Filter by date, dataset, patient count
   - Model performance metrics
   - Export capabilities

5. **Compare Predictions**
   - Select two predictions
   - Side-by-side metrics
   - Visual comparisons
   - Performance differences

6. **Export Options**
   - Download datasets history
   - Download predictions history
   - CSV format for analysis

## 📊 Data Requirements

### Clinical Data (EHR)

Required columns:
- `patient_id` - String
- `serum_creatinine` - Float (mg/dL)
- `egfr` - Float (mL/min/1.73m²)
- `albuminuria_acr` - Float (mg/g)
- `hba1c` - Float (%)
- `systolic_bp` - Integer (mmHg)
- `diastolic_bp` - Integer (mmHg)
- `bmi` - Float (kg/m²)

Optional columns:
- `heart_rate`, `has_diabetes`, `has_hypertension`, `has_cvd`,
  `medication_count`, `on_ace_inhibitor`, `on_arb`, `on_diuretic`,
  `icd10_codes`, `medications`

### Claims Data

Required columns:
- `patient_id` - String
- `er_visits_count` - Integer
- `hospital_admissions_count` - Integer
- `outpatient_visits_count` - Integer
- `primary_care_visits_count` - Integer

Optional columns:
- `specialist_visits_count`, `nephrology_referrals_count`,
  `dialysis_encounters_count`, `insurance_coverage_gap_days`,
  `claim_denials_count`, `prescription_fills_count`,
  `total_claims_amount`, `out_of_pocket_amount`

### SDOH Data (ZIP-code level)

Required columns:
- `zip_code` - String (5 digits)
- `median_household_income` - Float ($)
- `poverty_rate` - Float (%)
- `diabetes_prevalence` - Float (%)
- `adi_national_percentile` - Integer (1-100)

Optional columns:
- `unemployment_rate`, `high_school_grad_rate`,
  `bachelor_degree_rate`, `obesity_prevalence`,
  `physical_inactivity_rate`, `smoking_rate`,
  `food_desert_indicator`, `low_access_to_grocery_store`,
  `snap_participation_rate`, `adi_state_percentile`

### Retail Purchase Data

Required columns:
- `patient_id` - String
- `processed_food_purchases` - Integer
- `fresh_produce_purchases` - Integer
- `high_sodium_food_purchases` - Integer

Optional columns:
- `sugary_beverage_purchases`, `whole_grain_purchases`,
  `red_meat_purchases`, `bp_monitor_purchases`,
  `glucose_monitor_purchases`, `health_supplements_purchases`,
  `shopping_frequency_per_month`, `health_conscious_score`

### Labels (Optional)

Required columns:
- `patient_id` - String
- `has_ckd` - Boolean/Integer (0 or 1)
- `zip_code` - String
- `ckd_stage` - Integer (0-5)

## 🎨 Features & Visualizations

### Risk Categories

- **Low Risk**: Probability < 30%
- **Moderate Risk**: 30% ≤ Probability < 70%
- **High Risk**: Probability ≥ 70%

### Interactive Charts

1. **Pie Charts**: Risk distribution
2. **Histograms**: Probability distributions
3. **Bar Charts**: Feature importance
4. **Scatter Plots**: Timeline analysis
5. **Line Charts**: Trend analysis
6. **Gauge Charts**: Performance metrics

### Export Options

- Patient predictions (CSV)
- Filtered results (CSV)
- Datasets history (CSV)
- Predictions history (CSV)

## 🗄️ Database Schema

The application uses SQLite with three main tables:

### datasets
Stores uploaded dataset information
- id, upload_date, dataset_name
- record counts per data source
- dataset_hash for deduplication
- file_paths (JSON)

### predictions
Stores prediction run results
- id, dataset_id, prediction_date
- patient counts and CKD rates
- model performance metrics
- feature_importance (JSON)

### patient_predictions
Stores individual patient predictions
- id, prediction_id, patient_id
- prediction, probability
- risk_category
- top_risk_factors (JSON)

## 🔧 Configuration

### Streamlit Config

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
port = 8501
headless = true
enableCORS = false
```

### Model Configuration

Modify `config/config.yaml` to adjust:
- XGBoost hyperparameters
- Feature engineering options
- SHAP computation settings

## 🛡️ Security & Privacy

**Important Notes**:

1. **Data Privacy**
   - All data is stored locally
   - No external API calls
   - Patient IDs should be de-identified

2. **HIPAA Compliance**
   - Use de-identified data only
   - Ensure secure environment
   - Follow institutional IRB requirements

3. **Model Validation**
   - This is a research prototype
   - Always validate with clinical expertise
   - Not approved for clinical use

## 🐛 Troubleshooting

### Common Issues

**Issue**: "Model not found" error
**Solution**: Run `python src/train_model.py` first

**Issue**: Upload fails with validation error
**Solution**: Check column names match requirements exactly

**Issue**: Database locked error
**Solution**: Close other instances of the app

**Issue**: Streamlit won't start
**Solution**: Check if port 8501 is already in use

### Getting Help

1. Check the data templates
2. Review error messages carefully
3. Ensure all required columns are present
4. Verify data types are correct

## 📝 Development

### Project Structure

```
app/
├── app.py              # Main application (Home page)
├── pages/              # Multi-page app pages
│   ├── 1_📤_Upload_Data.py
│   ├── 2_📊_Dashboard.py
│   └── 3_📜_History.py
├── utils/              # Utility modules
│   ├── database.py     # SQLite operations
│   └── helpers.py      # Helper functions
├── database/           # SQLite database storage
└── README.md           # This file
```

### Adding New Features

1. Modify existing pages in `pages/`
2. Add utility functions to `utils/helpers.py`
3. Extend database schema in `utils/database.py`
4. Update this README

## 🎓 Citation

If you use this dashboard in your research:

```bibtex
@software{ckd_dashboard_2024,
  title={CKD Risk Prediction Dashboard},
  author={Your Name},
  year={2024},
  note={Multi-Modal XGBoost Classification System}
}
```

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- Built with Streamlit
- Powered by XGBoost
- Visualizations by Plotly
- Data storage with SQLite

---

**For questions or issues, please refer to the main project README.**
