# Open Two-Stage CKD Surveillance & Risk-Stratification Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> A fully reproducible CKD pipeline using **only public datasets** (no synthetic data, no private EHR cohorts), targeting submission to **JMIR Public Health & Surveillance**.

## What this is

Two complementary models in one repo:

| Stage | Unit | Outcome | Data sources | Cross-validated metric |
|---|---|---|---|---|
| **1. Ecological surveillance** | U.S. census tract (n=60,445) | CDC PLACES tract-level CKD prevalence (real, observed) | UW Neighborhood Atlas ADI 2020 (block-group); CDC PLACES 2022 | R² = **0.450 ± 0.006**, MAE = 0.45 pp |
| **2. Patient-level screening** | NHANES adult (n=15,150) | KDIGO CKD = (eGFR<60 by CKD-EPI 2021, no race) OR (UACR≥30 mg/g) | NHANES 2017–2020 P_*, 2021–2022 _L | Survey-weighted AUROC = **0.772**, AUPRC = 0.403, Brier = 0.102 |

The patient-level classifier uses **no kidney biomarkers** as features (creatinine, albumin) — only demographics, BMI, blood pressure, and self-reported comorbidities — making it usable as a screening triage tool where lab testing is not available.

## Data Sources (all real, all public)

| File | Source | Rows |
|---|---|---|
| `data/adi_2020_national_blockgroup.csv` | UW Neighborhood Atlas, ADI 2020, 12-digit FIPS, National | 242,335 block groups |
| `data/cdc_places/places_ckd_2022_tract.csv` | CDC PLACES 2022 GIS Friendly Format (Socrata `shc3-fzig`) | 72,337 tracts |
| `data/census/zcta_tract_crosswalk_2020.csv` | U.S. Census Bureau 2020 ZCTA↔tract relationship file | 171,480 overlap pairs |
| `data/nhanes/nhanes_kidney_panel_2017_2023.csv` | CDC NHANES 2017-2020 pre-pandemic + 2021-2022 (XPT files) | 17,846 adults |
| `data/usrds_2023_tables/usrds_extracted_parameters.csv` | USRDS 2023 Annual Data Report (reference statistics only) | 10 parameters |

ADI requires a one-time email registration at neighborhoodatlas.medicine.wisc.edu/download. All other downloads are scripted in `src/data_processing/fetch_*.py` and require no authentication.

## Quick start

```bash
pip install -r requirements.txt
make all
```

`make all` runs end-to-end: fetch raw data → train both models → evaluate → regenerate every figure → rebuild the JMIR manuscript with live numbers.

Individual targets (in dependency order):

```bash
make data       # fetch CDC PLACES, ZCTA crosswalk, NHANES (ADI must be placed manually)
make train      # train ecological + NHANES models
make evaluate   # full evaluation, calibration, subgroup, decision curves
make figures    # regenerate all figures
make paper      # rewrite the JMIR docx from live metrics
```

## Repository layout

```
src/
  data_processing/
    fetch_cdc_places.py        # CDC PLACES tract-level CKD via Socrata
    fetch_zcta_crosswalk.py    # Census 2020 ZCTA-tract relationship file
    fetch_nhanes.py            # NHANES XPT files + computed eGFR/UACR/CKD label
  utils/
    geo.py                     # ZCTA → state → Census region
    clinical.py                # CKD-EPI 2021 eGFR (no race), UACR, KDIGO label
  train_ecological_model.py    # XGBoost regressor → CDC PLACES ground truth
  train_nhanes_model.py        # XGBoost classifier → KDIGO CKD label
  generate_ecological_figures.py
  evaluate_nhanes_model.py
  rewrite_white_paper.py       # JMIR docx generator (numbers from JSON)
white_paper/
  CKD_Paper_JMIR_PHS_v3.docx           # current submission target
  CKD_Paper_Final_JMIR_05022026.docx   # original (preserved)
results/
  figures/                     # 13 PNGs embedded in the manuscript
  metrics/                     # full performance JSON/CSV
models/
  ckd_ecological_model.pkl
  ckd_nhanes_classifier.pkl
```

## Reproducibility

- All random seeds = 42
- Python 3.11, XGBoost 3.2, scikit-learn 1.8, pandas 3.0, pyreadstat 1.3
- All figures regenerable from `make figures`
- Manuscript text auto-populated from `results/metrics/*.json` — body text and figures cannot drift

## Citation

See [CITATION.cff](CITATION.cff). When citing this work please also cite the underlying public datasets (Neighborhood Atlas, CDC PLACES, NHANES, USRDS) as listed there.

## Disclaimer

This is a research artefact. Neither model is a medical device or clinical decision-support system. The patient-level classifier is intended as a low-cost screening triage tool; definitive CKD diagnosis requires creatinine and urine albumin measurement.
