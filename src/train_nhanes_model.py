"""
Train a patient-level CKD classifier on real NHANES 2017-2022 data.
Stage 2 of the two-stage validation in the JMIR submission.

Inputs (real, no synthetic):
  - data/nhanes/nhanes_kidney_panel_2017_2023.csv

Outcome: ckd = 1 if (eGFR < 60 mL/min/1.73m² via CKD-EPI 2021 [no race]) OR
                 (UACR >= 30 mg/g)  per KDIGO 2024 criteria.

Features used (NO direct kidney biomarkers — those define the label):
  - RIDAGEYR        age in years
  - RIAGENDR        sex (1=male, 2=female)
  - RIDRETH3        race/ethnicity (NHANES coding)
  - INDFMPIR        family income-to-poverty ratio
  - BMXBMI          body mass index
  - mean_sbp        mean oscillometric SBP across up to 3 readings
  - mean_dbp        mean oscillometric DBP
  - BPQ020          self-reported diagnosed hypertension
  - DIQ010          self-reported diagnosed diabetes
  - MCQ160B         self-reported heart failure
  - MCQ160F         self-reported stroke

XGBoost classifier with NHANES MEC weights for survey-weighted CV.
Stratified 5-fold CV on the binary CKD label.

Output:
  - models/ckd_nhanes_classifier.pkl
  - results/metrics/nhanes_training_metrics.json
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    average_precision_score, brier_score_loss, roc_auc_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "nhanes" / "nhanes_kidney_panel_2017_2023.csv"
MODEL_OUT = ROOT / "models" / "ckd_nhanes_classifier.pkl"
METRICS_OUT = ROOT / "results" / "metrics" / "nhanes_training_metrics.json"

FEATURES = [
    "RIDAGEYR", "RIAGENDR", "RIDRETH3", "INDFMPIR", "BMXBMI",
    "mean_sbp", "mean_dbp",
    "BPQ020", "DIQ010", "MCQ160B", "MCQ160F",
]


def main():
    print("Loading NHANES kidney panel...")
    df = pd.read_csv(DATA_CSV)
    print(f"  Total rows: {len(df):,}")

    # Drop rows missing the label or the unified survey weight
    df = df.dropna(subset=["ckd", "survey_weight"])
    print(f"  With CKD label and weight: {len(df):,}")
    print(f"  Cycle distribution:")
    print(df["nhanes_cycle"].value_counts().to_string())

    # When combining multiple cycles each weighted to the US population, divide
    # by the number of cycles so the population total is not double-counted.
    n_cycles = df["nhanes_cycle"].nunique()
    df["weight_combined"] = df["survey_weight"] / n_cycles

    X = df[FEATURES].astype(float)
    # Median-impute features (XGBoost can handle NaN natively, but for survey-weighted
    # logistic comparisons we want a complete matrix).  Keep NaN — XGBoost tolerates it.
    y = df["ckd"].astype(int).values
    w = df["weight_combined"].values

    print(f"\nClass balance: {y.sum():,} CKD-positive ({y.mean() * 100:.2f}%)")

    # Stratified 5-fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_model = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    print("\nCross-validating (5-fold stratified)...")

    # Manually loop folds so we can pass sample_weight to fit
    proba_oof = np.zeros(len(y), dtype=float)
    fold_metrics = []
    for k, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        m = XGBClassifier(**base_model.get_params())
        m.fit(X.iloc[train_idx], y[train_idx], sample_weight=w[train_idx])
        p = m.predict_proba(X.iloc[test_idx])[:, 1]
        proba_oof[test_idx] = p
        # Survey-weighted AUC via roc_auc_score(sample_weight=...)
        auroc = roc_auc_score(y[test_idx], p, sample_weight=w[test_idx])
        auprc = average_precision_score(y[test_idx], p, sample_weight=w[test_idx])
        brier = brier_score_loss(y[test_idx], p, sample_weight=w[test_idx])
        fold_metrics.append({
            "fold": k, "n": int(len(test_idx)),
            "AUROC": float(auroc), "AUPRC": float(auprc), "Brier": float(brier),
        })
        print(f"  Fold {k}: AUROC={auroc:.3f}  AUPRC={auprc:.3f}  Brier={brier:.4f}")

    # Survey-weighted overall metrics (out-of-fold predictions)
    auroc_oof = roc_auc_score(y, proba_oof, sample_weight=w)
    auprc_oof = average_precision_score(y, proba_oof, sample_weight=w)
    brier_oof = brier_score_loss(y, proba_oof, sample_weight=w)

    print(f"\nOOF AUROC: {auroc_oof:.4f}")
    print(f"OOF AUPRC: {auprc_oof:.4f}")
    print(f"OOF Brier: {brier_oof:.4f}")

    # Sensitivity / NPV at 90% specificity (operationally meaningful)
    from sklearn.metrics import roc_curve
    fpr, tpr, thresh = roc_curve(y, proba_oof, sample_weight=w)
    # find threshold giving FPR <= 0.10
    target = np.argmin(np.abs((1 - fpr) - 0.90))
    sens_at_90spec = tpr[target]
    thresh_at_90spec = thresh[target]
    print(f"Sensitivity at 90% specificity: {sens_at_90spec:.3f} "
          f"(threshold={thresh_at_90spec:.3f})")

    # Final fit on full data
    print("\nFitting final classifier on all data...")
    final_model = XGBClassifier(**base_model.get_params())
    final_model.fit(X, y, sample_weight=w)

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final_model,
        "feature_names": FEATURES,
        "outcome_definition": "eGFR<60 (CKD-EPI 2021, no race) OR UACR>=30 mg/g",
        "training_data": "NHANES 2017-2020 pre-pandemic + 2021-2022 (n=" f"{len(df):,})",
        "n_total": int(len(df)),
        "n_ckd_positive": int(y.sum()),
        "ckd_prevalence_unweighted": float(y.mean()),
        "ckd_prevalence_weighted": float(np.average(y, weights=w)),
        "oof_predictions_oof_idx_aligned": True,
    }, MODEL_OUT)

    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps({
        "n_total": int(len(df)),
        "n_ckd_positive": int(y.sum()),
        "ckd_prevalence_unweighted": float(y.mean()),
        "ckd_prevalence_weighted": float(np.average(y, weights=w)),
        "fold_metrics": fold_metrics,
        "oof_AUROC": float(auroc_oof),
        "oof_AUPRC": float(auprc_oof),
        "oof_Brier": float(brier_oof),
        "sens_at_90_spec": float(sens_at_90spec),
        "threshold_at_90_spec": float(thresh_at_90spec),
        "feature_importance": {
            n: float(v) for n, v in zip(FEATURES, final_model.feature_importances_)
        },
    }, indent=2))

    # Persist OOF predictions for downstream calibration / DCA / subgroup analysis
    pd.DataFrame({
        "SEQN": df["SEQN"].values,
        "ckd_true": y,
        "ckd_proba_oof": proba_oof,
        "weight": w,
        "RIDAGEYR": df["RIDAGEYR"].values,
        "RIAGENDR": df["RIAGENDR"].values,
        "RIDRETH3": df["RIDRETH3"].values,
        "nhanes_cycle": df["nhanes_cycle"].values,
    }).to_csv(ROOT / "results" / "nhanes_oof_predictions.csv", index=False)

    print(f"\nWrote: {MODEL_OUT.relative_to(ROOT)}")
    print(f"Wrote: {METRICS_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
