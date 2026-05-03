"""
Train the ecological CKD prevalence model on real data (Stage 1 of two-stage validation).

Inputs (all real, no synthetic):
  - data/adi_2020_national_blockgroup.csv  (UW Neighborhood Atlas, 242,335 BGs)
  - data/cdc_places/places_ckd_2022_tract.csv  (CDC PLACES, 72,337 tracts)
  - data/census/state_to_region.csv  (state FIPS -> Census region)

Unit of analysis: Census tract (~60,944 tracts that match between ADI and PLACES).

Outcome  : CDC PLACES tract-level crude CKD prevalence (real, observed).
Features : ADI national rank, ADI state rank, ADI quintile mean (aggregated from BGs).
Model    : XGBoost regressor.

Output:
  - models/ckd_ecological_model.pkl  (model + feature names + metadata)
  - results/tract_ckd_predictions.csv  (per-tract predicted prevalence)
  - results/training_metrics_ecological.json
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
ADI_CSV = ROOT / "data" / "adi_2020_national_blockgroup.csv"
PLACES_CSV = ROOT / "data" / "cdc_places" / "places_ckd_2022_tract.csv"
STATE_REGION_CSV = ROOT / "data" / "census" / "state_to_region.csv"

MODEL_OUT = ROOT / "models" / "ckd_ecological_model.pkl"
PRED_OUT = ROOT / "results" / "tract_ckd_predictions.csv"
METRICS_OUT = ROOT / "results" / "training_metrics_ecological.json"


def load_features() -> pd.DataFrame:
    print("Loading ADI block-group file...")
    adi = pd.read_csv(ADI_CSV, dtype={"FIPS": str, "GISJOIN": str})
    for col in ("ADI_NATRANK", "ADI_STATERNK"):
        adi[col] = pd.to_numeric(adi[col], errors="coerce")
    adi = adi.dropna(subset=["ADI_NATRANK", "ADI_STATERNK"]).copy()
    adi["tract_fips"] = adi["FIPS"].str[:11]
    print(f"  {len(adi):,} usable block groups -> {adi['tract_fips'].nunique():,} tracts")

    # ADI quintile from national rank deciles (1-20 = Q1, ..., 81-100 = Q5)
    adi["adi_quintile_bg"] = pd.cut(adi["ADI_NATRANK"], bins=[0, 20, 40, 60, 80, 100],
                                    labels=[1, 2, 3, 4, 5]).astype(int)

    tract_features = adi.groupby("tract_fips").agg(
        adi_natrank=("ADI_NATRANK", "mean"),
        adi_staternk=("ADI_STATERNK", "mean"),
        adi_quintile=("adi_quintile_bg", "mean"),
        n_blockgroups=("FIPS", "size"),
    ).reset_index()
    tract_features["state_fips"] = tract_features["tract_fips"].str[:2]
    return tract_features


def load_outcome() -> pd.DataFrame:
    print("Loading CDC PLACES tract-level CKD prevalence...")
    places = pd.read_csv(PLACES_CSV, dtype={"tract_fips": str, "state_fips": str,
                                             "county_fips": str})
    print(f"  {len(places):,} tracts with CKD prevalence")
    return places[["tract_fips", "ckd_pct", "ckd_lci", "ckd_uci", "total_pop"]]


def add_region(df: pd.DataFrame) -> pd.DataFrame:
    sr = pd.read_csv(STATE_REGION_CSV, dtype={"state_fips": str})
    return df.merge(sr[["state_fips", "state_abbr", "census_region"]],
                    on="state_fips", how="left")


def main():
    feats = load_features()
    out = load_outcome()
    df = feats.merge(out, on="tract_fips", how="inner")
    df = add_region(df)
    df = df.dropna(subset=["census_region"])  # drop AS/PR/VI/GU/MP territories
    print(f"\nMatched tracts (ADI ∩ PLACES, CONUS+AK+HI): {len(df):,}")
    print("By region:")
    print(df["census_region"].value_counts().to_string())

    feature_cols = ["adi_natrank", "adi_staternk", "adi_quintile"]
    X = df[feature_cols].astype(float)
    y = df["ckd_pct"].astype(float)  # outcome in percentage points (0-100)

    # ---------------- 1. Standard 5-fold CV (random tracts) ----------------
    print("\n=== Random 5-fold CV (tract-shuffled) ===")
    model = XGBRegressor(
        n_estimators=600, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1,
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_random = cross_val_score(model, X, y, scoring="r2", cv=cv, n_jobs=-1)
    mae_random = -cross_val_score(model, X, y, scoring="neg_mean_absolute_error",
                                  cv=cv, n_jobs=-1)
    print(f"  R²:  {r2_random.mean():.4f} ± {r2_random.std():.4f}")
    print(f"  MAE: {mae_random.mean():.4f} ± {mae_random.std():.4f}")

    # ---------------- 2. Leave-one-Census-region-out CV ----------------
    print("\n=== Leave-one-Census-region-out CV ===")
    loro_results = []
    for region in sorted(df["census_region"].unique()):
        train_idx = df["census_region"] != region
        test_idx = df["census_region"] == region
        m = XGBRegressor(**model.get_params())
        m.fit(X[train_idx], y[train_idx])
        p = m.predict(X[test_idx])
        r = r2_score(y[test_idx], p)
        mae = mean_absolute_error(y[test_idx], p)
        # Calibration slope/intercept on the held-out region
        slope, intercept = np.polyfit(p, y[test_idx], 1)
        loro_results.append({
            "held_out_region": region,
            "n_train": int(train_idx.sum()),
            "n_test": int(test_idx.sum()),
            "R2": float(r),
            "MAE": float(mae),
            "RMSE": float(np.sqrt(np.mean((p - y[test_idx]) ** 2))),
            "calibration_slope": float(slope),
            "calibration_intercept": float(intercept),
        })
        print(f"  Held-out {region:<10} (n={test_idx.sum():,}): "
              f"R²={r:.3f}  MAE={mae:.3f}  slope={slope:.3f}")

    # ---------------- 3. Sensitivity sweep on hyperparameters ----------------
    print("\n=== Sensitivity to model capacity ===")
    sensitivity = []
    for n_est, depth, lr in [(200, 3, 0.1), (400, 4, 0.05), (600, 5, 0.05),
                             (800, 6, 0.03), (1000, 7, 0.03)]:
        m = XGBRegressor(n_estimators=n_est, max_depth=depth, learning_rate=lr,
                         subsample=0.9, colsample_bytree=0.9,
                         random_state=42, n_jobs=-1)
        r2_ = cross_val_score(m, X, y, scoring="r2", cv=cv, n_jobs=-1).mean()
        mae_ = -cross_val_score(m, X, y, scoring="neg_mean_absolute_error",
                                cv=cv, n_jobs=-1).mean()
        sensitivity.append({
            "n_estimators": n_est, "max_depth": depth, "learning_rate": lr,
            "CV_R2": float(r2_), "CV_MAE": float(mae_),
        })
        print(f"  n_est={n_est:4d} depth={depth} lr={lr:<5}  R²={r2_:.4f}  MAE={mae_:.4f}")

    # ---------------- Final fit on all tracts ----------------
    print("\nFitting final model on all matched tracts...")
    model.fit(X, y)
    df["ckd_predicted_pct"] = model.predict(X)
    train_r2 = r2_score(y, df["ckd_predicted_pct"])
    train_mae = mean_absolute_error(y, df["ckd_predicted_pct"])
    print(f"  Train R²:  {train_r2:.4f}")
    print(f"  Train MAE: {train_mae:.4f}")

    # Save artefacts
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_names": feature_cols,
        "outcome": "CDC PLACES 2022 tract-level CKD crude prevalence (%)",
        "data_sources": {
            "adi": "data/adi_2020_national_blockgroup.csv (UW Neighborhood Atlas 2020)",
            "places": "data/cdc_places/places_ckd_2022_tract.csv (CDC PLACES 2022 release, BRFSS 2020)",
        },
    }, MODEL_OUT)

    df.to_csv(PRED_OUT, index=False)

    metrics = {
        "n_tracts_trained": int(len(df)),
        "by_region": df["census_region"].value_counts().to_dict(),
        "outcome_mean_pct": float(y.mean()),
        "outcome_std_pct": float(y.std()),
        "outcome_min_pct": float(y.min()),
        "outcome_max_pct": float(y.max()),
        "random_5fold_cv": {
            "R2_mean": float(r2_random.mean()),
            "R2_std": float(r2_random.std()),
            "MAE_mean": float(mae_random.mean()),
            "MAE_std": float(mae_random.std()),
        },
        "leave_one_region_out_cv": loro_results,
        "sensitivity_to_capacity": sensitivity,
        "train_full_fit": {"R2": float(train_r2), "MAE": float(train_mae)},
    }
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(metrics, indent=2, default=float))
    print(f"\nWrote: {MODEL_OUT.relative_to(ROOT)}")
    print(f"Wrote: {PRED_OUT.relative_to(ROOT)}  ({len(df):,} rows)")
    print(f"Wrote: {METRICS_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
