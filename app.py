"""
Two-stage CKD demo (JMIR Public Health & Surveillance submission).

Stage 1 - Ecological surveillance: predict tract-level CKD prevalence from
          ADI features.  User enters a 5-digit ZIP code; we look up the
          most-likely matching tract via the ZCTA->tract relationship file
          and return the model's predicted prevalence.

Stage 2 - Patient-level screening: classifier on NHANES non-kidney features
          (demographics, BMI, BP, self-reported HTN/DM/CHF/stroke).  Returns
          a calibrated CKD probability with a recommended-screening flag.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
ECOL_MODEL_PKL = ROOT / "models" / "ckd_ecological_model.pkl"
NHANES_MODEL_PKL = ROOT / "models" / "ckd_nhanes_classifier.pkl"
ECOL_METRICS = ROOT / "results" / "training_metrics_ecological.json"
NHANES_METRICS = ROOT / "results" / "metrics" / "nhanes_performance.json"
ADI_BG_CSV = ROOT / "data" / "adi_2020_national_blockgroup.csv"
ZCTA_TRACT_CSV = ROOT / "data" / "census" / "zcta_tract_crosswalk_2020.csv"
PRED_CSV = ROOT / "results" / "tract_ckd_predictions.csv"

st.set_page_config(
    page_title="CKD Surveillance + Screening (JMIR PHS)",
    page_icon="🏥",
    layout="wide",
)


@st.cache_resource
def load_ecological():
    bundle = joblib.load(ECOL_MODEL_PKL)
    metrics = json.loads(ECOL_METRICS.read_text())
    return bundle, metrics


@st.cache_resource
def load_nhanes():
    bundle = joblib.load(NHANES_MODEL_PKL)
    metrics = json.loads(NHANES_METRICS.read_text())
    return bundle, metrics


@st.cache_resource
def load_predictions():
    """Pre-computed predicted CKD prevalence for each tract."""
    return pd.read_csv(PRED_CSV, dtype={"tract_fips": str, "state_fips": str})


@st.cache_resource
def load_zcta_lookup():
    """ZCTA -> list of tract_fips, with land area for weighting."""
    df = pd.read_csv(ZCTA_TRACT_CSV, dtype={"zcta": str, "tract_fips": str})
    return df


def predict_for_zcta(zcta: str, predictions: pd.DataFrame, lookup: pd.DataFrame) -> dict | None:
    """Look up CKD prevalence for a 5-digit ZCTA via the ZCTA->tract crosswalk.
    Returns area-weighted mean across constituent tracts."""
    zcta = str(zcta).zfill(5)
    rows = lookup[lookup["zcta"] == zcta]
    if rows.empty:
        return None
    merged = rows.merge(predictions, on="tract_fips", how="inner")
    if merged.empty:
        return None
    if "arealand_part_m2" in merged.columns and merged["arealand_part_m2"].notna().any():
        w = merged["arealand_part_m2"].fillna(0)
        if w.sum() > 0:
            pred = float(np.average(merged["ckd_predicted_pct"], weights=w))
            obs = float(np.average(merged["ckd_pct"], weights=w))
        else:
            pred = float(merged["ckd_predicted_pct"].mean())
            obs = float(merged["ckd_pct"].mean())
    else:
        pred = float(merged["ckd_predicted_pct"].mean())
        obs = float(merged["ckd_pct"].mean())
    return {
        "zcta": zcta,
        "predicted_pct": pred,
        "observed_pct": obs,
        "n_tracts": int(len(merged)),
        "tracts": merged["tract_fips"].tolist()[:5],
        "states": merged["state_fips"].unique().tolist(),
        "regions": merged["census_region"].unique().tolist(),
        "adi_quintile_mean": float(merged["adi_quintile"].mean()),
    }


def render_gauge(value: float, vmin: float, vmax: float, title: str, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        gauge={
            "axis": {"range": [vmin, vmax]},
            "bar": {"color": color},
            "steps": [
                {"range": [vmin, (vmin + vmax) / 3], "color": "#e8f5e9"},
                {"range": [(vmin + vmax) / 3, 2 * (vmin + vmax) / 3], "color": "#fff8e1"},
                {"range": [2 * (vmin + vmax) / 3, vmax], "color": "#ffebee"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=10, r=10))
    return fig


# ============================================================ HEADER
st.markdown(
    """
    <h1 style='text-align:center;margin-bottom:0;'>CKD Two-Stage Surveillance + Screening</h1>
    <p style='text-align:center;color:#666;margin-top:0;'>
        Manuscript demo for JMIR Public Health &amp; Surveillance submission.<br/>
        All data: open. All code: reproducible. No synthetic data.
    </p>
    """,
    unsafe_allow_html=True,
)

# ============================================================ SIDEBAR
st.sidebar.title("Navigation")
page = st.sidebar.radio("Page",
                        ["🏥 Stage 1 — ZIP-level prevalence",
                         "🧬 Stage 2 — Patient screening",
                         "📊 Model performance",
                         "📚 About"])

# Load resources
ecol_bundle, ecol_metrics = load_ecological()
nhanes_bundle, nhanes_metrics = load_nhanes()


# ============================================================ STAGE 1
if page.startswith("🏥"):
    st.subheader("Stage 1 — Tract-level CKD prevalence by ZIP code")
    st.caption("Looks up each constituent census tract for the entered ZCTA, "
               "averages the model-predicted prevalence (area-weighted), and "
               "compares to the CDC PLACES ground truth for the same tracts.")

    zcta_input = st.text_input("Enter a 5-digit ZIP code / ZCTA (e.g. 60601, 90210, 10001):",
                               value="60601")

    if zcta_input:
        predictions = load_predictions()
        lookup = load_zcta_lookup()
        result = predict_for_zcta(zcta_input, predictions, lookup)
        if result is None:
            st.warning(f"No matched tracts for ZCTA {zcta_input.zfill(5)}. Try a different ZIP.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted CKD prevalence",
                      f"{result['predicted_pct']:.2f}%",
                      delta=f"vs CDC obs {result['observed_pct']:.2f}%")
            c2.metric("Constituent tracts", result["n_tracts"])
            c3.metric("Mean ADI quintile",
                      f"{result['adi_quintile_mean']:.2f}",
                      help="1 = least deprived, 5 = most")

            st.plotly_chart(
                render_gauge(result["predicted_pct"], 0, 15,
                             "Predicted CKD prevalence (%)", "#1f77b4"),
                use_container_width=True,
            )

            st.write(f"**State(s):** {', '.join(result['states'])}    "
                     f"**Region(s):** {', '.join(filter(None, result['regions']))}    "
                     f"**Sample tracts:** {', '.join(result['tracts'])}")

            with st.expander("Underlying model details"):
                st.write(f"- Trained on **{ecol_metrics['n_tracts_trained']:,}** "
                         f"matched ADI ∩ CDC PLACES tracts")
                st.write(f"- Cross-validated R² = "
                         f"**{ecol_metrics['random_5fold_cv']['R2_mean']:.3f}** "
                         f"± {ecol_metrics['random_5fold_cv']['R2_std']:.3f}")
                st.write(f"- MAE = "
                         f"**{ecol_metrics['random_5fold_cv']['MAE_mean']:.3f} "
                         f"percentage points**")


# ============================================================ STAGE 2
elif page.startswith("🧬"):
    st.subheader("Stage 2 — Patient-level CKD screening")
    st.caption("XGBoost classifier trained on NHANES 2017-2022 (n=15,150 adults). "
               "Uses NO kidney biomarkers; intended as a low-cost screening triage "
               "where creatinine and urine albumin are not yet measured.")

    with st.form("nhanes_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", 18, 100, 55)
            sex = st.selectbox("Sex", ["Female", "Male"])
            race = st.selectbox("Race / Ethnicity",
                                ["Non-Hispanic White", "Non-Hispanic Black",
                                 "Mexican American", "Other Hispanic",
                                 "Non-Hispanic Asian", "Other / Multi-racial"])
        with c2:
            bmi = st.number_input("BMI (kg/m²)", 10.0, 60.0, 28.0, 0.1)
            sbp = st.number_input("Mean SBP (mmHg)", 70, 220, 130)
            dbp = st.number_input("Mean DBP (mmHg)", 40, 130, 78)
            ipr = st.number_input("Income-to-poverty ratio", 0.0, 5.0, 2.0, 0.1)
        with c3:
            htn = st.checkbox("Diagnosed hypertension", value=True)
            dm = st.checkbox("Diagnosed diabetes")
            chf = st.checkbox("Heart failure")
            stroke = st.checkbox("Stroke history")
        submit = st.form_submit_button("Compute CKD probability")

    if submit:
        race_map = {"Mexican American": 1, "Other Hispanic": 2,
                    "Non-Hispanic White": 3, "Non-Hispanic Black": 4,
                    "Non-Hispanic Asian": 6, "Other / Multi-racial": 7}
        x = pd.DataFrame([{
            "RIDAGEYR": age,
            "RIAGENDR": 1 if sex == "Male" else 2,
            "RIDRETH3": race_map[race],
            "INDFMPIR": ipr,
            "BMXBMI": bmi,
            "mean_sbp": sbp,
            "mean_dbp": dbp,
            "BPQ020": 1 if htn else 0,
            "DIQ010": 1 if dm else 0,
            "MCQ160B": 1 if chf else 0,
            "MCQ160F": 1 if stroke else 0,
        }])[nhanes_bundle["feature_names"]]

        proba = float(nhanes_bundle["model"].predict_proba(x)[0, 1])
        threshold_screen = nhanes_metrics["threshold_at_90_spec"]
        flag = proba >= threshold_screen

        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(
                render_gauge(proba * 100, 0, 100,
                             "CKD probability (%)",
                             "#d62728" if flag else "#1f77b4"),
                use_container_width=True,
            )
        with c2:
            st.metric("Predicted CKD probability", f"{proba * 100:.1f}%")
            st.metric("Screening threshold (90% specificity)",
                      f"{threshold_screen * 100:.1f}%")
            if flag:
                st.error("✓ Above the 90%-specificity threshold — "
                         "recommend confirmatory creatinine + UACR testing.")
            else:
                st.success("Below screening threshold — routine follow-up.")
            st.caption("Note: this is a research artefact, not clinical "
                       "decision support. Definitive CKD diagnosis requires "
                       "lab measurement.")


# ============================================================ PERFORMANCE
elif page.startswith("📊"):
    st.subheader("Model performance")
    st.markdown("### Stage 1 — Ecological model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tracts trained",
              f"{ecol_metrics['n_tracts_trained']:,}")
    c2.metric("CV R²",
              f"{ecol_metrics['random_5fold_cv']['R2_mean']:.3f}")
    c3.metric("CV MAE (pp)",
              f"{ecol_metrics['random_5fold_cv']['MAE_mean']:.3f}")
    c4.metric("Calibration slope range (LORO)",
              f"{min(r['calibration_slope'] for r in ecol_metrics['leave_one_region_out_cv']):.2f}–"
              f"{max(r['calibration_slope'] for r in ecol_metrics['leave_one_region_out_cv']):.2f}")

    loro_df = pd.DataFrame(ecol_metrics["leave_one_region_out_cv"])
    st.write("**Leave-one-Census-region-out CV**")
    st.dataframe(loro_df.set_index("held_out_region")
                 [["n_train", "n_test", "R2", "MAE", "calibration_slope"]],
                 use_container_width=True)

    st.markdown("### Stage 2 — NHANES patient classifier")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", f"{nhanes_metrics['n_total']:,}")
    c2.metric("AUROC (OOF)", f"{nhanes_metrics['oof_AUROC']:.3f}")
    c3.metric("AUPRC (OOF)", f"{nhanes_metrics['oof_AUPRC']:.3f}")
    c4.metric("Brier score", f"{nhanes_metrics['oof_Brier']:.3f}")

    st.write("**Subgroup AUROC**")
    st.dataframe(pd.DataFrame(nhanes_metrics["subgroup_AUROC"]),
                 use_container_width=True)

    st.image(str(ROOT / "results" / "figures" / "performance_matrix.png"),
             caption="Ecological-model performance matrix")
    st.image(str(ROOT / "results" / "figures" / "nhanes_performance_matrix.png"),
             caption="NHANES classifier performance matrix")


# ============================================================ ABOUT
else:
    st.subheader("About")
    st.markdown(
        """
        This demo accompanies the manuscript

        > **An Open Two-Stage Surveillance and Risk-Stratification Pipeline for
        > Chronic Kidney Disease: Linking Neighborhood Deprivation, CDC PLACES,
        > and NHANES**

        targeted for **JMIR Public Health & Surveillance**.

        **Data sources (all public, no synthetic data):**

        - University of Wisconsin Neighborhood Atlas — Area Deprivation Index
          2020 national, 12-digit FIPS (n=242,335 block groups).
          Citation: Kind & Buckingham, NEJM 2018.
        - CDC PLACES 2022 release — census-tract chronic kidney disease
          prevalence (Socrata `shc3-fzig`).
        - U.S. Census Bureau 2020 — ZCTA ↔ tract relationship file.
        - CDC NHANES 2017-2020 pre-pandemic combined release + 2021-2022 cycle.
        - USRDS 2023 Annual Data Report (reference statistics).

        **Reproducibility:**

        - All code at `src/`.  Random seed 42.  `make all` from repo root
          regenerates everything from the raw downloads.
        - Manuscript text auto-populated from `results/metrics/*.json` —
          numbers cannot drift from the model output.

        **Disclaimer:**

        Research artefact only.  Not a medical device.  Not clinical decision
        support.  Definitive CKD diagnosis requires creatinine and urine
        albumin measurement per KDIGO 2024 guidelines.
        """
    )
