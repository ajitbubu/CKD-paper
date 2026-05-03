"""
Comprehensive evaluation of the NHANES patient-level CKD classifier.

Inputs:
  - models/ckd_nhanes_classifier.pkl
  - results/nhanes_oof_predictions.csv

Outputs:
  - results/metrics/nhanes_performance.json   complete metrics
  - results/figures/nhanes_roc.png            ROC curve
  - results/figures/nhanes_pr.png             precision-recall curve
  - results/figures/nhanes_calibration.png    calibration (deciles + isotonic)
  - results/figures/nhanes_subgroup_auroc.png subgroup performance
  - results/figures/nhanes_decision_curve.png decision curve analysis
  - results/figures/nhanes_performance_matrix.png  scorecard
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score, brier_score_loss,
    precision_recall_curve, roc_auc_score, roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PKL = ROOT / "models" / "ckd_nhanes_classifier.pkl"
OOF_CSV = ROOT / "results" / "nhanes_oof_predictions.csv"

FIG_DIR = ROOT / "results" / "figures"
MET_DIR = ROOT / "results" / "metrics"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MET_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")

# NHANES race/ethnicity codes (RIDRETH3)
RACE_LABELS = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other / Multi-racial",
}
SEX_LABELS = {1: "Male", 2: "Female"}


def age_band(age: float) -> str:
    if age < 50:
        return "<50"
    if age < 65:
        return "50-64"
    return "≥65"


def fig_roc(y, p, w):
    fpr, tpr, _ = roc_curve(y, p, sample_weight=w)
    auroc = roc_auc_score(y, p, sample_weight=w)
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.plot(fpr, tpr, color="#1f77b4", linewidth=2.5, label=f"NHANES classifier (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=2, label="Chance")
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC — NHANES patient-level CKD classifier")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nhanes_roc.png", dpi=150)
    plt.close(fig)


def fig_pr(y, p, w):
    pr, rec, _ = precision_recall_curve(y, p, sample_weight=w)
    auprc = average_precision_score(y, p, sample_weight=w)
    prev = np.average(y, weights=w)
    fig, ax = plt.subplots(figsize=(8, 7.5))
    ax.plot(rec, pr, color="#ff7f0e", linewidth=2.5,
            label=f"NHANES classifier (AUPRC = {auprc:.3f})")
    ax.axhline(prev, color="r", linestyle="--", linewidth=2,
               label=f"Prevalence = {prev:.3f}")
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (positive predictive value)")
    ax.set_title("Precision-Recall — NHANES classifier")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nhanes_pr.png", dpi=150)
    plt.close(fig)


def fig_calibration(y, p, w):
    df = pd.DataFrame({"y": y, "p": p, "w": w})
    df["decile"] = pd.qcut(df["p"], 10, labels=False, duplicates="drop")
    g = df.groupby("decile").apply(
        lambda x: pd.Series({
            "p_mean": np.average(x["p"], weights=x["w"]),
            "y_mean": np.average(x["y"], weights=x["w"]),
            "n": len(x),
            "se": np.sqrt(np.average((x["y"] - np.average(x["y"], weights=x["w"])) ** 2,
                                     weights=x["w"]) / len(x)),
        }), include_groups=False
    ).reset_index()
    slope, intercept = np.polyfit(g["p_mean"], g["y_mean"], 1)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.errorbar(g["p_mean"], g["y_mean"], yerr=1.96 * g["se"],
                fmt="o", markersize=10, color="#1f77b4", capsize=5,
                linewidth=2, label="Decile mean ± 95% CI")
    lo = min(g["p_mean"].min(), g["y_mean"].min()) * 0.9
    hi = max(g["p_mean"].max(), g["y_mean"].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect calibration")
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, slope * xs + intercept, "g-", linewidth=2,
            label=f"Fit: slope={slope:.3f}, intercept={intercept:.3f}")
    ax.set_xlabel("Predicted CKD probability — decile mean")
    ax.set_ylabel("Observed CKD frequency (survey-weighted)")
    ax.set_title("Calibration — NHANES classifier (10 deciles)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nhanes_calibration.png", dpi=150)
    plt.close(fig)
    return slope, intercept


def fig_subgroup(df: pd.DataFrame):
    """AUROC by age band, sex, race/ethnicity."""
    rows = []
    df = df.copy()
    df["age_band"] = df["RIDAGEYR"].map(age_band)
    df["sex"] = df["RIAGENDR"].map(SEX_LABELS)
    df["race"] = df["RIDRETH3"].map(RACE_LABELS)

    for col, label in [("age_band", "Age"), ("sex", "Sex"), ("race", "Race/Ethnicity")]:
        for level, sub in df.groupby(col, observed=True):
            if sub["ckd_true"].sum() < 10:  # need cases
                continue
            try:
                auroc = roc_auc_score(sub["ckd_true"], sub["ckd_proba_oof"],
                                      sample_weight=sub["weight"])
            except ValueError:
                continue
            rows.append({"group": label, "level": str(level), "n": len(sub),
                         "n_pos": int(sub["ckd_true"].sum()), "AUROC": float(auroc)})
    sub_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    levels = sub_df["group"] + ":  " + sub_df["level"]
    bars = ax.barh(levels, sub_df["AUROC"],
                   color=[{"Age": "#1f77b4", "Sex": "#ff7f0e",
                           "Race/Ethnicity": "#2ca02c"}[g] for g in sub_df["group"]],
                   edgecolor="black")
    overall_auroc = roc_auc_score(df["ckd_true"], df["ckd_proba_oof"],
                                  sample_weight=df["weight"])
    ax.axvline(overall_auroc, color="red", linestyle="--", linewidth=2,
               label=f"Overall AUROC = {overall_auroc:.3f}")
    for b, v, n in zip(bars, sub_df["AUROC"], sub_df["n"]):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2,
                f"{v:.3f}  (n={n:,})", va="center", fontsize=11)
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Survey-weighted AUROC")
    ax.set_title("Subgroup performance (NHANES classifier)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nhanes_subgroup_auroc.png", dpi=150)
    plt.close(fig)
    return sub_df


def fig_decision_curve(y, p, w):
    """Vickers 2006 net benefit at thresholds 1%-50%."""
    thresholds = np.arange(0.01, 0.51, 0.01)
    n = len(y)
    prev = np.average(y, weights=w)
    nb_model, nb_all, nb_none = [], [], []
    for t in thresholds:
        pred_pos = p >= t
        tp = float(np.sum(w[pred_pos & (y == 1)]))
        fp = float(np.sum(w[pred_pos & (y == 0)]))
        n_w = float(np.sum(w))
        nb = (tp / n_w) - (fp / n_w) * (t / (1 - t))
        nb_model.append(nb)
        # treat-all benefit
        nb_all.append(prev - (1 - prev) * (t / (1 - t)))
        nb_none.append(0.0)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.plot(thresholds, nb_model, color="#1f77b4", linewidth=2.5, label="Model")
    ax.plot(thresholds, nb_all, color="#2ca02c", linewidth=2, label="Treat all")
    ax.plot(thresholds, nb_none, color="red", linestyle="--", linewidth=2, label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis (Vickers 2006)")
    ax.set_ylim(min(nb_all + nb_model) * 1.1, max(nb_model + nb_all + [0.05]) * 1.1)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nhanes_decision_curve.png", dpi=150)
    plt.close(fig)


def fig_performance_matrix(metrics: dict, sub_df: pd.DataFrame):
    rows = [
        ("n (training cohort)", f"{metrics['n_total']:,}"),
        ("CKD positive (unweighted)", f"{metrics['n_ckd_positive']:,}  ({metrics['ckd_prevalence_unweighted'] * 100:.2f}%)"),
        ("CKD prevalence (survey-weighted)", f"{metrics['ckd_prevalence_weighted'] * 100:.2f}%"),
        ("OOF AUROC (5-fold CV, weighted)", f"{metrics['oof_AUROC']:.4f}"),
        ("OOF AUPRC (5-fold CV, weighted)", f"{metrics['oof_AUPRC']:.4f}"),
        ("OOF Brier score (weighted)", f"{metrics['oof_Brier']:.4f}"),
        ("Calibration slope (deciles)", f"{metrics['calibration_slope']:.4f}"),
        ("Calibration intercept (deciles)", f"{metrics['calibration_intercept']:.4f}"),
        ("Sensitivity at 90% specificity", f"{metrics['sens_at_90_spec'] * 100:.1f}%"),
        ("Threshold at 90% specificity", f"{metrics['threshold_at_90_spec']:.3f}"),
        ("AUROC range across folds", f"{metrics['auroc_min']:.3f}–{metrics['auroc_max']:.3f}"),
        ("AUROC across subgroups (min–max)",
         f"{sub_df['AUROC'].min():.3f}–{sub_df['AUROC'].max():.3f}"),
    ]
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                     loc="center", cellLoc="left", colLoc="left",
                     colWidths=[0.55, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 1.85)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f77b4")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f7f7f7" if r % 2 else "white")
    ax.set_title("NHANES patient-level CKD classifier — performance matrix\n"
                 "(Real NHANES 2017-2022, no kidney biomarkers as features)",
                 pad=18, fontsize=18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nhanes_performance_matrix.png", dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(OOF_CSV)
    print(f"Loaded {len(df):,} OOF predictions")
    y = df["ckd_true"].values
    p = df["ckd_proba_oof"].values
    w = df["weight"].values

    fig_roc(y, p, w)
    fig_pr(y, p, w)
    cal_slope, cal_intercept = fig_calibration(y, p, w)
    sub_df = fig_subgroup(df)
    fig_decision_curve(y, p, w)

    # Aggregate metrics
    fold_metrics = json.loads((MET_DIR / "nhanes_training_metrics.json").read_text())["fold_metrics"]
    aurocs = [f["AUROC"] for f in fold_metrics]
    metrics = {
        "n_total": int(len(y)),
        "n_ckd_positive": int(y.sum()),
        "ckd_prevalence_unweighted": float(y.mean()),
        "ckd_prevalence_weighted": float(np.average(y, weights=w)),
        "oof_AUROC": float(roc_auc_score(y, p, sample_weight=w)),
        "oof_AUPRC": float(average_precision_score(y, p, sample_weight=w)),
        "oof_Brier": float(brier_score_loss(y, p, sample_weight=w)),
        "calibration_slope": float(cal_slope),
        "calibration_intercept": float(cal_intercept),
        "auroc_min": float(min(aurocs)),
        "auroc_max": float(max(aurocs)),
        "auroc_per_fold": aurocs,
        "subgroup_AUROC": sub_df.to_dict(orient="records"),
    }
    # Sensitivity at 90% spec
    fpr, tpr, thr = roc_curve(y, p, sample_weight=w)
    idx = int(np.argmin(np.abs((1 - fpr) - 0.90)))
    metrics["sens_at_90_spec"] = float(tpr[idx])
    metrics["threshold_at_90_spec"] = float(thr[idx])

    fig_performance_matrix(metrics, sub_df)

    (MET_DIR / "nhanes_performance.json").write_text(json.dumps(metrics, indent=2))
    sub_df.to_csv(MET_DIR / "nhanes_subgroup.csv", index=False)
    print("\nWrote NHANES evaluation outputs:")
    for f in sorted(FIG_DIR.glob("nhanes_*.png")):
        print(" -", f.relative_to(ROOT))
    print(" -", (MET_DIR / "nhanes_performance.json").relative_to(ROOT))
    print(" -", (MET_DIR / "nhanes_subgroup.csv").relative_to(ROOT))

    print("\n=== Subgroup AUROC ===")
    print(sub_df.to_string(index=False))


if __name__ == "__main__":
    main()
