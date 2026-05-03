"""
Figures for the real-data tract-level ecological CKD model.

Reads:
  - models/ckd_ecological_model.pkl
  - results/tract_ckd_predictions.csv
  - results/training_metrics_ecological.json

Writes (replacing the stale ZCTA-anchor figures):
  - results/figures/predicted_vs_observed.png        (parity plot, real ground truth)
  - results/figures/calibration_plot.png             (deciles)
  - results/figures/leave_one_region_out.png         (per-region R²/MAE/slope)
  - results/figures/sensitivity_to_capacity.png      (hyperparameter sweep)
  - results/figures/prevalence_by_adi_quintile.png   (predicted vs observed by ADI)
  - results/figures/feature_importance.png           (3 ADI features)
  - results/figures/performance_matrix.png           (full scorecard table)
  - results/figures/risk_distribution.png            (distribution by quintile)
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
MODEL_PKL = ROOT / "models" / "ckd_ecological_model.pkl"
PRED_CSV = ROOT / "results" / "tract_ckd_predictions.csv"
METRICS_JSON = ROOT / "results" / "training_metrics_ecological.json"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
QUINTILE_PALETTE = sns.color_palette("RdYlBu_r", 5)


def load():
    bundle = joblib.load(MODEL_PKL)
    pred = pd.read_csv(PRED_CSV, dtype={"tract_fips": str, "state_fips": str})
    metrics = json.loads(METRICS_JSON.read_text())
    return bundle, pred, metrics


def fig_parity(pred: pd.DataFrame, metrics: dict):
    fig, ax = plt.subplots(figsize=(8.5, 8))
    s = pred.sample(min(8000, len(pred)), random_state=0)
    ax.scatter(s["ckd_pct"], s["ckd_predicted_pct"], s=8, alpha=0.4, color="#1f77b4")
    lo = min(pred["ckd_pct"].min(), pred["ckd_predicted_pct"].min())
    hi = max(pred["ckd_pct"].max(), pred["ckd_predicted_pct"].max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect")
    r2 = metrics["random_5fold_cv"]["R2_mean"]
    mae = metrics["random_5fold_cv"]["MAE_mean"]
    ax.set_xlabel("CDC PLACES observed CKD prevalence (%)")
    ax.set_ylabel("Model predicted CKD prevalence (%)")
    ax.set_title(f"Tract-level parity vs CDC PLACES ground truth\nCV R² = {r2:.3f}    MAE = {mae:.2f} pp    n = {len(pred):,} tracts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "predicted_vs_observed.png", dpi=150)
    plt.close(fig)


def fig_calibration(pred: pd.DataFrame):
    pred = pred.copy()
    pred["decile"] = pd.qcut(pred["ckd_predicted_pct"], 10, labels=False, duplicates="drop")
    g = pred.groupby("decile").agg(
        pred_mean=("ckd_predicted_pct", "mean"),
        obs_mean=("ckd_pct", "mean"),
        obs_se=("ckd_pct", lambda x: x.std() / np.sqrt(len(x))),
        n=("ckd_pct", "size"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.errorbar(g["pred_mean"], g["obs_mean"], yerr=1.96 * g["obs_se"],
                fmt="o", color="#1f77b4", markersize=10, capsize=5,
                linewidth=2, label="Decile mean ± 95% CI")
    lo = min(g["pred_mean"].min(), g["obs_mean"].min()) * 0.95
    hi = max(g["pred_mean"].max(), g["obs_mean"].max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="Perfect calibration")
    # Linear fit
    slope, intercept = np.polyfit(g["pred_mean"], g["obs_mean"], 1)
    xs = np.linspace(lo, hi, 50)
    ax.plot(xs, slope * xs + intercept, "g-", linewidth=2,
            label=f"Fit: slope={slope:.3f}, intercept={intercept:.3f}")
    ax.set_xlabel("Predicted CKD prevalence (%) — decile mean")
    ax.set_ylabel("Observed CKD prevalence (%) — CDC PLACES")
    ax.set_title("Calibration plot (10-decile binning, all tracts)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration_plot.png", dpi=150)
    plt.close(fig)


def fig_leave_one_region_out(metrics: dict):
    df = pd.DataFrame(metrics["leave_one_region_out_cv"])
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    regions = df["held_out_region"].tolist()
    region_palette = sns.color_palette("Set2", len(regions))

    bars = axes[0].bar(regions, df["R2"], color=region_palette, edgecolor="black")
    for b, v in zip(bars, df["R2"]):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                     ha="center", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("R²")
    axes[0].set_title("Held-out region R²")
    axes[0].set_ylim(0, max(df["R2"]) * 1.25)

    bars = axes[1].bar(regions, df["MAE"], color=region_palette, edgecolor="black")
    for b, v in zip(bars, df["MAE"]):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                     ha="center", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("MAE (percentage points)")
    axes[1].set_title("Held-out region MAE")
    axes[1].set_ylim(0, max(df["MAE"]) * 1.25)

    bars = axes[2].bar(regions, df["calibration_slope"], color=region_palette,
                       edgecolor="black")
    axes[2].axhline(1.0, color="red", linestyle="--", linewidth=2, label="Perfect (1.0)")
    for b, v in zip(bars, df["calibration_slope"]):
        axes[2].text(b.get_x() + b.get_width() / 2, v + 0.04, f"{v:.3f}",
                     ha="center", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Calibration slope")
    axes[2].set_title("Held-out region calibration slope")
    axes[2].set_ylim(0, max(df["calibration_slope"]) * 1.3)
    axes[2].legend(loc="lower right")

    fig.suptitle("Leave-one-Census-region-out cross-validation", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "leave_one_region_out.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_sensitivity(metrics: dict):
    df = pd.DataFrame(metrics["sensitivity_to_capacity"])
    df["label"] = df.apply(
        lambda r: f"n={int(r.n_estimators)}\nd={int(r.max_depth)}\nlr={r.learning_rate:.2f}",
        axis=1)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(df["label"], df["CV_R2"], color="#2ca02c", edgecolor="black", alpha=0.85)
    for b, v in zip(bars, df["CV_R2"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.4f}",
                ha="center", fontsize=11)
    ax.set_ylabel("Cross-validated R² (5-fold)")
    ax.set_xlabel("Model capacity (n_estimators, depth, learning_rate)")
    ax.set_title("Sensitivity of R² to XGBoost capacity")
    ax.set_ylim(min(df["CV_R2"]) * 0.985, max(df["CV_R2"]) * 1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sensitivity_to_capacity.png", dpi=150)
    plt.close(fig)


def fig_prevalence_by_quintile(pred: pd.DataFrame):
    pred = pred.copy()
    pred["adi_quintile_int"] = pred["adi_quintile"].round().clip(1, 5).astype(int)
    g = pred.groupby("adi_quintile_int").agg(
        observed_mean=("ckd_pct", "mean"),
        observed_std=("ckd_pct", "std"),
        predicted_mean=("ckd_predicted_pct", "mean"),
        predicted_std=("ckd_predicted_pct", "std"),
        n=("tract_fips", "size"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(1, 6)
    w = 0.38
    bars1 = ax.bar(x - w / 2, g["observed_mean"], w, yerr=g["observed_std"],
                   color="#1f77b4", edgecolor="black", capsize=5,
                   label="CDC PLACES observed")
    bars2 = ax.bar(x + w / 2, g["predicted_mean"], w, yerr=g["predicted_std"],
                   color="#ff7f0e", edgecolor="black", capsize=5,
                   label="Model predicted")
    for bars in (bars1, bars2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05,
                    f"{b.get_height():.2f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xlabel("ADI quintile (1 = least deprived, 5 = most deprived)")
    ax.set_ylabel("CKD prevalence (%)")
    ax.set_title("Predicted vs observed CKD prevalence by ADI quintile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "prevalence_by_adi_quintile.png", dpi=150)
    plt.close(fig)


def fig_feature_importance(bundle: dict):
    model = bundle["model"]
    names = bundle["feature_names"]
    importance = model.feature_importances_
    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh([names[i] for i in order], [importance[i] for i in order],
                   color=sns.color_palette("viridis", len(names)), edgecolor="black")
    for b, v in zip(bars, [importance[i] for i in order]):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center")
    ax.set_xlabel("XGBoost gain importance")
    ax.set_title("Feature importance (real CDC PLACES outcome)")
    ax.set_xlim(0, max(importance) * 1.15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
    plt.close(fig)


def fig_performance_matrix(metrics: dict):
    rows = [
        ("Tracts trained on", f"{metrics['n_tracts_trained']:,}"),
        ("Outcome mean (CDC PLACES)", f"{metrics['outcome_mean_pct']:.2f}%"),
        ("Outcome range", f"{metrics['outcome_min_pct']:.2f}% — {metrics['outcome_max_pct']:.2f}%"),
        ("CV R² (5-fold random, mean ± SD)",
         f"{metrics['random_5fold_cv']['R2_mean']:.4f} ± {metrics['random_5fold_cv']['R2_std']:.4f}"),
        ("CV MAE (percentage points)",
         f"{metrics['random_5fold_cv']['MAE_mean']:.4f} ± {metrics['random_5fold_cv']['MAE_std']:.4f}"),
        ("Train-fit R²", f"{metrics['train_full_fit']['R2']:.4f}"),
        ("Train-fit MAE", f"{metrics['train_full_fit']['MAE']:.4f}"),
    ]
    for r in metrics["leave_one_region_out_cv"]:
        rows.append((f"LORO held-out {r['held_out_region']:<10}: R² | MAE | slope",
                    f"{r['R2']:.3f} | {r['MAE']:.3f} | {r['calibration_slope']:.3f}"))

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=["Metric", "Value"],
                     loc="center", cellLoc="left", colLoc="left",
                     colWidths=[0.55, 0.45])
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 1.9)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1f77b4")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#f7f7f7" if r % 2 else "white")
    ax.set_title("Ecological CKD model — performance matrix\n(Real CDC PLACES ground truth, 60,445 tracts)",
                 pad=18, fontsize=18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "performance_matrix.png", dpi=150)
    plt.close(fig)


def fig_risk_distribution(pred: pd.DataFrame):
    pred = pred.copy()
    pred["adi_quintile_int"] = pred["adi_quintile"].round().clip(1, 5).astype(int)
    fig, ax = plt.subplots(figsize=(11, 6))
    for q in sorted(pred["adi_quintile_int"].unique()):
        sub = pred[pred["adi_quintile_int"] == q]["ckd_predicted_pct"]
        ax.hist(sub, bins=40, alpha=0.55, label=f"Q{q}",
                color=QUINTILE_PALETTE[q - 1], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Predicted CKD prevalence (%)")
    ax.set_ylabel("Number of tracts")
    ax.set_title("Distribution of predicted CKD prevalence by ADI quintile")
    ax.legend(title="ADI Quintile")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "risk_distribution.png", dpi=150)
    plt.close(fig)


def fig_residuals(pred: pd.DataFrame):
    res = pred["ckd_predicted_pct"] - pred["ckd_pct"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(res, bins=60, color="#2ca02c", edgecolor="black", alpha=0.85)
    axes[0].axvline(0, color="red", linestyle="--")
    axes[0].set_xlabel("Residual (predicted − observed, percentage points)")
    axes[0].set_ylabel("Number of tracts")
    axes[0].set_title("Residual distribution")
    s = pred.sample(min(8000, len(pred)), random_state=0)
    axes[1].scatter(s["ckd_predicted_pct"], s["ckd_predicted_pct"] - s["ckd_pct"],
                    s=6, alpha=0.35, color="#ff7f0e")
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Predicted prevalence (%)")
    axes[1].set_ylabel("Residual (pp)")
    axes[1].set_title("Residuals vs predicted")
    fig.suptitle(f"Residual diagnostics  |  Bias = {res.mean():+.4f} pp  |  SD = {res.std():.3f}",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    bundle, pred, metrics = load()

    # Remove stale ZCTA-anchor figures
    for stale in ["adi_rank_vs_risk.png", "cv_folds.png", "per_quintile_metrics.png",
                  "top_bottom_zctas.png", "training_metrics_card.png"]:
        p = FIG_DIR / stale
        if p.exists():
            p.unlink()

    fig_parity(pred, metrics)
    fig_calibration(pred)
    fig_leave_one_region_out(metrics)
    fig_sensitivity(metrics)
    fig_prevalence_by_quintile(pred)
    fig_feature_importance(bundle)
    fig_performance_matrix(metrics)
    fig_risk_distribution(pred)
    fig_residuals(pred)

    print("Wrote ecological figures to:", FIG_DIR)
    for f in sorted(FIG_DIR.glob("*.png")):
        print(" -", f.name)


if __name__ == "__main__":
    main()
