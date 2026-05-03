"""Clinical formulas for CKD."""
import numpy as np
import pandas as pd


def egfr_ckd_epi_2021(scr_mg_dl: pd.Series, age_yr: pd.Series, sex_male: pd.Series) -> pd.Series:
    """
    eGFR by 2021 CKD-EPI creatinine equation (no race coefficient).

    Reference: Inker LA et al. New creatinine- and cystatin C-based equations
    to estimate GFR without race. NEJM 2021;385:1737-1749.

    Parameters
    ----------
    scr_mg_dl : serum creatinine in mg/dL
    age_yr    : age in years
    sex_male  : boolean / 0-1 vector; True/1 for male, False/0 for female

    Returns
    -------
    eGFR (mL/min/1.73 m^2)
    """
    scr = pd.to_numeric(scr_mg_dl, errors="coerce").astype(float)
    age = pd.to_numeric(age_yr, errors="coerce").astype(float)
    male = pd.Series(sex_male).astype(bool).to_numpy()

    kappa = np.where(male, 0.9, 0.7)
    alpha = np.where(male, -0.302, -0.241)
    sex_coef = np.where(male, 1.0, 1.012)

    s_over_k = scr.to_numpy() / kappa
    min_term = np.minimum(s_over_k, 1.0) ** alpha
    max_term = np.maximum(s_over_k, 1.0) ** -1.200
    egfr = 142 * min_term * max_term * (0.9938 ** age.to_numpy()) * sex_coef

    return pd.Series(egfr, index=scr.index)


def uacr(urine_albumin_mg_l: pd.Series, urine_creatinine_mg_dl: pd.Series) -> pd.Series:
    """
    Urine albumin-to-creatinine ratio in mg/g.
    UACR = (albumin mg/L * 100) / (creatinine mg/dL * 1000 / 1000)
         = albumin mg/L / (creatinine mg/dL / 100)
         = albumin mg/L * 100 / creatinine mg/dL  -> divide by 1000? let's be explicit.

    Conventional: UACR (mg/g) = urine_albumin (mg/dL) / urine_creatinine (g/dL)
    NHANES gives albumin in ug/mL == mg/L  and creatinine in mg/dL.

      mg/L  -> mg/dL  : divide by 10
      mg/dL -> g/dL   : divide by 1000
    UACR = (alb_mg_L / 10) / (cr_mg_dL / 1000) = alb_mg_L * 100 / cr_mg_dL
    """
    alb = pd.to_numeric(urine_albumin_mg_l, errors="coerce")
    cr = pd.to_numeric(urine_creatinine_mg_dl, errors="coerce")
    return alb * 100.0 / cr


def ckd_label(egfr: pd.Series, uacr_mg_g: pd.Series) -> pd.Series:
    """
    CKD = eGFR < 60 OR UACR >= 30  (KDIGO 2024 criteria).
    Returns nullable Int64 series; NaN where both inputs are missing.
    """
    egfr_low = egfr < 60
    uacr_high = uacr_mg_g >= 30
    label = (egfr_low | uacr_high).astype("Int64")
    label[egfr.isna() & uacr_mg_g.isna()] = pd.NA
    return label
