"""
Fetch NHANES 2017-2023 kidney panel + computed eGFR/UACR/CKD label.

NHANES cycles and CDC file suffixes:
  2017-2018         _J         (standard cycle)
  2017-2020 (pre-pandemic)  P_*  (combined release; replaces 2019-2020 standalone)
  2021-2022         _L         (replacement for 2019-2020 standalone)
  2023-2024         _M         (newest cycle, may be partial)

Files needed per cycle:
  DEMO_*       SEQN, RIDAGEYR, RIAGENDR, RIDRETH3, INDFMPIR, WTMEC2YR, SDMVPSU, SDMVSTRA
  BMX_*        BMXBMI
  BIOPRO_*     LBXSCR  (serum creatinine, mg/dL)
  ALB_CR_*     URXUMA, URXUCR  (urine albumin, urine creatinine)
  BPQ_*        BPQ020  (told you have HTN)
  DIQ_*        DIQ010  (told you have diabetes)
  MCQ_*        MCQ160B (heart failure), MCQ160F (stroke)
  BPXO_*       BPXOSY1/2/3, BPXODI1/2/3  (oscillometric BP, 2017+)

Output: data/nhanes/nhanes_kidney_panel_2017_2023.csv
"""
import io
import json
import sys
from pathlib import Path

import pandas as pd
import pyreadstat
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.clinical import egfr_ckd_epi_2021, uacr, ckd_label  # noqa: E402

OUT_CSV = ROOT / "data" / "nhanes" / "nhanes_kidney_panel_2017_2023.csv"
META_JSON = ROOT / "data" / "nhanes" / "nhanes_kidney_panel_2017_2023.meta.json"
CACHE_DIR = ROOT / "data" / "nhanes" / "_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CYCLES = [
    # (start_year_for_url, suffix-or-prefix mode, label)
    # URL pattern: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{file}.xpt
    # CDC merges 2017-2018 + partial 2019-2020 into the "2017-2020 pre-pandemic"
    # release (P_*); using both that file AND the 2017-2018 standalone (_J) would
    # double-count participants.  We use only P_* and the 2021-2022 (_L) cycle.
    # The 2023-2024 (_M) cycle is included opportunistically; if not yet
    # published it is silently skipped.
    ("2017", "P_",  "2017-2020 pre-pandemic"),
    ("2021", "_L",  "2021-2022"),
    ("2023", "_M",  "2023-2024"),
]
FILES = ["DEMO", "BMX", "BIOPRO", "ALB_CR", "BPQ", "DIQ", "MCQ", "BPXO"]

KEEP_VARS = {
    "SEQN", "RIDAGEYR", "RIAGENDR", "RIDRETH3", "INDFMPIR",
    "WTMEC2YR", "WTMECPRP",   # pre-pandemic combined cycles use WTMECPRP
    "SDMVPSU", "SDMVSTRA",
    "BMXBMI",
    "LBXSCR",
    "URXUMA", "URXUCR",
    "BPQ020",
    "DIQ010",
    "MCQ160B", "MCQ160F",
    "BPXOSY1", "BPXOSY2", "BPXOSY3",
    "BPXODI1", "BPXODI2", "BPXODI3",
}


def fname(file: str, suffix_mode: str) -> str:
    return f"P_{file}.XPT" if suffix_mode == "P_" else f"{file}{suffix_mode}.XPT"


def download_xpt(year_str: str, file_name: str) -> Path | None:
    cache = CACHE_DIR / f"{year_str}__{file_name}"
    if cache.exists() and cache.stat().st_size > 50_000:  # full XPT, not the HTML stub
        return cache
    url = f"https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year_str}/DataFiles/{file_name}"
    try:
        r = requests.get(url, timeout=180,
                         headers={"User-Agent": "Mozilla/5.0 (research script)"})
    except Exception as e:
        print(f"    NETWORK ERR {url}: {e}")
        return None
    ctype = r.headers.get("content-type", "")
    # 'text/html' = CDC's "Page Not Found" stub disguised as 200 OK
    if r.status_code != 200 or "html" in ctype.lower() or len(r.content) < 50_000:
        return None
    cache.write_bytes(r.content)
    return cache


def load_xpt(path: Path) -> pd.DataFrame:
    # Some NHANES cycles (notably _L / 2021-2022) embed Windows-1252 / Latin-1
    # bytes in label fields, which break the default UTF-8 decoder.
    last_err = None
    for enc in (None, "latin1", "windows-1252"):
        try:
            df, _ = pyreadstat.read_xport(str(path), encoding=enc)
            break
        except UnicodeDecodeError as e:
            last_err = e
            continue
    else:
        raise last_err
    keep = [c for c in df.columns if c in KEEP_VARS]
    return df[keep]


def merge_cycle(cycle_path: str, suffix_mode: str, label: str) -> pd.DataFrame | None:
    print(f"Cycle {label} ({cycle_path}, suffix={suffix_mode!r}):")
    parts = []
    for file in FILES:
        fn = fname(file, suffix_mode)
        path = download_xpt(cycle_path, fn)
        if path is None:
            print(f"  - {fn}: not available")
            continue
        df = load_xpt(path)
        print(f"  - {fn}: {df.shape[0]:,} rows, {df.shape[1]} keep vars")
        parts.append(df)
    if not parts:
        return None
    # First df must contain SEQN; merge on SEQN
    parts = [p for p in parts if "SEQN" in p.columns]
    base = parts[0]
    for p in parts[1:]:
        base = base.merge(p, on="SEQN", how="left", suffixes=("", "_dup"))
        base = base.loc[:, ~base.columns.str.endswith("_dup")]
    base["nhanes_cycle"] = label
    return base


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cycle_dfs = []
    for cycle_path, suffix_mode, label in CYCLES:
        df = merge_cycle(cycle_path, suffix_mode, label)
        if df is not None:
            cycle_dfs.append(df)
    if not cycle_dfs:
        raise SystemExit("No NHANES cycles loaded - check network connectivity to wwwn.cdc.gov")

    full = pd.concat(cycle_dfs, ignore_index=True, sort=False)
    print(f"\nMerged NHANES rows across cycles: {len(full):,}")

    # Compute mean BP from up to 3 readings (oscillometric)
    sbp_cols = [c for c in ["BPXOSY1", "BPXOSY2", "BPXOSY3"] if c in full.columns]
    dbp_cols = [c for c in ["BPXODI1", "BPXODI2", "BPXODI3"] if c in full.columns]
    full["mean_sbp"] = full[sbp_cols].mean(axis=1) if sbp_cols else pd.NA
    full["mean_dbp"] = full[dbp_cols].mean(axis=1) if dbp_cols else pd.NA

    # eGFR + UACR + CKD label
    full["sex_male"] = (full["RIAGENDR"] == 1).astype(int)  # 1=male, 2=female
    full["egfr"] = egfr_ckd_epi_2021(
        full["LBXSCR"], full["RIDAGEYR"], full["sex_male"]
    )
    full["uacr_mg_g"] = uacr(full["URXUMA"], full["URXUCR"])
    full["ckd"] = ckd_label(full["egfr"], full["uacr_mg_g"])

    # Drop adults only (>=18)
    full = full[full["RIDAGEYR"] >= 18].copy()

    # Light cleanup of self-reported binary vars (1 yes / 2 no / 7,9 refused/dk -> NaN)
    for c in ["BPQ020", "DIQ010", "MCQ160B", "MCQ160F"]:
        if c in full.columns:
            full[c] = full[c].where(full[c].isin([1, 2]))
            full[c] = (full[c] == 1).astype(int)

    # Unify survey weights into a single 'survey_weight' column.
    # Pre-pandemic combined cycles (2017-2020 P_) use WTMECPRP (~2-yr equivalent
    # already adjusted by CDC); standalone cycles use WTMEC2YR.
    if "WTMECPRP" in full.columns:
        full["survey_weight"] = full["WTMECPRP"].fillna(full.get("WTMEC2YR"))
    else:
        full["survey_weight"] = full["WTMEC2YR"]

    print(f"  After adult filter: {len(full):,}")
    print(f"  CKD positive: {(full['ckd'] == 1).sum():,} "
          f"({(full['ckd'] == 1).mean() * 100:.2f}%)")
    print(f"  CKD missing labels: {full['ckd'].isna().sum():,}")
    print(f"  With survey_weight:  {full['survey_weight'].notna().sum():,}")

    full.to_csv(OUT_CSV, index=False)
    META_JSON.write_text(json.dumps({
        "cycles": [c[2] for c in CYCLES],
        "n_total": int(len(full)),
        "n_ckd_positive": int((full["ckd"] == 1).sum()),
        "ckd_prevalence": float((full["ckd"] == 1).mean()),
        "ckd_definition": "eGFR < 60 (CKD-EPI 2021, no race) OR UACR >= 30 mg/g",
        "variables_kept": sorted(full.columns.tolist()),
        "source_root": "https://wwwn.cdc.gov/Nchs/Nhanes/",
    }, indent=2, default=str))
    print(f"\nWrote: {OUT_CSV.relative_to(ROOT)}  ({len(full):,} rows)")


if __name__ == "__main__":
    main()
