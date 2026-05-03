"""
Fetch CDC PLACES census-tract CKD prevalence.

CDC PLACES publishes model-based estimates of chronic disease prevalence at the
census tract level. The 2024 and 2025 PLACES tract releases DROPPED chronic
kidney disease as a tract-level measure - the most recent tract-level release
that still includes CKD is the 2022 release (BRFSS 2020 underlying data).

Dataset: shc3-fzig - "PLACES: Census Tract Data (GIS Friendly Format), 2022 release"
Source: chronicdata.cdc.gov, Socrata SODA API.

Output: data/cdc_places/places_ckd_2022_tract.csv
Columns: state_fips, county_fips, tract_fips, ckd_pct, ckd_lci, ckd_uci, total_pop
"""
import json
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "data" / "cdc_places" / "places_ckd_2022_tract.csv"
META_JSON = ROOT / "data" / "cdc_places" / "places_ckd_2022_tract.meta.json"

# 2022 GIS-friendly tract release (latest tract release that still includes CKD)
ENDPOINT_PRIMARY = "https://chronicdata.cdc.gov/resource/shc3-fzig.json"
PAGE_SIZE = 50_000


def fetch_paged(endpoint: str, where: str) -> pd.DataFrame:
    rows = []
    offset = 0
    while True:
        params = {
            "$where": where,
            "$limit": PAGE_SIZE,
            "$offset": offset,
        }
        url = f"{endpoint}?{urlencode(params)}"
        print(f"  GET {url[:120]}{'...' if len(url) > 120 else ''}")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return pd.DataFrame(rows)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """2022 GIS-friendly schema: CKD lives in `kidney_crudeprev` / `kidney_crude95ci`
    columns, with `tractfips` as the 11-digit tract identifier."""
    pct = pd.to_numeric(df["kidney_crudeprev"], errors="coerce")
    ci = df["kidney_crude95ci"].astype(str).str.extract(
        r"\(?\s*([\d.]+)\s*[,\-]\s*([\d.]+)\s*\)?"
    )
    pop_col = "totalpopulation" if "totalpopulation" in df.columns else "totalpop18plus"

    out = pd.DataFrame({
        "tract_fips": df["tractfips"].astype(str).str.zfill(11),
        "ckd_pct": pct,
        "ckd_lci": pd.to_numeric(ci[0], errors="coerce"),
        "ckd_uci": pd.to_numeric(ci[1], errors="coerce"),
        "total_pop": pd.to_numeric(df[pop_col], errors="coerce"),
    })
    out["state_fips"] = out["tract_fips"].str[:2]
    out["county_fips"] = out["tract_fips"].str[:5]
    return out[["state_fips", "county_fips", "tract_fips", "ckd_pct",
                "ckd_lci", "ckd_uci", "total_pop"]].dropna(subset=["ckd_pct"])


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching from {ENDPOINT_PRIMARY}")
    df = fetch_paged(ENDPOINT_PRIMARY, where="kidney_crudeprev IS NOT NULL")

    print(f"Raw rows: {len(df):,}")
    out = normalise(df)
    print(f"Normalised: {len(out):,} tracts; CKD pct range "
          f"{out['ckd_pct'].min():.2f}-{out['ckd_pct'].max():.2f}%, mean {out['ckd_pct'].mean():.2f}%")

    out.to_csv(OUT_CSV, index=False)
    META_JSON.write_text(json.dumps({
        "source": "CDC PLACES 2022 release - census tract level (GIS friendly)",
        "dataset_id": "shc3-fzig",
        "measure": "Chronic kidney disease among adults aged >=18 years (crude prevalence, BRFSS 2020)",
        "n_tracts": int(len(out)),
        "ckd_pct_mean": float(out["ckd_pct"].mean()),
        "ckd_pct_min": float(out["ckd_pct"].min()),
        "ckd_pct_max": float(out["ckd_pct"].max()),
        "note": ("2024/2025 PLACES tract releases dropped CKD; 2022 release is the most "
                 "recent tract-level source with CKD prevalence."),
    }, indent=2))
    print(f"Wrote: {OUT_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
