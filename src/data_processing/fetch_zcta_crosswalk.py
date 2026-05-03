"""
Fetch Census 2020 ZCTA-to-tract relationship file.

Source: U.S. Census Bureau Geography FTP
  https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/
  tab20_zcta520_tract20_natl.txt   (national, pipe-delimited)

The relationship file gives, for every (ZCTA, Tract) pair that overlap,
the population in the overlap. We aggregate to ZCTA-level by population-
weighting the tract-level CDC PLACES CKD estimates and ADI block-group
values.

Output: data/census/zcta_tract_crosswalk_2020.csv
Columns: zcta, tract_fips, pop_in_overlap_2020, area_overlap_landm
"""
import io
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "data" / "census" / "zcta_tract_crosswalk_2020.csv"
META_JSON = ROOT / "data" / "census" / "zcta_tract_crosswalk_2020.meta.json"

URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/"
    "tab20_zcta520_tract20_natl.txt"
)


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    print(f"GET {URL}")
    r = requests.get(URL, timeout=300)
    r.raise_for_status()
    print(f"  bytes: {len(r.content):,}")

    df = pd.read_csv(io.BytesIO(r.content), sep="|", dtype=str)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows: {len(df):,}")

    # Schema (Census 2020 relationship files):
    #   GEOID_ZCTA5_20, NAMELSAD_ZCTA5_20, AREALAND_ZCTA5_20, AREAWATER_ZCTA5_20,
    #   GEOID_TRACT_20,  NAMELSAD_TRACT_20, STATE_TRACT_20, AREALAND_TRACT_20,
    #   AREAWATER_TRACT_20, AREALAND_PART, AREAWATER_PART
    # Older versions may have different column names; map flexibly.
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "zcta" in lc and "geoid" in lc:
            col_map["zcta"] = c
        elif "tract" in lc and "geoid" in lc:
            col_map["tract"] = c
        elif lc.startswith("arealand_part") or lc == "arealand_part":
            col_map["arealand_part"] = c
        elif "pop" in lc and "part" in lc:
            col_map["pop_part"] = c
    print(f"  Resolved columns: {col_map}")

    out = pd.DataFrame({
        "zcta": df[col_map["zcta"]].str.zfill(5),
        "tract_fips": df[col_map["tract"]].str.zfill(11),
    })
    if "arealand_part" in col_map:
        out["arealand_part_m2"] = pd.to_numeric(df[col_map["arealand_part"]], errors="coerce")
    if "pop_part" in col_map:
        out["pop_in_overlap_2020"] = pd.to_numeric(df[col_map["pop_part"]], errors="coerce")
    out.to_csv(OUT_CSV, index=False)

    META_JSON.write_text(json.dumps({
        "source_url": URL,
        "n_overlaps": int(len(out)),
        "n_unique_zcta": int(out["zcta"].nunique()),
        "n_unique_tract": int(out["tract_fips"].nunique()),
        "columns_resolved": col_map,
    }, indent=2))
    print(f"Wrote: {OUT_CSV.relative_to(ROOT)}")
    print(f"  Unique ZCTAs:  {out['zcta'].nunique():,}")
    print(f"  Unique tracts: {out['tract_fips'].nunique():,}")


if __name__ == "__main__":
    main()
