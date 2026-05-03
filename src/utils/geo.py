"""Geographic helpers for the CKD pipeline."""
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
STATE_REGION_CSV = ROOT / "data" / "census" / "state_to_region.csv"
ZCTA_CROSSWALK_CSV = ROOT / "data" / "census" / "zcta_tract_crosswalk_2020.csv"

# ZCTA -> state assignment is approximate.  ZIP-code prefix ranges (USPS / Census)
# give us a state for any 5-digit ZCTA without needing the full crosswalk.
# Source: USPS ZIP code prefix tables (first 3 digits).
_ZIP_PREFIX_TO_STATE = [
    ((  0,   4), "PR"),  # 005 = Puerto Rico (treated as outside CONUS)
    (( 10,  27), "MA"),  ((  5,   5), "NH"),  ((  6,   6), "CT"),  ((  7,   8), "NJ"),
    ((  9,   9), "MA"),  (( 28,  29), "RI"),  (( 30,  38), "VT"),  (( 39,  49), "NH"),
    (( 50,  54), "VT"),  (( 55,  56), "MA"),  (( 57,  59), "VT"),  (( 60,  69), "MA"),
    (( 70,  89), "ME"),  (( 90,  99), "PR"),  ((100, 149), "NY"),  ((150, 196), "PA"),
    ((197, 199), "DE"),  ((200, 205), "DC"),  ((206, 219), "MD"),  ((220, 246), "VA"),
    ((247, 268), "WV"),  ((270, 289), "NC"),  ((290, 299), "SC"),  ((300, 319), "GA"),
    ((320, 349), "FL"),  ((350, 369), "AL"),  ((370, 385), "TN"),  ((386, 397), "MS"),
    ((398, 399), "GA"),  ((400, 427), "KY"),  ((430, 459), "OH"),  ((460, 479), "IN"),
    ((480, 499), "MI"),  ((500, 528), "IA"),  ((530, 549), "WI"),  ((550, 567), "MN"),
    ((570, 577), "SD"),  ((580, 588), "ND"),  ((590, 599), "MT"),  ((600, 629), "IL"),
    ((630, 658), "MO"),  ((660, 679), "KS"),  ((680, 693), "NE"),  ((700, 714), "LA"),
    ((716, 729), "AR"),  ((730, 749), "OK"),  ((750, 799), "TX"),  ((800, 816), "CO"),
    ((820, 831), "WY"),  ((832, 838), "ID"),  ((840, 847), "UT"),  ((850, 865), "AZ"),
    ((870, 884), "NM"),  ((889, 898), "NV"),  ((900, 961), "CA"),  ((967, 968), "HI"),
    ((970, 979), "OR"),  ((980, 994), "WA"),  ((995, 999), "AK"),
]


@lru_cache(maxsize=1)
def state_region_table() -> pd.DataFrame:
    return pd.read_csv(STATE_REGION_CSV, dtype={"state_fips": str})


def state_for_zcta(zcta: int | str) -> str | None:
    """Return USPS state abbr for a 5-digit ZCTA, or None for non-CONUS / unknown."""
    try:
        z = int(zcta)
    except (TypeError, ValueError):
        return None
    p = z // 100  # first 3 digits
    for (lo, hi), state in _ZIP_PREFIX_TO_STATE:
        if lo <= p <= hi:
            return state
    return None


@lru_cache(maxsize=200_000)
def region_for_zcta(zcta: int | str) -> str | None:
    """Return Census region (Northeast/Midwest/South/West) for a ZCTA."""
    state = state_for_zcta(zcta)
    if state is None:
        return None
    tbl = state_region_table()
    row = tbl[tbl["state_abbr"] == state]
    if row.empty:
        return None
    return row.iloc[0]["census_region"]


def add_region_column(df: pd.DataFrame, zcta_col: str = "ZCTA") -> pd.DataFrame:
    """Add a 'census_region' column to a frame keyed by ZCTA."""
    df = df.copy()
    df["census_region"] = df[zcta_col].map(region_for_zcta)
    return df
