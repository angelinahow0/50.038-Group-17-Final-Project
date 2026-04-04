"""
01_fetch_raw_data.py
=====================
Fetches raw data from all 4 sources using their real schemas.

Meteostat
---------
Replicates the user's exact bulk pipeline:
  1. Load station list from https://bulk.meteostat.net/v2/stations/lite.json.gz
  2. Filter stations that have monthly inventory overlapping START–END
  3. Download monthly CSVs from https://data.meteostat.net/monthly/{id}.csv.gz
  4. Strip _source columns, filter date range, build date from year+month
  5. Inject: station_id, station_name, country (pycountry alpha_2→name),
             latitude, longitude, elevation

City → station mapping
-----------------------
The parquet has no city column. City assignment is done by nearest-neighbour
spatial join: for each WTC city (lat/lon), find the closest station by
Haversine distance and record the mapping in meteostat_city_station_map.csv.

Kaggle Tourism Images
---------------------
Extract metadata from labelled iStockPhoto JPEG files stored in data/raw/images/.
Descriptions are read from the XMP dc:description block (Windows Properties →
Details → Subject).  The text is stored as-is for scoring in 02_clean.
Output: kaggle_images_raw.csv  (filename, description)

Kaggle CSVs (manual download required)
---------------------------------------
  cost_of_living_raw.csv          → data/raw/
  worldwide_travel_cities_raw.csv → data/raw/

Parquet schema (meteostat_monthly_raw.parquet)
----------------------------------------------
  date, tavg, tmin, tmax, prcp, snow, wdir, wspd, wpgt, pres, tsun,
  station_id, station_name, country, latitude, longitude, elevation

Note: 'tavg/tmin/tmax' are the real Meteostat monthly column names.
      'tsun' is total sunshine minutes for the month.
      '_source' columns are stripped during fetch.
      'city' is NOT in this file — added via spatial join in 02_clean.
"""

import os
import re
import html
import json
import gzip
import math
import time
import warnings
import requests
import pandas as pd
import pycountry
from datetime import date
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

BASE = os.path.join(os.path.dirname(__file__), "..")
RAW  = os.path.join(BASE, "data", "raw")
os.makedirs(RAW, exist_ok=True)

# ── Config (mirrors user's script exactly) ─────────────────────
START       = date(2010, 1, 1)
END         = date(2023, 12, 31)
MAX_WORKERS = 20
MIN_ROWS    = 10
# ───────────────────────────────────────────────────────────────

country_lookup = {c.alpha_2: c.name for c in pycountry.countries}


# ═══════════════════════════════════════════════════════
# HELPER: Haversine distance (km) between two lat/lon points
# ═══════════════════════════════════════════════════════
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ═══════════════════════════════════════════════════════
# 1. METEOSTAT – bulk monthly pipeline
# ═══════════════════════════════════════════════════════
print("\n== [1] Meteostat: bulk monthly fetch ==")

def fetch_meteostat():

    # ── Step 1: Load station list (exact replica of user's code) ──
    print("  Loading station list...")
    url = "https://bulk.meteostat.net/v2/stations/lite.json.gz"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    station_list = json.loads(gzip.decompress(response.content))

    rows = []
    for s in station_list:
        monthly_inv = s.get("inventory", {}).get("monthly", {})
        inv_start   = monthly_inv.get("start")
        inv_end     = monthly_inv.get("end")
        if inv_start is None or inv_end is None:
            continue
        if int(inv_end) < START.year or int(inv_start) > END.year:
            continue
        rows.append({
            "id":        s.get("id"),
            "name":      s.get("name", {}).get("en", ""),
            "country":   s.get("country"),
            "latitude":  s.get("location", {}).get("latitude"),
            "longitude": s.get("location", {}).get("longitude"),
            "elevation": s.get("location", {}).get("elevation"),
        })

    all_stations = pd.DataFrame(rows).dropna(subset=["latitude", "longitude"])
    print(f"  Found {len(all_stations)} stations with monthly data in range")

    # ── Step 2: Download monthly CSVs (exact replica of user's fetch_station) ──
    def fetch_station(row):
        station_id = row.id
        try:
            r = requests.get(
                f"https://data.meteostat.net/monthly/{station_id}.csv.gz",
                timeout=10
            )
            if r.status_code != 200:
                return None

            df = pd.read_csv(BytesIO(r.content), compression="gzip")

            # Drop _source columns (exact replica)
            df = df[[c for c in df.columns if not c.endswith("_source")]]

            # Filter to date range using year/month columns (exact replica)
            df = df[(df["year"] >= START.year) & (df["year"] <= END.year)]

            if len(df) < MIN_ROWS:
                return None

            # Build date column from year + month (exact replica)
            df["date"] = pd.to_datetime(df[["year", "month"]].assign(day=1))
            df = df.drop(columns=["year", "month"])

            # Inject station metadata (exact replica)
            df["station_id"]   = station_id
            df["station_name"] = row.name
            df["country"]      = country_lookup.get(str(row.country), str(row.country))
            df["latitude"]     = row.latitude
            df["longitude"]    = row.longitude
            df["elevation"]    = row.elevation
            return df

        except Exception:
            return None

    print("  Fetching monthly data...")
    results = []
    station_rows = list(all_stations.itertuples())

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_station, row): row for row in station_rows}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is not None:
                results.append(result)
            if (i + 1) % 500 == 0:
                print(f"    Processed {i+1}/{len(station_rows)} stations, "
                      f"{len(results)} with data...")

    # ── Step 3: Combine and save (exact replica) ──
    print("  Combining results...")
    combined = pd.concat(results)
    combined = combined.sort_values(["station_id", "date"]).reset_index(drop=True)

    out_path = os.path.join(RAW, "meteostat_monthly_raw.parquet")
    combined.to_parquet(out_path, index=False)
    print(f"  Saved {len(combined):,} rows from {len(results)} stations")
    print(f"  Columns: {combined.columns.tolist()}")
    print(combined.head(2))

    # ── Step 4: City → station spatial mapping ──────────────────
    # Load WTC cities for lat/lon; find nearest station per city by Haversine
    wtc_path = os.path.join(RAW, "worldwide_travel_cities_raw.csv")
    if not os.path.exists(wtc_path):
        print("  [WARN] worldwide_travel_cities_raw.csv not found — "
              "skipping city→station map. Place it in data/raw/ first.")
        return combined, all_stations

    wtc = pd.read_csv(wtc_path)[["id", "city", "country", "latitude", "longitude"]].dropna(
        subset=["latitude", "longitude"]
    )
    wtc["latitude"]  = pd.to_numeric(wtc["latitude"],  errors="coerce")
    wtc["longitude"] = pd.to_numeric(wtc["longitude"], errors="coerce")
    wtc = wtc.dropna(subset=["latitude", "longitude"])

    # Build a station coordinate array for vectorised nearest-neighbour
    # (pure Python loop is fine for ~1000 cities × ~20k stations)
    station_coords = list(zip(
        all_stations["latitude"].values,
        all_stations["longitude"].values,
        all_stations["id"].values,
        all_stations["name"].values,
    ))

    city_station_rows = []
    for _, city_row in wtc.iterrows():
        clat, clon = city_row["latitude"], city_row["longitude"]
        best_dist, best_sid, best_sname = float("inf"), None, None
        for slat, slon, sid, sname in station_coords:
            try:
                d = haversine(clat, clon, float(slat), float(slon))
            except Exception:
                continue
            if d < best_dist:
                best_dist, best_sid, best_sname = d, sid, sname

        city_station_rows.append({
            "wtc_id":          city_row["id"],
            "city":            city_row["city"],
            "city_country":    city_row["country"],
            "city_lat":        clat,
            "city_lon":        clon,
            "station_id":      best_sid,
            "station_name":    best_sname,
            "distance_km":     round(best_dist, 1),
        })

    map_df = pd.DataFrame(city_station_rows)

    # Flag distant matches (> 150 km) as unreliable
    map_df["station_reliable"] = map_df["distance_km"] <= 150
    unreliable = (~map_df["station_reliable"]).sum()
    if unreliable:
        print(f"  [WARN] {unreliable} cities matched to station > 150 km away "
              f"(flagged station_reliable=False)")

    map_path = os.path.join(RAW, "meteostat_city_station_map.csv")
    map_df.to_csv(map_path, index=False)
    print(f"  Saved meteostat_city_station_map.csv ({len(map_df)} cities mapped)")
    print(map_df[["city", "station_id", "station_name", "distance_km",
                   "station_reliable"]].head(10).to_string(index=False))

    return combined, all_stations

try:
    df_meteo, df_stations = fetch_meteostat()
except requests.exceptions.ConnectionError:
    print("  [SKIP] No network access — place meteostat_monthly_raw.parquet "
          "and meteostat_city_station_map.csv in data/raw/ manually.")
except Exception as e:
    print(f"  [ERROR] Meteostat fetch failed: {e}")


# ═══════════════════════════════════════════════════════
# 2. KAGGLE TOURISM IMAGES – extract XMP / IPTC metadata
# ═══════════════════════════════════════════════════════
print("\n== [2] Kaggle Tourism Images: extract XMP metadata ==")

IMAGES_DIR = os.path.join(RAW, "images")


def extract_xmp_description(img_path):
    """Return the human-readable description embedded in a JPEG's XMP block.

    Windows Properties → Details field mapping for iStockPhoto JPEGs:
      'Subject'  → XMP dc:description  (primary — contains the caption text)
      'Title'    → XMP dc:title        (fallback)
      (headline) → photoshop:Headline  (last resort)
    """
    try:
        with open(img_path, "rb") as f:
            raw_bytes = f.read()
        xmp_start = raw_bytes.find(b"<x:xmpmeta")
        if xmp_start == -1:
            return ""
        xmp_end = raw_bytes.find(b"</x:xmpmeta>", xmp_start) + 12
        xmp_str = raw_bytes[xmp_start:xmp_end].decode("utf-8", errors="ignore")

        # dc:description — Windows "Subject" (primary)
        desc = re.findall(
            r"<dc:description>.*?<rdf:Alt>.*?<rdf:li[^>]*>(.*?)</rdf:li>",
            xmp_str, re.DOTALL)
        if desc:
            return html.unescape(desc[0].strip())

        # dc:title — Windows "Title" (fallback)
        title = re.findall(
            r"<dc:title>.*?<rdf:Alt>.*?<rdf:li[^>]*>(.*?)</rdf:li>",
            xmp_str, re.DOTALL)
        if title:
            return html.unescape(title[0].strip())

        # photoshop:Headline — last resort
        headline = re.findall(
            r"<photoshop:Headline>(.*?)</photoshop:Headline>",
            xmp_str, re.DOTALL)
        if headline:
            return html.unescape(headline[0].strip())
    except Exception:
        pass
    return ""


def fetch_kaggle_images():
    if not os.path.isdir(IMAGES_DIR):
        print(f"  [SKIP] Images folder not found: {IMAGES_DIR}")
        return None

    img_files = sorted(
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not img_files:
        print("  [SKIP] No image files found in data/raw/images/")
        return None

    records = []
    for fname in img_files:
        path = os.path.join(IMAGES_DIR, fname)
        description = extract_xmp_description(path)
        records.append({
            "filename":    fname,
            "filepath":    path,
            "description": description,
        })

    df = pd.DataFrame(records)
    has_desc = df["description"].str.strip().ne("").sum()
    print(f"  Found {len(df)} images | {has_desc} with non-empty descriptions")
    print("  Sample descriptions:")
    for _, row in df[df["description"].str.strip().ne("")].head(5).iterrows():
        print(f"    {row['filename']}: {row['description'][:80]}")

    out_path = os.path.join(RAW, "kaggle_images_raw.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved kaggle_images_raw.csv  ({len(df):,} rows)")
    return df


try:
    df_images = fetch_kaggle_images()
except Exception as e:
    import traceback
    print(f"  [ERROR] Kaggle image extraction failed: {type(e).__name__}: {e}")
    traceback.print_exc()


# ═══════════════════════════════════════════════════════
# 3. COST OF LIVING – Kaggle CSV (country-level)
# ═══════════════════════════════════════════════════════
print("\n== [3] Cost of Living: validate Kaggle CSV ==")

# Real columns (exact):
#   Rank, Country,
#   Cost of Living Index, Rent Index, Cost of Living Plus Rent Index,
#   Groceries Index, Restaurant Price Index, Local Purchasing Power Index

EXPECTED_COL_COLS = [
    "Rank", "Country",
    "Cost of Living Index", "Rent Index",
    "Cost of Living Plus Rent Index",
    "Groceries Index", "Restaurant Price Index",
    "Local Purchasing Power Index",
]

col_path = os.path.join(RAW, "cost_of_living_raw.csv")
if not os.path.exists(col_path):
    print(f"  [SKIP] Not found: {col_path}")
    print("  Download from: https://www.kaggle.com/code/olgaluzhetska/cost-of-living-analysis")
    print("  Save as: data/raw/cost_of_living_raw.csv")
else:
    # Find the first non-empty row to use as header.
    # The raw file has one or more leading empty/blank rows before the real header.
    raw = pd.read_csv(col_path, header=None)
    header_row = next(
        i for i, row in raw.iterrows()
        if row.notna().any() and row.astype(str).str.strip().ne("").any()
    )
    df_col = pd.read_csv(col_path, header=header_row)
    # Strip any accidental whitespace from column names introduced by the skip
    df_col.columns = df_col.columns.str.strip()
    # Drop any fully empty rows that may follow the header skip
    df_col = df_col.dropna(how="all").reset_index(drop=True)

    missing = [c for c in EXPECTED_COL_COLS if c not in df_col.columns]
    if missing:
        print(f"  [ERROR] Missing columns: {missing}")
        print(f"  Found: {df_col.columns.tolist()}")
    else:
        print(f"  OK — {len(df_col)} countries, columns validated")
        print(f"  Sample countries: {df_col['Country'].head(5).tolist()}")


# ═══════════════════════════════════════════════════════
# 4. WORLDWIDE TRAVEL CITIES – Kaggle CSV (city-level)
# ═══════════════════════════════════════════════════════
print("\n== [4] Worldwide Travel Cities: validate Kaggle CSV ==")

# Real columns (exact):
#   id, city, country, region, short_description,
#   latitude, longitude,
#   avg_temp_monthly  ← JSON {"1":{"avg","max","min"}, ..., "12":{}}
#   ideal_durations   ← JSON array ["Short trip","One week",...]
#   budget_level      ← "Budget" | "Mid-range" | "Luxury"
#   culture, adventure, nature, beaches, nightlife,
#   cuisine, wellness, urban, seclusion  ← int 1-5

EXPECTED_WTC_COLS = [
    "id", "city", "country", "region", "short_description",
    "latitude", "longitude", "avg_temp_monthly", "ideal_durations",
    "budget_level", "culture", "adventure", "nature", "beaches",
    "nightlife", "cuisine", "wellness", "urban", "seclusion",
]

wtc_path = os.path.join(RAW, "worldwide_travel_cities_raw.csv")
if not os.path.exists(wtc_path):
    print(f"  [SKIP] Not found: {wtc_path}")
    print("  Download from: https://www.kaggle.com/datasets/furkanima/worldwide-travel-cities-ratings-and-climate")
    print("  Save as: data/raw/worldwide_travel_cities_raw.csv")
else:
    df_wtc = pd.read_csv(wtc_path)
    missing = [c for c in EXPECTED_WTC_COLS if c not in df_wtc.columns]
    if missing:
        print(f"  [ERROR] Missing columns: {missing}")
    else:
        print(f"  OK — {len(df_wtc)} cities, columns validated")
        print(f"  Regions: {sorted(df_wtc['region'].dropna().unique().tolist())}")
        print(f"  Budget levels: {df_wtc['budget_level'].value_counts().to_dict()}")


# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("  FETCH SUMMARY")
print("="*60)
checks = [
    ("meteostat_monthly_raw.parquet",    "Meteostat monthly bulk"),
    ("meteostat_city_station_map.csv",   "City → station spatial map"),
    ("kaggle_images_raw.csv",            "Kaggle tourism image metadata"),
    ("cost_of_living_raw.csv",           "Cost of Living (Kaggle)"),
    ("worldwide_travel_cities_raw.csv",  "Worldwide Travel Cities (Kaggle)"),
]
for fname, label in checks:
    fpath = os.path.join(RAW, fname)
    if os.path.exists(fpath):
        size = os.path.getsize(fpath)
        print(f"  [OK]   {label:40s}  {size/1024/1024:.1f} MB")
    else:
        print(f"  [MISS] {label:40s}  not yet generated")
