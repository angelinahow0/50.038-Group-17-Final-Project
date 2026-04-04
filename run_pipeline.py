#!/usr/bin/env python3
"""
run_pipeline.py
================
Master runner. Steps:
  scripts/01_fetch_raw_data.py           - Meteostat bulk fetch + Kaggle image metadata
  scripts/02_clean_datasets.py           - clean each source; score images
  scripts/03_merge_datasets.py           - join all sources into city_master
  scripts/04_generate_synthetic_users.py - 5,000 user profiles + interactions
  notebooks/05_eda_notebook.py           - EDA figures (saved to notebooks/eda_figures/)

Before running:
  pip install pandas numpy matplotlib seaborn Pillow
  Place Kaggle CSVs at:
    data/raw/cost_of_living_raw.csv
    data/raw/worldwide_travel_cities_raw.csv
  Place tourism images at:
    data/raw/images/
"""
import subprocess, sys, os, time

SCRIPTS = [
    ("scripts",   "01_fetch_raw_data.py"),
    ("scripts",   "02_clean_datasets.py"),
    ("scripts",   "03_merge_datasets.py"),
    ("scripts",   "04_generate_synthetic_users.py"),
    ("notebooks", "05_eda_notebook.py"),
]

BASE = os.path.dirname(__file__)

def run(folder, script):
    path = os.path.join(BASE, folder, script)
    print(f"\n{'─'*60}\n  Running: {folder}/{script}\n{'─'*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, path])
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[FAILED] {script} (exit {result.returncode})")
        sys.exit(1)
    print(f"\n  Done in {elapsed:.1f}s")

if __name__ == "__main__":
    t_start = time.time()
    for folder, s in SCRIPTS:
        run(folder, s)
    print(f"\n{'='*60}\n  Pipeline complete in {time.time()-t_start:.1f}s\n{'='*60}")
