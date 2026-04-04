# Travel Recommender – Data Processing Pipeline
## Real-schema pipeline for 4 data sources

### Quick Start
```bash
pip install meteostat pandas numpy matplotlib seaborn
# Set FLICKR_API_KEY in your environment, then:
python3 run_pipeline.py
```

### Data Source Setup

| Source | How to get data |
|--------|-----------------|
| **Meteostat** | Auto-fetched via `pip install meteostat` (station lookup by lat/lon) |
| **Flickr** | Auto-scraped via Flickr API using `FLICKR_API_KEY` and travel keyword search |
| **Cost of Living** | Download from [Kaggle](https://www.kaggle.com/code/olgaluzhetska/cost-of-living-analysis) → save as `data/raw/cost_of_living_raw.csv` |
| **Worldwide Travel Cities** | Download from [Kaggle](https://www.kaggle.com/datasets/furkanima/worldwide-travel-cities-ratings-and-climate) → save as `data/raw/worldwide_travel_cities_raw.csv` |

### Real Schemas Used

**Meteostat hourly** (fetched via SDK):
`time, temp, dwpt, rhum, prcp, snow, wdir, wspd, wpgt, pres, tsun, coco`
→ tsun is in minutes/hour; aggregated to monthly sunshine hours

**Flickr** (scraped by Flickr API):
`owner_username, post_url, date_utc, caption, likes, comments, typename (Photo|Video), location_lat, location_lng`

**Cost of Living** (country-level, Numbeo):
`Rank, Country, Cost of Living Index, Rent Index, Cost of Living Plus Rent Index,
Groceries Index, Restaurant Price Index, Local Purchasing Power Index`
→ Joined to cities by country name. Cities with no country match are **dropped**.

**Worldwide Travel Cities** (city-level):
`id, city, country, region, short_description, latitude, longitude,
avg_temp_monthly (JSON), ideal_durations (JSON array), budget_level,
culture, adventure, nature, beaches, nightlife, cuisine, wellness, urban, seclusion`
→ Activity scores are integers 1–5. JSON columns are expanded during cleaning.

### Pipeline Steps

| Script | Input | Output |
|--------|-------|--------|
| `01_fetch_raw_data.py` | APIs + Kaggle CSVs | `data/raw/*.csv` |
| `02_clean_datasets.py` | raw CSVs | `data/cleaned/*.csv` |
| `03_merge_datasets.py` | cleaned CSVs | `data/merged/city_master.csv` |
| `04_generate_synthetic_users.py` | city_master_slim.csv | `synthetic_users.csv`, `synthetic_user_interactions.csv` |
| `05_eda_notebook.py` | all cleaned + merged | `notebooks/eda_figures/*.png` (22 figures) |

### Key Design Decisions
- **CoL join is strict**: cities whose country has no Numbeo entry are dropped from master
- **Climate hierarchy**: Meteostat hourly data preferred; WTC `avg_temp_monthly` JSON used as fallback
- **Flickr**: left join (cities with no scraped posts get NaN, not dropped)
- **tsun field**: Meteostat returns minutes per hour observation window, not hours/day — converted during aggregation


