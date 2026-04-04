# """
# 04_generate_synthetic_users.py
# ================================
# Generates 5,000 realistic synthetic traveler profiles with:
#   • Demographics (age, gender, nationality, income tier)
#   • Travel behaviour (frequency, budget, trip type, duration)
#   • Preference vectors (climate, activities, food, safety priority …)
#   • Interaction history (cities visited, wishlist, reviews given)
#   • Persona labels (7 archetype clusters)
#   • Feature-space coordinates for downstream ML / recommender

# Output: data/merged/synthetic_users.csv
#          data/merged/synthetic_user_interactions.csv  (user × city ratings)

# No external libraries needed – pure numpy / random / pandas.
# """

import pandas as pd
import numpy as np
import random
import os

np.random.seed(2024)
random.seed(2024)

BASE      = os.path.join(os.path.dirname(__file__), '..')
SYNTHETIC = os.path.join(BASE, 'data', 'synthetic')
os.makedirs(SYNTHETIC, exist_ok=True)

N_USERS = 5_000

# ─────────────────────────────────────────────────────
# PERSONA ARCHETYPES  (7 clusters)
# Each defines probability weights for preference dims
# ─────────────────────────────────────────────────────
PERSONAS = {
    "Budget Backpacker": dict(
        age_range=(19, 30), income_tier_weights=[0.6, 0.3, 0.1, 0.0],
        trips_per_year=(1, 4), avg_trip_days=(14, 30),
        budget_daily_usd=(20, 60), pref_adventure=0.8, pref_culture=0.6,
        pref_beach=0.5, pref_nightlife=0.5, pref_luxury=0.05,
        pref_family=0.1, pref_nature=0.7, pref_food=0.6,
        climate_pref="warm", solo_prob=0.55, group_prob=0.35,
    ),
    "Cultural Explorer": dict(
        age_range=(28, 55), income_tier_weights=[0.1, 0.4, 0.4, 0.1],
        trips_per_year=(2, 5), avg_trip_days=(7, 14),
        budget_daily_usd=(80, 200), pref_adventure=0.3, pref_culture=0.95,
        pref_beach=0.25, pref_nightlife=0.3, pref_luxury=0.3,
        pref_family=0.3, pref_nature=0.5, pref_food=0.85,
        climate_pref="any", solo_prob=0.35, group_prob=0.45,
    ),
    "Luxury Traveler": dict(
        age_range=(35, 65), income_tier_weights=[0.0, 0.05, 0.35, 0.6],
        trips_per_year=(3, 8), avg_trip_days=(5, 14),
        budget_daily_usd=(300, 1000), pref_adventure=0.2, pref_culture=0.55,
        pref_beach=0.65, pref_nightlife=0.5, pref_luxury=0.98,
        pref_family=0.35, pref_nature=0.4, pref_food=0.9,
        climate_pref="warm", solo_prob=0.15, group_prob=0.35,
    ),
    "Family Vacationer": dict(
        age_range=(30, 50), income_tier_weights=[0.05, 0.4, 0.45, 0.1],
        trips_per_year=(1, 3), avg_trip_days=(7, 21),
        budget_daily_usd=(100, 300), pref_adventure=0.4, pref_culture=0.55,
        pref_beach=0.7, pref_nightlife=0.1, pref_luxury=0.25,
        pref_family=0.98, pref_nature=0.6, pref_food=0.65,
        climate_pref="warm", solo_prob=0.02, group_prob=0.95,
    ),
    "Digital Nomad": dict(
        age_range=(24, 42), income_tier_weights=[0.2, 0.5, 0.25, 0.05],
        trips_per_year=(4, 12), avg_trip_days=(14, 60),
        budget_daily_usd=(40, 120), pref_adventure=0.5, pref_culture=0.7,
        pref_beach=0.45, pref_nightlife=0.45, pref_luxury=0.15,
        pref_family=0.05, pref_nature=0.5, pref_food=0.75,
        climate_pref="warm", solo_prob=0.70, group_prob=0.20,
    ),
    "Adventure Seeker": dict(
        age_range=(20, 45), income_tier_weights=[0.2, 0.5, 0.25, 0.05],
        trips_per_year=(2, 6), avg_trip_days=(10, 21),
        budget_daily_usd=(60, 180), pref_adventure=0.98, pref_culture=0.4,
        pref_beach=0.4, pref_nightlife=0.3, pref_luxury=0.1,
        pref_family=0.15, pref_nature=0.9, pref_food=0.55,
        climate_pref="any", solo_prob=0.45, group_prob=0.45,
    ),
    "Weekend Tripper": dict(
        age_range=(25, 50), income_tier_weights=[0.1, 0.4, 0.4, 0.1],
        trips_per_year=(4, 12), avg_trip_days=(2, 5),
        budget_daily_usd=(100, 350), pref_adventure=0.3, pref_culture=0.65,
        pref_beach=0.4, pref_nightlife=0.55, pref_luxury=0.4,
        pref_family=0.35, pref_nature=0.3, pref_food=0.8,
        climate_pref="any", solo_prob=0.3, group_prob=0.55,
    ),
}
PERSONA_NAMES = list(PERSONAS.keys())
# Assign personas with probability distribution
PERSONA_PROBS = [0.18, 0.16, 0.12, 0.14, 0.13, 0.14, 0.13]

NATIONALITIES = [
    ("American", "USA", 0.20), ("British", "UK", 0.09), ("German", "Germany", 0.08),
    ("French", "France", 0.07), ("Australian", "Australia", 0.06),
    ("Canadian", "Canada", 0.05), ("Japanese", "Japan", 0.05),
    ("Chinese", "China", 0.07), ("Indian", "India", 0.06),
    ("Brazilian", "Brazil", 0.04), ("Spanish", "Spain", 0.04),
    ("Italian", "Italy", 0.04), ("Dutch", "Netherlands", 0.02),
    ("Korean", "South Korea", 0.03), ("Singaporean", "Singapore", 0.02),
    ("Other", "Other", 0.08),
]
NAT_NAMES  = [n[0] for n in NATIONALITIES]
NAT_COUNTRIES = [n[1] for n in NATIONALITIES]
NAT_PROBS  = [n[2] for n in NATIONALITIES]

INCOME_TIERS = ["<30k", "30-70k", "70-150k", "150k+"]
GENDERS = ["Male", "Female", "Non-binary", "Prefer not to say"]
GENDER_PROBS = [0.46, 0.48, 0.04, 0.02]

TRAVEL_COMPANIONS = ["Solo", "Partner", "Family", "Friends", "Mixed group"]
ACCOMMODATION_TYPES = ["Hostel", "Budget Hotel", "Mid-range Hotel",
                       "Luxury Hotel", "Airbnb/VRBO", "Boutique Hotel",
                       "Resort", "Camping/Glamping"]
BOOKING_ADVANCE_DAYS = [0, 7, 14, 30, 60, 90, 180]
TRANSPORT_PREFS = ["Flights only", "Trains preferred", "Road trips", "Any"]

CLIMATE_PREFS = ["Tropical/Hot", "Mediterranean/Warm", "Temperate/Mild",
                 "Cold/Snow", "Any/Flexible"]
ACTIVITY_TAGS = [
    "beaches", "museums", "hiking", "food_tours", "nightlife", "shopping",
    "historical_sites", "wildlife", "yoga_wellness", "photography",
    "water_sports", "skiing", "cycling", "art_galleries", "live_music",
    "local_markets", "temples", "architecture", "wine_tasting", "volunteering"
]

# Load city list from master (needed for interaction matrix)
master_path = os.path.join(BASE, "data", "merged", "city_master_slim.csv")
if os.path.exists(master_path):
    _raw = pd.read_csv(master_path)
    _want = ["city", "country", "region", "master_travel_score",
            "col_cost_of_living_index", "col_affordability_tier",
            "culture", "adventure", "nature", "beaches",
            "nightlife", "cuisine", "wellness", "urban", "seclusion"]
    cities_df = _raw[[c for c in _want if c in _raw.columns]].copy()
    # Estimate mid-range daily budget from Cost of Living Index (NYC=100 baseline ~$150/day)
    if "col_cost_of_living_index" in cities_df.columns:
        cities_df["col_daily_budget_mid"] = (
            cities_df["col_cost_of_living_index"].fillna(50) / 100 * 150
        ).round(0)
    else:
        cities_df["col_daily_budget_mid"] = 100
    CITY_LIST = cities_df["city"].tolist()
else:
    CITY_LIST = ["Paris", "London", "Tokyo", "New York", "Bangkok",
                 "Dubai", "Barcelona", "Rome", "Sydney", "Singapore"]
    cities_df = pd.DataFrame({"city": CITY_LIST})

print(f"City pool for interactions: {len(CITY_LIST)} cities")


# ═════════════════════════════════════════════
# GENERATE USERS
# ═════════════════════════════════════════════
print(f"\nGenerating {N_USERS:,} synthetic users …")

def sample_pref(base_prob, noise=0.15):
    v = base_prob + np.random.normal(0, noise)
    return round(float(np.clip(v, 0, 1)), 3)

users = []
interactions = []

for uid in range(1, N_USERS + 1):
    # Persona
    persona_name = np.random.choice(PERSONA_NAMES, p=PERSONA_PROBS)
    p = PERSONAS[persona_name]

    # Demographics
    age = int(np.random.uniform(*p["age_range"]))
    gender = np.random.choice(GENDERS, p=GENDER_PROBS)
    nat_idx = np.random.choice(len(NATIONALITIES), p=NAT_PROBS)
    nationality = NAT_NAMES[nat_idx]
    home_country = NAT_COUNTRIES[nat_idx]
    income_tier = np.random.choice(INCOME_TIERS, p=p["income_tier_weights"])

    # Travel behaviour
    trips_per_year = int(np.random.uniform(*p["trips_per_year"]))
    avg_trip_days = int(np.random.uniform(*p["avg_trip_days"]))
    budget_daily = round(np.random.uniform(*p["budget_daily_usd"]) * (1 + np.random.normal(0, 0.1)), 0)
    budget_total_typical = round(budget_daily * avg_trip_days, 0)
    booking_advance = random.choice(BOOKING_ADVANCE_DAYS)

    # Companion type
    r = random.random()
    if r < p["solo_prob"]:
        companion = "Solo"
    elif r < p["solo_prob"] + p["group_prob"]:
        companion = random.choice(["Friends", "Mixed group"])
    else:
        companion = random.choice(["Partner", "Family"])

    # Accommodation
    budget_cat = INCOME_TIERS.index(income_tier)
    accom_weights = [
        [0.4, 0.3, 0.2, 0.0, 0.05, 0.02, 0.0, 0.03],
        [0.1, 0.25, 0.35, 0.05, 0.15, 0.05, 0.02, 0.03],
        [0.0, 0.05, 0.3, 0.2, 0.2, 0.15, 0.05, 0.05],
        [0.0, 0.0, 0.1, 0.45, 0.1, 0.15, 0.15, 0.05],
    ][budget_cat]
    accommodation_pref = np.random.choice(ACCOMMODATION_TYPES, p=accom_weights)

    # Preference scores (0–1)
    pref_adventure     = sample_pref(p["pref_adventure"])
    pref_culture       = sample_pref(p["pref_culture"])
    pref_beach         = sample_pref(p["pref_beach"])
    pref_nightlife     = sample_pref(p["pref_nightlife"])
    pref_luxury        = sample_pref(p["pref_luxury"])
    pref_family_friendly = sample_pref(p["pref_family"])
    pref_nature        = sample_pref(p["pref_nature"])
    pref_food          = sample_pref(p["pref_food"])
    pref_safety        = sample_pref(0.3 + budget_cat * 0.15)
    pref_digital_nomad = sample_pref(0.7 if persona_name == "Digital Nomad" else 0.15)
    pref_instagram     = sample_pref(0.6 if age < 35 else 0.35)
    pref_value_for_money = sample_pref(1 - pref_luxury)

    # Climate preference
    if p["climate_pref"] == "warm":
        climate_pref = np.random.choice(CLIMATE_PREFS[:2] + ["Any/Flexible"],
                                        p=[0.4, 0.4, 0.2])
    elif p["climate_pref"] == "any":
        climate_pref = np.random.choice(CLIMATE_PREFS, p=[0.25, 0.25, 0.2, 0.1, 0.2])
    else:
        climate_pref = np.random.choice(CLIMATE_PREFS)

    # Activity tags (multi-label): pick top 3–7
    activity_scores = {
        "beaches": pref_beach, "museums": pref_culture, "hiking": pref_adventure,
        "food_tours": pref_food, "nightlife": pref_nightlife, "shopping": pref_luxury * 0.6,
        "historical_sites": pref_culture, "wildlife": pref_nature, "yoga_wellness": sample_pref(0.2),
        "photography": pref_instagram, "water_sports": pref_beach * pref_adventure,
        "skiing": pref_adventure * 0.4, "cycling": pref_nature * 0.5,
        "art_galleries": pref_culture * 0.7, "live_music": pref_nightlife * 0.8,
        "local_markets": pref_food * 0.8, "temples": pref_culture * 0.8,
        "architecture": pref_culture * 0.7, "wine_tasting": pref_food * pref_luxury,
        "volunteering": sample_pref(0.1),
    }
    sorted_acts = sorted(activity_scores.items(), key=lambda x: -x[1])
    n_acts = random.randint(3, 7)
    # Pick from top half with probability weighted by score
    top_acts = sorted_acts[:12]
    probs_acts = np.array([v for _, v in top_acts])
    probs_acts = probs_acts / probs_acts.sum()
    chosen = np.random.choice(
        [k for k, _ in top_acts],
        size=min(n_acts, len(top_acts)), replace=False, p=probs_acts
    )
    activity_tags = "|".join(sorted(chosen))

    # Past trips (0–15 cities from city list)
    n_visited = min(len(CITY_LIST), int(np.random.lognormal(1.5, 0.8)))
    visited_cities = random.sample(CITY_LIST, min(n_visited, len(CITY_LIST)))
    # Wishlist (3–10 cities not yet visited)
    remaining = [c for c in CITY_LIST if c not in visited_cities]
    n_wish = min(len(remaining), random.randint(3, 10))
    wishlist_cities = random.sample(remaining, n_wish) if remaining else []

    # User record
    users.append({
        "user_id": f"U{uid:05d}",
        "persona": persona_name,
        "age": age,
        "gender": gender,
        "nationality": nationality,
        "home_country": home_country,
        "income_tier": income_tier,
        "trips_per_year": trips_per_year,
        "avg_trip_days": avg_trip_days,
        "typical_daily_budget_usd": budget_daily,
        "typical_trip_budget_usd": budget_total_typical,
        "preferred_accommodation": accommodation_pref,
        "preferred_companion": companion,
        "transport_preference": np.random.choice(TRANSPORT_PREFS),
        "avg_booking_advance_days": booking_advance,
        "climate_preference": climate_pref,
        "activity_tags": activity_tags,
        # Preference scores
        "pref_adventure": pref_adventure,
        "pref_culture": pref_culture,
        "pref_beach": pref_beach,
        "pref_nightlife": pref_nightlife,
        "pref_luxury": pref_luxury,
        "pref_family_friendly": pref_family_friendly,
        "pref_nature": pref_nature,
        "pref_food": pref_food,
        "pref_safety": pref_safety,
        "pref_digital_nomad": pref_digital_nomad,
        "pref_instagram_worthy": pref_instagram,
        "pref_value_for_money": pref_value_for_money,
        # History
        "n_countries_visited": len(set(
            cities_df.set_index("city").loc[
                [c for c in visited_cities if c in cities_df["city"].values], "country"
            ].tolist()
        )) if visited_cities else 0,
        "cities_visited_count": len(visited_cities),
        "cities_visited": "|".join(visited_cities),
        "wishlist_count": len(wishlist_cities),
        "wishlist": "|".join(wishlist_cities),
        # Derived cluster-ready embedding (flat preference vector)
        "emb_0": pref_adventure,
        "emb_1": pref_culture,
        "emb_2": pref_beach,
        "emb_3": pref_nightlife,
        "emb_4": pref_luxury,
        "emb_5": pref_family_friendly,
        "emb_6": pref_nature,
        "emb_7": pref_food,
        "emb_8": pref_safety,
        "emb_9": pref_digital_nomad,
        "emb_10": pref_instagram,
        "emb_11": pref_value_for_money,
        "emb_12": budget_daily / 500,    # normalised budget
        "emb_13": avg_trip_days / 60,    # normalised trip length
    })

    # Interaction records (explicit ratings for visited cities)
    for city in visited_cities:
        # Rating is biased by how well city matches user prefs
        city_row = cities_df[cities_df["city"] == city]
        base_rating = 3.5
        if not city_row.empty and "master_travel_score" in city_row.columns:
            city_score = city_row["master_travel_score"].values[0]
            base_rating = 2.5 + city_score / 10 * 4   # map 0-10 → 2.5-6.5
        noise = np.random.normal(0, 0.7)
        rating = round(float(np.clip(base_rating + noise, 1, 5)), 1)

        interactions.append({
            "user_id": f"U{uid:05d}",
            "city": city,
            "rating": rating,
            "interaction_type": "visited_and_rated",
            "review_length_words": int(np.random.lognormal(3, 1)) if random.random() > 0.3 else 0,
            "would_revisit": rating >= 4.0,
            "recommended_to_others": rating >= 3.5,
        })

    # Wishlist interactions (implicit signal)
    for city in wishlist_cities:
        interactions.append({
            "user_id": f"U{uid:05d}",
            "city": city,
            "rating": np.nan,
            "interaction_type": "wishlisted",
            "review_length_words": 0,
            "would_revisit": np.nan,
            "recommended_to_others": np.nan,
        })

    if uid % 1000 == 0:
        print(f"  … {uid:,} users generated")

df_users = pd.DataFrame(users)
df_inter = pd.DataFrame(interactions)

print(f"\nUsers:        {len(df_users):,} rows × {len(df_users.columns)} cols")
print(f"Interactions: {len(df_inter):,} rows × {len(df_inter.columns)} cols")
print(f"Persona distribution:")
print(df_users["persona"].value_counts().to_string())

# ════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════
df_users.to_csv(os.path.join(SYNTHETIC, "synthetic_users.csv"), index=False)
df_inter.to_csv(os.path.join(SYNTHETIC, "synthetic_user_interactions.csv"), index=False)

print(f"\n✓ Saved synthetic_users.csv ({len(df_users):,} users)")
print(f"✓ Saved synthetic_user_interactions.csv ({len(df_inter):,} interactions)")
print("✅ Synthetic user generation complete!")
