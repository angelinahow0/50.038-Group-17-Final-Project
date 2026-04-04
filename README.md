# 50.038-Group-17-Final-Project# Match-and-Go! Model Card

**Project**: 50.038 Computational Data Science · Group 17  
**Owner**: Person 2

---

## V1 — Weighted Baseline (`baseline.py`)

| Property | Value |
|---|---|
| Type | Rule-based weighted scorer |
| Formula | `0.25×Budget + 0.20×Climate + 0.35×Activity + 0.20×Visual` |
| Training data | None required |
| Input | User profile dict + destinations DataFrame |
| Output | Ranked DataFrame with sub-scores |

**Sub-score descriptions:**
- **Budget score** — Compatibility between user budget tier and city cost-of-living index (1.0 = perfect match, -0.35 per tier gap)
- **Climate score** — Temperature match between user preferred climate and destination's monthly average (Gaussian falloff)
- **Activity score** — Jaccard-style overlap between user activity preferences and destination tags
- **Attraction score** — Normalised tourism/satisfaction rating from Kaggle datasets
- **Visual score** — Keyword match between user visual preference and destination metadata

---

## V2 — Learned Ranker (`ranker.py`, `v2_ranker.pkl`)

| Property | Value |
|---|---|
| Type | LambdaRank (XGBoost or LightGBM, best selected at training time) |
| Objective | `rank:ndcg` / `lambdarank` |
| Features | 5 sub-scores from V1 |
| Training data | 5,000 synthetic users × ~50 destinations ≈ 250k rows |
| Label source | V1 composite rank positions (inverted for relevance) |
| Split | 80/10/10 train/val/test (split on `user_id` to prevent leakage) |
| Hyperparameters | n_estimators=300, max_depth=4, lr=0.05, subsample=0.8 |
| Output file | `models/v2_ranker.pkl` |

**How V2 improves on V1:**  
V1 uses fixed, hand-tuned weights. V2 learns non-linear interactions between sub-scores — e.g. that a perfect climate match combined with an activity match is worth more than the sum of their individual contributions.

---

## Training Pipeline

```bash
# First-time run (generates all data + trains model, ~5-10 min):
python run_pipeline.py

# If labels already exist:
python run_pipeline.py --skip-labels

# If labels + split already exist:
python run_pipeline.py --skip-labels --skip-split

# Skip ablation study (faster):
python run_pipeline.py --skip-ablation
```

**Prerequisites:**  
- `data/processed/destinations.csv` must exist (Person 1 deliverable)
- All packages in `requirements.txt` installed

---

## Using the Predict Wrapper

```python
from models.ranker import predict, load_model
import pandas as pd

dest = pd.read_csv("data/processed/destinations.csv")
model = load_model()

user = {
    "budget_tier":  "50-100",
    "activities":   ["beach", "food"],
    "travel_month": 7,
    "climate_pref": "warm",
    "trip_duration": 7,
    "visual_pref":  "tropical",
}

top5 = predict(user, dest, model, top_n=5)
print(top5[["city", "v2_score", "budget_score", "climate_score"]])
```

---

## Evaluation Results

See `evaluation/results/final_metrics.csv` after running the pipeline.

| Metric | V1 Baseline | V2 Ranker |
|--------|-------------|-----------|
| NDCG@5 | — | — |
| Precision@3 | — | — |
| MRR | — | — |

*Populated automatically by `run_pipeline.py`*
