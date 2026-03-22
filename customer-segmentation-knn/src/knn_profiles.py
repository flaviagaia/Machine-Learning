from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import DATA_PATH, ensure_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "knn_profiles.joblib"
SUMMARY_PATH = ARTIFACTS_DIR / "profile_summary.json"
NEIGHBORS_PATH = ARTIFACTS_DIR / "neighbor_examples.csv"

FEATURE_COLUMNS = [
    "age",
    "annual_income",
    "purchase_frequency_monthly",
    "average_ticket",
    "digital_engagement_score",
    "return_rate_pct",
    "support_tickets_quarter",
    "loyalty_months",
    "discount_sensitivity_score",
    "web_visits_monthly",
    "app_sessions_weekly",
    "tenure_months",
]


@dataclass
class NeighborResult:
    customer_id: str
    segment_label: str
    distance: float


def build_profile_engine(n_neighbors: int = 5, random_state: int = 42) -> dict:
    dataset = ensure_dataset(path=DATA_PATH, random_state=random_state)
    x = dataset[FEATURE_COLUMNS]

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("knn", NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")),
        ]
    )
    pipeline.fit(x)

    scaled_x = pipeline.named_steps["scaler"].transform(x)
    silhouette = silhouette_score(scaled_x, dataset["segment_label"])

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_columns": FEATURE_COLUMNS}, MODEL_PATH)

    query_customer = dataset.iloc[0]
    neighbors = find_similar_customers(pipeline, dataset, query_customer["customer_id"], top_k=n_neighbors)
    pd.DataFrame(neighbors).to_csv(NEIGHBORS_PATH, index=False)

    summary = {
        "dataset_path": str(DATA_PATH),
        "n_customers": int(len(dataset)),
        "n_segments": int(dataset["segment_label"].nunique()),
        "knn_neighbors": n_neighbors,
        "feature_columns": FEATURE_COLUMNS,
        "silhouette_score": round(float(silhouette), 4),
        "sample_customer_id": query_customer["customer_id"],
        "sample_customer_segment": query_customer["segment_label"],
        "sample_neighbors": neighbors,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def find_similar_customers(pipeline: Pipeline, dataset: pd.DataFrame, customer_id: str, top_k: int = 5) -> list[dict]:
    customer_rows = dataset.loc[dataset["customer_id"] == customer_id]
    if customer_rows.empty:
        raise ValueError(f"Customer '{customer_id}' not found.")

    query_vector = customer_rows.iloc[0][FEATURE_COLUMNS].to_frame().T
    scaled_query = pipeline.named_steps["scaler"].transform(query_vector)
    distances, indices = pipeline.named_steps["knn"].kneighbors(scaled_query, n_neighbors=top_k + 1)

    results = []
    for distance, idx in zip(distances[0], indices[0]):
        candidate = dataset.iloc[idx]
        if candidate["customer_id"] == customer_id:
            continue
        results.append(
            {
                "customer_id": candidate["customer_id"],
                "segment_label": candidate["segment_label"],
                "distance": round(float(distance), 4),
            }
        )
        if len(results) == top_k:
            break
    return results
