from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "fraud_dataset.parquet"
SOURCE_URL = "https://huggingface.co/datasets/h0d4r1/fraud_dataset"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the public fraud dataset before training."
        )

    frame = pd.read_parquet(path).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame["hour"] = frame["timestamp"].dt.hour
    frame["day"] = frame["timestamp"].dt.day
    frame["day_of_week"] = frame["timestamp"].dt.dayofweek
    frame["month"] = frame["timestamp"].dt.month
    frame["sender_prefix"] = frame["sender"].str.extract(r"([A-Z]+)")
    frame["receiver_prefix"] = frame["receiver"].str.extract(r"([A-Z]+)")
    return frame
