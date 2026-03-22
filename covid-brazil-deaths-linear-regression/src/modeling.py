from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data_pipeline import DAILY_DATA_PATH, load_daily_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "linear_regression_covid_deaths.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"
PLOT_PATH = ARTIFACTS_DIR / "actual_vs_predicted.png"


def create_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy().sort_values("date").reset_index(drop=True)
    data = data[(data["date"] >= "2024-01-01") & (data["date"] <= "2024-12-31")].reset_index(drop=True)
    data["target_deaths_7d_avg"] = data["covid_deaths"].rolling(7).mean()

    for lag in [1, 2, 3, 7, 14]:
        data[f"lag_{lag}"] = data["target_deaths_7d_avg"].shift(lag)

    data["rolling_mean_7"] = data["target_deaths_7d_avg"].shift(1).rolling(7).mean()
    data["rolling_mean_14"] = data["target_deaths_7d_avg"].shift(1).rolling(14).mean()
    data["rolling_std_7"] = data["target_deaths_7d_avg"].shift(1).rolling(7).std()
    data["trend_1_7"] = data["target_deaths_7d_avg"].shift(1) - data["target_deaths_7d_avg"].shift(7)
    data["day_of_week"] = data["date"].dt.dayofweek
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
    data["month"] = data["date"].dt.month

    data = data.dropna().reset_index(drop=True)
    return data


def train_linear_regression(test_size: float = 0.2) -> dict:
    frame = load_daily_dataset(DAILY_DATA_PATH)
    featured = create_features(frame)

    feature_columns = [
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "lag_14",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_std_7",
        "trend_1_7",
        "day_of_week",
        "week_of_year",
        "month",
    ]

    split_index = int(len(featured) * (1 - test_size))
    train = featured.iloc[:split_index].copy()
    test = featured.iloc[split_index:].copy()

    x_train = train[feature_columns]
    y_train = train["target_deaths_7d_avg"]
    x_test = test[feature_columns]
    y_test = test["target_deaths_7d_avg"]

    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    predictions = np.maximum(predictions, 0)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_columns": feature_columns}, MODEL_PATH)

    predictions_frame = test[["date", "covid_deaths", "target_deaths_7d_avg"]].copy()
    predictions_frame["predicted_deaths_7d_avg"] = np.round(predictions, 2)
    predictions_frame.to_csv(PREDICTIONS_PATH, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(
        predictions_frame["date"],
        predictions_frame["target_deaths_7d_avg"],
        label="Actual 7-day average",
        linewidth=2,
    )
    plt.plot(
        predictions_frame["date"],
        predictions_frame["predicted_deaths_7d_avg"],
        label="Predicted 7-day average",
        linewidth=2,
    )
    plt.title("COVID-19 deaths in Brazil: actual vs predicted 7-day average")
    plt.xlabel("Date")
    plt.ylabel("Deaths")
    plt.legend(frameon=False)
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=160)
    plt.close()

    metrics = {
        "data_path": str(DAILY_DATA_PATH),
        "n_observations": int(len(featured)),
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "target_definition": "7-day moving average of daily COVID-19 deaths",
        "feature_columns": feature_columns,
        "r2_score": round(float(r2), 4),
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
