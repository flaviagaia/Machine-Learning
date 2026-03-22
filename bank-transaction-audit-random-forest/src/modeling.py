from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data_pipeline import DATA_PATH, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "random_forest_audit_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
IMPORTANCE_PATH = ARTIFACTS_DIR / "feature_importance.csv"
IMPORTANCE_PLOT_PATH = ARTIFACTS_DIR / "feature_importance.png"
PREDICTIONS_PATH = ARTIFACTS_DIR / "predictions.csv"

TARGET_COLUMN = "risk_score"


def build_training_frame() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    frame = load_dataset(DATA_PATH)

    feature_columns = [
        "sender",
        "receiver",
        "amount",
        "transaction_type",
        "location",
        "device_type",
        "is_foreign_transaction",
        "time_of_day",
        "hour",
        "day",
        "day_of_week",
        "month",
        "sender_prefix",
        "receiver_prefix",
    ]

    numeric_features = ["amount", "is_foreign_transaction", "time_of_day", "hour", "day", "day_of_week", "month"]
    categorical_features = [
        "sender",
        "receiver",
        "transaction_type",
        "location",
        "device_type",
        "sender_prefix",
        "receiver_prefix",
    ]

    x = frame[feature_columns].copy()
    y = frame[TARGET_COLUMN].astype(int)
    return x, y, numeric_features, categorical_features


def train_random_forest(random_state: int = 42) -> dict:
    x, y, numeric_features, categorical_features = build_training_frame()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=350,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_state,
    )

    pipeline.fit(x_train, y_train)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)
    report = classification_report(y_test, predictions, output_dict=True)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": pipeline.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(IMPORTANCE_PATH, index=False)

    top_importances = importances.head(15).sort_values("importance")
    plt.figure(figsize=(10, 6))
    plt.barh(top_importances["feature"], top_importances["importance"], color="#1f6f5f")
    plt.title("Random Forest feature importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(IMPORTANCE_PLOT_PATH, dpi=160)
    plt.close()

    predictions_frame = x_test.copy()
    predictions_frame["actual_risk_score"] = y_test.values
    predictions_frame["predicted_risk_score"] = predictions
    predictions_frame["fraud_probability"] = probabilities.round(4)
    predictions_frame.to_csv(PREDICTIONS_PATH, index=False)

    metrics = {
        "data_path": str(DATA_PATH),
        "n_rows": int(len(x)),
        "n_features_before_encoding": int(x.shape[1]),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "positive_rate": round(float(y.mean()), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "precision_class_1": round(float(report["1"]["precision"]), 4),
        "recall_class_1": round(float(report["1"]["recall"]), 4),
        "top_features": importances.head(10).to_dict(orient="records"),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
