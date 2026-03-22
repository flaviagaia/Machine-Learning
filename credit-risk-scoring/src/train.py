from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .data import DATA_PATH, ensure_dataset
from .modeling import TARGET_COLUMN, build_candidate_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "best_credit_risk_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"


@dataclass
class ModelResult:
    model_name: str
    threshold: float
    f1_score: float
    precision: float
    recall: float
    roc_auc: float
    pr_auc: float


def _find_best_threshold(y_true: pd.Series, probabilities) -> float:
    precision_values, recall_values, thresholds = precision_recall_curve(y_true, probabilities)

    best_threshold = 0.5
    best_f1 = -1.0

    for precision_value, recall_value, threshold in zip(precision_values[1:], recall_values[1:], thresholds):
        if precision_value + recall_value == 0:
            continue
        current_f1 = (2 * precision_value * recall_value) / (precision_value + recall_value)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = float(threshold)

    return round(best_threshold, 4)


def _evaluate_model(model, x_valid: pd.DataFrame, y_valid: pd.Series, model_name: str) -> ModelResult:
    probabilities = model.predict_proba(x_valid)[:, 1]
    threshold = _find_best_threshold(y_valid, probabilities)
    predictions = (probabilities >= threshold).astype(int)
    return ModelResult(
        model_name=model_name,
        threshold=threshold,
        f1_score=f1_score(y_valid, predictions),
        precision=precision_score(y_valid, predictions),
        recall=recall_score(y_valid, predictions),
        roc_auc=roc_auc_score(y_valid, probabilities),
        pr_auc=average_precision_score(y_valid, probabilities),
    )


def train_and_evaluate(random_state: int = 42, refresh_dataset: bool = False) -> dict:
    dataset = ensure_dataset(path=DATA_PATH, random_state=random_state, refresh=refresh_dataset)

    x = dataset.drop(columns=[TARGET_COLUMN])
    y = dataset[TARGET_COLUMN]

    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=random_state,
    )

    results: list[ModelResult] = []
    trained_models = {}

    for candidate in build_candidate_models(random_state=random_state):
        candidate.pipeline.fit(x_train, y_train)
        result = _evaluate_model(candidate.pipeline, x_valid, y_valid, candidate.name)
        results.append(result)
        trained_models[candidate.name] = candidate.pipeline

    best_result = max(results, key=lambda item: (item.f1_score, item.pr_auc, item.roc_auc))
    best_model = trained_models[best_result.model_name]

    probabilities = best_model.predict_proba(x_valid)[:, 1]
    predictions = (probabilities >= best_result.threshold).astype(int)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    prediction_frame = x_valid.copy()
    prediction_frame["actual_defaulted"] = y_valid.to_numpy()
    prediction_frame["predicted_defaulted"] = predictions
    prediction_frame["default_probability"] = probabilities.round(4)
    prediction_frame.to_csv(PREDICTIONS_PATH, index=False)

    metrics = {
        "dataset_path": str(DATA_PATH),
        "validation_size": int(len(x_valid)),
        "default_rate": round(float(y.mean()), 4),
        "candidate_results": [asdict(result) for result in results],
        "best_model": asdict(best_result),
        "classification_report": classification_report(y_valid, predictions, output_dict=True),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
