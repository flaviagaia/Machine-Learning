from __future__ import annotations

from src.train import MODEL_PATH, METRICS_PATH, PREDICTIONS_PATH, train_and_evaluate


def main() -> None:
    metrics = train_and_evaluate()
    best_model = metrics["best_model"]

    print("Credit Risk Scoring")
    print("-" * 40)
    print(f"Best model: {best_model['model_name']}")
    print(f"Selected threshold: {best_model['threshold']:.3f}")
    print(f"Validation F1-score: {best_model['f1_score']:.3f}")
    print(f"Validation Precision: {best_model['precision']:.3f}")
    print(f"Validation Recall: {best_model['recall']:.3f}")
    print(f"Validation ROC-AUC: {best_model['roc_auc']:.3f}")
    print(f"Validation PR-AUC: {best_model['pr_auc']:.3f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
