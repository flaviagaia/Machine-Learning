from __future__ import annotations

from src.modeling import (
    IMPORTANCE_PATH,
    IMPORTANCE_PLOT_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PREDICTIONS_PATH,
    train_random_forest,
)


def main() -> None:
    metrics = train_random_forest()

    print("Bank Transaction Audit Random Forest")
    print("-" * 44)
    print(f"Rows: {metrics['n_rows']}")
    print(f"Train size: {metrics['train_size']}")
    print(f"Test size: {metrics['test_size']}")
    print(f"Positive rate: {metrics['positive_rate']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(f"Feature importance saved to: {IMPORTANCE_PATH}")
    print(f"Feature importance plot saved to: {IMPORTANCE_PLOT_PATH}")


if __name__ == "__main__":
    main()
