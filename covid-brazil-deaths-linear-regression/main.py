from __future__ import annotations

from src.modeling import METRICS_PATH, MODEL_PATH, PLOT_PATH, PREDICTIONS_PATH, train_linear_regression


def main() -> None:
    metrics = train_linear_regression()

    print("COVID-19 Brazil Deaths Linear Regression")
    print("-" * 44)
    print(f"Observations: {metrics['n_observations']}")
    print(f"Train size: {metrics['train_size']}")
    print(f"Test size: {metrics['test_size']}")
    print(f"R2 score: {metrics['r2_score']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(f"Plot saved to: {PLOT_PATH}")


if __name__ == "__main__":
    main()
