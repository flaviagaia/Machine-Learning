from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.train import METRICS_PATH, MODEL_PATH, PREDICTIONS_PATH, train_and_evaluate


class CreditRiskPipelineTest(unittest.TestCase):
    def test_training_pipeline_produces_artifacts_and_metrics(self):
        metrics = train_and_evaluate(random_state=42)

        self.assertIn("best_model", metrics)
        self.assertIn(metrics["best_model"]["model_name"], {"logistic_regression", "random_forest"})
        self.assertGreater(metrics["best_model"]["f1_score"], 0.65)
        self.assertTrue(MODEL_PATH.exists())
        self.assertTrue(METRICS_PATH.exists())
        self.assertTrue(PREDICTIONS_PATH.exists())

        stored_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(stored_metrics["best_model"]["model_name"], metrics["best_model"]["model_name"])


if __name__ == "__main__":
    unittest.main()
