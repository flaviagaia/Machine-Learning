from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import METRICS_PATH, MODEL_PATH, PLOT_PATH, PREDICTIONS_PATH, train_linear_regression


class CovidDeathsRegressionPipelineTest(unittest.TestCase):
    def test_training_pipeline_generates_metrics_and_artifacts(self):
        metrics = train_linear_regression()

        self.assertIn("r2_score", metrics)
        self.assertGreater(metrics["n_observations"], 100)
        self.assertGreater(metrics["r2_score"], 0.80)
        self.assertTrue(MODEL_PATH.exists())
        self.assertTrue(METRICS_PATH.exists())
        self.assertTrue(PREDICTIONS_PATH.exists())
        self.assertTrue(PLOT_PATH.exists())

        stored_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(stored_metrics["r2_score"], metrics["r2_score"])


if __name__ == "__main__":
    unittest.main()
