from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import (
    IMPORTANCE_PATH,
    IMPORTANCE_PLOT_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PREDICTIONS_PATH,
    train_random_forest,
)


class BankTransactionAuditRandomForestTest(unittest.TestCase):
    def test_training_pipeline_generates_artifacts(self):
        metrics = train_random_forest()

        self.assertGreater(metrics["n_rows"], 1000)
        self.assertGreater(metrics["f1_score"], 0.65)
        self.assertGreater(metrics["roc_auc"], 0.8)
        self.assertTrue(MODEL_PATH.exists())
        self.assertTrue(METRICS_PATH.exists())
        self.assertTrue(PREDICTIONS_PATH.exists())
        self.assertTrue(IMPORTANCE_PATH.exists())
        self.assertTrue(IMPORTANCE_PLOT_PATH.exists())

        stored_metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        self.assertEqual(stored_metrics["f1_score"], metrics["f1_score"])


if __name__ == "__main__":
    unittest.main()
