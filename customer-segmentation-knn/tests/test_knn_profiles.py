from __future__ import annotations

import json
import unittest
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.knn_profiles import MODEL_PATH, NEIGHBORS_PATH, PCA_PLOT_PATH, SUMMARY_PATH, build_profile_engine


class KNNProfilesTest(unittest.TestCase):
    def test_profile_engine_builds_artifacts(self):
        summary = build_profile_engine(random_state=42)

        self.assertGreater(summary["n_customers"], 500)
        self.assertEqual(summary["knn_neighbors"], 5)
        self.assertGreater(summary["silhouette_score"], 0.15)
        self.assertEqual(len(summary["sample_neighbors"]), 5)
        self.assertEqual(len(summary["pca_explained_variance_ratio"]), 2)
        self.assertTrue(MODEL_PATH.exists())
        self.assertTrue(SUMMARY_PATH.exists())
        self.assertTrue(NEIGHBORS_PATH.exists())
        self.assertTrue(PCA_PLOT_PATH.exists())

        stored_summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(stored_summary["sample_customer_id"], summary["sample_customer_id"])


if __name__ == "__main__":
    unittest.main()
