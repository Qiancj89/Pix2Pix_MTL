import sys
import unittest
import warnings
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

evaluation = import_module("09_evaluate_classification")


class ClassificationEvaluationTests(unittest.TestCase):
    def test_zero_predicted_positive_is_explicit_without_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = evaluation.binary_metrics([0, 1, 1], [0.1, 0.2, 0.3], threshold=0.5)
        undefined = [item for item in caught if issubclass(item.category, UndefinedMetricWarning)]
        self.assertEqual(undefined, [])
        self.assertEqual(result["reference_positive_count"], 2)
        self.assertEqual(result["predicted_positive_count"], 0)
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["f1"], 0.0)

    def test_validation_threshold_uses_validation_probabilities(self):
        frame = pd.DataFrame({
            "Task_true": [0, 0, 1, 1],
            "Task_prob": [0.05, 0.20, 0.40, 0.45],
        })
        threshold, status = evaluation.validation_youden_threshold(frame, "Task")
        self.assertEqual(status, "validation_youden_j")
        self.assertAlmostEqual(threshold, 0.40)

    def test_cluster_bootstrap_never_splits_patient_rows(self):
        frame = pd.DataFrame({"patient_id": ["p1", "p1", "p2"]})
        plans = evaluation.cluster_bootstrap_indices(frame, iterations=20, seed=7)
        self.assertEqual(len(plans), 20)
        for indices in plans:
            p1_rows = int(np.isin(indices, [0, 1]).sum())
            self.assertIn(p1_rows, [0, 2, 4])

    def test_vectorized_bootstrap_matches_point_metrics_for_unit_weights(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.4, 0.35, 0.8])
        point = evaluation.binary_metrics(y, p, threshold=0.5, include_calibration=False)
        arrays = evaluation.bootstrap_binary_metric_arrays(y, p, 0.5, np.ones((1, len(y))))
        for metric in ["prevalence", "accuracy", "sensitivity", "specificity", "precision", "f1",
                       "kappa", "roc_auc", "pr_auc", "brier"]:
            self.assertAlmostEqual(arrays[metric][0], point[metric], places=12, msg=metric)


if __name__ == "__main__":
    unittest.main()
