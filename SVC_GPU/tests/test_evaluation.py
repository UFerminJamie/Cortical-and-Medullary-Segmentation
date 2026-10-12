"""Checks for the class-specific binary metrics reported after fitting."""

import unittest

import numpy as np

from SVC_GPU.evaluation import classification_metrics


class ClassificationMetricsTests(unittest.TestCase):
    def test_both_classes_use_their_own_positive_orientation(self):
        actual = np.array([0, 0, 0, 1])
        predicted = np.array([0, 0, 1, 1])
        medullary_scores = np.array([-2.0, -1.0, 0.3, 1.0])

        metrics = classification_metrics(actual, predicted, medullary_scores)

        self.assertEqual(metrics["confusion_matrix"], [[2, 1], [0, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        cortical = metrics["per_class"]["cortical"]
        medullary = metrics["per_class"]["medullary"]
        self.assertAlmostEqual(cortical["accuracy"], 0.75)
        self.assertAlmostEqual(cortical["f1_score"], 0.8)
        self.assertAlmostEqual(cortical["sensitivity"], 2 / 3)
        self.assertAlmostEqual(cortical["specificity"], 1.0)
        self.assertAlmostEqual(cortical["auc"], 1.0)
        self.assertAlmostEqual(medullary["accuracy"], 0.75)
        self.assertAlmostEqual(medullary["f1_score"], 2 / 3)
        self.assertAlmostEqual(medullary["sensitivity"], 1.0)
        self.assertAlmostEqual(medullary["specificity"], 2 / 3)
        self.assertAlmostEqual(medullary["auc"], 1.0)

    def test_auc_is_unavailable_when_only_one_true_class_is_present(self):
        metrics = classification_metrics(
            np.array([0, 0]), np.array([0, 0]), np.array([-2.0, -1.0])
        )
        self.assertIsNone(metrics["per_class"]["cortical"]["auc"])
        self.assertIsNone(metrics["per_class"]["medullary"]["auc"])
        self.assertIsNone(metrics["per_class"]["medullary"]["sensitivity"])


if __name__ == "__main__":
    unittest.main()
