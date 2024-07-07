import unittest
import numpy as np
import os
import pandas as pd
import sys
# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classification_metrics import (
    get_multi_label_confusion_matrix,
    get_confusion_matrix,
    confusion_matrix_to_dict,
    accuracy_percentage,
    g_mean,
    accuracy,
    tpr,
    tnr,
    fpr,
    fnr,
    precision,
    recall,
    generalized_f,
    matthews_correlation_coefficient,
    auc,
    calculate_multi_label_confusion_based_metrics,
    calculate_confusion_based_metrics,
    calculate_permutation_importance
)

class TestMetricsFunctions(unittest.TestCase):

    def setUp(self):
        # Sample Data
        self.df = pd.DataFrame({
            'prediction': [1, 0, 1, 1, 0, 0, 1, 1],
            'known': [1, 0, 1, 0, 0, 1, 1, 0]
        })

        self.cm_dict = {'tp': 50, 'tn': 40, 'fp': 10, 'fn': 5}
        self.cm_array = np.array([[50, 10], [5, 40]])

    def test_get_confusion_matrix(self):
        cm = get_confusion_matrix(self.df)
        self.assertEqual(cm.shape, (2, 2))  # Shape of confusion matrix should be (2, 2)

    def test_accuracy_percentage(self):
        acc = accuracy_percentage(self.df)
        expected_acc = 62.5  # Calculation: (5 correct / 8 total) * 100
        self.assertAlmostEqual(acc, expected_acc)

    def test_g_mean(self):
        g = g_mean(self.cm_dict)
        expected_g = np.sqrt(tpr(self.cm_dict) * tnr(self.cm_dict))
        self.assertAlmostEqual(g, expected_g)

    def test_accuracy(self):
        if self.cm_array.shape[0] == 2 and self.cm_array.shape[1] == 2:
            # Binary classification case
            acc = accuracy(self.cm_array)
            expected_acc = (self.cm_array[0, 0] + self.cm_array[1, 1]) / self.cm_array.sum()
            self.assertAlmostEqual(acc, expected_acc, places=7)
        else:
            # Multi-class classification case
            acc = accuracy(self.cm_array)
            expected_acc = np.diag(self.cm_array).sum() / self.cm_array.sum()
            self.assertAlmostEqual(acc, expected_acc, places=7)

    def test_tpr(self):
        true_positive_rate = tpr(self.cm_dict)
        expected_tpr = self.cm_dict["tp"] / (self.cm_dict["tp"] + self.cm_dict["fn"])
        self.assertAlmostEqual(true_positive_rate, expected_tpr)

    def test_tnr(self):
        true_negative_rate = tnr(self.cm_dict)
        expected_tnr = self.cm_dict["tn"] / (self.cm_dict["tn"] + self.cm_dict["fp"])
        self.assertAlmostEqual(true_negative_rate, expected_tnr)

    def test_fpr(self):
        false_positive_rate = fpr(self.cm_dict)
        expected_fpr = self.cm_dict["fp"] / (self.cm_dict["fp"] + self.cm_dict["tn"])
        self.assertAlmostEqual(false_positive_rate, expected_fpr)

    def test_fnr(self):
        false_negative_rate = fnr(self.cm_dict)
        expected_fnr = self.cm_dict["fn"] / (self.cm_dict["fn"] + self.cm_dict["tp"])
        self.assertAlmostEqual(false_negative_rate, expected_fnr)

    def test_precision(self):
        prec = precision(self.cm_dict)
        expected_prec = self.cm_dict["tp"] / (self.cm_dict["tp"] + self.cm_dict["fp"])
        self.assertAlmostEqual(prec, expected_prec)

    def test_recall(self):
        rec = recall(self.cm_dict)
        expected_rec = self.cm_dict["tp"] / (self.cm_dict["tp"] + self.cm_dict["fn"])
        self.assertAlmostEqual(rec, expected_rec)

    def test_generalized_f(self):
        f1 = generalized_f(self.cm_dict, beta=1.0)
        expected_f1 = (2 * precision(self.cm_dict) * recall(self.cm_dict)) / (precision(self.cm_dict) + recall(self.cm_dict))
        self.assertAlmostEqual(f1, expected_f1)

    def test_matthews_correlation_coefficient(self):
        mcc = matthews_correlation_coefficient(self.cm_dict)
        expected_mcc = ((self.cm_dict["tp"] * self.cm_dict["tn"]) - (self.cm_dict["fp"] * self.cm_dict["fn"])) / \
                       np.sqrt((self.cm_dict["tp"] + self.cm_dict["fp"]) * (self.cm_dict["tp"] + self.cm_dict["fn"]) * \
                               (self.cm_dict["tn"] + self.cm_dict["fp"]) * (self.cm_dict["tn"] + self.cm_dict["fn"]))
        self.assertAlmostEqual(mcc, expected_mcc)

    def test_auc(self):
        fpr = np.array([0, 0.1, 0.2, 0.3])
        tpr = np.array([0, 0.2, 0.5, 0.8])
        auc_value = auc(fpr, tpr)
        expected_auc = np.trapz(tpr, fpr)  # Using trapezoidal rule for area under the curve
        self.assertAlmostEqual(auc_value, expected_auc)

    def test_calculate_confusion_based_metrics(self):
        metrics = calculate_confusion_based_metrics(cmtx=self.cm_dict, df=self.df)
        self.assertIsInstance(metrics, dict)
        self.assertIn("accuracy", metrics)
        self.assertIn("tpr", metrics)
        self.assertIn("fpr", metrics)
        self.assertIn("tnr", metrics)
        self.assertIn("fnr", metrics)
        self.assertIn("g-mean", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("matthews_correlation_coefficient", metrics)

if __name__ == "__main__":
    unittest.main()

