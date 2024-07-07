import unittest
import pandas as pd
import numpy as np
import logging

# Add the parent directory to the sys.path
import sys

from sklearn.metrics import confusion_matrix
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from plots import roc_curve_data, plot_roc_curve, precision_recall_data, plot_pr_curve, plot_confusion_matrix, plot_metrics

class TestPlottingFunctions(unittest.TestCase):

    def setUp(self):
        """Set up test data"""
        self.y_test = np.array([0, 1, 0, 1, 0, 1])
        self.probs = np.array([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3], [0.6, 0.4], [0.8, 0.2], [0.4, 0.6]])
        self.df = pd.DataFrame({
            'prediction': [0, 1, 0, 1, 0, 1],
            'known': [0, 1, 0, 1, 0, 1]
        })

    def test_roc_curve_data(self):
        """Test the roc_curve_data function"""
        fp, tp, thresholds, roc_auc = roc_curve_data(self.y_test, self.probs[:, 1])
        self.assertEqual(len(fp), len(tp))
        self.assertEqual(len(fp), len(thresholds))
        self.assertGreaterEqual(roc_auc, 0)
        self.assertLessEqual(roc_auc, 1)

    def test_plot_roc_curve(self):
        """Test the plot_roc_curve function"""
        axes, data = plot_roc_curve(self.probs, self.y_test, return_raw_data=True)
        self.assertIn(1, data)
        self.assertIn("fpr", data[1])
        self.assertIn("tpr", data[1])
        self.assertIn("auc", data[1])
        self.assertGreaterEqual(data[1]["auc"], 0)
        self.assertLessEqual(data[1]["auc"], 1)

    def test_precision_recall_data(self):
        """Test the precision_recall_data function"""
        prec, rec, thresholds, average_precision = precision_recall_data(self.y_test, self.probs[:, 1])
        self.assertEqual(len(prec), len(rec))
        self.assertEqual(len(prec), len(thresholds) + 1)
        self.assertGreaterEqual(average_precision, 0)
        self.assertLessEqual(average_precision, 1)

    def test_plot_pr_curve(self):
        """Test the plot_pr_curve function"""
        axes, data = plot_pr_curve(self.probs, self.y_test, return_raw_data=True)
        self.assertIn(1, data)
        self.assertIn("precision", data[1])
        self.assertIn("recall", data[1])
        self.assertIn("ap", data[1])
        self.assertGreaterEqual(data[1]["ap"], 0)
        self.assertLessEqual(data[1]["ap"], 1)

    def test_plot_confusion_matrix(self):
        """Test the plot_confusion_matrix function"""
        cmx = confusion_matrix(self.df['known'], self.df['prediction'])
        axes = plot_confusion_matrix(cmx)
        self.assertIsNotNone(axes)

    def test_plot_metrics(self):
        """Test the plot_metrics function"""
        axes = plot_metrics(self.df, probabilities=self.probs, roc_curve=True, pr_curve=True)
        self.assertIsNotNone(axes)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()


