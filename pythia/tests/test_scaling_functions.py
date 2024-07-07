import unittest
import pandas as pd
import numpy as np
import os
import sys


# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scaling import autoscale, minmaxscale, logarithm2, logarithm10

class TestScalingFunctions(unittest.TestCase):

    def setUp(self):
        """Set up a DataFrame for testing"""
        self.df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [5, 6, 7, 8, 9],
            'C': [9, 10, 11, 12, 13]
        })

    def test_autoscale(self):
        """Test the autoscale function"""
        scaled_df = autoscale(self.df)
        self.assertAlmostEqual(scaled_df['A'].mean(), 0)
        self.assertAlmostEqual(scaled_df['B'].mean(), 0)
        self.assertAlmostEqual(scaled_df['C'].mean(), 0)

    def test_minmaxscale(self):
        """Test the minmaxscale function"""
        scaled_df = minmaxscale(self.df)
        self.assertAlmostEqual(scaled_df['A'].min(), 0)
        self.assertAlmostEqual(scaled_df['A'].max(), 1)
        self.assertAlmostEqual(scaled_df['B'].min(), 0)
        self.assertAlmostEqual(scaled_df['B'].max(), 1)
        self.assertAlmostEqual(scaled_df['C'].min(), 0)
        self.assertAlmostEqual(scaled_df['C'].max(), 1)

    def test_logarithm2(self):
        """Test the logarithm2 function"""
        df_positive = self.df + 1  # shift to ensure all values are positive
        log_df = logarithm2(df_positive)
        self.assertTrue(np.allclose(log_df, np.log2(df_positive)))

    def test_logarithm10(self):
        """Test the logarithm10 function"""
        df_positive = self.df + 1  # shift to ensure all values are positive
        log_df = logarithm10(df_positive)
        self.assertTrue(np.allclose(log_df, np.log10(df_positive)))

if __name__ == '__main__':
    unittest.main()

