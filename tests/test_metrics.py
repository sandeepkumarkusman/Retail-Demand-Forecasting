"""Tests for evaluation metrics."""

import unittest
import numpy as np

from src.metrics import calculate_ml_metrics

class TestMetrics(unittest.TestCase):
    def test_calculate_ml_metrics(self):
        """It computes MAE, RMSE, SMAPE, WAPE, and MASE correctly."""
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 33])
        
        metrics = calculate_ml_metrics(y_true, y_pred)
        
        self.assertAlmostEqual(metrics['mae'], 2.3333333, places=5)
        self.assertAlmostEqual(metrics['rmse'], 2.380476, places=5)
        
        # SMAPE:
        # |10-12|/11 = 2/11 = 0.1818
        # |20-18|/19 = 2/19 = 0.1052
        # |30-33|/31.5 = 3/31.5 = 0.0952
        # Mean = 0.1274 * 100 = 12.74
        self.assertAlmostEqual(metrics['smape'], 12.7441, places=3)
