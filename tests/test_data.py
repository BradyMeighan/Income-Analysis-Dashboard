# =============================================================================
# Unit Tests for Data Preprocessing
# =============================================================================

import unittest
import pandas as pd
from data.preprocessing import preprocess_data

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame({
            'workclass': [1, 2, 2, 3],
            'education': [0, 1, 1, 2],
            'marital-status': [1, 2, 2, 3],
            'occupation': [0, 1, 1, 2],
            'relationship': [0, 1, 1, 2],
            'race': [0, 1, 1, 2],
            'sex': [0, 1, 1, 0],
            'native-country': [0, 1, 1, 2],
            'age': [25, 30, 45, 22],
            'capital-gain': [0, 5000, 0, 0],
            'capital-loss': [0, 0, 0, 0],
            'hours-per-week': [40, 50, 60, 30],
            'income': [0, 1, 1, 0],
            'fnlwgt': [123456, 234567, 345678, 456789]
        })

    def test_preprocessing(self):
        preprocessed_data, metrics, num_cols, cat_cols, add_num_feats, means, stds = preprocess_data(self.data)
        # Check duplicates
        self.assertEqual(metrics['duplicates_found'], 0)
        self.assertEqual(metrics['duplicates_removed'], 0)
        # Check missing values
        self.assertEqual(metrics['missing_values_before'], 0)
        self.assertEqual(metrics['missing_values_after'], 0)
        # Check outliers
        self.assertEqual(metrics['outliers_detected'], 0)
        self.assertEqual(metrics['outliers_removed'], 0)
        # Check columns
        self.assertIn('income_numeric', preprocessed_data.columns)
        # Check types
        self.assertTrue(preprocessed_data['income_numeric'].dtype == 'int64')

if __name__ == '__main__':
    unittest.main()
