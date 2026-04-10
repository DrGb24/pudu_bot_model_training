"""
Unit tests for the predictive maintenance system
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_preparation import DataPreparation, create_synthetic_data
from tree_models import TreeBasedModels
from kpi_metrics import KPIMetrics


class TestDataPreparation(unittest.TestCase):
    """Test data preparation module"""
    
    def setUp(self):
        self.data_prep = DataPreparation()
        
    def test_synthetic_data_creation(self):
        """Test synthetic data generation"""
        df = create_synthetic_data(n_samples=100)
        self.assertEqual(len(df), 100)
        self.assertIn('failure', df.columns)
        self.assertIn('temperature', df.columns)
        
    def test_missing_value_handling(self):
        """Test missing value handling"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [5, np.nan, 7, 8]
        })
        df_cleaned = self.data_prep.handle_missing_values(df)
        self.assertEqual(df_cleaned.isnull().sum().sum(), 0)
        
    def test_categorical_encoding(self):
        """Test categorical feature encoding"""
        df = pd.DataFrame({'cat_col': ['A', 'B', 'A', 'C']})
        df_encoded = self.data_prep.encode_categorical_features(df, ['cat_col'])
        self.assertTrue(df_encoded['cat_col'].dtype in [np.int64, np.int32])


class TestTreeModels(unittest.TestCase):
    """Test tree-based models"""
    
    def setUp(self):
        self.models = TreeBasedModels()
        # Create synthetic data for testing
        self.X_train = np.random.randn(100, 5)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.randn(20, 5)
        self.y_test = np.random.randint(0, 2, 20)
        
    def test_model_initialization(self):
        """Test all models are initialized"""
        self.assertEqual(len(self.models.models), 5)
        self.assertIn('random_forest', self.models.models)
        self.assertIn('xgboost', self.models.models)
        self.assertIn('catboost', self.models.models)
        
    def test_model_training(self):
        """Test model training"""
        self.models.train_model('decision_tree', self.X_train, self.y_train)
        self.assertIn('decision_tree', self.models.trained_models)
        
    def test_prediction(self):
        """Test model prediction"""
        self.models.train_model('decision_tree', self.X_train, self.y_train)
        predictions = self.models.predict('decision_tree', self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(all(p in [0, 1] for p in predictions))
        
    def test_prediction_probability(self):
        """Test prediction probability"""
        self.models.train_model('decision_tree', self.X_train, self.y_train)
        proba = self.models.predict_proba('decision_tree', self.X_test)
        self.assertEqual(proba.shape, (len(self.X_test), 2))
        self.assertTrue(np.allclose(proba.sum(axis=1), 1.0))


class TestKPIMetrics(unittest.TestCase):
    """Test KPI metrics calculation"""
    
    def setUp(self):
        self.kpi = KPIMetrics()
        self.y_true = np.array([0, 1, 1, 0, 1, 0])
        self.y_pred = np.array([0, 1, 1, 0, 0, 0])
        self.y_proba = np.random.rand(6, 2)
        self.y_proba = self.y_proba / self.y_proba.sum(axis=1, keepdims=True)
        
    def test_model_performance_kpis(self):
        """Test model performance KPI calculation"""
        kpis = self.kpi.calculate_model_performance_kpis(self.y_true, self.y_pred)
        
        self.assertIn('prediction_accuracy', kpis)
        self.assertIn('recall', kpis)
        self.assertIn('precision', kpis)
        self.assertIn('f1_score', kpis)
        self.assertTrue(0 <= kpis['prediction_accuracy'] <= 1)
        
    def test_false_alarm_rate(self):
        """Test false alarm rate calculation"""
        kpis = self.kpi.calculate_model_performance_kpis(self.y_true, self.y_pred)
        self.assertTrue(0 <= kpis['false_alarm_rate'] <= 1)
        
    def test_threshold_checking(self):
        """Test if metrics meet target thresholds"""
        kpis = self.kpi.calculate_model_performance_kpis(self.y_true, self.y_pred)
        self.assertIn('accuracy_meets_target', kpis)
        self.assertIn('recall_meets_target', kpis)
        self.assertIsInstance(kpis['accuracy_meets_target'], (bool, np.bool_))


if __name__ == '__main__':
    unittest.main()
