import unittest
import numpy as np
from sklearn.datasets import make_classification, make_regression
from assignment import (train_linear_regression, train_logistic_regression, 
                       evaluate_regression_model, evaluate_classification_model,
                       cross_validate_model, compare_models)

class TestLinearLogisticRegression(unittest.TestCase):
    
    def test_linear_regression(self):
        """Test linear regression training and evaluation"""
        X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=42)
        
        result = train_linear_regression(X, y, test_size=0.2, random_state=42)
        
        # Check if result contains expected keys
        expected_keys = ['model', 'X_test', 'y_test', 'y_pred', 'metrics']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check if model is trained
        self.assertIsNotNone(result['model'])
        
        # Check if metrics are reasonable
        metrics = result['metrics']
        self.assertIn('mse', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('r2', metrics)
        self.assertGreater(metrics['r2'], 0.8)  # Should have good R² score
    
    def test_logistic_regression(self):
        """Test logistic regression training and evaluation"""
        X, y = make_classification(n_samples=200, n_features=5, n_classes=2, 
                                 random_state=42, n_informative=3)
        
        result = train_logistic_regression(X, y, test_size=0.2, random_state=42)
        
        # Check if result contains expected keys
        expected_keys = ['model', 'X_test', 'y_test', 'y_pred', 'metrics']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check if model is trained
        self.assertIsNotNone(result['model'])
        
        # Check if accuracy is reasonable
        metrics = result['metrics']
        self.assertIn('accuracy', metrics)
        self.assertGreater(metrics['accuracy'], 0.7)  # Should have decent accuracy
    
    def test_evaluate_regression_model(self):
        """Test regression model evaluation"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        
        metrics = evaluate_regression_model(y_true, y_pred)
        
        # Check if all expected metrics are present
        expected_metrics = ['mse', 'rmse', 'r2']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
        
        # Check if R² is reasonable for this simple case
        self.assertGreater(metrics['r2'], 0.9)
    
    def test_evaluate_classification_model(self):
        """Test classification model evaluation"""
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 1])
        
        metrics = evaluate_classification_model(y_true, y_pred)
        
        # Check if accuracy is present
        self.assertIn('accuracy', metrics)
        self.assertAlmostEqual(metrics['accuracy'], 0.8, places=1)
    
    def test_cross_validate_model(self):
        """Test cross-validation functionality"""
        from sklearn.linear_model import LinearRegression
        
        X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
        model = LinearRegression()
        
        cv_results = cross_validate_model(model, X, y, cv=3)
        
        # Check if results contain expected keys
        self.assertIn('mean_score', cv_results)
        self.assertIn('std_score', cv_results)
        
        # Check if scores are reasonable
        self.assertGreater(cv_results['mean_score'], 0.8)
        self.assertGreaterEqual(cv_results['std_score'], 0)
    
    def test_compare_models(self):
        """Test model comparison functionality"""
        # Test with regression data
        X, y = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)
        
        results = compare_models(X, y, test_size=0.2, random_state=42)
        
        # Check if both models are present
        self.assertIn('linear_regression', results)
        self.assertIn('logistic_regression', results)
        
        # Check if linear regression performs better on regression data
        linear_r2 = results['linear_regression']['metrics']['r2']
        logistic_acc = results['logistic_regression']['metrics']['accuracy']
        
        self.assertGreater(linear_r2, 0.8)
        self.assertGreater(logistic_acc, 0.5)  # Logistic should still work reasonably


if __name__ == '__main__':
    unittest.main()
