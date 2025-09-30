"""
Test suite for Iris ML project.

This module contains unit tests for the main functionality of the iris
classification project including data loading, model training, and predictions.
"""

import unittest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Import modules to test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_iris_data, split_data, save_data_info
from src.train import create_model, train_model, save_model
from src.evaluate import load_model, evaluate_model
from src.predict import predict_single, predict_batch, load_model_for_prediction


class TestDataLoader(unittest.TestCase):
    """Test cases for data_loader.py functions."""
    
    def setUp(self):
        """Set up test data."""
        self.X, self.y, self.feature_names, self.target_names = load_iris_data()
    
    def test_load_iris_data(self):
        """Test iris data loading function."""
        X, y, feature_names, target_names = load_iris_data()
        
        # Test data shapes
        self.assertEqual(X.shape, (150, 4))
        self.assertEqual(y.shape, (150,))
        self.assertEqual(len(feature_names), 4)
        self.assertEqual(len(target_names), 3)
        
        # Test data types
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(feature_names, list)
        self.assertTrue(hasattr(target_names, '__len__'))  # Could be list or array
    
    def test_split_data_ratios(self):
        """Test train/test split ratios."""
        X_train, X_test, y_train, y_test = split_data(self.X, self.y, test_size=0.2)
        
        # Test split ratios
        self.assertEqual(X_train.shape[0], 120)  # 80% of 150
        self.assertEqual(X_test.shape[0], 30)    # 20% of 150
        self.assertEqual(y_train.shape[0], 120)
        self.assertEqual(y_test.shape[0], 30)
        
        # Test feature dimensions preserved
        self.assertEqual(X_train.shape[1], 4)
        self.assertEqual(X_test.shape[1], 4)
    
    def test_split_data_custom_ratio(self):
        """Test custom train/test split ratio."""
        X_train, X_test, y_train, y_test = split_data(self.X, self.y, test_size=0.3)
        
        self.assertEqual(X_train.shape[0], 105)  # 70% of 150
        self.assertEqual(X_test.shape[0], 45)    # 30% of 150
    
    @patch('builtins.print')
    def test_save_data_info(self, mock_print):
        """Test data info printing function."""
        save_data_info(self.X, self.y, self.feature_names, self.target_names)
        
        # Verify that print was called (basic smoke test)
        self.assertTrue(mock_print.called)


class TestModelTraining(unittest.TestCase):
    """Test cases for train.py functions."""
    
    def setUp(self):
        """Set up test data."""
        self.X, self.y, _, _ = load_iris_data()
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.X, self.y)
    
    def test_create_model_random_forest(self):
        """Test random forest model creation."""
        model, scaler = create_model('random_forest')
        
        self.assertIsNotNone(model)
        self.assertIsNone(scaler)  # RF doesn't need scaling
        self.assertEqual(type(model).__name__, 'RandomForestClassifier')
    
    def test_create_model_svm(self):
        """Test SVM model creation."""
        model, scaler = create_model('svm')
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)  # SVM needs scaling
        self.assertEqual(type(model).__name__, 'SVC')
        self.assertEqual(type(scaler).__name__, 'StandardScaler')
    
    def test_create_model_logistic_regression(self):
        """Test logistic regression model creation."""
        model, scaler = create_model('logistic_regression')
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)  # LR benefits from scaling
        self.assertEqual(type(model).__name__, 'LogisticRegression')
    
    def test_create_model_invalid_type(self):
        """Test invalid model type raises error."""
        with self.assertRaises(ValueError):
            create_model('invalid_model')
    
    def test_train_model_without_scaler(self):
        """Test model training without scaler."""
        model, scaler = create_model('random_forest')
        trained_model, fitted_scaler = train_model(model, self.X_train, self.y_train, scaler)
        
        self.assertIsNotNone(trained_model)
        self.assertIsNone(fitted_scaler)
        
        # Test that model can make predictions
        predictions = trained_model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertTrue(all(0 <= pred <= 2 for pred in predictions))  # Valid class labels
    
    def test_train_model_with_scaler(self):
        """Test model training with scaler."""
        model, scaler = create_model('svm')
        trained_model, fitted_scaler = train_model(model, self.X_train, self.y_train, scaler)
        
        self.assertIsNotNone(trained_model)
        self.assertIsNotNone(fitted_scaler)
        
        # Test that model can make predictions with scaled data
        X_test_scaled = fitted_scaler.transform(self.X_test)
        predictions = trained_model.predict(X_test_scaled)
        self.assertEqual(len(predictions), len(self.y_test))
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.pkl')
            
            # Train and save model
            model, scaler = create_model('random_forest')
            trained_model, fitted_scaler = train_model(model, self.X_train, self.y_train, scaler)
            save_model(trained_model, fitted_scaler, model_path)
            
            # Verify file was created
            self.assertTrue(os.path.exists(model_path))
            
            # Load and test model
            loaded_model, loaded_scaler, metadata = load_model(model_path)
            self.assertIsNotNone(loaded_model)
            self.assertIsNone(loaded_scaler)  # RF doesn't use scaler
            self.assertIn('model_type', metadata)


class TestModelEvaluation(unittest.TestCase):
    """Test cases for evaluate.py functions."""
    
    def setUp(self):
        """Set up test data and trained model."""
        self.X, self.y, _, self.target_names = load_iris_data()
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.X, self.y)
        
        # Train a simple model for testing
        model, scaler = create_model('random_forest')
        self.trained_model, self.fitted_scaler = train_model(model, self.X_train, self.y_train, scaler)
    
    def test_evaluate_model(self):
        """Test model evaluation function."""
        metrics = evaluate_model(self.trained_model, self.X_test, self.y_test, self.fitted_scaler)
        
        # Test that all expected metrics are present
        expected_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'predictions']
        for key in expected_keys:
            self.assertIn(key, metrics)
        
        # Test metric ranges
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        self.assertGreaterEqual(metrics['precision'], 0.0)
        self.assertLessEqual(metrics['precision'], 1.0)
        
        # Test predictions shape
        self.assertEqual(len(metrics['predictions']), len(self.y_test))
    
    def test_load_model_file_not_found(self):
        """Test load_model with non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_model('non_existent_model.pkl')


class TestPredictions(unittest.TestCase):
    """Test cases for predict.py functions."""
    
    def setUp(self):
        """Set up test data and trained model."""
        self.X, self.y, _, self.target_names = load_iris_data()
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(self.X, self.y)
        
        # Train a model for testing
        model, scaler = create_model('random_forest')
        self.trained_model, self.fitted_scaler = train_model(model, self.X_train, self.y_train, scaler)
    
    def test_predict_single(self):
        """Test single prediction function."""
        test_features = [5.1, 3.5, 1.4, 0.2]  # Typical setosa measurements
        
        result = predict_single(self.trained_model, test_features, 
                               self.fitted_scaler, self.target_names)
        
        # Test result structure
        expected_keys = ['predicted_species', 'predicted_class', 'confidence_scores', 'max_confidence']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Test prediction validity
        self.assertIn(result['predicted_class'], [0, 1, 2])
        self.assertIn(result['predicted_species'], self.target_names)
        self.assertGreaterEqual(result['max_confidence'], 0.0)
        self.assertLessEqual(result['max_confidence'], 1.0)
        
        # Test confidence scores
        self.assertEqual(len(result['confidence_scores']), 3)
        confidence_sum = sum(result['confidence_scores'].values())
        self.assertAlmostEqual(confidence_sum, 1.0, places=5)  # Should sum to 1
    
    def test_predict_single_invalid_features(self):
        """Test single prediction with invalid feature count."""
        invalid_features = [5.1, 3.5, 1.4]  # Only 3 features instead of 4
        
        with self.assertRaises(ValueError):
            predict_single(self.trained_model, invalid_features, 
                          self.fitted_scaler, self.target_names)
    
    def test_predict_batch(self):
        """Test batch prediction function."""
        test_features = [
            [5.1, 3.5, 1.4, 0.2],
            [6.0, 3.0, 4.5, 1.5],
            [6.5, 3.0, 5.5, 2.0]
        ]
        
        results = predict_batch(self.trained_model, test_features, 
                               self.fitted_scaler, self.target_names)
        
        # Test result count
        self.assertEqual(len(results), 3)
        
        # Test each result structure
        for i, result in enumerate(results):
            self.assertEqual(result['sample_index'], i)
            self.assertIn(result['predicted_class'], [0, 1, 2])
            self.assertIn(result['predicted_species'], self.target_names)
            self.assertEqual(len(result['confidence_scores']), 3)
    
    def test_predict_batch_invalid_shape(self):
        """Test batch prediction with invalid array shape."""
        invalid_features = [[5.1, 3.5, 1.4]]  # Only 3 features per sample
        
        with self.assertRaises(ValueError):
            predict_batch(self.trained_model, invalid_features, 
                         self.fitted_scaler, self.target_names)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    def test_complete_workflow(self):
        """Test the complete ML workflow from data loading to prediction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'integration_test_model.pkl')
            
            # 1. Load data
            X, y, feature_names, target_names = load_iris_data()
            self.assertEqual(X.shape, (150, 4))
            
            # 2. Split data
            X_train, X_test, y_train, y_test = split_data(X, y)
            
            # 3. Create and train model
            model, scaler = create_model('random_forest')
            trained_model, fitted_scaler = train_model(model, X_train, y_train, scaler)
            
            # 4. Save model
            save_model(trained_model, fitted_scaler, model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # 5. Load model
            loaded_model, loaded_scaler, metadata = load_model(model_path)
            
            # 6. Evaluate model
            metrics = evaluate_model(loaded_model, X_test, y_test, loaded_scaler)
            self.assertGreater(metrics['accuracy'], 0.8)  # Should be high accuracy
            
            # 7. Make predictions
            test_sample = X_test[0]
            result = predict_single(loaded_model, test_sample, loaded_scaler, target_names)
            self.assertIn(result['predicted_species'], target_names)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)