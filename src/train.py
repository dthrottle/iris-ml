"""
Model training module for Iris ML project.

This module provides functions to create, train, and save machine learning
models for iris flower classification.
"""

import argparse
import os
from datetime import datetime
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
try:
    from data_loader import load_iris_data, split_data
except ImportError:
    from src.data_loader import load_iris_data, split_data


def create_model(model_type='random_forest'):
    """
    Initialize a machine learning model.
    
    Args:
        model_type (str): Type of model to create
            Options: 'random_forest', 'svm', 'logistic_regression'
    
    Returns:
        tuple: (model, scaler) where scaler is None for random_forest
    
    Raises:
        ValueError: If model_type is not supported
    """
    scaler = None
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'svm':
        model = SVC(kernel='rbf', random_state=42, probability=True)
        scaler = StandardScaler()  # SVM needs feature scaling
    elif model_type == 'logistic_regression':
        model = LogisticRegression(random_state=42, max_iter=1000)
        scaler = StandardScaler()  # Logistic regression benefits from scaling
    else:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Choose from: 'random_forest', 'svm', 'logistic_regression'")
    
    return model, scaler


def train_model(model, X_train, y_train, scaler=None):
    """
    Train the machine learning model.
    
    Args:
        model: Sklearn model instance
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels  
        scaler: StandardScaler instance (optional)
    
    Returns:
        tuple: (trained_model, fitted_scaler) where fitted_scaler is None 
               if no scaler was provided
    """
    # Scale features if scaler is provided
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
        return model, scaler
    else:
        model.fit(X_train, y_train)
        return model, None


def save_model(model, scaler=None, filepath='models/iris_model.pkl'):
    """
    Save trained model and scaler using joblib.
    
    Args:
        model: Trained sklearn model
        scaler: Fitted StandardScaler (optional)
        filepath (str): Path to save the model
    """
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save both model and scaler together
    model_data = {
        'model': model,
        'scaler': scaler,
        'timestamp': datetime.now().isoformat(),
        'model_type': type(model).__name__
    }
    
    joblib.dump(model_data, filepath)
    print(f"Model saved to: {filepath}")


def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(description='Train Iris classification model')
    parser.add_argument('--model', 
                       choices=['random_forest', 'svm', 'logistic_regression'],
                       default='random_forest',
                       help='Type of model to train')
    parser.add_argument('--save', 
                       default='models/iris_model.pkl',
                       help='Path to save the trained model')
    
    args = parser.parse_args()
    
    print("Loading Iris dataset...")
    X, y, feature_names, target_names = load_iris_data()
    
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"Creating {args.model} model...")
    model, scaler = create_model(args.model)
    
    print("Training model...")
    trained_model, fitted_scaler = train_model(model, X_train, y_train, scaler)
    
    print("Saving model...")
    save_model(trained_model, fitted_scaler, args.save)
    
    # Quick training accuracy check
    if fitted_scaler is not None:
        X_train_scaled = fitted_scaler.transform(X_train)  
        train_accuracy = trained_model.score(X_train_scaled, y_train)
    else:
        train_accuracy = trained_model.score(X_train, y_train)
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()