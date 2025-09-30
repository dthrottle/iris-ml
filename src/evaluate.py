"""
Model evaluation module for Iris ML project.

This module provides functions to load trained models, evaluate their 
performance, and visualize results.
"""

import argparse
import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support
)
try:
    from data_loader import load_iris_data, split_data
except ImportError:
    from src.data_loader import load_iris_data, split_data


def load_model(filepath):
    """
    Load saved model and scaler from file.
    
    Args:
        filepath (str): Path to the saved model file
        
    Returns:
        tuple: (model, scaler, metadata) where scaler may be None
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model_data = joblib.load(filepath)
    
    # Handle both old format (just model) and new format (dict with model/scaler)
    if isinstance(model_data, dict):
        model = model_data['model']
        scaler = model_data.get('scaler', None)
        metadata = {
            'timestamp': model_data.get('timestamp', 'Unknown'),
            'model_type': model_data.get('model_type', 'Unknown')
        }
    else:
        # Old format - just the model
        model = model_data
        scaler = None
        metadata = {'timestamp': 'Unknown', 'model_type': type(model).__name__}
    
    return model, scaler, metadata


def evaluate_model(model, X_test, y_test, scaler=None):
    """
    Evaluate model performance and calculate metrics.
    
    Args:
        model: Trained sklearn model
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        scaler: Fitted StandardScaler (optional)
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1-score
    """
    # Scale test features if scaler is provided
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred
    }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, target_names, save_path=None):
    """
    Create and display confusion matrix visualization.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        target_names (list): List of class names
        save_path (str, optional): Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def print_classification_report(y_true, y_pred, target_names):
    """
    Print detailed classification report.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels  
        target_names (list): List of class names
    """
    print("\n" + "="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    
    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)


def print_evaluation_summary(metrics, metadata):
    """
    Print a summary of evaluation results.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics
        metadata (dict): Model metadata
    """
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model Type: {metadata['model_type']}")
    print(f"Trained: {metadata['timestamp']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")


def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(description='Evaluate Iris classification model')
    parser.add_argument('--model', 
                       default='models/iris_model.pkl',
                       help='Path to the trained model file')
    parser.add_argument('--save-plot',
                       help='Path to save confusion matrix plot')
    
    args = parser.parse_args()
    
    print("Loading model...")
    try:
        model, scaler, metadata = load_model(args.model)
        print(f"Loaded {metadata['model_type']} model from {args.model}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train a model first using: python src/train.py")
        return
    
    print("Loading test data...")
    X, y, feature_names, target_names = load_iris_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test, scaler)
    
    # Print results
    print_evaluation_summary(metrics, metadata)
    print_classification_report(y_test, metrics['predictions'], target_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, metrics['predictions'], target_names, args.save_plot)
    
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()