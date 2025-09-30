"""
Data loading and preprocessing module for Iris ML project.

This module provides functions to load the Iris dataset, split it into
training and testing sets, and display dataset information.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_iris_data():
    """
    Load iris dataset from sklearn.
    
    Returns:
        tuple: X (features), y (labels), feature_names, target_names
            - X: Feature matrix of shape (150, 4)
            - y: Target vector of shape (150,)
            - feature_names: List of feature names
            - target_names: List of target class names
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    return X, y, feature_names, target_names


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        test_size (float): Proportion of dataset to include in test split
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test


def save_data_info(X, y, feature_names, target_names):
    """
    Print dataset statistics and information.
    
    Args:
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target vector
        feature_names (list): List of feature names
        target_names (list): List of target class names
    """
    print("=" * 50)
    print("IRIS DATASET INFORMATION")
    print("=" * 50)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(target_names)}")
    
    print("\nFeature names:")
    for i, name in enumerate(feature_names):
        print(f"  {i}: {name}")
    
    print("\nTarget classes:")
    for i, name in enumerate(target_names):
        count = np.sum(y == i)
        print(f"  {i}: {name} ({count} samples)")
    
    print("\nFeature statistics:")
    df = pd.DataFrame(X, columns=feature_names)
    print(df.describe())
    
    print("\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for class_idx, count in zip(unique, counts):
        print(f"  {target_names[class_idx]}: {count} samples ({count/len(y)*100:.1f}%)")


if __name__ == "__main__":
    # Example usage
    X, y, feature_names, target_names = load_iris_data()
    save_data_info(X, y, feature_names, target_names)
    
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"\nTrain set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")