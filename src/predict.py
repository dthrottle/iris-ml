"""
Prediction module for Iris ML project.

This module provides functions to make predictions on new iris flower 
measurements using trained models.
"""

import argparse
import numpy as np
import joblib
try:
    from data_loader import load_iris_data
except ImportError:
    from src.data_loader import load_iris_data


def load_model_for_prediction(filepath):
    """
    Load saved model and scaler for making predictions.
    
    Args:
        filepath (str): Path to the saved model file
        
    Returns:
        tuple: (model, scaler, target_names) where scaler may be None
        
    Raises:
        FileNotFoundError: If model file doesn't exist
    """
    try:
        model_data = joblib.load(filepath)
        
        # Handle both old format (just model) and new format (dict)
        if isinstance(model_data, dict):
            model = model_data['model']
            scaler = model_data.get('scaler', None)
        else:
            model = model_data
            scaler = None
        
        # Load target names from dataset
        _, _, _, target_names = load_iris_data()
        
        return model, scaler, target_names
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {filepath}")


def predict_single(model, features, scaler=None, target_names=None):
    """
    Predict species for a single flower.
    
    Args:
        model: Trained sklearn model
        features (list or np.ndarray): List/array of 4 measurements 
                                     [sepal_length, sepal_width, petal_length, petal_width]
        scaler: Fitted StandardScaler (optional)
        target_names (list): List of class names
        
    Returns:
        dict: Dictionary containing predicted species name and confidence scores
    """
    # Convert to numpy array and reshape for single prediction
    features = np.array(features).reshape(1, -1)
    
    # Validate input
    if features.shape[1] != 4:
        raise ValueError("Features must contain exactly 4 measurements: "
                        "[sepal_length, sepal_width, petal_length, petal_width]")
    
    # Scale features if scaler is provided
    if scaler is not None:
        features = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Get species name
    species_name = target_names[prediction] if target_names is not None else f"Class {prediction}"
    
    # Create confidence scores dictionary
    confidence_scores = {}
    if target_names is not None:
        for i, name in enumerate(target_names):
            confidence_scores[name] = probabilities[i]
    else:
        for i, prob in enumerate(probabilities):
            confidence_scores[f"Class {i}"] = prob
    
    result = {
        'predicted_species': species_name,
        'predicted_class': int(prediction),
        'confidence_scores': confidence_scores,
        'max_confidence': float(np.max(probabilities))
    }
    
    return result


def predict_batch(model, features_array, scaler=None, target_names=None):
    """
    Predict species for multiple flowers.
    
    Args:
        model: Trained sklearn model
        features_array (np.ndarray): Array of shape (n_samples, 4) with measurements
        scaler: Fitted StandardScaler (optional)
        target_names (list): List of class names
        
    Returns:
        list: List of prediction dictionaries
    """
    features_array = np.array(features_array)
    
    # Validate input
    if features_array.ndim != 2 or features_array.shape[1] != 4:
        raise ValueError("Features array must have shape (n_samples, 4)")
    
    # Scale features if scaler is provided
    if scaler is not None:
        features_array = scaler.transform(features_array)
    
    # Make predictions
    predictions = model.predict(features_array)
    probabilities = model.predict_proba(features_array)
    
    results = []
    for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
        species_name = target_names[pred] if target_names is not None else f"Class {pred}"
        
        confidence_scores = {}
        if target_names is not None:
            for j, name in enumerate(target_names):
                confidence_scores[name] = probs[j]
        else:
            for j, prob in enumerate(probs):
                confidence_scores[f"Class {j}"] = prob
        
        result = {
            'sample_index': i,
            'predicted_species': species_name,
            'predicted_class': int(pred),
            'confidence_scores': confidence_scores,
            'max_confidence': float(np.max(probs))
        }
        results.append(result)
    
    return results


def interactive_predict(model, scaler=None, target_names=None):
    """
    Interactive CLI for making predictions.
    
    Args:
        model: Trained sklearn model
        scaler: Fitted StandardScaler (optional)
        target_names (list): List of class names
    """
    print("\n" + "="*60)
    print("INTERACTIVE IRIS PREDICTION")
    print("="*60)
    print("Enter measurements for an iris flower.")
    print("Features needed:")
    print("  1. Sepal length (cm)")
    print("  2. Sepal width (cm)")
    print("  3. Petal length (cm)")
    print("  4. Petal width (cm)")
    print("\nType 'quit' or 'exit' to stop.")
    print("-"*60)
    
    while True:
        try:
            # Get user input
            user_input = input("\nEnter measurements (comma-separated) or 'quit': ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Parse measurements
            measurements = [float(x.strip()) for x in user_input.split(',')]
            
            if len(measurements) != 4:
                print("Error: Please enter exactly 4 measurements.")
                continue
            
            # Make prediction
            result = predict_single(model, measurements, scaler, target_names)
            
            # Display results
            print(f"\nPrediction Results:")
            print(f"  Predicted Species: {result['predicted_species']}")
            print(f"  Confidence: {result['max_confidence']:.3f}")
            print(f"\nAll Confidence Scores:")
            for species, confidence in result['confidence_scores'].items():
                bar = "█" * int(confidence * 20)  # Simple bar chart
                print(f"  {species:15}: {confidence:.3f} {bar}")
            
        except ValueError as e:
            if "could not convert string to float" in str(e):
                print("Error: Invalid input. Please enter 4 numeric values separated by commas.")
            else:
                print(f"Error: {e}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main function with CLI support."""
    parser = argparse.ArgumentParser(description='Make predictions with Iris classification model')
    parser.add_argument('--model', 
                       default='models/iris_model.pkl',
                       help='Path to the trained model file')
    parser.add_argument('--features',
                       help='Comma-separated feature values: sepal_length,sepal_width,petal_length,petal_width')
    parser.add_argument('--interactive', 
                       action='store_true',
                       help='Start interactive prediction mode')
    
    args = parser.parse_args()
    
    print("Loading model...")
    try:
        model, scaler, target_names = load_model_for_prediction(args.model)
        print(f"Model loaded successfully from {args.model}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train a model first using: python src/train.py")
        return
    
    if args.interactive:
        # Interactive mode
        interactive_predict(model, scaler, target_names)
    
    elif args.features:
        # Single prediction mode
        try:
            features = [float(x.strip()) for x in args.features.split(',')]
            
            if len(features) != 4:
                print("Error: Please provide exactly 4 feature values.")
                return
            
            print(f"Making prediction for features: {features}")
            result = predict_single(model, features, scaler, target_names)
            
            print(f"\nPrediction Results:")
            print(f"  Predicted Species: {result['predicted_species']}")
            print(f"  Confidence: {result['max_confidence']:.3f}")
            print(f"\nConfidence Scores:")
            for species, confidence in result['confidence_scores'].items():
                print(f"  {species}: {confidence:.3f}")
                
        except ValueError:
            print("Error: Invalid feature values. Please provide numeric values.")
            
    else:
        # No arguments provided - show example usage
        print("\nNo prediction mode specified. Choose one of the following:")
        print(f"  1. Single prediction: python src/predict.py --model {args.model} --features 5.1,3.5,1.4,0.2")
        print(f"  2. Interactive mode: python src/predict.py --model {args.model} --interactive")


if __name__ == "__main__":
    main()