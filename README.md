# Iris ML Project

A comprehensive machine learning project for classifying iris flowers using scikit-learn. This project demonstrates the complete ML workflow including data loading, model training, evaluation, and prediction with multiple model options and a user-friendly CLI interface.

## Project Description

This project implements a classification system for iris flowers based on their physical measurements. It uses the famous Iris dataset and provides three different machine learning algorithms:
- **Random Forest** (default) - Robust ensemble method
- **Support Vector Machine (SVM)** - Powerful kernel-based classifier  
- **Logistic Regression** - Simple yet effective linear classifier

The project achieves **95-98% accuracy** on the test set and provides both programmatic APIs and command-line interfaces for all operations.

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository:**
```bash
git clone <repository-url>
cd iris-ml
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv iris-env
source iris-env/bin/activate  # On Windows: iris-env\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Dataset Information

The Iris dataset contains measurements of 150 iris flowers from three species:

### Features (4 measurements per flower):
- **Sepal Length** (cm) - Length of the sepal
- **Sepal Width** (cm) - Width of the sepal  
- **Petal Length** (cm) - Length of the petal
- **Petal Width** (cm) - Width of the petal

### Classes (3 species):
- **Setosa** (50 samples) - Typically smaller petals
- **Versicolor** (50 samples) - Medium-sized features
- **Virginica** (50 samples) - Generally larger flowers

### Dataset Statistics:
- **Total samples:** 150
- **Features per sample:** 4
- **Classes:** 3 (balanced dataset)
- **Missing values:** None
- **Data split:** 80% training (120 samples), 20% testing (30 samples)

## Usage Examples

### 1. Training a Model

**Train with default Random Forest:**
```bash
python src/train.py
```

**Train with specific model type:**
```bash
# Random Forest
python src/train.py --model random_forest --save models/rf_model.pkl

# Support Vector Machine
python src/train.py --model svm --save models/svm_model.pkl

# Logistic Regression  
python src/train.py --model logistic_regression --save models/lr_model.pkl
```

### 2. Evaluating Model Performance

**Evaluate default model:**
```bash
python src/evaluate.py
```

**Evaluate specific model:**
```bash
python src/evaluate.py --model models/svm_model.pkl
```

**Save confusion matrix plot:**
```bash
python src/evaluate.py --model models/rf_model.pkl --save-plot confusion_matrix.png
```

### 3. Making Predictions

**Single prediction:**
```bash
# Predict for a flower with measurements: sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2
python src/predict.py --model models/iris_model.pkl --features 5.1,3.5,1.4,0.2
```

**Interactive prediction mode:**
```bash
python src/predict.py --model models/iris_model.pkl --interactive
```

Example interactive session:
```
Enter measurements (comma-separated) or 'quit': 5.1,3.5,1.4,0.2

Prediction Results:
  Predicted Species: setosa
  Confidence: 0.980

All Confidence Scores:
  setosa         : 0.980 ████████████████████
  versicolor     : 0.020 █
  virginica      : 0.000 
```

### 4. Viewing Dataset Information

```bash
python src/data_loader.py
```

This displays comprehensive dataset statistics, feature descriptions, and class distributions.

## Model Performance

### Expected Accuracy by Model Type:
- **Random Forest:** 95-98% (typically 96-97%)
- **SVM:** 95-98% (typically 96-97%)  
- **Logistic Regression:** 93-97% (typically 95-96%)

### Performance Characteristics:
- **Training time:** < 1 second for all models
- **Model size:** < 1 MB
- **Inference time:** < 1ms per prediction
- **Memory usage:** Minimal (<50MB during training)

### Sample Evaluation Output:
```
EVALUATION SUMMARY
==================================================
Model Type: RandomForestClassifier
Trained: 2024-01-15T10:30:45
Accuracy: 0.9667
Precision: 0.9667
Recall: 0.9667
F1-Score: 0.9667

CLASSIFICATION REPORT
==================================================
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.91      1.00      0.95        10
   virginica       1.00      0.90      0.95        10

    accuracy                           0.97        30
   macro avg       0.97      0.97      0.97        30
weighted avg       0.97      0.97      0.97        30
```

## Project Structure

```
iris-ml/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── iris_ml_spec.md          # Project specification
├── LICENSE                  # License file
├── data/                    # Data directory (optional)
├── src/                     # Source code
│   ├── __init__.py         # Package initialization
│   ├── data_loader.py      # Data loading and preprocessing
│   ├── train.py            # Model training with CLI
│   ├── evaluate.py         # Model evaluation with CLI  
│   └── predict.py          # Prediction with CLI
├── models/                  # Saved models directory
│   └── iris_model.pkl      # Default trained model
├── notebooks/               # Jupyter notebooks (optional)
│   └── exploration.ipynb   # Data exploration notebook
└── tests/                   # Unit tests
    └── test_model.py       # Comprehensive test suite
```

## Running Tests

Run the complete test suite:
```bash
python -m pytest tests/ -v
```

Or run tests directly:
```bash
python tests/test_model.py
```

### Test Coverage:
- Data loading functions
- Model creation and training
- Model evaluation metrics
- Prediction accuracy
- File I/O operations
- Error handling
- Complete workflow integration

## API Reference

### Data Loading (`data_loader.py`)
- `load_iris_data()` - Load dataset from sklearn
- `split_data(X, y, test_size, random_state)` - Create train/test splits
- `save_data_info(X, y, feature_names, target_names)` - Display dataset info

### Model Training (`train.py`)  
- `create_model(model_type)` - Initialize ML model
- `train_model(model, X_train, y_train, scaler)` - Train model
- `save_model(model, scaler, filepath)` - Save trained model

### Model Evaluation (`evaluate.py`)
- `load_model(filepath)` - Load saved model
- `evaluate_model(model, X_test, y_test, scaler)` - Calculate metrics
- `plot_confusion_matrix(y_true, y_pred, target_names)` - Visualize results

### Predictions (`predict.py`)
- `predict_single(model, features, scaler, target_names)` - Single prediction
- `predict_batch(model, features_array, scaler, target_names)` - Batch predictions
- `interactive_predict(model, scaler, target_names)` - Interactive CLI

## Future Improvements

### Planned Enhancements:
1. **Hyperparameter Tuning** - GridSearchCV for optimal parameters
2. **Feature Importance Analysis** - Visualization of feature contributions  
3. **Model Comparison Dashboard** - Side-by-side performance comparison
4. **Web Interface** - Flask/Streamlit app for easy predictions
5. **Docker Container** - Containerized deployment
6. **CI/CD Pipeline** - GitHub Actions for automated testing
7. **Data Visualization** - PCA plots and feature distributions  
8. **Model Explainability** - SHAP values for prediction interpretation
9. **REST API** - HTTP endpoint for predictions
10. **Model Monitoring** - Performance tracking over time

### Potential Extensions:
- Support for other iris datasets
- Ensemble methods combining multiple models
- Deep learning approaches with TensorFlow/PyTorch
- Real-time prediction streaming
- Mobile app integration

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Iris Dataset:** Originally collected by Edgar Anderson and made famous by Ronald Fisher
- **Scikit-learn:** For providing excellent machine learning tools
- **Python Community:** For the amazing ecosystem of data science libraries

---

**Quick Start:**
```bash
# Install dependencies
pip install -r requirements.txt

# Train a model
python src/train.py

# Evaluate performance  
python src/evaluate.py

# Make a prediction
python src/predict.py --features 5.1,3.5,1.4,0.2
```
Iris ML
