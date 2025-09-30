# Iris ML Project Specification

## Project Overview
A simple machine learning project for classifying iris flowers using scikit-learn. This project demonstrates the complete ML workflow: data loading, training, evaluation, and prediction.

## Project Structure
```
iris-ml-project/
├── README.md
├── requirements.txt
├── data/
│   └── iris.csv (optional, can load from sklearn)
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   └── (trained models saved here)
├── notebooks/
│   └── exploration.ipynb (optional)
└── tests/
    └── test_model.py
```

## File Specifications

### requirements.txt
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
```

### src/data_loader.py
**Purpose**: Load and preprocess the Iris dataset

**Functions**:
- `load_iris_data()` - Load iris dataset from sklearn
  - Returns: X (features), y (labels), feature_names, target_names
- `split_data(X, y, test_size=0.2, random_state=42)` - Split into train/test sets
  - Returns: X_train, X_test, y_train, y_test
- `save_data_info(X, y, feature_names, target_names)` - Print dataset statistics

**Output**: Clean train/test splits ready for modeling

### src/train.py
**Purpose**: Train the ML model

**Functions**:
- `create_model(model_type='random_forest')` - Initialize model
  - Supported types: 'random_forest', 'svm', 'logistic_regression'
- `train_model(model, X_train, y_train)` - Fit model to training data
  - Returns: Trained model
- `save_model(model, filepath='models/iris_model.pkl')` - Save trained model using joblib

**CLI Usage**:
```bash
python src/train.py --model random_forest --save models/iris_model.pkl
```

### src/evaluate.py
**Purpose**: Evaluate model performance

**Functions**:
- `load_model(filepath)` - Load saved model
- `evaluate_model(model, X_test, y_test)` - Calculate metrics
  - Returns: Dictionary with accuracy, precision, recall, F1-score
- `plot_confusion_matrix(y_true, y_pred, target_names)` - Visualize results
- `print_classification_report(y_true, y_pred, target_names)` - Detailed metrics

**CLI Usage**:
```bash
python src/evaluate.py --model models/iris_model.pkl
```

**Expected Output**:
- Accuracy score
- Classification report (precision, recall, F1 per class)
- Confusion matrix visualization

### src/predict.py
**Purpose**: Make predictions on new data

**Functions**:
- `predict_single(model, features)` - Predict single flower
  - Input: List or array of 4 measurements [sepal_length, sepal_width, petal_length, petal_width]
  - Returns: Species name and confidence scores
- `predict_batch(model, features_array)` - Predict multiple flowers
- `interactive_predict()` - Interactive CLI for predictions

**CLI Usage**:
```bash
# Predict single sample
python src/predict.py --model models/iris_model.pkl --features 5.1,3.5,1.4,0.2

# Interactive mode
python src/predict.py --model models/iris_model.pkl --interactive
```

## README.md Contents

### Sections Required:
1. **Project Description** - Brief overview of iris classification
2. **Installation** - How to install dependencies
3. **Dataset Info** - Description of features and classes
4. **Usage Examples**:
   - Training a model
   - Evaluating performance
   - Making predictions
5. **Model Performance** - Expected accuracy (~95-98%)
6. **Project Structure** - File organization
7. **Future Improvements** - Possible enhancements

## Implementation Details

### Model Configuration
- **Default Model**: Random Forest Classifier
- **Alternative Models**: SVM, Logistic Regression
- **Hyperparameters**: Use sklearn defaults initially
- **Train/Test Split**: 80/20
- **Random State**: 42 (for reproducibility)

### Data Processing
- **Scaling**: StandardScaler for SVM and Logistic Regression
- **No scaling needed**: For Random Forest
- **No missing values**: Iris dataset is clean

### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision/Recall/F1**: Per-class metrics
- **Confusion Matrix**: Visualization of predictions
- **Cross-validation**: Optional 5-fold CV

### Model Persistence
- **Format**: joblib pickle files
- **Location**: `models/` directory
- **Naming**: `iris_<model_type>_<timestamp>.pkl`

## Testing Requirements

### tests/test_model.py
- Test data loading functions
- Test train/test split ratios
- Test model prediction shapes
- Test model saving/loading
- Verify prediction outputs are valid class labels

## Optional Enhancements
1. **Hyperparameter tuning** - GridSearchCV
2. **Feature importance** - Visualization
3. **Multiple model comparison** - Compare RF, SVM, LR
4. **Web interface** - Simple Flask/Streamlit app
5. **Docker container** - Containerized deployment
6. **CI/CD** - GitHub Actions for testing
7. **Data visualization** - PCA plot, feature distributions

## Expected Performance
- **Training time**: < 1 second
- **Accuracy**: 95-98% on test set
- **Model size**: < 1 MB
- **Inference time**: < 1ms per prediction

## Git Workflow
1. Initial commit with project structure
2. Implement data_loader.py
3. Implement train.py
4. Implement evaluate.py
5. Implement predict.py
6. Add tests
7. Add documentation
8. Final polish and README

## Success Criteria
- [ ] All functions documented with docstrings
- [ ] CLI works for all modules
- [ ] Model achieves >95% accuracy
- [ ] Code follows PEP 8
- [ ] README has clear usage examples
- [ ] Tests pass
- [ ] Can train, save, load, and predict successfully