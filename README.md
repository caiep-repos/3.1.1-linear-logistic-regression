# Linear and Logistic Regression with Scikit-Learn

## Problem Description

In this assignment, you will learn to use scikit-learn for linear and logistic regression. You'll implement practical machine learning workflows including model training, evaluation, cross-validation, and model comparison.

## Learning Objectives

- Learn to use scikit-learn's LinearRegression and LogisticRegression
- Understand model evaluation metrics for regression and classification
- Implement train-test split and cross-validation
- Compare different models on the same dataset
- Build end-to-end ML workflows

## Setup Instructions

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Assignment Tasks

Open the `assignment.py` file and implement the following functions:

### 1. Linear Regression (`train_linear_regression`)
- **Purpose**: Train a linear regression model using sklearn
- **Features**: Train-test split, model training, prediction, evaluation
- **Returns**: Model, predictions, and regression metrics (MSE, RMSE, R²)

### 2. Logistic Regression (`train_logistic_regression`)
- **Purpose**: Train a logistic regression model using sklearn
- **Features**: Train-test split, model training, prediction, evaluation
- **Returns**: Model, predictions, and classification metrics (accuracy, etc.)

### 3. Model Evaluation (`evaluate_regression_model` & `evaluate_classification_model`)
- **Purpose**: Calculate appropriate metrics for model performance
- **Regression**: MSE, RMSE, R² score
- **Classification**: Accuracy, classification report

### 4. Cross-Validation (`cross_validate_model`)
- **Purpose**: Perform k-fold cross-validation on any sklearn model
- **Features**: Calculate mean and standard deviation of scores

### 5. Model Comparison (`compare_models`)
- **Purpose**: Compare linear and logistic regression on the same dataset
- **Features**: Automatically determine if data is regression or classification

## Implementation Tips

- Use `train_test_split` for data splitting
- Apply `StandardScaler` if needed for logistic regression
- Use appropriate metrics for each problem type
- Handle both numpy arrays and pandas DataFrames
- Return dictionaries with consistent key names

## Testing Your Solution

Run the test file to verify your implementation:

```bash
python test.py
```

The test suite includes:
- Individual function testing
- Model performance validation
- Cross-validation testing
- Model comparison testing

## Expected Output

All tests should pass when your implementation is correct. Each function should return properly structured dictionaries with the expected keys and reasonable performance metrics.
