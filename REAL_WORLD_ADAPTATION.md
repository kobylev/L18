# Real-World Dataset Adaptation Guide

This document provides guidance on adapting the Logistic Regression implementation to work with real-world, complex datasets instead of synthetic data.

## Overview

The current implementation uses synthetic 2D data with two well-separated clusters. Real-world datasets present additional challenges:

- **Categorical features** (e.g., gender, country, product type)
- **High dimensionality** (hundreds or thousands of features)
- **Missing values** (incomplete data)
- **Mixed data types** (numerical and categorical)
- **Class imbalance** (unequal distribution of classes)
- **Feature scaling requirements**
- **Non-linear relationships**

## Step-by-Step Adaptation Process

### 1. Data Loading and Preparation

#### Loading Real-World Data

Replace the synthetic data generation with actual data loading:

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Example: Load CSV data
def load_real_world_data(filepath: str, target_column: str):
    """
    Load real-world dataset from CSV.

    Args:
        filepath: Path to CSV file
        target_column: Name of the target/label column

    Returns:
        X: Feature matrix
        y: Target vector
    """
    df = pd.read_csv(filepath)

    # Separate features and target
    y = df[target_column].values
    X = df.drop(columns=[target_column])

    return X, y

# Usage
X_raw, y = load_real_world_data('data/customer_data.csv', target_column='churned')
```

### 2. Handling Missing Values

Real-world data often contains missing values that must be addressed:

```python
from sklearn.impute import SimpleImputer

def handle_missing_values(X: pd.DataFrame, strategy: str = 'mean'):
    """
    Handle missing values in the dataset.

    Strategies:
        - 'mean': Replace with column mean (for numerical features)
        - 'median': Replace with column median (for numerical features)
        - 'most_frequent': Replace with mode (for categorical features)
        - 'constant': Replace with a constant value

    Args:
        X: Feature DataFrame with possible missing values
        strategy: Imputation strategy

    Returns:
        X_imputed: DataFrame with no missing values
        imputer: Fitted imputer (to apply to test data)
    """
    # For numerical features
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    imputer_num = SimpleImputer(strategy=strategy)
    X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])

    # For categorical features
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    imputer_cat = SimpleImputer(strategy='most_frequent')
    X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])

    return X, {'numerical': imputer_num, 'categorical': imputer_cat}

# Usage
X_filled, imputers = handle_missing_values(X_raw)
```

### 3. Encoding Categorical Features

Logistic Regression requires numerical inputs. Convert categorical variables:

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

def encode_categorical_features(X: pd.DataFrame, encoding_method: str = 'onehot'):
    """
    Encode categorical features as numerical values.

    Methods:
        - 'label': Label encoding (0, 1, 2, ...) for ordinal features
        - 'onehot': One-hot encoding for nominal features (recommended)

    Args:
        X: Feature DataFrame with categorical columns
        encoding_method: Encoding strategy

    Returns:
        X_encoded: Encoded feature matrix
        encoders: Dictionary of fitted encoders
    """
    X_encoded = X.copy()
    encoders = {}

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    if encoding_method == 'onehot':
        # One-hot encoding creates binary columns for each category
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        encoders['method'] = 'onehot'
        encoders['columns'] = categorical_cols.tolist()

    elif encoding_method == 'label':
        # Label encoding assigns integers to categories
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            encoders[col] = le

    return X_encoded, encoders

# Usage
X_encoded, encoders = encode_categorical_features(X_filled, encoding_method='onehot')
```

### 4. Feature Scaling (Z-Score Normalization)

**Critical for Gradient Ascent stability!**

Use the built-in normalization functions from `data_utils.py`:

```python
from src.data_utils import normalize_features, apply_normalization

# Normalize training data with Z-Score (recommended)
X_train_normalized, norm_params = normalize_features(
    X_train,
    method='zscore'  # or 'minmax' for [0,1] scaling
)

# Apply same normalization to test data
X_test_normalized = apply_normalization(X_test, norm_params)

# Why Z-Score is better:
# - Centers data around 0 (mean = 0, std = 1)
# - Makes gradient ascent more stable
# - Prevents features with large ranges from dominating
# - Faster convergence
```

### 5. Handling High Dimensionality

For datasets with many features (e.g., 100+ columns):

```python
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

def reduce_dimensionality(X: np.ndarray, method: str = 'pca', n_features: int = 10):
    """
    Reduce feature dimensionality to improve performance.

    Methods:
        - 'pca': Principal Component Analysis (unsupervised)
        - 'selectk': Select K best features (supervised)

    Args:
        X: Feature matrix
        method: Dimensionality reduction method
        n_features: Number of features to keep

    Returns:
        X_reduced: Reduced feature matrix
        reducer: Fitted reducer object
    """
    if method == 'pca':
        # PCA: Create new features that capture most variance
        reducer = PCA(n_components=n_features)
        X_reduced = reducer.fit_transform(X)
        print(f"Explained variance: {reducer.explained_variance_ratio_.sum():.2%}")

    elif method == 'selectk':
        # SelectKBest: Choose features most correlated with target
        reducer = SelectKBest(score_func=f_classif, k=n_features)
        X_reduced = reducer.fit_transform(X, y_train)

    return X_reduced, reducer

# Usage (for high-dimensional data)
if X_train.shape[1] > 50:
    X_train_reduced, reducer = reduce_dimensionality(X_train, method='pca', n_features=20)
    X_test_reduced = reducer.transform(X_test)
```

### 6. Handling Class Imbalance

When one class is much more common than the other:

```python
from sklearn.utils import resample

def balance_classes(X: np.ndarray, y: np.ndarray, method: str = 'oversample'):
    """
    Balance class distribution in training data.

    Methods:
        - 'oversample': Duplicate minority class samples
        - 'undersample': Remove majority class samples
        - 'smote': Synthetic Minority Over-sampling (requires imblearn)

    Args:
        X: Feature matrix
        y: Labels
        method: Balancing method

    Returns:
        X_balanced, y_balanced: Balanced dataset
    """
    if method == 'oversample':
        # Oversample minority class
        mask_majority = (y == 0)
        mask_minority = (y == 1)

        X_majority = X[mask_majority]
        X_minority = X[mask_minority]
        y_majority = y[mask_majority]
        y_minority = y[mask_minority]

        # Resample minority to match majority
        X_minority_upsampled, y_minority_upsampled = resample(
            X_minority, y_minority,
            n_samples=len(y_majority),
            random_state=42
        )

        X_balanced = np.vstack([X_majority, X_minority_upsampled])
        y_balanced = np.concatenate([y_majority, y_minority_upsampled])

    elif method == 'undersample':
        # Undersample majority class
        # Similar implementation but downsample majority
        pass

    return X_balanced, y_balanced

# Check class balance
unique, counts = np.unique(y_train, return_counts=True)
print(f"Class distribution: {dict(zip(unique, counts))}")

# Balance if needed
if counts[0] / counts[1] > 2 or counts[1] / counts[0] > 2:
    X_train, y_train = balance_classes(X_train, y_train, method='oversample')
```

### 7. Complete Real-World Pipeline

Here's a complete example workflow:

```python
from src.model import LogisticRegression
from src.data_utils import normalize_features, apply_normalization
from src.evaluation import calculate_classification_metrics, print_classification_report
import pandas as pd
import numpy as np

# Step 1: Load data
df = pd.read_csv('data/real_world_data.csv')
y = df['target'].values
X = df.drop(columns=['target', 'id'])  # Remove target and non-predictive columns

# Step 2: Split train/test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Handle missing values
X_train_filled, imputers = handle_missing_values(X_train, strategy='median')
X_test_filled = X_test.copy()
X_test_filled[X_test.select_dtypes(include=['int64', 'float64']).columns] = \
    imputers['numerical'].transform(X_test[X_test.select_dtypes(include=['int64', 'float64']).columns])

# Step 4: Encode categorical features
X_train_encoded, encoders = encode_categorical_features(X_train_filled, encoding_method='onehot')
X_test_encoded = pd.get_dummies(X_test_filled, columns=encoders['columns'], drop_first=True)

# Ensure test has same columns as train
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Step 5: Convert to numpy
X_train_np = X_train_encoded.values
X_test_np = X_test_encoded.values

# Step 6: Normalize features (CRITICAL!)
X_train_norm, norm_params = normalize_features(X_train_np, method='zscore')
X_test_norm = apply_normalization(X_test_np, norm_params)

# Step 7: Train model with Mini-Batch Gradient Ascent
model = LogisticRegression(
    learning_rate=0.01,      # May need tuning for real data
    n_iterations=5000,
    tolerance=1e-6,
    batch_size=256           # Use mini-batch for large datasets
)

model.fit(X_train_norm, y_train, verbose=True)

# Step 8: Evaluate
y_pred_train = model.predict(X_train_norm)
y_pred_test = model.predict(X_test_norm)

train_metrics = calculate_classification_metrics(y_train, y_pred_train)
test_metrics = calculate_classification_metrics(y_test, y_pred_test)

print_classification_report(train_metrics, "Training")
print_classification_report(test_metrics, "Test")
```

## Hyperparameter Tuning for Real Data

The default hyperparameters may not work well for all datasets:

```python
def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """
    Manually test different hyperparameter combinations.

    For automated tuning, consider using GridSearchCV or RandomizedSearchCV
    from scikit-learn (requires wrapping our model).
    """
    learning_rates = [0.001, 0.01, 0.1]
    batch_sizes = [32, 128, 512, None]  # None = full batch

    best_f1 = 0
    best_params = {}

    for lr in learning_rates:
        for bs in batch_sizes:
            model = LogisticRegression(
                learning_rate=lr,
                n_iterations=1000,
                batch_size=bs
            )
            model.fit(X_train, y_train, verbose=False)

            y_pred = model.predict(X_val)
            metrics = calculate_classification_metrics(y_val, y_pred)

            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_params = {'learning_rate': lr, 'batch_size': bs}

    print(f"Best params: {best_params}, F1: {best_f1:.4f}")
    return best_params
```

## Example Real-World Datasets

### 1. Titanic Survival (Beginner)
- **Features**: Age, Sex, Class, Fare, Siblings/Spouses, Parents/Children
- **Target**: Survived (0 or 1)
- **Challenges**: Missing values (Age, Cabin), categorical features, class imbalance
- **Source**: [Kaggle Titanic](https://www.kaggle.com/c/titanic)

### 2. Credit Card Fraud Detection (Intermediate)
- **Features**: 28 PCA-transformed features, Time, Amount
- **Target**: Fraud (0 or 1)
- **Challenges**: Severe class imbalance (0.17% fraud), high dimensionality
- **Source**: [Kaggle Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)

### 3. Adult Income Prediction (Intermediate)
- **Features**: Age, Education, Occupation, Hours/week, etc.
- **Target**: Income >50K (0 or 1)
- **Challenges**: Mixed numerical/categorical, missing values
- **Source**: [UCI Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

### 4. Customer Churn (Advanced)
- **Features**: Customer demographics, usage patterns, service features
- **Target**: Churned (0 or 1)
- **Challenges**: Imbalanced classes, many features, temporal aspects
- **Source**: Various telecom datasets

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Model doesn't converge | Lower learning rate, increase iterations, check normalization |
| Poor test performance | Add regularization, reduce features, get more data, check for leakage |
| Training too slow | Use mini-batch (batch_size=256), reduce features (PCA), vectorize code |
| Predictions always same class | Check class balance, adjust decision threshold, examine feature scaling |
| NaN/Inf in training | Check for missing values, ensure proper normalization, clip extreme values |

## Limitations of Binary Logistic Regression

Be aware of when this approach may not be suitable:

- **Multi-class problems**: Need multinomial logistic regression or one-vs-rest
- **Non-linear relationships**: Consider feature engineering or other algorithms (SVM, Random Forest, Neural Networks)
- **Very high dimensions**: May need regularization (L1/L2) or deep learning
- **Temporal/Sequential data**: Consider LSTM, GRU, or time-series models
- **Image/Text data**: Need specialized architectures (CNNs, Transformers)

## Summary Checklist

When adapting to real-world data, ensure you:

- [ ] Load and explore the dataset (check shape, types, missing values)
- [ ] Split data into train/validation/test sets
- [ ] Handle missing values appropriately
- [ ] Encode categorical features (one-hot or label encoding)
- [ ] **Normalize features using Z-Score normalization**
- [ ] Consider dimensionality reduction if needed
- [ ] Address class imbalance if present
- [ ] Train with appropriate hyperparameters (tune learning rate and batch size)
- [ ] Evaluate using multiple metrics (Accuracy, Precision, Recall, F1)
- [ ] Check for overfitting (compare train vs test performance)
- [ ] Visualize results and decision boundaries (if 2D)

## Additional Resources

- **Scikit-learn Preprocessing**: https://scikit-learn.org/stable/modules/preprocessing.html
- **Handling Imbalanced Data**: https://imbalanced-learn.org/
- **Feature Engineering**: https://feature-engine.readthedocs.io/
- **Pandas Data Cleaning**: https://pandas.pydata.org/docs/user_guide/missing_data.html
