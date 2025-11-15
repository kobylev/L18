"""
Data Utilities

Functions for generating and preparing synthetic datasets.
"""

import numpy as np
from typing import Tuple, Optional, Literal


def generate_synthetic_data(n_samples_per_class: int = 5000,
                           random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset with two well-separated clusters.

    Creates two clusters using multivariate normal distributions:
    - Class 0: Centered around (0.3, 0.3)
    - Class 1: Centered around (0.7, 0.7)

    Args:
        n_samples_per_class: Number of samples per class
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (features, labels)
        - features: shape (2*n_samples_per_class, 2) normalized to [0, 1]
        - labels: shape (2*n_samples_per_class,) with values 0 or 1
    """
    np.random.seed(random_state)

    # Class 0: Cluster centered around (0.3, 0.3)
    class_0_mean = [0.3, 0.3]
    class_0_cov = [[0.01, 0.002], [0.002, 0.01]]
    X_class_0 = np.random.multivariate_normal(class_0_mean, class_0_cov, n_samples_per_class)
    y_class_0 = np.zeros(n_samples_per_class)

    # Class 1: Cluster centered around (0.7, 0.7)
    class_1_mean = [0.7, 0.7]
    class_1_cov = [[0.01, 0.002], [0.002, 0.01]]
    X_class_1 = np.random.multivariate_normal(class_1_mean, class_1_cov, n_samples_per_class)
    y_class_1 = np.ones(n_samples_per_class)

    # Combine the two classes
    X = np.vstack([X_class_0, X_class_1])
    y = np.concatenate([y_class_0, y_class_1])

    # Ensure values are in [0, 1] range
    X = np.clip(X, 0, 1)

    # Shuffle the data
    shuffle_idx = np.random.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    return X, y


def normalize_features(X: np.ndarray,
                      method: Literal['minmax', 'zscore'] = 'zscore',
                      mean: Optional[np.ndarray] = None,
                      std: Optional[np.ndarray] = None,
                      min_val: Optional[np.ndarray] = None,
                      max_val: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
    """
    Normalize features using Min-Max scaling or Z-Score standardization.

    Z-Score Normalization (Recommended for Gradient Ascent):
        z = (x - μ) / σ
        where μ is the mean and σ is the standard deviation

    Min-Max Normalization:
        x_scaled = (x - min) / (max - min)

    Args:
        X: Feature matrix of shape (n_samples, n_features)
        method: 'zscore' for standardization or 'minmax' for [0,1] scaling
        mean: Pre-computed mean (for test data), if None computed from X
        std: Pre-computed std (for test data), if None computed from X
        min_val: Pre-computed min (for test data), if None computed from X
        max_val: Pre-computed max (for test data), if None computed from X

    Returns:
        Tuple of (normalized_X, normalization_params)
        normalization_params contains the statistics for applying to test data
    """
    X_normalized = X.copy()
    params = {'method': method}

    if method == 'zscore':
        # Z-Score Normalization (Standardization)
        if mean is None:
            mean = np.mean(X, axis=0)
        if std is None:
            std = np.std(X, axis=0)
            # Prevent division by zero
            std = np.where(std == 0, 1.0, std)

        X_normalized = (X - mean) / std
        params['mean'] = mean
        params['std'] = std

    elif method == 'minmax':
        # Min-Max Normalization
        if min_val is None:
            min_val = np.min(X, axis=0)
        if max_val is None:
            max_val = np.max(X, axis=0)

        # Prevent division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1.0, range_val)

        X_normalized = (X - min_val) / range_val
        params['min'] = min_val
        params['max'] = max_val

    else:
        raise ValueError(f"Unknown normalization method: {method}. Use 'zscore' or 'minmax'.")

    return X_normalized, params


def apply_normalization(X: np.ndarray, params: dict) -> np.ndarray:
    """
    Apply normalization to new data using pre-computed parameters.

    This is used for normalizing test data using statistics from training data.

    Args:
        X: Feature matrix to normalize
        params: Dictionary containing normalization parameters from normalize_features()

    Returns:
        Normalized feature matrix
    """
    method = params['method']

    if method == 'zscore':
        return (X - params['mean']) / params['std']
    elif method == 'minmax':
        range_val = params['max'] - params['min']
        range_val = np.where(range_val == 0, 1.0, range_val)
        return (X - params['min']) / range_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")
