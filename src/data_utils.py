"""
Data Utilities

Functions for generating and preparing synthetic datasets.
"""

import numpy as np
from typing import Tuple


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
