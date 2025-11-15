"""
Evaluation

Functions for model evaluation, testing, and results formatting.
"""

import numpy as np
import pandas as pd
from typing import Tuple
from .data_utils import generate_synthetic_data


def create_results_table(X: np.ndarray, y: np.ndarray,
                        p_hat: np.ndarray, n_display: int = 20) -> pd.DataFrame:
    """
    Create results table as specified in PRD.

    Args:
        X: Feature matrix
        y: True labels
        p_hat: Predicted probabilities
        n_display: Number of samples to display

    Returns:
        DataFrame with results
    """
    # Calculate squared errors
    error_squared = (y - p_hat) ** 2

    # Create DataFrame
    df = pd.DataFrame({
        'X1': X[:n_display, 0],
        'X2': X[:n_display, 1],
        'Y (Ground Truth)': y[:n_display].astype(int),
        'Sigmoid(P) (Prediction)': p_hat[:n_display],
        'Error Squared': error_squared[:n_display]
    })

    # Add MSE row at the bottom
    mse = np.mean(error_squared)
    mse_row = pd.DataFrame({
        'X1': [''],
        'X2': ['Average Error (MSE)'],
        'Y (Ground Truth)': [''],
        'Sigmoid(P) (Prediction)': [''],
        'Error Squared': [mse]
    })

    df = pd.concat([df, mse_row], ignore_index=True)

    return df


def test_on_unseen_data(model, n_test_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Test the model on unseen data to demonstrate generalization.

    Args:
        model: Trained LogisticRegression model
        n_test_samples: Number of test samples to generate

    Returns:
        Tuple of (X_test, y_test, p_test)
    """
    print("\n" + "="*70)
    print("TESTING ON UNSEEN DATA (GENERALIZATION)")
    print("="*70)

    # Generate new test data (different random state)
    X_test, y_test = generate_synthetic_data(n_samples_per_class=n_test_samples//2,
                                            random_state=123)

    # Predict on test data
    p_test = model.predict_proba(X_test)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    test_mse = np.mean((y_test - p_test) ** 2)
    accuracy = np.mean(y_pred_test == y_test)

    print(f"\nTest Set Size: {len(y_test)} samples")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Confusion matrix
    tp = np.sum((y_test == 1) & (y_pred_test == 1))
    tn = np.sum((y_test == 0) & (y_pred_test == 0))
    fp = np.sum((y_test == 0) & (y_pred_test == 1))
    fn = np.sum((y_test == 1) & (y_pred_test == 0))

    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn:4d}")
    print(f"  True Positives:  {tp:4d}")
    print(f"  False Positives: {fp:4d}")
    print(f"  False Negatives: {fn:4d}")

    return X_test, y_test, p_test
