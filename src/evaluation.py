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


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate standard classification metrics: Precision, Recall, F1 Score.

    Metrics:
        Precision = TP / (TP + FP) - How many predicted positives are actually positive
        Recall = TP / (TP + FN) - How many actual positives were correctly identified
        F1 Score = 2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary containing all metrics
    """
    # Confusion matrix components
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True Positives
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True Negatives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False Positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False Negatives

    # Calculate metrics with zero-division handling
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn},
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy
    }


def print_classification_report(metrics: dict, dataset_name: str = "Test"):
    """
    Print a formatted classification report.

    Args:
        metrics: Dictionary from calculate_classification_metrics()
        dataset_name: Name of the dataset (for display purposes)
    """
    cm = metrics['confusion_matrix']

    print(f"\n{'='*70}")
    print(f"{dataset_name.upper()} SET CLASSIFICATION REPORT")
    print(f"{'='*70}")

    print(f"\nConfusion Matrix:")
    print(f"                    Predicted Negative | Predicted Positive")
    print(f"  Actual Negative:        {cm['TN']:4d}         |       {cm['FP']:4d}")
    print(f"  Actual Positive:        {cm['FN']:4d}         |       {cm['TP']:4d}")

    print(f"\nPerformance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")

    print(f"\nMetric Explanations:")
    print(f"  - Precision: Of all positive predictions, {metrics['precision']*100:.1f}% were correct")
    print(f"  - Recall:    Of all actual positives, {metrics['recall']*100:.1f}% were found")
    print(f"  - F1 Score:  Harmonic mean of Precision and Recall (balanced metric)")


def test_on_unseen_data(model, n_test_samples: int = 1000,
                       normalization_params: dict = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Test the model on unseen data to demonstrate generalization.

    Args:
        model: Trained LogisticRegression model
        n_test_samples: Number of test samples to generate
        normalization_params: Parameters from training normalization (if any)

    Returns:
        Tuple of (X_test, y_test, p_test)
    """
    print("\n" + "="*70)
    print("TESTING ON UNSEEN DATA (GENERALIZATION)")
    print("="*70)

    # Generate new test data (different random state)
    from .data_utils import apply_normalization
    X_test, y_test = generate_synthetic_data(n_samples_per_class=n_test_samples//2,
                                            random_state=123)

    # Apply same normalization as training data
    if normalization_params is not None:
        X_test = apply_normalization(X_test, normalization_params)
        print(f"Applied {normalization_params['method']} normalization to test data")

    # Predict on test data
    p_test = model.predict_proba(X_test)
    y_pred_test = model.predict(X_test)

    # Calculate comprehensive metrics
    test_mse = np.mean((y_test - p_test) ** 2)
    metrics = calculate_classification_metrics(y_test, y_pred_test)

    print(f"\nTest Set Size: {len(y_test)} samples")
    print(f"Test MSE: {test_mse:.6f}")

    # Print detailed classification report
    print_classification_report(metrics, "Test")

    return X_test, y_test, p_test
