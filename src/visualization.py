"""
Visualization

Functions for creating plots and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict


def plot_classification(X: np.ndarray, y: np.ndarray, p_hat: np.ndarray,
                       model, output_dir: str = 'output'):
    """
    Create classification plot with decision boundary.

    Args:
        X: Feature matrix
        y: True labels
        p_hat: Predicted probabilities
        model: Trained LogisticRegression model
        output_dir: Directory to save the plot
    """
    save_path = os.path.join(output_dir, 'classification_plot.png')

    plt.figure(figsize=(12, 10))

    # Predict classes based on threshold 0.5
    y_pred = (p_hat >= 0.5).astype(int)

    # Create color map based on predicted class
    colors = np.array(['red', 'blue'])
    point_colors = colors[y_pred]

    # Plot points with shapes for true vs predicted
    # Ground Truth (X markers)
    for true_class in [0, 1]:
        mask = (y == true_class)
        plt.scatter(X[mask, 0], X[mask, 1],
                   c=point_colors[mask], marker='x', s=100, alpha=0.6,
                   label=f'True Class {true_class}', linewidths=2)

    # Predicted (circle markers) - only plot misclassified for clarity
    misclassified = (y != y_pred)
    if np.any(misclassified):
        plt.scatter(X[misclassified, 0], X[misclassified, 1],
                   c=point_colors[misclassified], marker='o', s=200,
                   facecolors='none', linewidths=3,
                   label='Misclassified (Predicted)', edgecolors=point_colors[misclassified])

    # Plot decision boundary
    x1_min, x1_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    x1_boundary, x2_boundary = model.get_decision_boundary((x1_min, x1_max))
    plt.plot(x1_boundary, x2_boundary, 'g-', linewidth=3, label='Decision Boundary (σ(Z)=0.5)')

    plt.xlabel('X₁', fontsize=14, fontweight='bold')
    plt.ylabel('X₂', fontsize=14, fontweight='bold')
    plt.title('Binary Classification Results\nLogistic Regression with Decision Boundary',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Classification plot saved to: {save_path}")
    plt.close()


def plot_convergence(history: Dict, output_dir: str = 'output'):
    """
    Create convergence plots for Log-Likelihood and MSE.

    Args:
        history: Training history dictionary
        output_dir: Directory to save the plot
    """
    save_path = os.path.join(output_dir, 'convergence_plot.png')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    iterations = range(len(history['log_likelihood']))

    # Plot 1: Log-Likelihood Progress
    ax1.plot(iterations, history['log_likelihood'], 'b-', linewidth=2)
    ax1.set_xlabel('Iteration Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Log-Likelihood Value', fontsize=12, fontweight='bold')
    ax1.set_title('Log-Likelihood Progress\n(Expected: Increasing)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # Plot 2: MSE Progress
    ax2.plot(iterations, history['mse'], 'r-', linewidth=2)
    ax2.set_xlabel('Iteration Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
    ax2.set_title('MSE Progress\n(Expected: Decreasing)',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Convergence plot saved to: {save_path}")
    plt.close()
