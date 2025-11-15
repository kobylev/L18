"""
Logistic Regression Implementation Package

A modular implementation of binary logistic regression using gradient ascent.
"""

__version__ = "1.0.0"
__author__ = "AI Expert"

from .model import LogisticRegression
from .data_utils import generate_synthetic_data, normalize_features, apply_normalization
from .visualization import plot_classification, plot_convergence
from .evaluation import (test_on_unseen_data, create_results_table,
                        calculate_classification_metrics, print_classification_report)
from .utils import create_output_directory
from .cross_validation import cross_validate, k_fold_split, stratified_k_fold_split, grid_search_cv

__all__ = [
    'LogisticRegression',
    'generate_synthetic_data',
    'normalize_features',
    'apply_normalization',
    'plot_classification',
    'plot_convergence',
    'test_on_unseen_data',
    'create_results_table',
    'calculate_classification_metrics',
    'print_classification_report',
    'create_output_directory',
    'cross_validate',
    'k_fold_split',
    'stratified_k_fold_split',
    'grid_search_cv',
]
