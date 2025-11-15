"""
Logistic Regression Implementation Package

A modular implementation of binary logistic regression using gradient ascent.
"""

__version__ = "1.0.0"
__author__ = "AI Expert"

from .model import LogisticRegression
from .data_utils import generate_synthetic_data
from .visualization import plot_classification, plot_convergence
from .evaluation import test_on_unseen_data, create_results_table
from .utils import create_output_directory

__all__ = [
    'LogisticRegression',
    'generate_synthetic_data',
    'plot_classification',
    'plot_convergence',
    'test_on_unseen_data',
    'create_results_table',
    'create_output_directory',
]
