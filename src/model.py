"""
Logistic Regression Model

Main model class combining base functionality and training logic.
"""

from .model_base import LogisticRegressionBase
from .trainer import LogisticRegressionTrainer


class LogisticRegression(LogisticRegressionBase, LogisticRegressionTrainer):
    """
    Complete Logistic Regression implementation using Gradient Ascent.

    This class combines:
    - Base mathematical functions (from LogisticRegressionBase)
    - Training logic (from LogisticRegressionTrainer)

    Attributes:
        learning_rate (float): Learning step (alpha) for gradient ascent
        n_iterations (int): Maximum number of iterations
        tolerance (float): Convergence tolerance for stopping criteria
        beta (np.ndarray): Model coefficients [beta_0, beta_1, beta_2]
        history (dict): Training history (log-likelihood, MSE, beta values)

    Example:
        >>> model = LogisticRegression(learning_rate=0.03)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    pass
