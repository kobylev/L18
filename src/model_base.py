"""
Base Model Components

Contains core mathematical functions and model structure for Logistic Regression.
"""

import numpy as np
from typing import Tuple


class LogisticRegressionBase:
    """
    Base class for Logistic Regression with core mathematical functions.

    Attributes:
        learning_rate (float): Learning step (alpha) for gradient ascent
        n_iterations (int): Maximum number of iterations
        tolerance (float): Convergence tolerance for stopping criteria
        beta (np.ndarray): Model coefficients [beta_0, beta_1, beta_2]
        history (dict): Training history (log-likelihood, MSE, beta values)
    """

    def __init__(self, learning_rate: float = 0.03, n_iterations: int = 10000,
                 tolerance: float = 1e-6, batch_size: int = None):
        """
        Initialize the Logistic Regression base model.

        Args:
            learning_rate: Learning step for gradient ascent
            n_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            batch_size: Size of mini-batches (None = full batch gradient ascent)
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.batch_size = batch_size
        self.beta = None
        self.history = {
            'log_likelihood': [],
            'mse': [],
            'beta_0': [],
            'beta_1': [],
            'beta_2': []
        }

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        σ(Z) = 1 / (1 + e^(-Z))

        Args:
            z: Weighted sum (linear combination)

        Returns:
            Probability values between 0 and 1
        """
        z = np.clip(z, -500, 500)  # Prevent overflow
        return 1 / (1 + np.exp(-z))

    def _add_bias_term(self, X: np.ndarray) -> np.ndarray:
        """
        Add bias term (column of ones) to feature matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Feature matrix with bias term (n_samples, n_features + 1)
        """
        n_samples = X.shape[0]
        return np.column_stack([np.ones(n_samples), X])

    def _compute_weighted_sum(self, X: np.ndarray) -> np.ndarray:
        """
        Compute weighted sum Z = β₀X₀ + β₁X₁ + β₂X₂

        Args:
            X: Feature matrix with bias term

        Returns:
            Weighted sum for each sample
        """
        return X @ self.beta

    def _compute_log_likelihood(self, y: np.ndarray, p_hat: np.ndarray) -> float:
        """
        Compute log-likelihood function.

        L = Σ[y_i * log(p̂_i) + (1 - y_i) * log(1 - p̂_i)]

        Args:
            y: True labels
            p_hat: Predicted probabilities

        Returns:
            Log-likelihood value
        """
        epsilon = 1e-15
        p_hat = np.clip(p_hat, epsilon, 1 - epsilon)
        return np.sum(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))

    def _compute_mse(self, y: np.ndarray, p_hat: np.ndarray) -> float:
        """
        Compute Mean Squared Error.

        MSE = (1/N) * Σ(y_i - p̂_i)²

        Args:
            y: True labels
            p_hat: Predicted probabilities

        Returns:
            MSE value
        """
        return np.mean((y - p_hat) ** 2)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for samples.

        Args:
            X: Feature matrix of shape (n_samples, 2)

        Returns:
            Predicted probabilities
        """
        if self.beta is None:
            raise ValueError("Model has not been fitted yet!")

        X_with_bias = self._add_bias_term(X)
        z = self._compute_weighted_sum(X_with_bias)
        return self.sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels for samples.

        Args:
            X: Feature matrix of shape (n_samples, 2)
            threshold: Decision threshold (default 0.5)

        Returns:
            Predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def get_decision_boundary(self, x1_range: Tuple[float, float],
                             n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate decision boundary line.

        Decision boundary is where σ(Z) = 0.5, which means Z = 0
        β₀ + β₁X₁ + β₂X₂ = 0
        X₂ = -(β₀ + β₁X₁) / β₂

        Args:
            x1_range: Range of X1 values (min, max)
            n_points: Number of points for the line

        Returns:
            Tuple of (X1 values, X2 values) for the boundary line
        """
        if self.beta is None:
            raise ValueError("Model has not been fitted yet!")

        x1 = np.linspace(x1_range[0], x1_range[1], n_points)

        if np.abs(self.beta[2]) < 1e-10:
            return x1, np.zeros_like(x1)

        x2 = -(self.beta[0] + self.beta[1] * x1) / self.beta[2]
        return x1, x2
