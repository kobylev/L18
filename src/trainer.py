"""
Model Trainer

Contains the training logic for Logistic Regression using Gradient Ascent.
"""

import numpy as np
from .model_base import LogisticRegressionBase


class LogisticRegressionTrainer:
    """
    Training mixin for Logistic Regression using Gradient Ascent.

    This class handles the gradient ascent optimization algorithm.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> 'LogisticRegressionTrainer':
        """
        Fit the logistic regression model using Gradient Ascent.

        The update rule (Gradient Ascent for maximizing log-likelihood):
        β_k^(t+1) = β_k^(t) + α * Σ[(y_i - p̂_i) * X_{k,i}]

        Args:
            X: Feature matrix of shape (n_samples, 2)
            y: Target vector of shape (n_samples,)
            verbose: Whether to print progress

        Returns:
            self
        """
        # Add bias term (X_0 = 1)
        X_with_bias = self._add_bias_term(X)
        n_samples, n_features = X_with_bias.shape

        # Initialize beta coefficients
        np.random.seed(42)
        self.beta = np.random.randn(n_features) * 0.01

        if verbose:
            self._print_training_header()

        # Gradient Ascent iterations
        for iteration in range(self.n_iterations):
            # Compute weighted sum and probability
            z = self._compute_weighted_sum(X_with_bias)
            p_hat = self.sigmoid(z)

            # Compute gradients
            error = y - p_hat
            gradients = X_with_bias.T @ error

            # Update beta coefficients (Gradient ASCENT)
            beta_old = self.beta.copy()
            self.beta = self.beta + self.learning_rate * gradients

            # Compute and store metrics
            log_likelihood = self._compute_log_likelihood(y, p_hat)
            mse = self._compute_mse(y, p_hat)

            self._store_history(log_likelihood, mse)

            # Check convergence
            beta_change = np.max(np.abs(self.beta - beta_old))

            if verbose:
                self._print_progress(iteration, log_likelihood, mse, beta_change)

            # Stop if converged
            if beta_change < self.tolerance:
                if verbose:
                    print(f"\nConverged at iteration {iteration}!")
                break

        if verbose:
            self._print_training_summary(log_likelihood, mse)

        return self

    def _store_history(self, log_likelihood: float, mse: float):
        """Store training metrics in history."""
        self.history['log_likelihood'].append(log_likelihood)
        self.history['mse'].append(mse)
        self.history['beta_0'].append(self.beta[0])
        self.history['beta_1'].append(self.beta[1])
        self.history['beta_2'].append(self.beta[2])

    def _print_training_header(self):
        """Print training initialization info."""
        print(f"Starting Gradient Ascent...")
        print(f"Initial beta: {self.beta}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Max iterations: {self.n_iterations}\n")

    def _print_progress(self, iteration: int, log_likelihood: float,
                       mse: float, beta_change: float):
        """Print training progress."""
        if iteration % 100 == 0 or iteration < 10:
            print(f"Iteration {iteration:4d} | "
                  f"Log-Likelihood: {log_likelihood:10.4f} | "
                  f"MSE: {mse:.6f} | "
                  f"beta change: {beta_change:.8f}")

    def _print_training_summary(self, log_likelihood: float, mse: float):
        """Print final training results."""
        print(f"\nFinal beta coefficients: {self.beta}")
        print(f"Final Log-Likelihood: {log_likelihood:.4f}")
        print(f"Final MSE: {mse:.6f}")
