"""
Test Loss Functions

Tests for log-likelihood and MSE calculations.
"""

import unittest
import numpy as np
import sys
sys.path.insert(0, '..')
from src import LogisticRegression


class TestLogLikelihood(unittest.TestCase):
    """Test cases for log-likelihood calculation."""

    def test_log_likelihood_perfect_predictions(self):
        """
        Perfect predictions should give maximum log-likelihood.

        When p̂ = 1 for y=1 and p̂ = 0 for y=0,
        log-likelihood = log(1) + log(1) = 0 (per sample)
        """
        model = LogisticRegression()

        y = np.array([1, 1, 0, 0])
        p_hat = np.array([0.9999, 0.9999, 0.0001, 0.0001])

        ll = model._compute_log_likelihood(y, p_hat)

        # Should be close to 0 (maximum possible)
        # Exact value: 4 * log(0.9999) ≈ -0.0004
        self.assertGreater(ll, -0.01)

    def test_log_likelihood_worst_predictions(self):
        """
        Worst predictions should give very negative log-likelihood.

        When p̂ = 0 for y=1 and p̂ = 1 for y=0,
        log-likelihood approaches -∞
        """
        model = LogisticRegression()

        y = np.array([1, 1, 0, 0])
        p_hat = np.array([0.0001, 0.0001, 0.9999, 0.9999])

        ll = model._compute_log_likelihood(y, p_hat)

        # Should be very negative
        self.assertLess(ll, -10)

    def test_log_likelihood_known_value(self):
        """Test log-likelihood against manually calculated value."""
        model = LogisticRegression()

        # Simple case:
        # y = [1, 0]
        # p̂ = [0.8, 0.3]
        #
        # LL = 1*log(0.8) + 0*log(0.2) + 0*log(0.7) + 1*log(0.3)
        #    = log(0.8) + log(0.3)
        #    = log(0.24)
        #    ≈ -1.427

        y = np.array([1, 0])
        p_hat = np.array([0.8, 0.3])

        ll = model._compute_log_likelihood(y, p_hat)

        expected = np.log(0.8) + np.log(0.7)
        self.assertAlmostEqual(ll, expected, places=6)

    def test_log_likelihood_random_predictions(self):
        """
        Random predictions (p̂ = 0.5 everywhere) should give:
        LL = N * log(0.5) = N * (-0.693)
        """
        model = LogisticRegression()

        n_samples = 100
        y = np.random.randint(0, 2, n_samples)
        p_hat = np.full(n_samples, 0.5)

        ll = model._compute_log_likelihood(y, p_hat)

        expected = n_samples * np.log(0.5)
        self.assertAlmostEqual(ll, expected, places=6)

    def test_log_likelihood_clipping(self):
        """
        Probabilities should be clipped to avoid log(0).

        Even if p̂ contains exact 0 or 1, function should not crash.
        """
        model = LogisticRegression()

        y = np.array([1, 0, 1, 0])
        p_hat = np.array([1.0, 0.0, 1.0, 0.0])  # Extreme values

        # Should not raise error
        try:
            ll = model._compute_log_likelihood(y, p_hat)
            # Should be finite
            self.assertTrue(np.isfinite(ll))
        except Exception as e:
            self.fail(f"Log-likelihood with extreme probabilities raised: {e}")


class TestMSE(unittest.TestCase):
    """Test cases for Mean Squared Error calculation."""

    def test_mse_perfect_predictions(self):
        """Perfect predictions should give MSE = 0."""
        model = LogisticRegression()

        y = np.array([1, 1, 0, 0])
        p_hat = np.array([1.0, 1.0, 0.0, 0.0])

        mse = model._compute_mse(y, p_hat)
        self.assertAlmostEqual(mse, 0.0, places=10)

    def test_mse_known_value(self):
        """Test MSE against manually calculated value."""
        model = LogisticRegression()

        # y = [1, 0, 1]
        # p̂ = [0.8, 0.3, 0.6]
        #
        # Errors: (1-0.8)² = 0.04, (0-0.3)² = 0.09, (1-0.6)² = 0.16
        # MSE = (0.04 + 0.09 + 0.16) / 3 = 0.29 / 3 ≈ 0.0967

        y = np.array([1, 0, 1])
        p_hat = np.array([0.8, 0.3, 0.6])

        mse = model._compute_mse(y, p_hat)

        expected = ((1-0.8)**2 + (0-0.3)**2 + (1-0.6)**2) / 3
        self.assertAlmostEqual(mse, expected, places=6)

    def test_mse_worst_predictions(self):
        """Worst predictions should give MSE = 1."""
        model = LogisticRegression()

        y = np.array([1, 1, 0, 0])
        p_hat = np.array([0.0, 0.0, 1.0, 1.0])

        mse = model._compute_mse(y, p_hat)
        self.assertAlmostEqual(mse, 1.0, places=10)

    def test_mse_random_predictions(self):
        """
        Random predictions (p̂ = 0.5 everywhere) should give:
        MSE = (1/N) * Σ(y_i - 0.5)²

        For balanced dataset: MSE = 0.25
        """
        model = LogisticRegression()

        # Balanced dataset
        y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        p_hat = np.full(8, 0.5)

        mse = model._compute_mse(y, p_hat)

        # MSE = 4*(1-0.5)² + 4*(0-0.5)² = 4*0.25 + 4*0.25 = 2.0
        # MSE = 2.0 / 8 = 0.25
        self.assertAlmostEqual(mse, 0.25, places=10)

    def test_mse_range(self):
        """MSE should always be in [0, 1] for binary classification."""
        model = LogisticRegression()

        # Test with random data
        np.random.seed(42)
        y = np.random.randint(0, 2, 100)
        p_hat = np.random.rand(100)

        mse = model._compute_mse(y, p_hat)

        self.assertGreaterEqual(mse, 0.0)
        self.assertLessEqual(mse, 1.0)


class TestLossImprovement(unittest.TestCase):
    """Test that training improves loss functions."""

    def test_training_improves_log_likelihood(self):
        """Training should increase log-likelihood."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticRegression(
            learning_rate=0.05,
            n_iterations=500
        )
        model.fit(X, y, verbose=False)

        # Initial LL (first iteration)
        ll_initial = model.history['log_likelihood'][0]

        # Final LL (last iteration)
        ll_final = model.history['log_likelihood'][-1]

        # Should improve (increase)
        self.assertGreater(ll_final, ll_initial,
                          "Training should increase log-likelihood")

    def test_training_reduces_mse(self):
        """Training should decrease MSE."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticRegression(
            learning_rate=0.05,
            n_iterations=500
        )
        model.fit(X, y, verbose=False)

        # Initial MSE (first iteration)
        mse_initial = model.history['mse'][0]

        # Final MSE (last iteration)
        mse_final = model.history['mse'][-1]

        # Should improve (decrease)
        self.assertLess(mse_final, mse_initial,
                       "Training should decrease MSE")

    def test_monotonic_improvement(self):
        """
        For well-behaved problems, log-likelihood should generally increase.

        Note: Mini-batch may have small fluctuations, so we check overall trend.
        """
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticRegression(
            learning_rate=0.03,
            n_iterations=1000,
            batch_size=None  # Full batch for monotonic improvement
        )
        model.fit(X, y, verbose=False)

        ll_history = model.history['log_likelihood']

        # Check that final LL is better than every early LL
        # (allows for small fluctuations but overall increasing trend)
        early_ll_max = max(ll_history[:50])  # Max in first 50 iterations
        final_ll = ll_history[-1]

        self.assertGreater(final_ll, early_ll_max,
                          "Final log-likelihood should exceed early values")


class TestLossWithRegularization(unittest.TestCase):
    """Test loss calculation with regularization."""

    def test_regularization_affects_training(self):
        """Regularization should lead to different final parameters."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        # Model without regularization
        model_no_reg = LogisticRegression(
            learning_rate=0.05,
            n_iterations=500,
            regularization=None
        )
        model_no_reg.fit(X, y, verbose=False)

        # Model with L2 regularization
        model_l2 = LogisticRegression(
            learning_rate=0.05,
            n_iterations=500,
            regularization='l2',
            lambda_reg=0.1
        )
        model_l2.fit(X, y, verbose=False)

        # Betas should be different
        beta_diff = np.linalg.norm(model_no_reg.beta - model_l2.beta)
        self.assertGreater(beta_diff, 0.01,
                          "Regularization should produce different parameters")

        # L2 regularization should produce smaller weights (shrinkage)
        beta_norm_no_reg = np.linalg.norm(model_no_reg.beta[1:])  # Exclude bias
        beta_norm_l2 = np.linalg.norm(model_l2.beta[1:])

        self.assertLess(beta_norm_l2, beta_norm_no_reg,
                       "L2 regularization should shrink weights")


if __name__ == '__main__':
    unittest.main()
