"""
Test Core Model Functions

Tests for sigmoid function and gradient calculations.
"""

import unittest
import numpy as np
import sys
sys.path.insert(0, '..')
from src import LogisticRegression


class TestSigmoidFunction(unittest.TestCase):
    """Test cases for the sigmoid activation function."""

    def test_sigmoid_at_zero(self):
        """Sigmoid of 0 should be exactly 0.5."""
        model = LogisticRegression()
        result = model.sigmoid(np.array([0.0]))
        self.assertAlmostEqual(result[0], 0.5, places=10)

    def test_sigmoid_positive_infinity(self):
        """Sigmoid of large positive number should approach 1."""
        model = LogisticRegression()
        result = model.sigmoid(np.array([100.0]))
        self.assertAlmostEqual(result[0], 1.0, places=5)

    def test_sigmoid_negative_infinity(self):
        """Sigmoid of large negative number should approach 0."""
        model = LogisticRegression()
        result = model.sigmoid(np.array([-100.0]))
        self.assertAlmostEqual(result[0], 0.0, places=5)

    def test_sigmoid_symmetry(self):
        """Sigmoid should be symmetric: σ(x) = 1 - σ(-x)."""
        model = LogisticRegression()
        x = 2.5
        result_pos = model.sigmoid(np.array([x]))
        result_neg = model.sigmoid(np.array([-x]))
        self.assertAlmostEqual(result_pos[0], 1 - result_neg[0], places=10)

    def test_sigmoid_known_values(self):
        """Test sigmoid against known values."""
        model = LogisticRegression()

        # σ(ln(3)) = 3/4 = 0.75
        ln_3 = np.log(3)
        result = model.sigmoid(np.array([ln_3]))
        self.assertAlmostEqual(result[0], 0.75, places=6)

        # σ(-ln(3)) = 1/4 = 0.25
        result = model.sigmoid(np.array([-ln_3]))
        self.assertAlmostEqual(result[0], 0.25, places=6)

    def test_sigmoid_array_input(self):
        """Sigmoid should work with numpy arrays."""
        model = LogisticRegression()
        z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = model.sigmoid(z)

        # Check shape
        self.assertEqual(result.shape, z.shape)

        # Check bounds
        self.assertTrue(np.all(result > 0))
        self.assertTrue(np.all(result < 1))

        # Check monotonicity (sigmoid is strictly increasing)
        self.assertTrue(np.all(np.diff(result) > 0))


class TestGradientCalculation(unittest.TestCase):
    """Test cases for gradient computation."""

    def test_gradient_ascent_update_increases_likelihood(self):
        """
        Gradient ascent should increase log-likelihood.

        For a simple case where we know the optimal direction,
        verify that the gradient points in the right direction.
        """
        # Create simple separable data
        np.random.seed(42)
        X = np.array([
            [1.0, 1.0],   # Class 1
            [2.0, 2.0],   # Class 1
            [-1.0, -1.0], # Class 0
            [-2.0, -2.0]  # Class 0
        ])
        y = np.array([1, 1, 0, 0])

        model = LogisticRegression(learning_rate=0.1, n_iterations=1)

        # Manually set beta to non-optimal value
        model.beta = np.array([0.0, 0.0, 0.0])

        # Compute initial log-likelihood
        X_with_bias = model._add_bias_term(X)
        z_initial = model._compute_weighted_sum(X_with_bias)
        p_hat_initial = model.sigmoid(z_initial)
        ll_initial = model._compute_log_likelihood(y, p_hat_initial)

        # Compute gradient
        error = y - p_hat_initial
        gradients = X_with_bias.T @ error

        # Update beta (one gradient ascent step)
        beta_updated = model.beta + 0.1 * gradients

        # Compute new log-likelihood
        model.beta = beta_updated
        z_new = model._compute_weighted_sum(X_with_bias)
        p_hat_new = model.sigmoid(z_new)
        ll_new = model._compute_log_likelihood(y, p_hat_new)

        # Log-likelihood should increase
        self.assertGreater(ll_new, ll_initial,
                          "Gradient ascent should increase log-likelihood")

    def test_gradient_at_convergence_is_small(self):
        """
        At convergence, gradients should be close to zero.
        """
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticRegression(
            learning_rate=0.05,
            n_iterations=1000,
            tolerance=1e-6
        )
        model.fit(X, y, verbose=False)

        # Compute gradient at final beta
        X_with_bias = model._add_bias_term(X)
        z = model._compute_weighted_sum(X_with_bias)
        p_hat = model.sigmoid(z)
        error = y - p_hat
        gradients = X_with_bias.T @ error

        # Gradient norm should be small
        gradient_norm = np.linalg.norm(gradients)
        self.assertLess(gradient_norm, 1.0,
                       "Gradient norm should be small at convergence")

    def test_bias_term_addition(self):
        """Test that bias term is correctly added."""
        model = LogisticRegression()
        X = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]])

        X_with_bias = model._add_bias_term(X)

        # Check shape
        self.assertEqual(X_with_bias.shape, (3, 3))

        # Check bias column (first column should be all ones)
        np.testing.assert_array_equal(X_with_bias[:, 0], np.ones(3))

        # Check original features preserved
        np.testing.assert_array_equal(X_with_bias[:, 1:], X)


class TestWeightedSum(unittest.TestCase):
    """Test weighted sum computation."""

    def test_weighted_sum_calculation(self):
        """Test Z = β₀X₀ + β₁X₁ + β₂X₂."""
        model = LogisticRegression()
        model.beta = np.array([1.0, 2.0, 3.0])  # β₀=1, β₁=2, β₂=3

        X_with_bias = np.array([
            [1.0, 1.0, 1.0],  # X₀=1 (bias), X₁=1, X₂=1
            [1.0, 2.0, 3.0]   # X₀=1 (bias), X₁=2, X₂=3
        ])

        z = model._compute_weighted_sum(X_with_bias)

        # For first sample: 1*1 + 2*1 + 3*1 = 6
        self.assertAlmostEqual(z[0], 6.0, places=10)

        # For second sample: 1*1 + 2*2 + 3*3 = 14
        self.assertAlmostEqual(z[1], 14.0, places=10)


class TestPredictions(unittest.TestCase):
    """Test prediction functions."""

    def test_predict_proba_output_range(self):
        """Predicted probabilities should be in [0, 1]."""
        np.random.seed(42)
        X_train = np.random.randn(50, 2)
        y_train = (X_train[:, 0] > 0).astype(int)

        model = LogisticRegression(n_iterations=100)
        model.fit(X_train, y_train, verbose=False)

        X_test = np.random.randn(20, 2)
        probas = model.predict_proba(X_test)

        # All probabilities should be in [0, 1]
        self.assertTrue(np.all(probas >= 0))
        self.assertTrue(np.all(probas <= 1))

    def test_predict_binary_output(self):
        """Predict should return binary labels."""
        np.random.seed(42)
        X_train = np.random.randn(50, 2)
        y_train = (X_train[:, 0] > 0).astype(int)

        model = LogisticRegression(n_iterations=100)
        model.fit(X_train, y_train, verbose=False)

        X_test = np.random.randn(20, 2)
        predictions = model.predict(X_test)

        # All predictions should be 0 or 1
        self.assertTrue(np.all((predictions == 0) | (predictions == 1)))

    def test_predict_threshold(self):
        """Test custom threshold for predictions."""
        np.random.seed(42)
        X_train = np.random.randn(50, 2)
        y_train = (X_train[:, 0] > 0).astype(int)

        model = LogisticRegression(n_iterations=100)
        model.fit(X_train, y_train, verbose=False)

        X_test = np.array([[0.5, 0.5]])

        # Get probability
        proba = model.predict_proba(X_test)[0]

        # Test different thresholds
        if proba > 0.3 and proba < 0.7:
            pred_low = model.predict(X_test, threshold=0.3)[0]
            pred_high = model.predict(X_test, threshold=0.7)[0]

            # Predictions should differ based on threshold
            self.assertNotEqual(pred_low, pred_high)


class TestModelNotFitted(unittest.TestCase):
    """Test error handling for unfitted model."""

    def test_predict_before_fit_raises_error(self):
        """Predicting before fitting should raise ValueError."""
        model = LogisticRegression()
        X = np.array([[1.0, 2.0]])

        with self.assertRaises(ValueError):
            model.predict_proba(X)

    def test_decision_boundary_before_fit_raises_error(self):
        """Getting decision boundary before fitting should raise ValueError."""
        model = LogisticRegression()

        with self.assertRaises(ValueError):
            model.get_decision_boundary((-1, 1))


if __name__ == '__main__':
    unittest.main()
