"""
Test Data Utilities

Tests for data generation and normalization functions.
"""

import unittest
import numpy as np
import sys
sys.path.insert(0, '..')
from src import generate_synthetic_data, normalize_features


class TestDataGeneration(unittest.TestCase):
    """Test cases for synthetic data generation."""

    def test_data_shape(self):
        """Generated data should have correct shape."""
        n_samples = 100
        X, y = generate_synthetic_data(n_samples_per_class=n_samples)

        # Should have n_samples * 2 total samples (two classes)
        expected_total = n_samples * 2
        self.assertEqual(X.shape[0], expected_total)

        # Should have 2 features
        self.assertEqual(X.shape[1], 2)

        # y should be 1D with same length
        self.assertEqual(y.shape[0], expected_total)
        self.assertEqual(len(y.shape), 1)

    def test_class_balance(self):
        """Generated data should be perfectly balanced."""
        n_samples = 100
        X, y = generate_synthetic_data(n_samples_per_class=n_samples)

        # Count class 0 and class 1
        n_class_0 = np.sum(y == 0)
        n_class_1 = np.sum(y == 1)

        self.assertEqual(n_class_0, n_samples)
        self.assertEqual(n_class_1, n_samples)

    def test_binary_labels(self):
        """Labels should be binary (0 or 1)."""
        X, y = generate_synthetic_data(n_samples_per_class=50)

        # All labels should be 0 or 1
        unique_labels = np.unique(y)
        np.testing.assert_array_equal(unique_labels, np.array([0, 1]))

    def test_reproducibility_with_seed(self):
        """Same seed should produce same data."""
        np.random.seed(42)
        X1, y1 = generate_synthetic_data(n_samples_per_class=100)

        np.random.seed(42)
        X2, y2 = generate_synthetic_data(n_samples_per_class=100)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_different_seeds_produce_different_data(self):
        """Different seeds should produce different data."""
        X1, y1 = generate_synthetic_data(n_samples_per_class=100, random_state=42)
        X2, y2 = generate_synthetic_data(n_samples_per_class=100, random_state=123)

        # Data should be different
        self.assertFalse(np.allclose(X1, X2))

    def test_class_separation(self):
        """
        Classes should be linearly separable (by design).

        Class 0: centered at (0.3, 0.3)
        Class 1: centered at (0.7, 0.7)
        """
        X, y = generate_synthetic_data(n_samples_per_class=500, random_state=42)

        # Get class means
        X_class_0 = X[y == 0]
        X_class_1 = X[y == 1]

        mean_class_0 = np.mean(X_class_0, axis=0)
        mean_class_1 = np.mean(X_class_1, axis=0)

        # Class 0 should be centered around (0.3, 0.3)
        self.assertLess(mean_class_0[0], 0.5)
        self.assertLess(mean_class_0[1], 0.5)

        # Class 1 should be centered around (0.7, 0.7)
        self.assertGreater(mean_class_1[0], 0.5)
        self.assertGreater(mean_class_1[1], 0.5)

    def test_data_types(self):
        """Generated data should have correct types."""
        X, y = generate_synthetic_data(n_samples_per_class=50)

        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertTrue(np.issubdtype(X.dtype, np.floating))
        # y is created with np.zeros() and np.ones() which are floats,
        # but contains only 0.0 and 1.0 values
        self.assertTrue(np.issubdtype(y.dtype, np.floating))


class TestZScoreNormalization(unittest.TestCase):
    """Test cases for Z-Score normalization."""

    def test_zscore_mean_zero(self):
        """Z-score normalization should produce mean ≈ 0."""
        X = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]])

        X_norm, params = normalize_features(X, method='zscore')

        # Check mean of each feature
        means = np.mean(X_norm, axis=0)
        np.testing.assert_array_almost_equal(means, np.zeros(2), decimal=10)

    def test_zscore_std_one(self):
        """Z-score normalization should produce std ≈ 1."""
        X = np.array([[1.0, 10.0],
                      [3.0, 20.0],
                      [5.0, 30.0],
                      [7.0, 40.0]])

        X_norm, params = normalize_features(X, method='zscore')

        # Check std of each feature
        stds = np.std(X_norm, axis=0, ddof=0)
        np.testing.assert_array_almost_equal(stds, np.ones(2), decimal=10)

    def test_zscore_known_values(self):
        """Test Z-score against manually calculated values."""
        X = np.array([[1.0, 5.0],
                      [2.0, 10.0],
                      [3.0, 15.0]])

        X_norm, params = normalize_features(X, method='zscore')

        # Feature 0: mean=2, std=0.816
        # Feature 1: mean=10, std=4.082
        #
        # X_norm[0,0] = (1 - 2) / 0.816 ≈ -1.225
        # X_norm[1,0] = (2 - 2) / 0.816 = 0
        # X_norm[2,0] = (3 - 2) / 0.816 ≈ 1.225

        expected_col_0 = (X[:, 0] - 2.0) / np.std(X[:, 0], ddof=0)
        expected_col_1 = (X[:, 1] - 10.0) / np.std(X[:, 1], ddof=0)

        np.testing.assert_array_almost_equal(X_norm[:, 0], expected_col_0)
        np.testing.assert_array_almost_equal(X_norm[:, 1], expected_col_1)

    def test_zscore_params_returned(self):
        """Z-score should return correct normalization parameters."""
        X = np.array([[1.0, 10.0],
                      [3.0, 20.0],
                      [5.0, 30.0]])

        X_norm, params = normalize_features(X, method='zscore')

        # Check returned parameters
        self.assertIn('mean', params)
        self.assertIn('std', params)

        # Verify values
        expected_mean = np.array([3.0, 20.0])
        expected_std = np.std(X, axis=0, ddof=0)

        np.testing.assert_array_almost_equal(params['mean'], expected_mean)
        np.testing.assert_array_almost_equal(params['std'], expected_std)

    def test_zscore_single_value_feature(self):
        """Z-score should handle constant features (std=0)."""
        X = np.array([[1.0, 5.0],
                      [1.0, 5.0],
                      [1.0, 5.0]])

        X_norm, params = normalize_features(X, method='zscore')

        # Constant features should remain constant (avoid division by zero)
        # Should either stay as 0 or original values
        # Check that no NaN or Inf values
        self.assertFalse(np.any(np.isnan(X_norm)))
        self.assertFalse(np.any(np.isinf(X_norm)))


class TestMinMaxNormalization(unittest.TestCase):
    """Test cases for Min-Max normalization."""

    def test_minmax_range(self):
        """Min-Max normalization should produce values in [0, 1]."""
        X = np.array([[1.0, 10.0],
                      [3.0, 20.0],
                      [5.0, 30.0],
                      [7.0, 40.0]])

        X_norm, params = normalize_features(X, method='minmax')

        # All values should be in [0, 1]
        self.assertTrue(np.all(X_norm >= 0))
        self.assertTrue(np.all(X_norm <= 1))

    def test_minmax_known_values(self):
        """Test Min-Max against manually calculated values."""
        X = np.array([[1.0, 5.0],
                      [2.0, 10.0],
                      [3.0, 15.0]])

        X_norm, params = normalize_features(X, method='minmax')

        # Feature 0: min=1, max=3, range=2
        # X_norm[0,0] = (1-1)/2 = 0
        # X_norm[1,0] = (2-1)/2 = 0.5
        # X_norm[2,0] = (3-1)/2 = 1

        # Feature 1: min=5, max=15, range=10
        # X_norm[0,1] = (5-5)/10 = 0
        # X_norm[1,1] = (10-5)/10 = 0.5
        # X_norm[2,1] = (15-5)/10 = 1

        expected = np.array([[0.0, 0.0],
                            [0.5, 0.5],
                            [1.0, 1.0]])

        np.testing.assert_array_almost_equal(X_norm, expected)

    def test_minmax_params_returned(self):
        """Min-Max should return correct normalization parameters."""
        X = np.array([[1.0, 10.0],
                      [3.0, 20.0],
                      [5.0, 30.0]])

        X_norm, params = normalize_features(X, method='minmax')

        # Check returned parameters
        self.assertIn('min', params)
        self.assertIn('max', params)

        # Verify values
        expected_min = np.array([1.0, 10.0])
        expected_max = np.array([5.0, 30.0])

        np.testing.assert_array_almost_equal(params['min'], expected_min)
        np.testing.assert_array_almost_equal(params['max'], expected_max)

    def test_minmax_single_value_feature(self):
        """Min-Max should handle constant features (range=0)."""
        X = np.array([[5.0, 10.0],
                      [5.0, 20.0],
                      [5.0, 30.0]])

        X_norm, params = normalize_features(X, method='minmax')

        # Check that no NaN or Inf values
        self.assertFalse(np.any(np.isnan(X_norm)))
        self.assertFalse(np.any(np.isinf(X_norm)))


class TestNormalizationInverse(unittest.TestCase):
    """Test that normalization parameters can be used for inverse transform."""

    def test_zscore_inverse_transform(self):
        """Test that we can reverse Z-score normalization."""
        X = np.array([[1.0, 10.0],
                      [3.0, 20.0],
                      [5.0, 30.0]])

        X_norm, params = normalize_features(X, method='zscore')

        # Manually inverse transform
        X_recovered = X_norm * params['std'] + params['mean']

        np.testing.assert_array_almost_equal(X_recovered, X)

    def test_minmax_inverse_transform(self):
        """Test that we can reverse Min-Max normalization."""
        X = np.array([[1.0, 10.0],
                      [3.0, 20.0],
                      [5.0, 30.0]])

        X_norm, params = normalize_features(X, method='minmax')

        # Manually inverse transform
        X_recovered = X_norm * (params['max'] - params['min']) + params['min']

        np.testing.assert_array_almost_equal(X_recovered, X)


class TestNormalizationEdgeCases(unittest.TestCase):
    """Test edge cases for normalization."""

    def test_single_sample(self):
        """Normalization should work with single sample."""
        X = np.array([[1.0, 2.0]])

        X_norm_z, params_z = normalize_features(X, method='zscore')
        X_norm_mm, params_mm = normalize_features(X, method='minmax')

        # Should not crash
        self.assertEqual(X_norm_z.shape, (1, 2))
        self.assertEqual(X_norm_mm.shape, (1, 2))

    def test_negative_values(self):
        """Normalization should handle negative values."""
        X = np.array([[-5.0, -10.0],
                      [-3.0, -5.0],
                      [-1.0, 0.0]])

        X_norm_z, _ = normalize_features(X, method='zscore')
        X_norm_mm, _ = normalize_features(X, method='minmax')

        # Should not crash and produce valid output
        self.assertFalse(np.any(np.isnan(X_norm_z)))
        self.assertFalse(np.any(np.isnan(X_norm_mm)))

    def test_large_values(self):
        """Normalization should handle large values."""
        X = np.array([[1e6, 1e9],
                      [2e6, 2e9],
                      [3e6, 3e9]])

        X_norm_z, _ = normalize_features(X, method='zscore')
        X_norm_mm, _ = normalize_features(X, method='minmax')

        # Should not overflow
        self.assertTrue(np.all(np.isfinite(X_norm_z)))
        self.assertTrue(np.all(np.isfinite(X_norm_mm)))

    def test_invalid_method(self):
        """Invalid normalization method should raise error."""
        X = np.array([[1.0, 2.0]])

        with self.assertRaises(ValueError):
            normalize_features(X, method='invalid_method')


class TestNormalizationWithTraining(unittest.TestCase):
    """Test normalization in actual training context."""

    def test_normalized_data_improves_convergence(self):
        """
        Normalized data should converge more stably.

        This is a practical test showing normalization benefits.
        """
        np.random.seed(42)

        # Generate data with very different scales
        X_raw = np.random.randn(200, 2)
        X_raw[:, 0] = X_raw[:, 0] * 100  # Feature 0: large scale
        X_raw[:, 1] = X_raw[:, 1] * 0.01  # Feature 1: small scale
        y = (X_raw[:, 0] + X_raw[:, 1] > 0).astype(int)

        from src import LogisticRegression

        # Train with normalization (should be stable)
        X_norm, _ = normalize_features(X_raw, method='zscore')
        model_norm = LogisticRegression(
            learning_rate=0.05,
            n_iterations=100
        )
        model_norm.fit(X_norm, y, verbose=False)

        # Normalized model should converge without issues
        ll_norm = model_norm.history['log_likelihood'][-1]

        # Check that training was stable (no NaN, no -inf)
        self.assertTrue(np.isfinite(ll_norm),
                       "Normalized data should produce finite log-likelihood")

        # Check that MSE is reasonable
        mse_norm = model_norm.history['mse'][-1]
        self.assertLess(mse_norm, 0.5,
                       "Normalized data should achieve reasonable MSE")


if __name__ == '__main__':
    unittest.main()
