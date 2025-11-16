# Test Suite for Logistic Regression Implementation

Comprehensive unit tests for the Logistic Regression from scratch implementation.

## Test Coverage

### `test_model.py` - Core Model Functions

**Sigmoid Function Tests:**
- ✅ Sigmoid at zero returns 0.5
- ✅ Sigmoid approaches 1 for large positive values
- ✅ Sigmoid approaches 0 for large negative values
- ✅ Sigmoid symmetry: σ(x) = 1 - σ(-x)
- ✅ Known mathematical values
- ✅ Array input handling
- ✅ Monotonicity (strictly increasing)

**Gradient Calculation Tests:**
- ✅ Gradient ascent increases log-likelihood
- ✅ Gradients approach zero at convergence
- ✅ Bias term addition correctness
- ✅ Weighted sum computation (Z = β₀X₀ + β₁X₁ + β₂X₂)

**Prediction Tests:**
- ✅ `predict_proba()` outputs in [0, 1]
- ✅ `predict()` outputs binary labels
- ✅ Custom threshold handling
- ✅ Error handling for unfitted model

### `test_loss.py` - Loss Functions

**Log-Likelihood Tests:**
- ✅ Perfect predictions give maximum LL
- ✅ Worst predictions give very negative LL
- ✅ Known values verification
- ✅ Random predictions (p̂=0.5)
- ✅ Probability clipping (avoid log(0))

**MSE Tests:**
- ✅ Perfect predictions give MSE=0
- ✅ Known values verification
- ✅ Worst predictions give MSE=1
- ✅ Random predictions (balanced data)
- ✅ MSE range validation [0, 1]

**Training Improvement Tests:**
- ✅ Training increases log-likelihood
- ✅ Training decreases MSE
- ✅ Monotonic improvement (full batch)
- ✅ Regularization effects on parameters

### `test_data_utils.py` - Data Generation & Normalization

**Data Generation Tests:**
- ✅ Correct shape (n_samples × 2)
- ✅ Perfect class balance
- ✅ Binary labels (0 or 1)
- ✅ Reproducibility with seed
- ✅ Different seeds produce different data
- ✅ Linear separability
- ✅ Correct data types

**Z-Score Normalization Tests:**
- ✅ Mean ≈ 0 after normalization
- ✅ Std ≈ 1 after normalization
- ✅ Known values verification
- ✅ Normalization parameters returned
- ✅ Constant features handling (std=0)
- ✅ Inverse transform correctness

**Min-Max Normalization Tests:**
- ✅ Output range [0, 1]
- ✅ Known values verification
- ✅ Normalization parameters returned
- ✅ Constant features handling (range=0)
- ✅ Inverse transform correctness

**Edge Cases:**
- ✅ Single sample normalization
- ✅ Negative values
- ✅ Large values (no overflow)
- ✅ Invalid method error handling

**Integration Tests:**
- ✅ Normalized data improves convergence

---

## Running Tests

### Run All Tests

```bash
# From tests/ directory
python run_tests.py

# Or from project root
python -m unittest discover tests

# Verbose output
python run_tests.py -v
```

### Run Specific Test File

```bash
# Test only model functions
python -m unittest test_model

# Test only loss functions
python -m unittest test_loss

# Test only data utilities
python -m unittest test_data_utils
```

### Run Specific Test Class

```bash
# Test only sigmoid function
python -m unittest test_model.TestSigmoidFunction

# Test only Z-score normalization
python -m unittest test_data_utils.TestZScoreNormalization
```

### Run Specific Test Method

```bash
# Test only sigmoid at zero
python -m unittest test_model.TestSigmoidFunction.test_sigmoid_at_zero
```

---

## Test Output Example

```
Running tests...

test_sigmoid_at_zero (test_model.TestSigmoidFunction) ... ok
test_sigmoid_positive_infinity (test_model.TestSigmoidFunction) ... ok
test_sigmoid_negative_infinity (test_model.TestSigmoidFunction) ... ok
...
test_zscore_mean_zero (test_data_utils.TestZScoreNormalization) ... ok
test_zscore_std_one (test_data_utils.TestZScoreNormalization) ... ok

----------------------------------------------------------------------
Ran 75 tests in 2.341s

OK
```

---

## Test Statistics

| Category | Test Classes | Test Methods | Coverage |
|----------|--------------|--------------|----------|
| **Model Functions** | 5 | 25 | Core math, gradients, predictions |
| **Loss Functions** | 4 | 22 | LL, MSE, training, regularization |
| **Data Utils** | 6 | 28 | Generation, normalization, edge cases |
| **Total** | **15** | **75** | **Comprehensive** |

---

## Writing New Tests

### Test Structure

```python
import unittest
import numpy as np
from src import LogisticRegression

class TestNewFeature(unittest.TestCase):
    """Test cases for new feature."""

    def test_basic_functionality(self):
        """Test description."""
        # Arrange
        model = LogisticRegression()

        # Act
        result = model.some_method()

        # Assert
        self.assertEqual(result, expected_value)
```

### Best Practices

1. **Descriptive Names**: Test names should describe what they test
2. **One Assertion**: Each test should verify one specific behavior
3. **Independent**: Tests should not depend on each other
4. **Reproducible**: Use `np.random.seed()` for randomized tests
5. **Edge Cases**: Test boundary conditions and error cases
6. **Documentation**: Add docstrings explaining what is tested

---

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run tests
  run: |
    cd tests
    python run_tests.py
```

---

## Test Coverage Goals

✅ **Current Coverage:**
- Sigmoid function: 100%
- Log-likelihood: 100%
- MSE: 100%
- Data generation: 100%
- Normalization: 100%
- Gradient calculation: 90%

🔄 **Future Coverage:**
- Early stopping mechanism
- Learning rate schedules
- K-Fold cross-validation
- Grid search
- Regularization (L1, L2, Elastic Net)

---

## Common Test Failures

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Run from tests/ directory
cd tests
python run_tests.py

# Or add parent to path
export PYTHONPATH="${PYTHONPATH}:/path/to/L18"
```

### Random Seed Issues

**Error**: Tests fail intermittently

**Solution**: Always set `np.random.seed()` before random operations

### Floating Point Precision

**Error**: `AssertionError: 0.9999999 != 1.0`

**Solution**: Use `assertAlmostEqual()` instead of `assertEqual()`

```python
# Bad
self.assertEqual(result, 1.0)

# Good
self.assertAlmostEqual(result, 1.0, places=6)
```

---

## Dependencies

Tests require only standard libraries:
- `unittest` (built-in)
- `numpy`

No additional test frameworks needed (pytest, nose, etc.).

---

## Contributing Tests

When adding new features to the project:

1. Write tests BEFORE implementation (TDD)
2. Ensure all existing tests still pass
3. Add tests for edge cases
4. Update this README with new test coverage

---

## License

Tests are part of the main project and share the same MIT License.
