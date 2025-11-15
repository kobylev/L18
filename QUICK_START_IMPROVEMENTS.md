# Quick Start Guide - Improved Logistic Regression

## What's New? 🚀

This improved implementation adds four key enhancements:

1. **Z-Score Normalization** - Better convergence stability
2. **Mini-Batch Gradient Ascent** - Scalability for large datasets
3. **Comprehensive Metrics** - Precision, Recall, F1 Score
4. **Real-World Guide** - Adapt to production datasets

## Quick Start

### Basic Usage (With All Improvements)

```python
from src import LogisticRegression
from src.data_utils import normalize_features, generate_synthetic_data
from src.evaluation import calculate_classification_metrics, print_classification_report

# 1. Generate/Load data
X_train, y_train = generate_synthetic_data(n_samples_per_class=5000)

# 2. Apply Z-Score normalization (NEW!)
X_train_norm, norm_params = normalize_features(X_train, method='zscore')

# 3. Train with Mini-Batch Gradient Ascent (NEW!)
model = LogisticRegression(
    learning_rate=0.03,
    n_iterations=10000,
    batch_size=512  # NEW: Mini-batch size
)
model.fit(X_train_norm, y_train)

# 4. Evaluate with comprehensive metrics (NEW!)
y_pred = model.predict(X_train_norm)
metrics = calculate_classification_metrics(y_train, y_pred)
print_classification_report(metrics, "Training")
```

### Test Set Evaluation

```python
from src.data_utils import apply_normalization

# Generate test data
X_test, y_test = generate_synthetic_data(n_samples_per_class=500, random_state=123)

# Apply same normalization as training
X_test_norm = apply_normalization(X_test, norm_params)

# Predict and evaluate
y_pred_test = model.predict(X_test_norm)
test_metrics = calculate_classification_metrics(y_test, y_pred_test)
print_classification_report(test_metrics, "Test")
```

## Feature Comparison

### Normalization Options

```python
# Z-Score (Recommended for Gradient Ascent)
X_norm, params = normalize_features(X, method='zscore')
# Result: mean=0, std=1

# Min-Max (For [0,1] range)
X_norm, params = normalize_features(X, method='minmax')
# Result: range=[0, 1]
```

### Batch Size Options

```python
# Full Batch (original implementation)
model = LogisticRegression(batch_size=None)

# Mini-Batch (recommended for large datasets)
model = LogisticRegression(batch_size=512)

# Small Mini-Batch (for very large datasets)
model = LogisticRegression(batch_size=128)
```

### Evaluation Metrics

**Before (Original):**
- Accuracy
- MSE
- Confusion Matrix components

**After (Improved):**
- Accuracy
- **Precision** (NEW)
- **Recall** (NEW)
- **F1 Score** (NEW)
- MSE
- Full Confusion Matrix

## Results You'll See

```
======================================================================
TEST SET CLASSIFICATION REPORT
======================================================================

Confusion Matrix:
                    Predicted Negative | Predicted Positive
  Actual Negative:         496         |          4
  Actual Positive:           0         |        500

Performance Metrics:
  Accuracy:  0.9960 (99.60%)
  Precision: 0.9921 (99.21%)
  Recall:    1.0000 (100.00%)
  F1 Score:  0.9960

Metric Explanations:
  - Precision: Of all positive predictions, 99.2% were correct
  - Recall:    Of all actual positives, 100.0% were found
  - F1 Score:  Harmonic mean of Precision and Recall (balanced metric)
```

## When to Use Each Improvement

### 1. Z-Score Normalization
**Use when:**
- Features have different scales
- Using gradient-based optimization
- Want faster convergence
- Working with real-world data

**Skip when:**
- Data already normalized
- All features same scale (rare)

### 2. Mini-Batch Gradient Ascent
**Use when:**
- Dataset > 10,000 samples
- Memory constraints
- Want faster iterations
- Training on large datasets

**Recommended batch sizes:**
- 10,000 samples: `batch_size=512`
- 100,000 samples: `batch_size=1024`
- 1,000,000+ samples: `batch_size=2048`

**Use Full Batch when:**
- Dataset < 1,000 samples
- Want most stable convergence
- Memory not a concern

### 3. Comprehensive Metrics
**Always use!** Especially when:
- Classes are imbalanced
- False positives/negatives have different costs
- Reporting to stakeholders
- Comparing models

### 4. Real-World Adaptation
**Use when:**
- Working with production data
- Have categorical features
- Dealing with missing values
- Need to handle high dimensionality

## Common Usage Patterns

### Pattern 1: Small Dataset (< 1,000 samples)

```python
# Use full batch, Z-Score normalization
X_norm, params = normalize_features(X, method='zscore')
model = LogisticRegression(
    learning_rate=0.03,
    batch_size=None  # Full batch
)
model.fit(X_norm, y)
```

### Pattern 2: Large Dataset (> 10,000 samples)

```python
# Use mini-batch, Z-Score normalization
X_norm, params = normalize_features(X, method='zscore')
model = LogisticRegression(
    learning_rate=0.01,  # Lower LR for mini-batch
    batch_size=512
)
model.fit(X_norm, y)
```

### Pattern 3: Real-World Dataset

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# See REAL_WORLD_ADAPTATION.md for complete guide

# 1. Load data
df = pd.read_csv('data.csv')

# 2. Preprocess (handle missing, encode categorical)
# ... (see REAL_WORLD_ADAPTATION.md)

# 3. Normalize
X_norm, params = normalize_features(X, method='zscore')

# 4. Train
model = LogisticRegression(learning_rate=0.01, batch_size=256)
model.fit(X_norm, y)

# 5. Comprehensive evaluation
metrics = calculate_classification_metrics(y_test, y_pred)
print_classification_report(metrics)
```

## Troubleshooting

### Issue: Model not converging with mini-batch

**Solution:** Lower learning rate or increase batch size
```python
model = LogisticRegression(
    learning_rate=0.01,  # Was 0.03
    batch_size=1024      # Was 512
)
```

### Issue: Poor test performance despite good training

**Solution:** Check normalization consistency
```python
# Make sure test uses TRAINING normalization params
X_test_norm = apply_normalization(X_test, norm_params)  # ✓ Correct
# NOT: X_test_norm, _ = normalize_features(X_test)      # ✗ Wrong!
```

### Issue: Low precision but high recall

**Solution:** Adjust decision threshold
```python
# Default threshold is 0.5
y_pred = model.predict(X_test, threshold=0.5)

# Increase threshold to improve precision
y_pred = model.predict(X_test, threshold=0.7)
```

## Files to Reference

1. **[IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)** - Detailed technical documentation
2. **[REAL_WORLD_ADAPTATION.md](REAL_WORLD_ADAPTATION.md)** - Complete guide for production data
3. **[main.py](main.py)** - Full working example with all improvements
4. **[src/data_utils.py](src/data_utils.py)** - Normalization functions
5. **[src/evaluation.py](src/evaluation.py)** - Metrics calculation

## Run Complete Demo

```bash
# Run the improved implementation
python main.py

# Output includes:
# - Z-Score normalization statistics
# - Mini-batch training progress
# - Comprehensive metrics (Precision, Recall, F1)
# - Test set evaluation
# - Generated plots
```

## Next Steps

1. ✅ Run `python main.py` to see all improvements in action
2. ✅ Read [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md) for technical details
3. ✅ Review [REAL_WORLD_ADAPTATION.md](REAL_WORLD_ADAPTATION.md) when ready for production data
4. ✅ Experiment with different `batch_size` values
5. ✅ Try both `zscore` and `minmax` normalization methods
6. ✅ Analyze Precision/Recall trade-offs with different thresholds
