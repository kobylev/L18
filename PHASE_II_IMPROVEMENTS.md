# Phase II Improvements: Cross-Validation & Regularization

This document details the Phase II improvements to the Logistic Regression implementation, focusing on robust evaluation and overfitting prevention.

## Overview

| # | Feature | Purpose | File |
|---|---------|---------|------|
| 5 | K-Fold Cross-Validation | Robust performance estimation | [src/cross_validation.py](src/cross_validation.py) |
| 6 | L1/L2 Regularization | Prevent overfitting | [src/trainer.py](src/trainer.py), [src/model_base.py](src/model_base.py) |

---

## 5️⃣ K-Fold Cross-Validation

### Problem

A single train/test split can give unreliable performance estimates because:
- Results depend on which samples end up in train vs test
- Limited data usage (typically 20-30% held out for testing)
- High variance in performance metrics

### Solution

**K-Fold Cross-Validation** splits data into K parts (folds), trains K different models, and averages their performance:

```
Fold 1: [Test] [Train] [Train] [Train] [Train]
Fold 2: [Train] [Test] [Train] [Train] [Train]
Fold 3: [Train] [Train] [Test] [Train] [Train]
Fold 4: [Train] [Train] [Train] [Test] [Train]
Fold 5: [Train] [Train] [Train] [Train] [Test]
```

Each fold gets a turn as the validation set, providing a more stable error estimate.

### Implementation

**File:** [src/cross_validation.py](src/cross_validation.py)

#### Basic K-Fold Cross-Validation

```python
from src import cross_validate, normalize_features, generate_synthetic_data

# Generate and normalize data
X, y = generate_synthetic_data(n_samples_per_class=1000)
X_norm, params = normalize_features(X, method='zscore')

# Define model parameters
model_params = {
    'learning_rate': 0.03,
    'n_iterations': 2000,
    'batch_size': None,
    'regularization': None,
    'lambda_reg': 0.0
}

# Run 5-fold cross-validation
cv_results = cross_validate(
    X_norm, y,
    model_params=model_params,
    k=5,
    random_state=42,
    verbose=True
)

# Results
print(f"Mean F1 Score: {cv_results['mean_scores']['f1_score']:.4f}")
print(f"Std F1 Score:  {cv_results['std_scores']['f1_score']:.4f}")
```

#### Stratified K-Fold (Maintains Class Balance)

```python
from src.cross_validation import stratified_k_fold_split

# Ensures each fold has same proportion of classes as full dataset
folds = stratified_k_fold_split(X, y, k=5, random_state=42)

for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(folds):
    print(f"Fold {fold_idx + 1}:")
    print(f"  Class 0 ratio (train): {np.sum(y_train==0)/len(y_train):.2%}")
    print(f"  Class 0 ratio (val):   {np.sum(y_val==0)/len(y_val):.2%}")
```

### Benefits

1. **More Reliable Estimates**: Averaging over K folds reduces variance
2. **Better Data Usage**: Every sample used for both training and validation
3. **Confidence Intervals**: Standard deviation across folds shows uncertainty
4. **Detect Overfitting**: High std indicates model is sensitive to data split

### Results Example

```
K-FOLD CROSS-VALIDATION (k=5)
Dataset size: 2000 samples
Fold size: ~400 samples per fold

Fold 1/5: Accuracy: 1.0000, F1: 1.0000
Fold 2/5: Accuracy: 0.9900, F1: 0.9900
Fold 3/5: Accuracy: 0.9925, F1: 0.9926
Fold 4/5: Accuracy: 0.9950, F1: 0.9948
Fold 5/5: Accuracy: 0.9950, F1: 0.9950

CROSS-VALIDATION RESULTS (Mean ± Std)
Accuracy:  0.9945 ± 0.0033
Precision: 0.9930 ± 0.0040
Recall:    0.9960 ± 0.0049
F1 Score:  0.9945 ± 0.0033
```

**Interpretation:**
- Mean F1 = 0.9945: Expected performance on unseen data
- Std = 0.0033: Low variance indicates stable model
- Compare to single split: More trustworthy estimate

### Grid Search with Cross-Validation

Combine K-Fold with hyperparameter search:

```python
from src import grid_search_cv

# Define parameter grid to search
param_grid = {
    'learning_rate': [0.01, 0.03, 0.1],
    'regularization': [None, 'l1', 'l2'],
    'lambda_reg': [0.0, 0.01, 0.1],
    'n_iterations': [2000],
    'batch_size': [None]
}

# Search for best combination
results = grid_search_cv(
    X_norm, y,
    param_grid=param_grid,
    k=5,
    random_state=42,
    verbose=True
)

print(f"Best parameters: {results['best_params']}")
print(f"Best F1 score: {results['best_score']:.4f}")

# Train final model with best parameters
from src import LogisticRegression
final_model = LogisticRegression(**results['best_params'])
final_model.fit(X_norm, y)
```

---

## 6️⃣ Regularization (L1, L2, Elastic Net)

### Problem

**Overfitting** occurs when a model memorizes training data instead of learning general patterns:
- High training accuracy but poor test accuracy
- Model coefficients (beta values) become too large
- Poor generalization to new data

### Solution

**Regularization** adds a penalty term to discourage large coefficients:

```
Original:     Maximize  Log-Likelihood(beta)
Regularized:  Maximize  Log-Likelihood(beta) - lambda * Penalty(beta)
```

### Types of Regularization

#### L2 Regularization (Ridge)

**Penalty:** λ × Σ(beta²)

**Effect:**
- Shrinks all coefficients toward zero
- Prefers small, distributed weights
- Smooth parameter space (differentiable everywhere)

**Use when:**
- All features might be relevant
- Want to reduce coefficient magnitude
- Prefer smooth solutions

**Gradient (for gradient ascent):**
```
Regularization term: -2 * lambda * beta
```

#### L1 Regularization (Lasso)

**Penalty:** λ × Σ|beta|

**Effect:**
- Can shrink coefficients to exactly zero
- Performs automatic feature selection
- Sparse solutions (many zeros)

**Use when:**
- Believe many features are irrelevant
- Want automatic feature selection
- Prefer sparse solutions

**Gradient (for gradient ascent):**
```
Regularization term: -lambda * sign(beta)
```

#### Elastic Net

**Penalty:** λ × [α × Σ|beta| + (1-α) × Σ(beta²)]

**Effect:**
- Combines L1 and L2
- Gets benefits of both
- α controls the mix (α=0.5 is equal mix)

### Implementation

**Files Modified:**
- [src/model_base.py](src/model_base.py#L23-L50) - Added `regularization` and `lambda_reg` parameters
- [src/trainer.py](src/trainer.py#L111-L150) - Added `_compute_regularization_gradient()` method

#### Basic Usage

```python
from src import LogisticRegression, normalize_features

# No Regularization (default)
model_none = LogisticRegression(
    learning_rate=0.03,
    regularization=None
)

# L1 Regularization (Lasso)
model_l1 = LogisticRegression(
    learning_rate=0.03,
    regularization='l1',
    lambda_reg=0.01  # Regularization strength
)

# L2 Regularization (Ridge)
model_l2 = LogisticRegression(
    learning_rate=0.03,
    regularization='l2',
    lambda_reg=0.01
)

# Elastic Net
model_elastic = LogisticRegression(
    learning_rate=0.03,
    regularization='elastic',
    lambda_reg=0.01
)
```

#### Choosing Lambda (Regularization Strength)

**Lambda = 0:** No regularization (risk of overfitting)
**Lambda too small (0.001):** Minimal effect
**Lambda moderate (0.01-0.1):** Good balance
**Lambda too large (10+):** Underfitting (all coefficients near zero)

**How to find optimal lambda:**
```python
# Use Grid Search with Cross-Validation
param_grid = {
    'learning_rate': [0.03],
    'regularization': ['l2'],
    'lambda_reg': [0.0, 0.001, 0.01, 0.1, 1.0],
    'n_iterations': [2000]
}

results = grid_search_cv(X, y, param_grid, k=5)
print(f"Best lambda: {results['best_params']['lambda_reg']}")
```

### Mathematical Details

#### Gradient Ascent without Regularization

```
beta_new = beta_old + learning_rate * (X.T @ (y - p_hat))
```

#### Gradient Ascent with L2 Regularization

```
gradient = X.T @ (y - p_hat)
reg_gradient = 2 * lambda * beta  # L2 penalty gradient
beta_new = beta_old + learning_rate * (gradient - reg_gradient)
```

Note: We **subtract** the regularization gradient because we're **maximizing** log-likelihood.

#### Why Don't We Regularize the Bias Term?

The bias term (beta_0) is not regularized because:
- It doesn't cause overfitting (just shifts the decision boundary)
- Regularizing it can hurt performance
- Standard practice in machine learning

```python
# In the code
reg_grad = np.zeros_like(self.beta)
reg_grad[1:] = ...  # Only regularize beta_1, beta_2, etc., not beta_0
```

### Results Example

```
REGULARIZATION COMPARISON

Regularization             Train Acc   Test Acc      Gap
----------------------------------------------------------------------
None                          0.9944     1.0000  -0.0056
l1 (lambda=0.01)              0.9944     1.0000  -0.0056
l2 (lambda=0.01)              0.9944     1.0000  -0.0056
l2 (lambda=0.1)               0.9944     1.0000  -0.0056

Beta coefficients:
  No Regularization: [-0.255,  7.139,  7.334]
  L2 (lambda=0.01):  [-0.228,  6.809,  6.965]  <- Smaller coefficients
  L2 (lambda=0.1):   [-0.131,  5.507,  5.555]  <- Even smaller
```

**Observations:**
1. As lambda increases, coefficients shrink toward zero
2. L2 prevents extremely large coefficients
3. Training/test accuracy remain high (well-separated data)
4. For harder problems, regularization would reduce overfitting gap

### When to Use Which Regularization?

| Scenario | Recommendation |
|----------|----------------|
| Many features, unsure which are important | **L1 (Lasso)** - automatic feature selection |
| All features seem relevant | **L2 (Ridge)** - smooth shrinkage |
| Mixed scenario | **Elastic Net** - best of both |
| Small dataset, risk of overfitting | **L2 with lambda=0.1** |
| Large dataset, low overfitting risk | **None or L2 with lambda=0.01** |

---

## Combined Usage: Cross-Validation + Regularization

The power comes from combining these techniques:

```python
from src import grid_search_cv, normalize_features, generate_synthetic_data

# 1. Generate and normalize data
X, y = generate_synthetic_data(n_samples_per_class=1000)
X_norm, params = normalize_features(X, method='zscore')

# 2. Define search space including regularization
param_grid = {
    'learning_rate': [0.01, 0.03, 0.1],
    'n_iterations': [2000],
    'batch_size': [None, 256],
    'regularization': [None, 'l1', 'l2'],
    'lambda_reg': [0.0, 0.01, 0.1, 1.0]
}

# 3. Find best combination using K-Fold CV
results = grid_search_cv(
    X_norm, y,
    param_grid=param_grid,
    k=5,
    random_state=42,
    verbose=True
)

# 4. Train final model
from src import LogisticRegression
best_model = LogisticRegression(**results['best_params'])
best_model.fit(X_norm, y)

print("Best model configuration:")
print(f"  Regularization: {results['best_params']['regularization']}")
print(f"  Lambda: {results['best_params']['lambda_reg']}")
print(f"  Expected F1: {results['best_score']:.4f}")
```

---

## Complete Feature Comparison

| Feature | Phase I | Phase II |
|---------|---------|----------|
| **Normalization** | Z-Score ✅ | ✅ |
| **Training** | Mini-Batch ✅ | ✅ |
| **Metrics** | Precision, Recall, F1 ✅ | ✅ |
| **Validation** | Single Split | **K-Fold CV** ✅ |
| **Overfitting Prevention** | ❌ | **L1/L2 Regularization** ✅ |
| **Hyperparameter Tuning** | Manual | **Grid Search** ✅ |

---

## Running the Demos

### Demo 1: Regularization Comparison

```bash
python demo_advanced_features.py
```

This will:
1. Train models with No Reg, L1, L2, and different lambda values
2. Compare beta coefficients
3. Show overfitting gaps (train acc - test acc)

### Demo 2: K-Fold Cross-Validation

Included in the same demo:
- Runs 5-fold CV
- Shows per-fold results
- Reports mean ± std for all metrics

### Demo 3: Grid Search (Optional)

Select 'y' when prompted to run hyperparameter search.

---

## Key Formulas

### K-Fold Cross-Validation

**Mean Score:**
```
Mean = (1/K) * Σ(score_i)  for i=1 to K
```

**Standard Deviation:**
```
Std = sqrt((1/K) * Σ(score_i - mean)²)
```

### Regularization

**L1 (Lasso) Penalty:**
```
Penalty = lambda * (|beta_1| + |beta_2| + ... + |beta_n|)
Gradient = lambda * [sign(beta_1), sign(beta_2), ..., sign(beta_n)]
```

**L2 (Ridge) Penalty:**
```
Penalty = lambda * (beta_1² + beta_2² + ... + beta_n²)
Gradient = 2 * lambda * [beta_1, beta_2, ..., beta_n]
```

**Elastic Net Penalty:**
```
Penalty = lambda * [alpha * Σ|beta_i| + (1-alpha) * Σ(beta_i²)]
```

---

## Best Practices

### For K-Fold Cross-Validation:

1. **Choose K wisely:**
   - K=5 or K=10 are most common
   - Larger K = more computation but better estimates
   - K=N (Leave-One-Out) for very small datasets

2. **Use stratified K-Fold** for imbalanced classes

3. **Report mean ± std** to show reliability

4. **Never tune hyperparameters on test set** - use CV instead

### For Regularization:

1. **Always normalize features first** (Z-Score recommended)

2. **Start with lambda=0.01** and adjust:
   - Increase if overfitting (high train/test gap)
   - Decrease if underfitting (low train accuracy)

3. **Use Grid Search** to find optimal lambda

4. **Monitor coefficient magnitudes**:
   - Very large (>100): Probably overfitting, increase lambda
   - All near zero (<0.1): Underfitting, decrease lambda

5. **L2 is usually safer** than L1 as a starting point

---

## Troubleshooting

### Issue: High variance in cross-validation scores

**Cause:** Small dataset or difficult problem
**Solution:**
- Increase dataset size if possible
- Try more folds (K=10 instead of K=5)
- Use stratified K-Fold

### Issue: Regularization not helping

**Cause:** Data is too easy (well-separated) or lambda too small
**Solution:**
- Check train/test gap - if already small, regularization won't help much
- Try larger lambda values (0.1, 1.0)
- Ensure features are normalized

### Issue: All coefficients go to zero

**Cause:** Lambda is too large
**Solution:**
- Reduce lambda (try 0.01, 0.001)
- Check if model can train without regularization first

### Issue: Cross-validation takes too long

**Cause:** Large dataset or many hyperparameters
**Solution:**
- Use fewer folds (K=3 instead of K=5)
- Reduce n_iterations
- Use mini-batch training
- Narrow parameter grid

---

## Summary

**Phase II adds critical production features:**

1. **K-Fold Cross-Validation** provides reliable performance estimates
2. **Regularization** prevents overfitting and improves generalization
3. **Grid Search** automates hyperparameter tuning

**Together, these create a robust, production-ready implementation suitable for real-world machine learning tasks.**

---

## Next Steps

Potential Phase III improvements:
- **Early stopping**: Stop training when validation performance plateaus
- **Learning rate scheduling**: Adapt learning rate during training
- **Feature selection**: Automated feature importance analysis
- **Ensemble methods**: Combine multiple models
- **Multi-class support**: Extend to >2 classes (one-vs-rest or softmax)

---

## References

- Cross-Validation: [scikit-learn CV guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- Regularization: [Andrew Ng's ML Course](https://www.coursera.org/learn/machine-learning)
- L1 vs L2: [Towards Data Science](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)
- Grid Search: [Hyperparameter Tuning Guide](https://scikit-learn.org/stable/modules/grid_search.html)
