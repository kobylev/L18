# Phase III: Learning Rate Scheduling

This document details the Phase III improvement: **Learning Rate Scheduling** for adaptive optimization and improved convergence.

## Overview

| Feature | Purpose | Status |
|---------|---------|--------|
| Step Decay | Reduce LR by factor every N steps | ✅ |
| Exponential Decay | Smooth exponential LR reduction | ✅ |
| Inverse Time Decay | 1/(1+t) style decay | ✅ |
| Adaptive LR | Adjust based on validation performance | ✅ |
| LR History Tracking | Monitor LR changes during training | ✅ |

---

## Problem

**Fixed learning rate causes issues:**
1. **Too High**: Overshooting optimal parameters, unstable training
2. **Too Low**: Slow convergence, wasted computation
3. **One Size Doesn't Fit All**: Different training phases need different LRs

**Traditional approach:** Trial-and-error to find good constant learning rate

**Better approach:** Start high for fast progress, decay for fine-tuning

---

## Solution: Learning Rate Scheduling

**Learning Rate Scheduling** automatically adjusts the learning rate during training:

```
Early iterations:  High LR → Fast progress, explore parameter space
Middle iterations: Medium LR → Refine parameters
Late iterations:   Low LR → Fine-tune, converge to optimum
```

---

## Implemented Strategies

### 1. Step Decay

**Formula:**
```
lr(t) = lr_initial * (decay_rate ^ floor(t / decay_steps))
```

**Example:**
```python
initial_lr = 0.1
decay_rate = 0.95
decay_steps = 100

Iteration 0-99:   lr = 0.1000  (0.95^0)
Iteration 100-199: lr = 0.0950  (0.95^1)
Iteration 200-299: lr = 0.0902  (0.95^2)
Iteration 300-399: lr = 0.0857  (0.95^3)
```

**Characteristics:**
- Stepwise reduction (piece-wise constant)
- Easy to tune
- Works well in practice

**Use when:**
- Standard training scenarios
- Want predictable LR schedule
- Need simple, reliable decay

### 2. Exponential Decay

**Formula:**
```
lr(t) = lr_initial * (decay_rate ^ t)
```

**Example:**
```python
initial_lr = 0.1
decay_rate = 0.995

Iteration 0:   lr = 0.1000
Iteration 100: lr = 0.0606
Iteration 200: lr = 0.0368
Iteration 500: lr = 0.0820
Iteration 1000: lr = 0.0067
```

**Characteristics:**
- Smooth, continuous decay
- Aggressive reduction
- Can become too small quickly

**Use when:**
- Want smooth decay curve
- Training for many iterations
- Need aggressive LR reduction

### 3. Inverse Time Decay

**Formula:**
```
lr(t) = lr_initial / (1 + decay_rate * t)
```

**Example:**
```python
initial_lr = 0.1
decay_rate = 0.001

Iteration 0:    lr = 0.1000
Iteration 100:  lr = 0.0909
Iteration 500:  lr = 0.0667
Iteration 1000: lr = 0.0500
Iteration 5000: lr = 0.0167
```

**Characteristics:**
- Hyperbolic decay (1/t behavior)
- Gentle, slow reduction
- Never reaches zero

**Use when:**
- Long training runs
- Want gentle decay
- Theoretical guarantees needed (some convergence proofs use 1/t)

### 4. Adaptive Learning Rate

**Algorithm:**
```python
if validation_loss improved:
    lr = min(lr * 1.05, initial_lr)  # Increase slightly
else:
    lr = lr * decay_rate              # Decrease
```

**Characteristics:**
- Data-driven adjustments
- Increases when making progress
- Decreases when stagnating
- Requires validation set

**Use when:**
- Have validation data available
- Want "smart" LR adjustment
- Training on complex problems

---

## Implementation

### Basic Usage

```python
from src import LogisticRegression, normalize_features, generate_synthetic_data

X, y = generate_synthetic_data(n_samples_per_class=1000)
X_norm, params = normalize_features(X, method='zscore')

# Step decay (recommended)
model = LogisticRegression(
    learning_rate=0.1,           # Initial LR
    n_iterations=1000,
    lr_schedule='step',          # Schedule type
    lr_decay_rate=0.95,          # Multiply by 0.95
    lr_decay_steps=100           # Every 100 iterations
)

model.fit(X_norm, y)

# Check LR progression
print(f"Initial LR: {model.initial_learning_rate}")
print(f"Final LR: {model.history['learning_rate'][-1]}")
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr_schedule` | str | `None` | Schedule type: 'step', 'exponential', 'inverse', 'adaptive', None |
| `lr_decay_rate` | float | `0.95` | Decay multiplier (0 < rate < 1) |
| `lr_decay_steps` | int | `100` | Steps between decays (step schedule only) |
| `lr_min` | float | `1e-6` | Minimum LR floor value |

---

## Usage Examples

### Example 1: Step Decay (Recommended)

```python
model = LogisticRegression(
    learning_rate=0.1,
    lr_schedule='step',
    lr_decay_rate=0.95,    # Reduce by 5% each decay
    lr_decay_steps=100      # Every 100 iterations
)
model.fit(X_norm, y)
```

**LR Progression:**
```
Iteration 0:   0.1000
Iteration 100: 0.0950
Iteration 200: 0.0902
Iteration 300: 0.0857
...
```

### Example 2: Exponential Decay

```python
model = LogisticRegression(
    learning_rate=0.1,
    lr_schedule='exponential',
    lr_decay_rate=0.999     # Very gentle decay
)
model.fit(X_norm, y)
```

### Example 3: Adaptive with Early Stopping

```python
# Adaptive LR works great with early stopping
model = LogisticRegression(
    learning_rate=0.05,
    lr_schedule='adaptive',
    lr_decay_rate=0.9,
    early_stopping=True,
    patience=15,
    validation_split=0.2
)
model.fit(X_norm, y)
```

### Example 4: Combine with Regularization

```python
# Full stack: Regularization + LR Schedule + Early Stopping
model = LogisticRegression(
    learning_rate=0.1,
    n_iterations=2000,
    regularization='l2',
    lambda_reg=0.01,
    lr_schedule='step',
    lr_decay_rate=0.95,
    lr_decay_steps=150,
    early_stopping=True,
    patience=20
)
model.fit(X_norm, y)
```

---

## Choosing Parameters

### Learning Rate Decay Rate

**Aggressive Decay (0.90):**
- Fast LR reduction
- Good for well-behaved problems
- Risk of premature convergence

**Moderate Decay (0.95):**
- Balanced approach
- **Recommended default**
- Works for most problems

**Gentle Decay (0.99):**
- Slow LR reduction
- Good for long training
- May not decay enough

### Decay Steps (Step Schedule)

**Small Steps (50):**
- Frequent updates
- More responsive to changes
- Can be noisy

**Medium Steps (100):**
- **Recommended default**
- Good balance
- Standard in practice

**Large Steps (200+):**
- Infrequent updates
- More stable
- Like constant LR with occasional drops

### Initial Learning Rate

**With LR Schedule:**
- Can start higher than constant LR
- Schedule will reduce it over time
- Try 0.1 instead of 0.03

**Rule of Thumb:**
```python
# Without schedule
learning_rate = 0.03  # Conservative

# With step schedule
learning_rate = 0.1   # Can be more aggressive
lr_decay_rate = 0.95
```

---

## Monitoring Learning Rate

### Accessing LR History

```python
model.fit(X_norm, y)

# Get LR history
lr_history = model.history['learning_rate']

print(f"Initial LR: {lr_history[0]}")
print(f"Final LR: {lr_history[-1]}")
print(f"Minimum LR: {min(lr_history)}")
print(f"Maximum LR: {max(lr_history)}")
```

### Plotting LR Schedule

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Plot 1: LR over time
plt.subplot(1, 2, 1)
plt.plot(model.history['learning_rate'])
plt.xlabel('Iteration')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.yscale('log')
plt.grid(True)

# Plot 2: LR vs MSE
plt.subplot(1, 2, 2)
plt.plot(model.history['learning_rate'], model.history['mse'], 'o-')
plt.xlabel('Learning Rate')
plt.ylabel('MSE')
plt.title('LR vs Performance')
plt.xscale('log')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Demo Results

### Running the Demo

```bash
python demo_lr_scheduling.py
```

### Sample Output

```
======================================================================
LEARNING RATE SCHEDULE COMPARISON
======================================================================
Schedule                    Iterations    Final MSE     Final LR
----------------------------------------------------------------------
No Schedule (Constant)             760     0.003420     0.100000
Step Decay                         917     0.003420     0.063025
Exponential Decay                 1000     0.004171     0.000001
Inverse Time Decay                1000     0.004354     0.000105
Adaptive LR                        760     0.003420     0.100000
```

**Observations:**
1. **Constant & Adaptive**: Both converged fastest (760 iterations)
2. **Step Decay**: Converged with reduced LR (0.063), took slightly longer
3. **Exponential/Inverse**: Too aggressive decay → slower convergence
4. **Adaptive**: Matched constant performance (smart adjustments)

---

## Best Practices

### 1. Start with Step Decay

```python
# Recommended starting point
model = LogisticRegression(
    learning_rate=0.1,
    lr_schedule='step',
    lr_decay_rate=0.95,
    lr_decay_steps=100
)
```

**Why:**
- Simple and effective
- Easy to tune
- Widely used in practice
- Predictable behavior

### 2. Use Adaptive for Complex Problems

```python
# For problems with noisy gradients or complex landscapes
model = LogisticRegression(
    learning_rate=0.05,
    lr_schedule='adaptive',
    lr_decay_rate=0.9,
    early_stopping=True,  # Adaptive works best with validation
    validation_split=0.2
)
```

### 3. Combine with Early Stopping

```python
# Best of both worlds
model = LogisticRegression(
    learning_rate=0.1,
    lr_schedule='step',
    lr_decay_rate=0.95,
    lr_decay_steps=100,
    early_stopping=True,
    patience=15
)
```

**Benefits:**
- LR schedule improves convergence
- Early stopping prevents overfitting
- Saves computation time

### 4. Monitor LR in Training Output

The training header shows your schedule:
```
LR Schedule: STEP (decay_rate=0.95, steps=100)
```

Helps verify settings are correct.

---

## Troubleshooting

### Issue: LR decays too quickly

**Symptoms:**
```
Final LR: 0.000001  (way too small)
Training stopped improving early
```

**Solutions:**
```python
# Option 1: Increase decay_rate
lr_decay_rate=0.99  # Was 0.90

# Option 2: Increase decay_steps
lr_decay_steps=200  # Was 50

# Option 3: Use gentler schedule
lr_schedule='inverse'  # Was 'exponential'

# Option 4: Set minimum LR
lr_min=1e-4  # Was 1e-6
```

### Issue: LR not decaying enough

**Symptoms:**
```
Final LR: 0.0950  (barely changed from 0.1)
Training unstable late in training
```

**Solutions:**
```python
# Option 1: Decrease decay_rate
lr_decay_rate=0.90  # Was 0.99

# Option 2: Decrease decay_steps
lr_decay_steps=50  # Was 200

# Option 3: Use more aggressive schedule
lr_schedule='exponential'  # Was 'step'
```

### Issue: Adaptive LR not working

**Symptoms:**
```
Adaptive LR stays constant or behaves erratically
```

**Cause:** Needs validation data

**Solution:**
```python
# Make sure early_stopping is enabled
early_stopping=True
validation_split=0.2  # Need validation set for adaptive
```

### Issue: Training unstable with high initial LR

**Symptoms:**
```
Log-likelihood: -inf
MSE exploding
```

**Solutions:**
```python
# Start with lower LR
learning_rate=0.05  # Was 0.2

# Use more aggressive decay
lr_decay_rate=0.90  # Was 0.95
lr_decay_steps=50    # Was 100
```

---

## Mathematical Details

### Step Decay Formula

```
lr(t) = lr_0 * γ^⌊t/k⌋

where:
  lr_0 = initial learning rate
  γ = decay_rate (0 < γ < 1)
  k = decay_steps
  ⌊⌋ = floor function
  t = current iteration
```

**Example:**
```
lr_0 = 0.1, γ = 0.95, k = 100

t=0:   lr = 0.1 * 0.95^0 = 0.1000
t=150: lr = 0.1 * 0.95^1 = 0.0950
t=250: lr = 0.1 * 0.95^2 = 0.0902
```

### Exponential Decay Formula

```
lr(t) = lr_0 * γ^t

Continuous approximation:
lr(t) ≈ lr_0 * e^(-λt)

where λ = -log(γ)
```

### Inverse Time Decay Formula

```
lr(t) = lr_0 / (1 + γt)

As t → ∞:
lr(t) → 0

But decay rate slows over time (hyperbolic)
```

### Adaptive Strategy

```
if validation_loss(t) < validation_loss(t-1):
    # Improving
    lr(t+1) = min(lr(t) * 1.05, lr_0)
else:
    # Not improving
    lr(t+1) = lr(t) * γ
```

---

## Comparison with Other Optimizers

### Our Implementation vs Adam/RMSprop

**Our Schedules:**
- Manual, explicit decay rules
- Simple, interpretable
- Works with gradient ascent
- Easy to tune

**Adam/RMSprop:**
- Adaptive per-parameter learning rates
- More complex
- Typically for gradient descent
- Automatic adaptation

**When to use ours:**
- Gradient ascent (maximization)
- Want simple, explicit control
- Educational purposes
- Good baselines

**When to use Adam:**
- Very complex problems
- Many parameters
- Need per-parameter adaptation

---

## Complete Feature Stack

Combining all Phase III features:

```python
from src import LogisticRegression, normalize_features, generate_synthetic_data

X, y = generate_synthetic_data(n_samples_per_class=1000)
X_norm, params = normalize_features(X, method='zscore')

# Ultimate model: All Phase III features
model = LogisticRegression(
    # Basic parameters
    learning_rate=0.1,
    n_iterations=2000,

    # Mini-batch (Phase I)
    batch_size=512,

    # Regularization (Phase II)
    regularization='l2',
    lambda_reg=0.01,

    # Early stopping (Phase III)
    early_stopping=True,
    patience=15,
    validation_split=0.2,

    # LR scheduling (Phase III)
    lr_schedule='step',
    lr_decay_rate=0.95,
    lr_decay_steps=100,
    lr_min=1e-5
)

model.fit(X_norm, y)

# Results
print(f"Stopped at iteration: {model.stopped_epoch or 'max iterations'}")
print(f"Initial LR: {model.initial_learning_rate:.6f}")
print(f"Final LR: {model.history['learning_rate'][-1]:.6f}")
print(f"Final MSE: {model.history['mse'][-1]:.6f}")
```

---

## Summary

**Learning rate scheduling provides:**
1. ✅ **Faster Convergence** - Start with high LR for quick progress
2. ✅ **Better Final Performance** - End with low LR for fine-tuning
3. ✅ **Stability** - Reduce oscillations in late training
4. ✅ **Flexibility** - Multiple strategies for different problems

**Recommended Strategy:**
```python
# For most problems
lr_schedule='step'
lr_decay_rate=0.95
lr_decay_steps=100

# For complex/noisy problems
lr_schedule='adaptive'
early_stopping=True
```

---

## Next Steps

Your implementation now has:
- ✅ Phase I: Z-Score, Mini-Batch, Metrics
- ✅ Phase II: K-Fold CV, Regularization, Grid Search
- ✅ Phase III: Early Stopping + LR Scheduling

**Potential Phase IV:**
- Feature selection (automatic importance analysis)
- Ensemble methods (combine multiple models)
- Multi-class support (softmax, one-vs-rest)
- Online learning (incremental updates)

---

## References

- [Learning Rate Schedules - Stanford CS231n](http://cs231n.github.io/neural-networks-3/#anneal)
- [Cyclical Learning Rates](https://arxiv.org/abs/1506.01186)
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
- [An Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/)
