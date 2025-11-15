# Quick Start Guide

## 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install numpy matplotlib pandas
```

## 2. Run the Implementation

**Modular version (recommended):**
```bash
python main.py
```

**Legacy single-file version:**
```bash
python logistic_regression.py
```

## 3. Check Results
All output files are in the `output/` folder:
- `output/results_table.csv` - Detailed predictions and errors
- `output/classification_plot.png` - Visual classification results
- `output/convergence_plot.png` - Training progress

## Expected Output

```
======================================================================
LOGISTIC REGRESSION - BINARY CLASSIFICATION
Manual Implementation using Gradient Ascent
======================================================================

======================================================================
PHASE 1: DATA PREPARATION
======================================================================
Created output directory: output/

Dataset created successfully!
  Total samples: 10000
  Class 0 samples: 5000
  Class 1 samples: 5000

======================================================================
PHASE 2: GRADIENT ASCENT IMPLEMENTATION
======================================================================
Starting Gradient Ascent...
Initial β: [0.0174 0.0040 0.0097]

Iteration    0 | Log-Likelihood:  -6931.4718 | MSE: 0.250000
...
Converged at iteration XXX!

Final β coefficients: [...]
Final Log-Likelihood: ...
Final MSE: 0.00XXXX

======================================================================
PHASE 3: GENERATING DELIVERABLES
======================================================================

--- Results Table ---
[Table with predictions and errors]

Classification plot saved to: output/classification_plot.png
Convergence plot saved to: output/convergence_plot.png

======================================================================
ADVANCED: TESTING ON UNSEEN DATA
======================================================================
Test MSE: 0.00XXXX
Test Accuracy: 0.99XX (99.XX%)
```

## What You Get

1. **Trained Model**: Optimized β coefficients for classification
2. **Visualizations**: High-quality plots showing results
3. **Metrics**: MSE, accuracy, confusion matrix
4. **Proof of Generalization**: Performance on unseen data

## Common Use Cases

### Adjust Learning Rate
```python
from src import LogisticRegression

model = LogisticRegression(learning_rate=0.05)  # Faster
model = LogisticRegression(learning_rate=0.01)  # More stable
```

### More Data
```python
from src import generate_synthetic_data

X, y = generate_synthetic_data(n_samples_per_class=10000)  # 20k total
```

### Custom Output Location
```python
from src import create_output_directory, plot_classification

output_dir = create_output_directory('my_results')
plot_classification(X, y, p_hat, model, output_dir)
```

## Troubleshooting

**Issue**: Dependencies not found
**Solution**: `pip install numpy matplotlib pandas`

**Issue**: Output folder not created
**Solution**: Script creates it automatically - check permissions

**Issue**: Slow convergence
**Solution**: Increase learning rate or check data separation

## Next Steps

1. Review the plots in the `output/` folder
2. Analyze the results table CSV
3. Experiment with different hyperparameters
4. Try with your own data (see README.md)

For detailed documentation, see [README.md](README.md)
