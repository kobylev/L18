# Architecture Documentation

## Modular Design Overview

The codebase has been refactored from a single monolithic file (~590 lines) into a clean, modular architecture with separated concerns. Each module is focused, maintainable, and typically under 250 lines.

## Design Principles

1. **Separation of Concerns**: Each module handles a specific responsibility
2. **Single Responsibility**: Each function/class has one clear purpose
3. **Short Files**: No file exceeds ~250 lines for easy navigation
4. **Clear Imports**: Package exports via `__init__.py` for clean API
5. **Maintainability**: Easy to test, modify, and extend

## Module Breakdown

### 1. `src/model.py` (~31 lines)
**Purpose**: Main model class

**Contents**:
- `LogisticRegression` class (combines base + trainer via multiple inheritance)

**Dependencies**: `model_base`, `trainer`

**Why separate?**: Clean interface that combines base functionality and training logic.

### 2. `src/model_base.py` (~174 lines)
**Purpose**: Core mathematical functions

**Contents**:
- `LogisticRegressionBase` class
  - Sigmoid function
  - Bias term handling
  - Weighted sum computation
  - Log-likelihood calculation
  - MSE calculation
  - Prediction methods
  - Decision boundary calculation

**Dependencies**: `numpy`

**Why separate?**: Core math functions are independent of training logic, making them easier to test and reuse.

### 3. `src/trainer.py` (~109 lines)
**Purpose**: Training algorithm

**Contents**:
- `LogisticRegressionTrainer` class
  - Gradient ascent implementation
  - Training loop
  - History tracking
  - Progress reporting

**Dependencies**: `numpy`, `model_base`

**Why separate?**: Training logic is complex enough to warrant its own module. This separation allows:
- Easy modification of training algorithm
- Testing training independently
- Alternative optimizers can be added

### 4. `src/data_utils.py` (~55 lines)
**Purpose**: Data generation and preparation

**Contents**:
- `generate_synthetic_data()`: Creates two separable clusters

**Dependencies**: `numpy`

**Why separate?**: Data generation is independent of the model. This allows easy swapping of real datasets or different synthetic generators.

### 3. `src/visualization.py` (~110 lines)
**Purpose**: All plotting and visualization

**Contents**:
- `plot_classification()`: Classification results with decision boundary
- `plot_convergence()`: Training progress plots

**Dependencies**: `numpy`, `matplotlib`, `os`

**Why separate?**: Visualization is a distinct concern. Separating it allows:
- Easy modification of plot styles
- Addition of new plot types
- Testing model without GUI dependencies

### 4. `src/evaluation.py` (~100 lines)
**Purpose**: Model evaluation and testing

**Contents**:
- `create_results_table()`: Format predictions as DataFrame
- `test_on_unseen_data()`: Test set validation and metrics

**Dependencies**: `numpy`, `pandas`, `data_utils`

**Why separate?**: Evaluation logic is distinct from training. This allows:
- Different evaluation strategies
- Multiple metric calculations
- Independent testing

### 5. `src/utils.py` (~25 lines)
**Purpose**: General helper functions

**Contents**:
- `create_output_directory()`: File system utilities

**Dependencies**: `os`

**Why separate?**: Utility functions are cross-cutting concerns used by multiple modules.

### 6. `main.py` (~105 lines)
**Purpose**: Orchestration and workflow

**Contents**:
- `main()`: Coordinates the complete pipeline

**Dependencies**: All `src` modules

**Why separate?**: Main entry point should be thin, just orchestrating the workflow without business logic.

### 7. `src/__init__.py` (~25 lines)
**Purpose**: Package interface

**Contents**:
- Exports all public functions/classes
- Version information

**Why needed?**: Provides clean import API (`from src import LogisticRegression`)

## Data Flow

```
main.py
  │
  ├─> data_utils.generate_synthetic_data()
  │     └─> Returns X, y
  │
  ├─> model.LogisticRegression()
  │     ├─> fit(X, y)
  │     └─> predict_proba(X)
  │
  ├─> evaluation.create_results_table()
  │     └─> Returns DataFrame
  │
  ├─> visualization.plot_classification()
  │     └─> Saves plot to output/
  │
  ├─> visualization.plot_convergence()
  │     └─> Saves plot to output/
  │
  └─> evaluation.test_on_unseen_data()
        └─> Prints metrics
```

## Benefits of Modular Architecture

### 1. Maintainability
- Each file is focused and easy to understand
- Changes to one module don't affect others
- Easier to locate bugs

### 2. Testability
- Each module can be tested independently
- Mock dependencies easily
- Unit tests are simpler

### 3. Reusability
- Import only what you need
- Use individual components in other projects
- Mix and match functionality

### 4. Extensibility
- Add new visualizations without touching model
- Swap data sources without changing algorithm
- Add new metrics without modifying training

### 5. Collaboration
- Multiple developers can work on different modules
- Reduced merge conflicts
- Clear ownership of components

## Migration from Monolithic Version

### Old (logistic_regression.py)
```python
# Everything in one file (~590 lines)
- LogisticRegression class
- generate_synthetic_data()
- create_results_table()
- plot_classification()
- plot_convergence()
- test_on_unseen_data()
- create_output_directory()
- main()
```

### New (Modular)
```python
# Separated into focused modules
src/model.py              # ~240 lines
src/data_utils.py         # ~60 lines
src/visualization.py      # ~110 lines
src/evaluation.py         # ~100 lines
src/utils.py              # ~25 lines
main.py                   # ~105 lines
```

**Result**: Same functionality, better organization!

## Usage Examples

### Import Everything
```python
from src import (
    LogisticRegression,
    generate_synthetic_data,
    plot_classification,
    plot_convergence,
    test_on_unseen_data,
    create_results_table,
    create_output_directory
)
```

### Import Selectively
```python
# Only need model
from src.model import LogisticRegression

# Only need data generation
from src.data_utils import generate_synthetic_data

# Only need plotting
from src.visualization import plot_classification, plot_convergence
```

### Direct Module Access
```python
import src.model as model
import src.data_utils as data

# Use with full module name
X, y = data.generate_synthetic_data()
lr = model.LogisticRegression()
lr.fit(X, y)
```

## Testing Strategy

Each module can be tested independently:

```python
# Test model
from src.model import LogisticRegression
import numpy as np

X = np.array([[0.3, 0.3], [0.7, 0.7]])
y = np.array([0, 1])
model = LogisticRegression()
model.fit(X, y, verbose=False)
assert model.beta is not None

# Test data generation
from src.data_utils import generate_synthetic_data

X, y = generate_synthetic_data(n_samples_per_class=100)
assert X.shape == (200, 2)
assert y.shape == (200,)

# Test utils
from src.utils import create_output_directory
import os

output_dir = create_output_directory('test_output')
assert os.path.exists('test_output')
```

## File Size Comparison

| File | Lines | Purpose |
|------|-------|---------|
| `logistic_regression.py` (old) | ~590 | Everything |
| `src/model.py` | ~240 | ML algorithm only |
| `src/data_utils.py` | ~60 | Data generation |
| `src/visualization.py` | ~110 | Plotting |
| `src/evaluation.py` | ~100 | Testing/metrics |
| `src/utils.py` | ~25 | Helpers |
| `main.py` | ~105 | Orchestration |
| **Total (new)** | **~640** | **Better organized** |

Note: Slightly more total lines due to module headers and imports, but each file is much shorter and focused.

## Future Extensions

The modular architecture makes it easy to add:

1. **New Models**: Add `src/models/ridge_regression.py`
2. **New Datasets**: Add `src/data_utils/real_data.py`
3. **New Visualizations**: Add functions to `src/visualization.py`
4. **New Metrics**: Add functions to `src/evaluation.py`
5. **Configuration**: Add `src/config.py` for hyperparameters
6. **Logging**: Add `src/logger.py` for better tracking

## Conclusion

The modular architecture provides:
- ✅ Shorter, focused files (all under 250 lines)
- ✅ Clear separation of concerns
- ✅ Easy to test and maintain
- ✅ Simple to extend
- ✅ Better code organization
- ✅ Preserved all original functionality

This design follows software engineering best practices while maintaining the simplicity and clarity of the original implementation.
