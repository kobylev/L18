"""
Cross-Validation

Functions for K-Fold Cross-Validation to provide robust model evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple
from .model import LogisticRegression
from .evaluation import calculate_classification_metrics


def k_fold_split(X: np.ndarray, y: np.ndarray, k: int = 5,
                 random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split data into K folds for cross-validation.

    Args:
        X: Feature matrix
        y: Target vector
        k: Number of folds
        random_state: Random seed for reproducibility

    Returns:
        List of (X_train, X_val, y_train, y_val) tuples for each fold
    """
    np.random.seed(random_state)
    n_samples = len(y)

    # Shuffle indices
    indices = np.random.permutation(n_samples)

    # Calculate fold size
    fold_size = n_samples // k

    folds = []
    for i in range(k):
        # Validation indices for this fold
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n_samples
        val_indices = indices[val_start:val_end]

        # Training indices (all except validation)
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

        # Split data
        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_val_fold = X[val_indices]
        y_val_fold = y[val_indices]

        folds.append((X_train_fold, X_val_fold, y_train_fold, y_val_fold))

    return folds


def cross_validate(X: np.ndarray, y: np.ndarray,
                   model_params: Dict = None,
                   k: int = 5,
                   random_state: int = 42,
                   verbose: bool = True) -> Dict:
    """
    Perform K-Fold Cross-Validation.

    This provides a more robust estimate of model performance than a single
    train/test split by training and evaluating on K different splits.

    Args:
        X: Feature matrix (should be pre-normalized)
        y: Target vector
        model_params: Dictionary of model hyperparameters
        k: Number of folds (default: 5)
        random_state: Random seed
        verbose: Whether to print progress

    Returns:
        Dictionary containing:
            - fold_scores: List of metrics for each fold
            - mean_scores: Average metrics across all folds
            - std_scores: Standard deviation of metrics
            - models: List of trained models (one per fold)
    """
    if model_params is None:
        model_params = {
            'learning_rate': 0.03,
            'n_iterations': 5000,
            'tolerance': 1e-6,
            'batch_size': None,
            'regularization': None,
            'lambda_reg': 0.0
        }

    if verbose:
        print("\n" + "="*70)
        print(f"K-FOLD CROSS-VALIDATION (k={k})")
        print("="*70)
        print(f"Model parameters: {model_params}")
        print(f"Dataset size: {len(y)} samples")
        print(f"Fold size: ~{len(y)//k} samples per fold\n")

    # Create K folds
    folds = k_fold_split(X, y, k=k, random_state=random_state)

    # Store results
    fold_scores = []
    models = []

    # Train and evaluate on each fold
    for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(folds):
        if verbose:
            print(f"Fold {fold_idx + 1}/{k}:")
            print(f"  Training samples: {len(y_train)}")
            print(f"  Validation samples: {len(y_val)}")

        # Create and train model
        model = LogisticRegression(**model_params)
        model.fit(X_train, y_train, verbose=False)

        # Predict on validation set
        y_val_pred = model.predict(X_val)

        # Calculate metrics
        metrics = calculate_classification_metrics(y_val, y_val_pred)

        if verbose:
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}\n")

        fold_scores.append(metrics)
        models.append(model)

    # Calculate mean and std across folds
    mean_scores = {
        'accuracy': np.mean([s['accuracy'] for s in fold_scores]),
        'precision': np.mean([s['precision'] for s in fold_scores]),
        'recall': np.mean([s['recall'] for s in fold_scores]),
        'f1_score': np.mean([s['f1_score'] for s in fold_scores])
    }

    std_scores = {
        'accuracy': np.std([s['accuracy'] for s in fold_scores]),
        'precision': np.std([s['precision'] for s in fold_scores]),
        'recall': np.std([s['recall'] for s in fold_scores]),
        'f1_score': np.std([s['f1_score'] for s in fold_scores])
    }

    if verbose:
        print("="*70)
        print("CROSS-VALIDATION RESULTS (Mean ± Std)")
        print("="*70)
        print(f"Accuracy:  {mean_scores['accuracy']:.4f} ± {std_scores['accuracy']:.4f}")
        print(f"Precision: {mean_scores['precision']:.4f} ± {std_scores['precision']:.4f}")
        print(f"Recall:    {mean_scores['recall']:.4f} ± {std_scores['recall']:.4f}")
        print(f"F1 Score:  {mean_scores['f1_score']:.4f} ± {std_scores['f1_score']:.4f}")
        print("="*70)

    return {
        'fold_scores': fold_scores,
        'mean_scores': mean_scores,
        'std_scores': std_scores,
        'models': models
    }


def stratified_k_fold_split(X: np.ndarray, y: np.ndarray, k: int = 5,
                            random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Split data into K folds with stratification (maintains class balance).

    Ensures each fold has approximately the same proportion of classes as the full dataset.

    Args:
        X: Feature matrix
        y: Target vector
        k: Number of folds
        random_state: Random seed

    Returns:
        List of (X_train, X_val, y_train, y_val) tuples for each fold
    """
    np.random.seed(random_state)

    # Get indices for each class
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]

    # Shuffle indices for each class
    np.random.shuffle(class_0_indices)
    np.random.shuffle(class_1_indices)

    # Calculate fold sizes for each class
    fold_size_0 = len(class_0_indices) // k
    fold_size_1 = len(class_1_indices) // k

    folds = []
    for i in range(k):
        # Validation indices for this fold (from each class)
        val_start_0 = i * fold_size_0
        val_end_0 = (i + 1) * fold_size_0 if i < k - 1 else len(class_0_indices)
        val_indices_0 = class_0_indices[val_start_0:val_end_0]

        val_start_1 = i * fold_size_1
        val_end_1 = (i + 1) * fold_size_1 if i < k - 1 else len(class_1_indices)
        val_indices_1 = class_1_indices[val_start_1:val_end_1]

        val_indices = np.concatenate([val_indices_0, val_indices_1])

        # Training indices (all except validation)
        train_indices_0 = np.concatenate([class_0_indices[:val_start_0], class_0_indices[val_end_0:]])
        train_indices_1 = np.concatenate([class_1_indices[:val_start_1], class_1_indices[val_end_1:]])
        train_indices = np.concatenate([train_indices_0, train_indices_1])

        # Split data
        X_train_fold = X[train_indices]
        y_train_fold = y[train_indices]
        X_val_fold = X[val_indices]
        y_val_fold = y[val_indices]

        folds.append((X_train_fold, X_val_fold, y_train_fold, y_val_fold))

    return folds


def grid_search_cv(X: np.ndarray, y: np.ndarray,
                   param_grid: Dict[str, List],
                   k: int = 5,
                   random_state: int = 42,
                   verbose: bool = True) -> Dict:
    """
    Perform Grid Search with K-Fold Cross-Validation to find best hyperparameters.

    Args:
        X: Feature matrix (should be pre-normalized)
        y: Target vector
        param_grid: Dictionary mapping parameter names to lists of values to try
        k: Number of folds
        random_state: Random seed
        verbose: Whether to print progress

    Returns:
        Dictionary containing:
            - best_params: Best hyperparameters found
            - best_score: Best F1 score achieved
            - results: List of all results
    """
    if verbose:
        print("\n" + "="*70)
        print("GRID SEARCH WITH K-FOLD CROSS-VALIDATION")
        print("="*70)

    # Generate all parameter combinations
    import itertools
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))

    if verbose:
        print(f"Testing {len(param_combinations)} parameter combinations")
        print(f"Parameters: {param_names}\n")

    results = []
    best_score = -1
    best_params = None

    for combo_idx, combo in enumerate(param_combinations):
        # Create parameter dictionary
        params = dict(zip(param_names, combo))

        if verbose:
            print(f"\n[{combo_idx + 1}/{len(param_combinations)}] Testing: {params}")

        # Run cross-validation with these parameters
        cv_results = cross_validate(X, y, model_params=params, k=k,
                                    random_state=random_state, verbose=False)

        mean_f1 = cv_results['mean_scores']['f1_score']
        std_f1 = cv_results['std_scores']['f1_score']

        if verbose:
            print(f"  F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")

        # Track best parameters
        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = params.copy()

        results.append({
            'params': params,
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'cv_results': cv_results
        })

    if verbose:
        print("\n" + "="*70)
        print("GRID SEARCH RESULTS")
        print("="*70)
        print(f"Best F1 Score: {best_score:.4f}")
        print(f"Best Parameters: {best_params}")
        print("="*70)

    return {
        'best_params': best_params,
        'best_score': best_score,
        'results': results
    }
