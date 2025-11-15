"""
Advanced Features Demo

Demonstrates K-Fold Cross-Validation and Regularization (L1/L2).
"""

import numpy as np
from src import (
    LogisticRegression,
    generate_synthetic_data,
    normalize_features,
    cross_validate,
    grid_search_cv
)


def demo_regularization():
    """Demonstrate L1, L2, and no regularization comparison."""
    print("\n" + "="*70)
    print("REGULARIZATION COMPARISON DEMO")
    print("="*70)

    # Generate and normalize data
    X, y = generate_synthetic_data(n_samples_per_class=1000)
    X_norm, norm_params = normalize_features(X, method='zscore')

    # Split into train/test
    train_size = int(0.8 * len(y))
    X_train, X_test = X_norm[:train_size], X_norm[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"\nDataset: {len(y_train)} train, {len(y_test)} test")

    regularizations = [
        (None, 0.0, "No Regularization"),
        ('l1', 0.01, "L1 (Lasso) lambda=0.01"),
        ('l2', 0.01, "L2 (Ridge) lambda=0.01"),
        ('l2', 0.1, "L2 (Ridge) lambda=0.1")
    ]

    results = []

    for reg_type, lambda_val, desc in regularizations:
        print(f"\n--- {desc} ---")

        model = LogisticRegression(
            learning_rate=0.03,
            n_iterations=2000,
            regularization=reg_type,
            lambda_reg=lambda_val
        )

        model.fit(X_train, y_train, verbose=False)

        # Evaluate
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        train_acc = np.mean(y_pred_train == y_train)
        test_acc = np.mean(y_pred_test == y_test)

        print(f"  Beta coefficients: {model.beta}")
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        print(f"  Overfitting Gap: {train_acc - test_acc:.4f}")

        results.append({
            'reg_type': reg_type,
            'lambda': lambda_val,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'beta': model.beta.copy()
        })

    # Summary
    print("\n" + "="*70)
    print("REGULARIZATION SUMMARY")
    print("="*70)
    print(f"{'Regularization':<25} {'Train Acc':>10} {'Test Acc':>10} {'Gap':>8}")
    print("-"*70)
    for r in results:
        reg_name = f"{r['reg_type'] or 'None'} (lambda={r['lambda']})" if r['reg_type'] else "None"
        gap = r['train_acc'] - r['test_acc']
        print(f"{reg_name:<25} {r['train_acc']:>10.4f} {r['test_acc']:>10.4f} {gap:>8.4f}")


def demo_k_fold_cv():
    """Demonstrate K-Fold Cross-Validation."""
    print("\n" + "="*70)
    print("K-FOLD CROSS-VALIDATION DEMO")
    print("="*70)

    # Generate and normalize data
    X, y = generate_synthetic_data(n_samples_per_class=1000)
    X_norm, norm_params = normalize_features(X, method='zscore')

    # Model parameters
    model_params = {
        'learning_rate': 0.03,
        'n_iterations': 2000,
        'tolerance': 1e-6,
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

    print("\nCross-validation provides more stable error estimates than")
    print("a single train/test split.")


def demo_grid_search():
    """Demonstrate Grid Search with Cross-Validation."""
    print("\n" + "="*70)
    print("GRID SEARCH CROSS-VALIDATION DEMO")
    print("="*70)

    # Generate and normalize data
    X, y = generate_synthetic_data(n_samples_per_class=500)  # Smaller for speed
    X_norm, norm_params = normalize_features(X, method='zscore')

    # Define parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.03, 0.1],
        'regularization': [None, 'l2'],
        'lambda_reg': [0.0, 0.01, 0.1],
        'n_iterations': [1000],
        'tolerance': [1e-6],
        'batch_size': [None]
    }

    print("Testing combinations of:")
    print("  - Learning rates: [0.01, 0.03, 0.1]")
    print("  - Regularization: [None, L2]")
    print("  - Lambda values: [0.0, 0.01, 0.1]")

    # Run grid search
    search_results = grid_search_cv(
        X_norm, y,
        param_grid=param_grid,
        k=3,  # 3-fold for speed
        random_state=42,
        verbose=True
    )

    print("\nBest hyperparameters found:")
    for param, value in search_results['best_params'].items():
        print(f"  {param}: {value}")


def main():
    """Run all demos."""
    print("="*70)
    print("ADVANCED FEATURES DEMONSTRATION")
    print("="*70)
    print("\nThis script demonstrates:")
    print("  1. Regularization (L1, L2) to prevent overfitting")
    print("  2. K-Fold Cross-Validation for robust evaluation")
    print("  3. Grid Search for hyperparameter tuning")
    print("="*70)

    # Demo 1: Regularization
    demo_regularization()

    # Demo 2: K-Fold Cross-Validation
    demo_k_fold_cv()

    # Demo 3: Grid Search (optional - takes longer)
    user_input = input("\nRun Grid Search demo? (slower) [y/N]: ")
    if user_input.lower() == 'y':
        demo_grid_search()
    else:
        print("\nSkipping Grid Search demo.")

    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  - Regularization prevents overfitting (reduces train/test gap)")
    print("  - K-Fold CV provides more reliable performance estimates")
    print("  - Grid Search automates hyperparameter tuning")
    print("="*70)


if __name__ == "__main__":
    main()
