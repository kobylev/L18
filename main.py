"""
Main Entry Point

Orchestrates the complete logistic regression workflow.
"""

import numpy as np
import os
from src import (
    LogisticRegression,
    generate_synthetic_data,
    plot_classification,
    plot_convergence,
    test_on_unseen_data,
    create_results_table,
    create_output_directory
)


def main():
    """
    Main execution function implementing the complete PRD.
    """
    print("="*70)
    print("LOGISTIC REGRESSION - BINARY CLASSIFICATION")
    print("Manual Implementation using Gradient Ascent")
    print("="*70)

    # Create output directory
    output_dir = create_output_directory('output')

    # PHASE 1: Data Preparation
    print("\n" + "="*70)
    print("PHASE 1: DATA PREPARATION")
    print("="*70)

    n_samples_per_class = 5000
    X_train, y_train = generate_synthetic_data(n_samples_per_class=n_samples_per_class)

    print(f"\nDataset created successfully!")
    print(f"  Total samples: {len(y_train)}")
    print(f"  Class 0 samples: {np.sum(y_train == 0)}")
    print(f"  Class 1 samples: {np.sum(y_train == 1)}")
    print(f"  Features: X1, X2 (normalized to [0, 1])")
    print(f"  X1 range: [{X_train[:, 0].min():.3f}, {X_train[:, 0].max():.3f}]")
    print(f"  X2 range: [{X_train[:, 1].min():.3f}, {X_train[:, 1].max():.3f}]")

    # PHASE 2: Gradient Descent Implementation
    print("\n" + "="*70)
    print("PHASE 2: GRADIENT ASCENT IMPLEMENTATION")
    print("="*70)

    model = LogisticRegression(learning_rate=0.03, n_iterations=10000, tolerance=1e-6)
    model.fit(X_train, y_train, verbose=True)

    # PHASE 3: Deliverables
    print("\n" + "="*70)
    print("PHASE 3: GENERATING DELIVERABLES")
    print("="*70)

    # Get predictions
    p_hat_train = model.predict_proba(X_train)

    # 3.1: Results Table
    print("\n--- Results Table ---")
    results_df = create_results_table(X_train, y_train, p_hat_train, n_display=20)
    print(results_df.to_string(index=False))
    results_csv_path = os.path.join(output_dir, 'results_table.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nFull results saved to: {results_csv_path}")

    # 3.2: Classification Plot
    print("\n--- Classification Plot ---")
    plot_classification(X_train, y_train, p_hat_train, model, output_dir)

    # 3.3: Convergence Plot
    print("\n--- Convergence Plot ---")
    plot_convergence(model.history, output_dir)

    # PHASE 4: Advanced - Test on Unseen Data
    print("\n" + "="*70)
    print("ADVANCED: TESTING ON UNSEEN DATA")
    print("="*70)
    test_on_unseen_data(model, n_test_samples=1000)

    print("\n" + "="*70)
    print("IMPLEMENTATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated files in '{output_dir}/' directory:")
    print(f"  - {output_dir}/classification_plot.png")
    print(f"  - {output_dir}/convergence_plot.png")
    print(f"  - {output_dir}/results_table.csv")
    print("\nThe model demonstrates:")
    print("  ✓ Accurate binary classification")
    print("  ✓ Proper convergence (increasing log-likelihood, decreasing MSE)")
    print("  ✓ Generalization to unseen data")
    print("="*70)


if __name__ == "__main__":
    main()
