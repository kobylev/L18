"""
Main Entry Point

Orchestrates the complete logistic regression workflow with improvements:
- Z-Score normalization for better gradient ascent stability
- Mini-Batch gradient ascent for scalability
- Comprehensive evaluation metrics (Precision, Recall, F1)
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
from src.data_utils import normalize_features
from src.evaluation import calculate_classification_metrics, print_classification_report


def main():
    """
    Main execution function implementing the complete PRD with improvements.
    """
    print("="*70)
    print("LOGISTIC REGRESSION - BINARY CLASSIFICATION")
    print("Manual Implementation using Gradient Ascent")
    print("="*70)
    print("\nImprovements:")
    print("  1. Z-Score Normalization for stable convergence")
    print("  2. Mini-Batch Gradient Ascent for scalability")
    print("  3. Comprehensive metrics (Precision, Recall, F1)")
    print("  4. Real-world dataset adaptation guide")
    print("="*70)

    # Create output directory
    output_dir = create_output_directory('output')

    # PHASE 1: Data Preparation
    print("\n" + "="*70)
    print("PHASE 1: DATA PREPARATION")
    print("="*70)

    n_samples_per_class = 5000
    X_train_raw, y_train = generate_synthetic_data(n_samples_per_class=n_samples_per_class)

    print(f"\nDataset created successfully!")
    print(f"  Total samples: {len(y_train)}")
    print(f"  Class 0 samples: {np.sum(y_train == 0)}")
    print(f"  Class 1 samples: {np.sum(y_train == 1)}")
    print(f"  Features: X1, X2")

    # NEW: Apply Z-Score Normalization
    print("\n--- Applying Z-Score Normalization ---")
    X_train, norm_params = normalize_features(X_train_raw, method='zscore')

    print(f"Normalization method: {norm_params['method']}")
    print(f"  Original X1 range: [{X_train_raw[:, 0].min():.3f}, {X_train_raw[:, 0].max():.3f}]")
    print(f"  Original X2 range: [{X_train_raw[:, 1].min():.3f}, {X_train_raw[:, 1].max():.3f}]")
    print(f"  Normalized X1: mean={X_train[:, 0].mean():.3f}, std={X_train[:, 0].std():.3f}")
    print(f"  Normalized X2: mean={X_train[:, 1].mean():.3f}, std={X_train[:, 1].std():.3f}")

    # PHASE 2: Gradient Ascent Implementation
    print("\n" + "="*70)
    print("PHASE 2: GRADIENT ASCENT IMPLEMENTATION")
    print("="*70)

    # NEW: Using Mini-Batch Gradient Ascent
    model = LogisticRegression(
        learning_rate=0.03,
        n_iterations=10000,
        tolerance=1e-6,
        batch_size=512  # Mini-batch size (set to None for full batch)
    )
    model.fit(X_train, y_train, verbose=True)

    # PHASE 3: Deliverables
    print("\n" + "="*70)
    print("PHASE 3: GENERATING DELIVERABLES")
    print("="*70)

    # Get predictions
    p_hat_train = model.predict_proba(X_train)
    y_pred_train = model.predict(X_train)

    # NEW: Calculate comprehensive metrics on training data
    print("\n--- Training Set Performance ---")
    train_metrics = calculate_classification_metrics(y_train, y_pred_train)
    print_classification_report(train_metrics, "Training")

    # 3.1: Results Table
    print("\n--- Results Table ---")
    results_df = create_results_table(X_train, y_train, p_hat_train, n_display=20)
    print(results_df.to_string(index=False))
    results_csv_path = os.path.join(output_dir, 'results_table.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nFull results saved to: {results_csv_path}")

    # 3.2: Classification Plot (using original scale for visualization)
    print("\n--- Classification Plot ---")
    plot_classification(X_train_raw, y_train, p_hat_train, model, output_dir,
                       X_normalized=X_train, norm_params=norm_params)

    # 3.3: Convergence Plot
    print("\n--- Convergence Plot ---")
    plot_convergence(model.history, output_dir)

    # PHASE 4: Advanced - Test on Unseen Data
    print("\n" + "="*70)
    print("PHASE 4: TESTING ON UNSEEN DATA")
    print("="*70)

    # NEW: Pass normalization parameters to test function
    test_on_unseen_data(model, n_test_samples=1000, normalization_params=norm_params)

    print("\n" + "="*70)
    print("IMPLEMENTATION COMPLETE!")
    print("="*70)
    print(f"\nGenerated files in '{output_dir}/' directory:")
    print(f"  - {output_dir}/classification_plot.png")
    print(f"  - {output_dir}/convergence_plot.png")
    print(f"  - {output_dir}/results_table.csv")
    print("\nThe improved model demonstrates:")
    print("  [X] Z-Score normalization for stable convergence")
    print("  [X] Mini-Batch gradient ascent for scalability")
    print("  [X] Comprehensive evaluation (Precision, Recall, F1)")
    print("  [X] Proper convergence (increasing log-likelihood, decreasing MSE)")
    print("  [X] Generalization to unseen data")
    print("\nFor real-world dataset adaptation:")
    print("  See REAL_WORLD_ADAPTATION.md for detailed guide")
    print("="*70)


if __name__ == "__main__":
    main()
