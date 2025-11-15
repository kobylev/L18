"""
Early Stopping Demo

Demonstrates the early stopping mechanism to prevent overfitting and
improve training efficiency.
"""

import numpy as np
import matplotlib.pyplot as plt
from src import (
    LogisticRegression,
    generate_synthetic_data,
    normalize_features,
    create_output_directory
)


def demo_early_stopping_comparison():
    """Compare training with and without early stopping."""
    print("\n" + "="*70)
    print("EARLY STOPPING DEMONSTRATION")
    print("="*70)

    # Generate and normalize data
    X, y = generate_synthetic_data(n_samples_per_class=1000)
    X_norm, norm_params = normalize_features(X, method='zscore')

    print("\nTraining two models:")
    print("  1. WITHOUT early stopping (runs all iterations)")
    print("  2. WITH early stopping (stops when validation performance plateaus)")

    # Model 1: Without early stopping
    print("\n" + "-"*70)
    print("MODEL 1: Without Early Stopping")
    print("-"*70)

    model_no_es = LogisticRegression(
        learning_rate=0.03,
        n_iterations=2000,
        regularization='l2',
        lambda_reg=0.01,
        early_stopping=False
    )

    model_no_es.fit(X_norm, y, verbose=False)

    print(f"Training completed after {len(model_no_es.history['log_likelihood'])} iterations")
    print(f"Final training MSE: {model_no_es.history['mse'][-1]:.6f}")

    # Model 2: With early stopping
    print("\n" + "-"*70)
    print("MODEL 2: With Early Stopping (patience=10)")
    print("-"*70)

    model_with_es = LogisticRegression(
        learning_rate=0.03,
        n_iterations=2000,
        regularization='l2',
        lambda_reg=0.01,
        early_stopping=True,
        patience=10,
        validation_split=0.2
    )

    model_with_es.fit(X_norm, y, verbose=True)

    print(f"\nEarly stopping triggered at iteration: {model_with_es.stopped_epoch}")
    print(f"Iterations saved: {2000 - (model_with_es.stopped_epoch or 2000)}")

    # Compare results
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Metric':<30} {'No Early Stop':>20} {'With Early Stop':>20}")
    print("-"*70)

    iterations_no_es = len(model_no_es.history['log_likelihood'])
    iterations_with_es = len(model_with_es.history['log_likelihood'])

    print(f"{'Iterations completed':<30} {iterations_no_es:>20d} {iterations_with_es:>20d}")
    print(f"{'Final training MSE':<30} {model_no_es.history['mse'][-1]:>20.6f} {model_with_es.history['mse'][-1]:>20.6f}")

    if model_with_es.history['val_mse']:
        print(f"{'Final validation MSE':<30} {'N/A':>20} {model_with_es.history['val_mse'][-1]:>20.6f}")

    efficiency_gain = (1 - iterations_with_es / iterations_no_es) * 100
    print(f"{'Efficiency gain':<30} {'baseline':>20} {efficiency_gain:>19.1f}%")

    # Plot comparison
    plot_early_stopping_comparison(model_no_es, model_with_es)


def plot_early_stopping_comparison(model_no_es, model_with_es):
    """Create visualization comparing training with and without early stopping."""
    output_dir = create_output_directory('output')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training Loss Comparison
    ax1 = axes[0, 0]
    ax1.plot(model_no_es.history['log_likelihood'], 'b-', label='No Early Stop', alpha=0.7)
    ax1.plot(model_with_es.history['log_likelihood'], 'r-', label='With Early Stop', alpha=0.7)

    if model_with_es.stopped_epoch:
        ax1.axvline(x=model_with_es.stopped_epoch, color='red', linestyle='--',
                   alpha=0.5, label=f'Early Stop (iter {model_with_es.stopped_epoch})')

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Log-Likelihood (Training)')
    ax1.set_title('Training Log-Likelihood Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training MSE Comparison
    ax2 = axes[0, 1]
    ax2.plot(model_no_es.history['mse'], 'b-', label='No Early Stop', alpha=0.7)
    ax2.plot(model_with_es.history['mse'], 'r-', label='With Early Stop', alpha=0.7)

    if model_with_es.stopped_epoch:
        ax2.axvline(x=model_with_es.stopped_epoch, color='red', linestyle='--',
                   alpha=0.5, label=f'Early Stop (iter {model_with_es.stopped_epoch})')

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('MSE (Training)')
    ax2.set_title('Training MSE Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Validation Metrics (Early Stopping Model)
    ax3 = axes[1, 0]
    if model_with_es.history['val_log_likelihood']:
        ax3.plot(model_with_es.history['log_likelihood'], 'b-',
                label='Training LL', alpha=0.7)
        ax3.plot(model_with_es.history['val_log_likelihood'], 'r-',
                label='Validation LL', alpha=0.7)

        if model_with_es.stopped_epoch:
            ax3.axvline(x=model_with_es.stopped_epoch, color='red', linestyle='--',
                       alpha=0.5, label=f'Early Stop')

        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Log-Likelihood')
        ax3.set_title('Early Stopping: Train vs Validation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Early stopping not enabled',
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Validation Metrics')

    # Plot 4: Validation MSE (Early Stopping Model)
    ax4 = axes[1, 1]
    if model_with_es.history['val_mse']:
        ax4.plot(model_with_es.history['mse'], 'b-',
                label='Training MSE', alpha=0.7)
        ax4.plot(model_with_es.history['val_mse'], 'r-',
                label='Validation MSE', alpha=0.7)

        if model_with_es.stopped_epoch:
            ax4.axvline(x=model_with_es.stopped_epoch, color='red', linestyle='--',
                       alpha=0.5, label=f'Early Stop')

        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('MSE')
        ax4.set_title('Early Stopping: MSE Train vs Validation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Early stopping not enabled',
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Validation MSE')

    plt.tight_layout()
    save_path = f'{output_dir}/early_stopping_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.close()


def demo_patience_comparison():
    """Compare different patience values."""
    print("\n" + "="*70)
    print("PATIENCE PARAMETER COMPARISON")
    print("="*70)

    # Generate data
    X, y = generate_synthetic_data(n_samples_per_class=1000)
    X_norm, norm_params = normalize_features(X, method='zscore')

    patience_values = [5, 10, 20]
    results = []

    for patience in patience_values:
        print(f"\nTraining with patience={patience}...")

        model = LogisticRegression(
            learning_rate=0.03,
            n_iterations=2000,
            regularization='l2',
            lambda_reg=0.01,
            early_stopping=True,
            patience=patience,
            validation_split=0.2
        )

        model.fit(X_norm, y, verbose=False)

        stopped_at = model.stopped_epoch if model.stopped_epoch else 2000

        results.append({
            'patience': patience,
            'stopped_at': stopped_at,
            'final_train_mse': model.history['mse'][-1],
            'final_val_mse': model.history['val_mse'][-1] if model.history['val_mse'] else None
        })

        print(f"  Stopped at iteration: {stopped_at}")
        print(f"  Final validation MSE: {model.history['val_mse'][-1] if model.history['val_mse'] else 'N/A'}")

    # Summary
    print("\n" + "="*70)
    print("PATIENCE COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Patience':>10} {'Stopped At':>15} {'Train MSE':>15} {'Val MSE':>15}")
    print("-"*70)
    for r in results:
        val_mse_str = f"{r['final_val_mse']:.6f}" if r['final_val_mse'] else "N/A"
        print(f"{r['patience']:>10} {r['stopped_at']:>15} {r['final_train_mse']:>15.6f} {val_mse_str:>15}")

    print("\nInterpretation:")
    print("  - Lower patience: Stops earlier, may underfit")
    print("  - Higher patience: More tolerant of plateaus, less aggressive stopping")
    print("  - Optimal patience depends on problem complexity and noise level")


def main():
    """Run all early stopping demonstrations."""
    print("="*70)
    print("EARLY STOPPING FEATURE DEMONSTRATION")
    print("="*70)
    print("\nEarly stopping prevents overfitting by monitoring validation performance")
    print("and stopping training when it stops improving.")
    print("\nBenefits:")
    print("  - Prevents overfitting (better generalization)")
    print("  - Saves computation time (fewer iterations)")
    print("  - Automatically finds optimal stopping point")
    print("  - Restores best parameters found during training")
    print("="*70)

    # Demo 1: Basic early stopping comparison
    demo_early_stopping_comparison()

    # Demo 2: Patience comparison (optional)
    try:
        user_input = input("\nRun patience comparison demo? [y/N]: ")
        if user_input.lower() == 'y':
            demo_patience_comparison()
        else:
            print("\nSkipping patience comparison demo.")
    except EOFError:
        print("\nSkipping patience comparison demo (no input available).")

    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  - Early stopping monitors validation loss to prevent overfitting")
    print("  - Patience controls how tolerant the stopping criterion is")
    print("  - Best parameters are automatically restored when stopping")
    print("  - Validation split should be 10-20% of training data")
    print("="*70)


if __name__ == "__main__":
    main()
