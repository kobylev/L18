"""
Learning Rate Scheduling Demo

Demonstrates different learning rate decay strategies for improved
convergence and training stability.
"""

import numpy as np
import matplotlib.pyplot as plt
from src import (
    LogisticRegression,
    generate_synthetic_data,
    normalize_features,
    create_output_directory
)


def demo_lr_schedules_comparison():
    """Compare different learning rate schedules."""
    print("\n" + "="*70)
    print("LEARNING RATE SCHEDULING DEMONSTRATION")
    print("="*70)

    # Generate and normalize data
    X, y = generate_synthetic_data(n_samples_per_class=1000)
    X_norm, norm_params = normalize_features(X, method='zscore')

    print("\nComparing 5 learning rate strategies:")
    print("  1. No schedule (constant)")
    print("  2. Step decay (reduce every N steps)")
    print("  3. Exponential decay (smooth exponential reduction)")
    print("  4. Inverse time decay (1/t style)")
    print("  5. Adaptive (increase if improving, decrease if not)")

    schedules = [
        (None, "No Schedule (Constant)"),
        ('step', "Step Decay"),
        ('exponential', "Exponential Decay"),
        ('inverse', "Inverse Time Decay"),
        ('adaptive', "Adaptive LR")
    ]

    models = []
    results = []

    for schedule, name in schedules:
        print(f"\n{'-'*70}")
        print(f"Training with: {name}")
        print(f"{'-'*70}")

        model = LogisticRegression(
            learning_rate=0.1,          # Higher initial LR to see decay effect
            n_iterations=1000,
            regularization='l2',
            lambda_reg=0.01,
            lr_schedule=schedule,
            lr_decay_rate=0.95,
            lr_decay_steps=100,
            early_stopping=False        # Disable to see full schedule
        )

        model.fit(X_norm, y, verbose=False)

        # Get final metrics
        final_mse = model.history['mse'][-1]
        final_ll = model.history['log_likelihood'][-1]
        iterations = len(model.history['mse'])

        print(f"  Iterations: {iterations}")
        print(f"  Final MSE: {final_mse:.6f}")
        print(f"  Final Log-Likelihood: {final_ll:.4f}")
        print(f"  Initial LR: {model.initial_learning_rate:.6f}")
        print(f"  Final LR: {model.history['learning_rate'][-1]:.6f}")

        models.append(model)
        results.append({
            'schedule': name,
            'final_mse': final_mse,
            'final_ll': final_ll,
            'iterations': iterations,
            'initial_lr': model.initial_learning_rate,
            'final_lr': model.history['learning_rate'][-1]
        })

    # Summary
    print("\n" + "="*70)
    print("LEARNING RATE SCHEDULE COMPARISON")
    print("="*70)
    print(f"{'Schedule':<25} {'Iterations':>12} {'Final MSE':>12} {'Final LR':>12}")
    print("-"*70)
    for r in results:
        print(f"{r['schedule']:<25} {r['iterations']:>12} {r['final_mse']:>12.6f} {r['final_lr']:>12.6f}")

    # Plot comparison
    plot_lr_schedules(models, schedules)

    return models, results


def plot_lr_schedules(models, schedules):
    """Visualize learning rate schedules and their effects."""
    output_dir = create_output_directory('output')

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    # Plot 1: Learning Rate Over Time
    ax1 = axes[0]
    for idx, (model, (schedule, name)) in enumerate(zip(models, schedules)):
        lr_history = model.history['learning_rate']
        ax1.plot(lr_history, label=name, color=colors[idx], alpha=0.7)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Learning Rate')
    ax1.set_title('Learning Rate Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Log-Likelihood Progress
    ax2 = axes[1]
    for idx, (model, (schedule, name)) in enumerate(zip(models, schedules)):
        ll_history = model.history['log_likelihood']
        ax2.plot(ll_history, label=name, color=colors[idx], alpha=0.7)

    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Log-Likelihood')
    ax2.set_title('Log-Likelihood Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: MSE Progress
    ax3 = axes[2]
    for idx, (model, (schedule, name)) in enumerate(zip(models, schedules)):
        mse_history = model.history['mse']
        ax3.plot(mse_history, label=name, color=colors[idx], alpha=0.7)

    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('MSE')
    ax3.set_title('MSE Progress')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: LR vs MSE Scatter
    ax4 = axes[3]
    for idx, (model, (schedule, name)) in enumerate(zip(models, schedules)):
        final_lr = model.history['learning_rate'][-1]
        final_mse = model.history['mse'][-1]
        ax4.scatter(final_lr, final_mse, label=name, color=colors[idx], s=100)

    ax4.set_xlabel('Final Learning Rate')
    ax4.set_ylabel('Final MSE')
    ax4.set_title('Final LR vs Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')

    # Plot 5: Individual Schedule Detail - Step
    ax5 = axes[4]
    step_model = models[1]  # Step decay model
    ax5.plot(step_model.history['learning_rate'], 'b-', label='Learning Rate')
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Learning Rate', color='b')
    ax5.tick_params(axis='y', labelcolor='b')
    ax5.set_title('Step Decay Detail')
    ax5.grid(True, alpha=0.3)

    ax5_twin = ax5.twinx()
    ax5_twin.plot(step_model.history['mse'], 'r-', label='MSE', alpha=0.7)
    ax5_twin.set_ylabel('MSE', color='r')
    ax5_twin.tick_params(axis='y', labelcolor='r')

    # Plot 6: Individual Schedule Detail - Adaptive
    ax6 = axes[5]
    adaptive_model = models[4]  # Adaptive model
    ax6.plot(adaptive_model.history['learning_rate'], 'b-', label='Learning Rate')
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Learning Rate', color='b')
    ax6.tick_params(axis='y', labelcolor='b')
    ax6.set_title('Adaptive LR Detail')
    ax6.grid(True, alpha=0.3)

    ax6_twin = ax6.twinx()
    ax6_twin.plot(adaptive_model.history['mse'], 'r-', label='MSE', alpha=0.7)
    ax6_twin.set_ylabel('MSE', color='r')
    ax6_twin.tick_params(axis='y', labelcolor='r')

    plt.tight_layout()
    save_path = f'{output_dir}/lr_scheduling_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.close()


def demo_optimal_decay_rate():
    """Find optimal decay rate for step schedule."""
    print("\n" + "="*70)
    print("OPTIMAL DECAY RATE SEARCH")
    print("="*70)

    X, y = generate_synthetic_data(n_samples_per_class=1000)
    X_norm, norm_params = normalize_features(X, method='zscore')

    decay_rates = [0.90, 0.93, 0.95, 0.97, 0.99]
    results = []

    print("\nTesting decay rates: [0.90, 0.93, 0.95, 0.97, 0.99]")

    for decay_rate in decay_rates:
        model = LogisticRegression(
            learning_rate=0.1,
            n_iterations=1000,
            lr_schedule='step',
            lr_decay_rate=decay_rate,
            lr_decay_steps=100,
            regularization='l2',
            lambda_reg=0.01
        )

        model.fit(X_norm, y, verbose=False)

        final_mse = model.history['mse'][-1]
        final_ll = model.history['log_likelihood'][-1]

        results.append({
            'decay_rate': decay_rate,
            'final_mse': final_mse,
            'final_ll': final_ll,
            'final_lr': model.history['learning_rate'][-1]
        })

        print(f"  Decay rate {decay_rate:.2f}: MSE={final_mse:.6f}, LL={final_ll:.4f}, Final LR={model.history['learning_rate'][-1]:.6f}")

    # Find best
    best = min(results, key=lambda x: x['final_mse'])
    print(f"\nBest decay rate: {best['decay_rate']:.2f} with MSE={best['final_mse']:.6f}")


def demo_adaptive_vs_constant():
    """Compare adaptive LR with constant LR on challenging problem."""
    print("\n" + "="*70)
    print("ADAPTIVE LR vs CONSTANT LR")
    print("="*70)

    X, y = generate_synthetic_data(n_samples_per_class=1000)
    X_norm, norm_params = normalize_features(X, method='zscore')

    print("\nTraining two models on same data:")
    print("  1. Constant LR (traditional)")
    print("  2. Adaptive LR (adjusts based on performance)")

    # Constant LR
    print(f"\n{'-'*70}")
    print("Model 1: Constant Learning Rate")
    print(f"{'-'*70}")

    model_constant = LogisticRegression(
        learning_rate=0.03,
        n_iterations=1000,
        regularization='l2',
        lambda_reg=0.01,
        early_stopping=True,
        patience=15,
        validation_split=0.2,
        lr_schedule=None
    )

    model_constant.fit(X_norm, y, verbose=False)

    print(f"  Iterations: {len(model_constant.history['mse'])}")
    print(f"  Final MSE: {model_constant.history['mse'][-1]:.6f}")
    print(f"  Learning Rate: {model_constant.learning_rate:.6f} (constant)")
    if model_constant.stopped_epoch:
        print(f"  Early stopped at: {model_constant.stopped_epoch}")

    # Adaptive LR
    print(f"\n{'-'*70}")
    print("Model 2: Adaptive Learning Rate")
    print(f"{'-'*70}")

    model_adaptive = LogisticRegression(
        learning_rate=0.03,
        n_iterations=1000,
        regularization='l2',
        lambda_reg=0.01,
        early_stopping=True,
        patience=15,
        validation_split=0.2,
        lr_schedule='adaptive',
        lr_decay_rate=0.9
    )

    model_adaptive.fit(X_norm, y, verbose=False)

    print(f"  Iterations: {len(model_adaptive.history['mse'])}")
    print(f"  Final MSE: {model_adaptive.history['mse'][-1]:.6f}")
    print(f"  Initial LR: {model_adaptive.initial_learning_rate:.6f}")
    print(f"  Final LR: {model_adaptive.history['learning_rate'][-1]:.6f}")
    if model_adaptive.stopped_epoch:
        print(f"  Early stopped at: {model_adaptive.stopped_epoch}")

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    mse_improvement = (model_constant.history['mse'][-1] - model_adaptive.history['mse'][-1]) / model_constant.history['mse'][-1] * 100
    iter_diff = len(model_constant.history['mse']) - len(model_adaptive.history['mse'])

    print(f"MSE Improvement: {mse_improvement:+.2f}%")
    print(f"Iteration Difference: {iter_diff:+d}")


def main():
    """Run all learning rate scheduling demonstrations."""
    print("="*70)
    print("LEARNING RATE SCHEDULING FEATURE DEMONSTRATION")
    print("="*70)
    print("\nLearning rate scheduling adapts the learning rate during training")
    print("to improve convergence speed and final performance.")
    print("\nStrategies:")
    print("  - Step decay: Reduce LR every N steps")
    print("  - Exponential decay: Smooth exponential reduction")
    print("  - Inverse time decay: 1/(1+t) style decay")
    print("  - Adaptive: Increase if improving, decrease if stagnating")
    print("="*70)

    # Demo 1: Compare all schedules
    models, results = demo_lr_schedules_comparison()

    # Demo 2: Optimal decay rate (optional)
    try:
        user_input = input("\nRun optimal decay rate search? [y/N]: ")
        if user_input.lower() == 'y':
            demo_optimal_decay_rate()
        else:
            print("\nSkipping decay rate search.")
    except EOFError:
        print("\nSkipping decay rate search (no input available).")

    # Demo 3: Adaptive vs Constant
    try:
        user_input = input("\nRun adaptive vs constant comparison? [y/N]: ")
        if user_input.lower() == 'y':
            demo_adaptive_vs_constant()
        else:
            print("\nSkipping adaptive comparison.")
    except EOFError:
        print("\nSkipping adaptive comparison (no input available).")

    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("  - Step decay: Simple, works well for most cases")
    print("  - Exponential decay: Smooth, aggressive reduction")
    print("  - Inverse time decay: Gentler, good for long training")
    print("  - Adaptive: Smart, adjusts to actual performance")
    print("\nBest Practice:")
    print("  - Start with step decay (decay_rate=0.95, steps=100)")
    print("  - Try adaptive for complex problems")
    print("  - Combine with early stopping for best results")
    print("="*70)


if __name__ == "__main__":
    main()
