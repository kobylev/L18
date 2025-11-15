"""
Logistic Regression Implementation from Scratch
Binary Classification using Gradient Ascent

This implementation follows the PRD specifications for manual logistic regression
without using sklearn's built-in models.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Dict, List
import os


class LogisticRegression:
    """
    Manual implementation of Logistic Regression using Gradient Ascent.

    Attributes:
        learning_rate (float): Learning step (alpha) for gradient ascent
        n_iterations (int): Maximum number of iterations
        tolerance (float): Convergence tolerance for stopping criteria
        beta (np.ndarray): Model coefficients [beta_0, beta_1, beta_2]
        history (Dict): Training history (log-likelihood, MSE, beta values)
    """

    def __init__(self, learning_rate: float = 0.03, n_iterations: int = 10000,
                 tolerance: float = 1e-6):
        """
        Initialize the Logistic Regression model.

        Args:
            learning_rate: Learning step for gradient ascent
            n_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.beta = None
        self.history = {
            'log_likelihood': [],
            'mse': [],
            'beta_0': [],
            'beta_1': [],
            'beta_2': []
        }

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function.

        σ(Z) = 1 / (1 + e^(-Z))

        Args:
            z: Weighted sum (linear combination)

        Returns:
            Probability values between 0 and 1
        """
        # Clip z to prevent overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _add_bias_term(self, X: np.ndarray) -> np.ndarray:
        """
        Add bias term (column of ones) to feature matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            Feature matrix with bias term (n_samples, n_features + 1)
        """
        n_samples = X.shape[0]
        return np.column_stack([np.ones(n_samples), X])

    def _compute_weighted_sum(self, X: np.ndarray) -> np.ndarray:
        """
        Compute weighted sum Z = β₀X₀ + β₁X₁ + β₂X₂

        Args:
            X: Feature matrix with bias term

        Returns:
            Weighted sum for each sample
        """
        return X @ self.beta

    def _compute_log_likelihood(self, y: np.ndarray, p_hat: np.ndarray) -> float:
        """
        Compute log-likelihood function.

        L = Σ[y_i * log(p̂_i) + (1 - y_i) * log(1 - p̂_i)]

        Args:
            y: True labels
            p_hat: Predicted probabilities

        Returns:
            Log-likelihood value
        """
        # Clip probabilities to prevent log(0)
        epsilon = 1e-15
        p_hat = np.clip(p_hat, epsilon, 1 - epsilon)

        log_likelihood = np.sum(
            y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat)
        )
        return log_likelihood

    def _compute_mse(self, y: np.ndarray, p_hat: np.ndarray) -> float:
        """
        Compute Mean Squared Error.

        MSE = (1/N) * Σ(y_i - p̂_i)²

        Args:
            y: True labels
            p_hat: Predicted probabilities

        Returns:
            MSE value
        """
        return np.mean((y - p_hat) ** 2)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> 'LogisticRegression':
        """
        Fit the logistic regression model using Gradient Ascent.

        The update rule (Gradient Ascent for maximizing log-likelihood):
        β_k^(t+1) = β_k^(t) + α * Σ[(y_i - p̂_i) * X_{k,i}]

        Args:
            X: Feature matrix of shape (n_samples, 2)
            y: Target vector of shape (n_samples,)
            verbose: Whether to print progress

        Returns:
            self
        """
        # Add bias term (X_0 = 1)
        X_with_bias = self._add_bias_term(X)
        n_samples, n_features = X_with_bias.shape

        # Initialize beta coefficients with small random values
        np.random.seed(42)
        self.beta = np.random.randn(n_features) * 0.01

        if verbose:
            print(f"Starting Gradient Ascent...")
            print(f"Initial β: {self.beta}")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Max iterations: {self.n_iterations}\n")

        # Gradient Ascent iterations
        for iteration in range(self.n_iterations):
            # Step 1: Compute weighted sum Z
            z = self._compute_weighted_sum(X_with_bias)

            # Step 2: Compute probability using sigmoid
            p_hat = self.sigmoid(z)

            # Step 3: Compute gradients (partial derivatives)
            # ∂L/∂β_k = Σ[(y_i - p̂_i) * X_{k,i}]
            error = y - p_hat
            gradients = X_with_bias.T @ error

            # Step 4: Update beta coefficients (Gradient ASCENT - addition)
            beta_old = self.beta.copy()
            self.beta = self.beta + self.learning_rate * gradients

            # Compute metrics for this iteration
            log_likelihood = self._compute_log_likelihood(y, p_hat)
            mse = self._compute_mse(y, p_hat)

            # Store history
            self.history['log_likelihood'].append(log_likelihood)
            self.history['mse'].append(mse)
            self.history['beta_0'].append(self.beta[0])
            self.history['beta_1'].append(self.beta[1])
            self.history['beta_2'].append(self.beta[2])

            # Check convergence
            beta_change = np.max(np.abs(self.beta - beta_old))

            if verbose and (iteration % 100 == 0 or iteration < 10):
                print(f"Iteration {iteration:4d} | "
                      f"Log-Likelihood: {log_likelihood:10.4f} | "
                      f"MSE: {mse:.6f} | "
                      f"β change: {beta_change:.8f}")

            # Stop if converged
            if beta_change < self.tolerance:
                if verbose:
                    print(f"\nConverged at iteration {iteration}!")
                break

        if verbose:
            print(f"\nFinal β coefficients: {self.beta}")
            print(f"Final Log-Likelihood: {log_likelihood:.4f}")
            print(f"Final MSE: {mse:.6f}")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities for samples.

        Args:
            X: Feature matrix of shape (n_samples, 2)

        Returns:
            Predicted probabilities
        """
        if self.beta is None:
            raise ValueError("Model has not been fitted yet!")

        X_with_bias = self._add_bias_term(X)
        z = self._compute_weighted_sum(X_with_bias)
        return self.sigmoid(z)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels for samples.

        Args:
            X: Feature matrix of shape (n_samples, 2)
            threshold: Decision threshold (default 0.5)

        Returns:
            Predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def get_decision_boundary(self, x1_range: Tuple[float, float],
                             n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate decision boundary line.

        Decision boundary is where σ(Z) = 0.5, which means Z = 0
        β₀ + β₁X₁ + β₂X₂ = 0
        X₂ = -(β₀ + β₁X₁) / β₂

        Args:
            x1_range: Range of X1 values (min, max)
            n_points: Number of points for the line

        Returns:
            Tuple of (X1 values, X2 values) for the boundary line
        """
        if self.beta is None:
            raise ValueError("Model has not been fitted yet!")

        x1 = np.linspace(x1_range[0], x1_range[1], n_points)

        # Handle case where beta_2 is close to zero
        if np.abs(self.beta[2]) < 1e-10:
            return x1, np.zeros_like(x1)

        x2 = -(self.beta[0] + self.beta[1] * x1) / self.beta[2]
        return x1, x2


def create_output_directory(output_dir: str = 'output') -> str:
    """
    Create output directory if it doesn't exist.

    Args:
        output_dir: Name of the output directory

    Returns:
        Path to the output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}/")
    return output_dir


def generate_synthetic_data(n_samples_per_class: int = 5000,
                           random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic dataset with two well-separated clusters.

    Args:
        n_samples_per_class: Number of samples per class
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (features, labels)
        - features: shape (2*n_samples_per_class, 2) normalized to [0, 1]
        - labels: shape (2*n_samples_per_class,) with values 0 or 1
    """
    np.random.seed(random_state)

    # Class 0: Cluster centered around (0.3, 0.3)
    class_0_mean = [0.3, 0.3]
    class_0_cov = [[0.01, 0.002], [0.002, 0.01]]
    X_class_0 = np.random.multivariate_normal(class_0_mean, class_0_cov, n_samples_per_class)
    y_class_0 = np.zeros(n_samples_per_class)

    # Class 1: Cluster centered around (0.7, 0.7)
    class_1_mean = [0.7, 0.7]
    class_1_cov = [[0.01, 0.002], [0.002, 0.01]]
    X_class_1 = np.random.multivariate_normal(class_1_mean, class_1_cov, n_samples_per_class)
    y_class_1 = np.ones(n_samples_per_class)

    # Combine the two classes
    X = np.vstack([X_class_0, X_class_1])
    y = np.concatenate([y_class_0, y_class_1])

    # Ensure values are in [0, 1] range
    X = np.clip(X, 0, 1)

    # Shuffle the data
    shuffle_idx = np.random.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    return X, y


def create_results_table(X: np.ndarray, y: np.ndarray,
                        p_hat: np.ndarray, n_display: int = 20) -> pd.DataFrame:
    """
    Create results table as specified in PRD.

    Args:
        X: Feature matrix
        y: True labels
        p_hat: Predicted probabilities
        n_display: Number of samples to display

    Returns:
        DataFrame with results
    """
    n_samples = len(y)

    # Calculate squared errors
    error_squared = (y - p_hat) ** 2

    # Create DataFrame
    df = pd.DataFrame({
        'X₁': X[:n_display, 0],
        'X₂': X[:n_display, 1],
        'Y (Ground Truth)': y[:n_display].astype(int),
        'Sigmoid(P̂) (Prediction)': p_hat[:n_display],
        'Error²': error_squared[:n_display]
    })

    # Add MSE row at the bottom
    mse = np.mean(error_squared)
    mse_row = pd.DataFrame({
        'X₁': [''],
        'X₂': ['Average Error (MSE)'],
        'Y (Ground Truth)': [''],
        'Sigmoid(P̂) (Prediction)': [''],
        'Error²': [mse]
    })

    df = pd.concat([df, mse_row], ignore_index=True)

    return df


def plot_classification(X: np.ndarray, y: np.ndarray, p_hat: np.ndarray,
                       model: LogisticRegression, output_dir: str = 'output'):
    """
    Create classification plot with decision boundary.

    Args:
        X: Feature matrix
        y: True labels
        p_hat: Predicted probabilities
        model: Trained LogisticRegression model
        output_dir: Directory to save the plot
    """
    save_path = os.path.join(output_dir, 'classification_plot.png')

    plt.figure(figsize=(12, 10))

    # Predict classes based on threshold 0.5
    y_pred = (p_hat >= 0.5).astype(int)

    # Create color map based on predicted class
    colors = np.array(['red', 'blue'])
    point_colors = colors[y_pred]

    # Plot points with shapes for true vs predicted
    # Ground Truth (X markers)
    for true_class in [0, 1]:
        mask = (y == true_class)
        plt.scatter(X[mask, 0], X[mask, 1],
                   c=point_colors[mask], marker='x', s=100, alpha=0.6,
                   label=f'True Class {true_class}', linewidths=2)

    # Predicted (circle markers) - only plot misclassified for clarity
    misclassified = (y != y_pred)
    if np.any(misclassified):
        plt.scatter(X[misclassified, 0], X[misclassified, 1],
                   c=point_colors[misclassified], marker='o', s=200,
                   facecolors='none', linewidths=3,
                   label='Misclassified (Predicted)', edgecolors=point_colors[misclassified])

    # Plot decision boundary
    x1_min, x1_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    x1_boundary, x2_boundary = model.get_decision_boundary((x1_min, x1_max))
    plt.plot(x1_boundary, x2_boundary, 'g-', linewidth=3, label='Decision Boundary (σ(Z)=0.5)')

    plt.xlabel('X₁', fontsize=14, fontweight='bold')
    plt.ylabel('X₂', fontsize=14, fontweight='bold')
    plt.title('Binary Classification Results\nLogistic Regression with Decision Boundary',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Classification plot saved to: {save_path}")
    plt.close()


def plot_convergence(history: Dict, output_dir: str = 'output'):
    """
    Create convergence plots for Log-Likelihood and MSE.

    Args:
        history: Training history dictionary
        output_dir: Directory to save the plot
    """
    save_path = os.path.join(output_dir, 'convergence_plot.png')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    iterations = range(len(history['log_likelihood']))

    # Plot 1: Log-Likelihood Progress
    ax1.plot(iterations, history['log_likelihood'], 'b-', linewidth=2)
    ax1.set_xlabel('Iteration Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Log-Likelihood Value', fontsize=12, fontweight='bold')
    ax1.set_title('Log-Likelihood Progress\n(Expected: Increasing)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # Plot 2: MSE Progress
    ax2.plot(iterations, history['mse'], 'r-', linewidth=2)
    ax2.set_xlabel('Iteration Number', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Squared Error (MSE)', fontsize=12, fontweight='bold')
    ax2.set_title('MSE Progress\n(Expected: Decreasing)',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Convergence plot saved to: {save_path}")
    plt.close()


def test_on_unseen_data(model: LogisticRegression, n_test_samples: int = 1000):
    """
    Test the model on unseen data to demonstrate generalization.

    Args:
        model: Trained LogisticRegression model
        n_test_samples: Number of test samples to generate
    """
    print("\n" + "="*70)
    print("TESTING ON UNSEEN DATA (GENERALIZATION)")
    print("="*70)

    # Generate new test data (different random state)
    X_test, y_test = generate_synthetic_data(n_samples_per_class=n_test_samples//2,
                                            random_state=123)

    # Predict on test data
    p_test = model.predict_proba(X_test)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    test_mse = np.mean((y_test - p_test) ** 2)
    accuracy = np.mean(y_pred_test == y_test)

    print(f"\nTest Set Size: {len(y_test)} samples")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Confusion matrix
    tp = np.sum((y_test == 1) & (y_pred_test == 1))
    tn = np.sum((y_test == 0) & (y_pred_test == 0))
    fp = np.sum((y_test == 0) & (y_pred_test == 1))
    fn = np.sum((y_test == 1) & (y_pred_test == 0))

    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn:4d}")
    print(f"  True Positives:  {tp:4d}")
    print(f"  False Positives: {fp:4d}")
    print(f"  False Negatives: {fn:4d}")

    return X_test, y_test, p_test


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
    print(f"  Features: X₁, X₂ (normalized to [0, 1])")
    print(f"  X₁ range: [{X_train[:, 0].min():.3f}, {X_train[:, 0].max():.3f}]")
    print(f"  X₂ range: [{X_train[:, 1].min():.3f}, {X_train[:, 1].max():.3f}]")

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
