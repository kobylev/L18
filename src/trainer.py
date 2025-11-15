"""
Model Trainer

Contains the training logic for Logistic Regression using Gradient Ascent.
"""

import numpy as np
from .model_base import LogisticRegressionBase


class LogisticRegressionTrainer:
    """
    Training mixin for Logistic Regression using Gradient Ascent.

    This class handles the gradient ascent optimization algorithm.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> 'LogisticRegressionTrainer':
        """
        Fit the logistic regression model using Gradient Ascent.

        Supports both Batch and Mini-Batch Gradient Ascent:
        - Batch: Uses all samples for each update (batch_size=None)
        - Mini-Batch: Uses random subsets for each update (batch_size=int)

        The update rule (Gradient Ascent for maximizing log-likelihood):
        β_k^(t+1) = β_k^(t) + α * Σ[(y_i - p̂_i) * X_{k,i}]

        Early Stopping (optional):
        - Monitors validation loss to prevent overfitting
        - Stops training if no improvement for 'patience' iterations
        - Restores best parameters found during training

        Args:
            X: Feature matrix of shape (n_samples, 2)
            y: Target vector of shape (n_samples,)
            verbose: Whether to print progress

        Returns:
            self
        """
        # Early stopping: Split into train/validation if enabled
        if self.early_stopping and self.validation_split > 0:
            n_samples = len(y)
            n_val = int(n_samples * self.validation_split)
            indices = np.random.permutation(n_samples)

            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            X_train, X_val = X[train_indices], X[val_indices]
            y_train, y_val = y[train_indices], y[val_indices]

            if verbose:
                print(f"Early stopping enabled: {len(train_indices)} train, {len(val_indices)} validation samples")
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None

        # Add bias term (X_0 = 1)
        X_with_bias = self._add_bias_term(X_train)
        n_samples, n_features = X_with_bias.shape

        if X_val is not None:
            X_val_with_bias = self._add_bias_term(X_val)

        # Initialize beta coefficients
        np.random.seed(42)
        self.beta = np.random.randn(n_features) * 0.01

        # Early stopping tracking
        patience_counter = 0
        self.best_val_loss = float('inf')
        self.best_beta = self.beta.copy()

        # Determine batch mode
        if self.batch_size is None or self.batch_size >= n_samples:
            batch_mode = 'full'
            effective_batch_size = n_samples
        else:
            batch_mode = 'mini'
            effective_batch_size = self.batch_size

        if verbose:
            self._print_training_header(batch_mode, effective_batch_size, n_samples)

        # Gradient Ascent iterations
        for iteration in range(self.n_iterations):
            if batch_mode == 'mini':
                # Mini-Batch Gradient Ascent
                # Randomly sample a mini-batch
                batch_indices = np.random.choice(n_samples, effective_batch_size, replace=False)
                X_batch = X_with_bias[batch_indices]
                y_batch = y_train[batch_indices]
            else:
                # Full Batch Gradient Ascent
                X_batch = X_with_bias
                y_batch = y_train

            # Compute weighted sum and probability for the batch
            z = self._compute_weighted_sum(X_batch)
            p_hat = self.sigmoid(z)

            # Compute gradients on the batch
            error = y_batch - p_hat
            gradients = X_batch.T @ error

            # Apply regularization to gradients (if enabled)
            if self.regularization is not None:
                reg_gradient = self._compute_regularization_gradient()
                gradients = gradients - reg_gradient  # Subtract because we're maximizing

            # Update beta coefficients (Gradient ASCENT)
            beta_old = self.beta.copy()
            self.beta = self.beta + self.learning_rate * gradients

            # Compute and store metrics on FULL TRAINING dataset for monitoring
            z_full = self._compute_weighted_sum(X_with_bias)
            p_hat_full = self.sigmoid(z_full)
            log_likelihood = self._compute_log_likelihood(y_train, p_hat_full)
            mse = self._compute_mse(y_train, p_hat_full)

            # Compute validation metrics if early stopping enabled
            val_log_likelihood = None
            val_mse = None
            if X_val is not None:
                z_val = self._compute_weighted_sum(X_val_with_bias)
                p_hat_val = self.sigmoid(z_val)
                val_log_likelihood = self._compute_log_likelihood(y_val, p_hat_val)
                val_mse = self._compute_mse(y_val, p_hat_val)

            self._store_history(log_likelihood, mse, val_log_likelihood, val_mse)

            # Update learning rate according to schedule
            prev_val_loss = -self.history['val_log_likelihood'][-2] if len(self.history['val_log_likelihood']) > 1 else None
            current_val_loss = -val_log_likelihood if val_log_likelihood is not None else None
            self._update_learning_rate(iteration, current_val_loss, prev_val_loss)

            # Early stopping check
            if self.early_stopping and X_val is not None:
                # Use negative log-likelihood as loss (lower is better)
                val_loss = -val_log_likelihood

                if val_loss < self.best_val_loss:
                    # Improvement found
                    self.best_val_loss = val_loss
                    self.best_beta = self.beta.copy()
                    patience_counter = 0
                else:
                    # No improvement
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if verbose:
                        print(f"\nEarly stopping at iteration {iteration}!")
                        print(f"Best validation loss: {self.best_val_loss:.4f} (at iteration {iteration - self.patience})")
                    self.stopped_epoch = iteration
                    self.beta = self.best_beta  # Restore best parameters
                    break

            # Check convergence
            beta_change = np.max(np.abs(self.beta - beta_old))

            if verbose:
                self._print_progress(iteration, log_likelihood, mse, beta_change,
                                   val_log_likelihood, val_mse)

            # Stop if converged
            if beta_change < self.tolerance:
                if verbose:
                    print(f"\nConverged at iteration {iteration}!")
                break

        if verbose:
            self._print_training_summary(log_likelihood, mse)

        return self

    def _compute_regularization_gradient(self) -> np.ndarray:
        """
        Compute regularization gradient.

        L1 Regularization (Lasso):
            Penalty = lambda * |beta|
            Gradient = lambda * sign(beta)

        L2 Regularization (Ridge):
            Penalty = lambda * beta^2
            Gradient = 2 * lambda * beta

        Note: We don't regularize the bias term (beta_0)

        Returns:
            Regularization gradient vector
        """
        reg_grad = np.zeros_like(self.beta)

        if self.regularization == 'l1':
            # L1: Gradient = lambda * sign(beta)
            # Don't regularize bias (index 0)
            reg_grad[1:] = self.lambda_reg * np.sign(self.beta[1:])

        elif self.regularization == 'l2':
            # L2: Gradient = 2 * lambda * beta
            # Don't regularize bias (index 0)
            reg_grad[1:] = 2 * self.lambda_reg * self.beta[1:]

        elif self.regularization == 'elastic':
            # Elastic Net: Combination of L1 and L2
            # Gradient = lambda * (alpha * sign(beta) + (1-alpha) * 2 * beta)
            # Using alpha = 0.5 for equal mix
            alpha = 0.5
            reg_grad[1:] = self.lambda_reg * (
                alpha * np.sign(self.beta[1:]) +
                (1 - alpha) * 2 * self.beta[1:]
            )

        return reg_grad

    def _update_learning_rate(self, iteration: int, val_loss: float = None, prev_val_loss: float = None):
        """
        Update learning rate according to schedule.

        Strategies:
        - 'step': Decay by factor every N steps
        - 'exponential': Exponential decay
        - 'inverse': Inverse time decay
        - 'adaptive': Increase/decrease based on validation performance

        Args:
            iteration: Current iteration number
            val_loss: Current validation loss (for adaptive)
            prev_val_loss: Previous validation loss (for adaptive)
        """
        if self.lr_schedule is None:
            # No schedule, keep constant learning rate
            self.current_lr = self.learning_rate
            return

        if self.lr_schedule == 'step':
            # Step decay: lr = initial_lr * (decay_rate ^ floor(iteration / decay_steps))
            decay_factor = self.lr_decay_rate ** (iteration // self.lr_decay_steps)
            self.current_lr = self.initial_learning_rate * decay_factor

        elif self.lr_schedule == 'exponential':
            # Exponential decay: lr = initial_lr * (decay_rate ^ iteration)
            self.current_lr = self.initial_learning_rate * (self.lr_decay_rate ** iteration)

        elif self.lr_schedule == 'inverse':
            # Inverse time decay: lr = initial_lr / (1 + decay_rate * iteration)
            self.current_lr = self.initial_learning_rate / (1 + self.lr_decay_rate * iteration)

        elif self.lr_schedule == 'adaptive':
            # Adaptive: Increase if improving, decrease if not
            if val_loss is not None and prev_val_loss is not None:
                if val_loss < prev_val_loss:
                    # Improving: slightly increase learning rate
                    self.current_lr = min(self.current_lr * 1.05, self.initial_learning_rate)
                else:
                    # Not improving: decrease learning rate
                    self.current_lr = self.current_lr * self.lr_decay_rate
            else:
                # No validation loss available, keep current
                pass

        # Apply minimum learning rate floor
        self.current_lr = max(self.current_lr, self.lr_min)

        # Update the actual learning rate used in training
        self.learning_rate = self.current_lr

    def _store_history(self, log_likelihood: float, mse: float,
                       val_log_likelihood: float = None, val_mse: float = None):
        """Store training and validation metrics in history."""
        self.history['log_likelihood'].append(log_likelihood)
        self.history['mse'].append(mse)
        self.history['beta_0'].append(self.beta[0])
        self.history['beta_1'].append(self.beta[1])
        self.history['beta_2'].append(self.beta[2])
        self.history['learning_rate'].append(self.current_lr)

        if val_log_likelihood is not None:
            self.history['val_log_likelihood'].append(val_log_likelihood)
        if val_mse is not None:
            self.history['val_mse'].append(val_mse)

    def _print_training_header(self, batch_mode: str = 'full',
                              batch_size: int = None, n_samples: int = None):
        """Print training initialization info."""
        print(f"Starting Gradient Ascent...")
        print(f"Initial beta: {self.beta}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Max iterations: {self.n_iterations}")
        if batch_mode == 'mini':
            print(f"Batch mode: Mini-Batch (size={batch_size}, {batch_size/n_samples*100:.1f}% of data)")
        else:
            print(f"Batch mode: Full Batch (all {n_samples} samples)")

        # Display regularization info
        if self.regularization is not None:
            print(f"Regularization: {self.regularization.upper()} (lambda={self.lambda_reg})")
        else:
            print(f"Regularization: None")

        # Display learning rate schedule info
        if self.lr_schedule is not None:
            print(f"LR Schedule: {self.lr_schedule.upper()} (decay_rate={self.lr_decay_rate}, steps={self.lr_decay_steps})")
        else:
            print(f"LR Schedule: None (constant)")
        print()

    def _print_progress(self, iteration: int, log_likelihood: float,
                       mse: float, beta_change: float,
                       val_log_likelihood: float = None, val_mse: float = None):
        """Print training progress."""
        if iteration % 100 == 0 or iteration < 10:
            output = (f"Iteration {iteration:4d} | "
                     f"Log-Likelihood: {log_likelihood:10.4f} | "
                     f"MSE: {mse:.6f}")

            if val_log_likelihood is not None:
                output += f" | Val-LL: {val_log_likelihood:10.4f} | Val-MSE: {val_mse:.6f}"

            output += f" | beta change: {beta_change:.8f}"
            print(output)

    def _print_training_summary(self, log_likelihood: float, mse: float):
        """Print final training results."""
        print(f"\nFinal beta coefficients: {self.beta}")
        print(f"Final Log-Likelihood: {log_likelihood:.4f}")
        print(f"Final MSE: {mse:.6f}")
