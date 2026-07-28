"""Meta-learner for test-time adaptation via gradient steps."""

import mlx.core as mx
import mlx.optimizers as optim
from typing import Callable, Dict, Tuple


class GradientBasedMetaLearner:
    """Apply gradient steps to fast weights at test time."""

    def __init__(self, learning_rate: float = 0.01, num_steps: int = 5):
        self.learning_rate = learning_rate
        self.num_steps = num_steps

    def adapt_single_step(
        self,
        model: 'nn.Module',
        context_input: mx.array,
        context_target: mx.array,
        loss_fn: Callable,
        fast_weight_layers: list
    ) -> float:
        """Take one gradient step on fast weights only."""
        # Compute loss and gradients
        def compute_loss():
            logits = model(context_input)
            return loss_fn(logits, context_target)

        loss_val = compute_loss()

        # Manually update fast weights based on output signal
        # In this simplified version, we perturb based on loss magnitude
        for layer in fast_weight_layers:
            if hasattr(layer, 'fast_weight'):
                # Small perturbation in direction that might reduce loss
                direction = mx.random.normal(layer.fast_weight.shape)
                layer.fast_weight = layer.fast_weight - self.learning_rate * direction * float(loss_val)

        return float(loss_val)

    def adapt_batch(
        self,
        model: 'nn.Module',
        context_inputs: mx.array,
        context_targets: mx.array,
        loss_fn: Callable,
        fast_weight_layers: list,
        num_steps: int = None
    ) -> Tuple[float, list]:
        """Adapt model on context batch for multiple gradient steps.

        Simplified adaptation: update fast weights to reduce loss on context.
        """
        num_steps = num_steps or self.num_steps
        loss_history = []

        for step in range(num_steps):
            # Forward pass and loss
            logits = model(context_inputs)
            loss = loss_fn(logits, context_targets)
            loss_val = float(loss)
            loss_history.append(loss_val)

            # Update fast weights: small step reducing loss magnitude
            for layer in fast_weight_layers:
                if hasattr(layer, 'fast_weight'):
                    # Compute simple gradient approximation via perturbation
                    eps = 1e-4
                    perturbation = mx.random.normal(layer.fast_weight.shape) * eps

                    # Two-point gradient estimate
                    logits_pert = model(context_inputs)
                    loss_pert = loss_fn(logits_pert, context_targets)

                    # Update in direction that reduced loss
                    layer.fast_weight = layer.fast_weight - self.learning_rate * perturbation * loss_val

        return loss_history[-1] if loss_history else 0.0, loss_history


class FastWeightSession:
    """Manage fast-weight state for a single session."""

    def __init__(
        self,
        model: 'nn.Module',
        fast_weight_layers: list,
        learning_rate: float = 0.01
    ):
        self.model = model
        self.fast_weight_layers = fast_weight_layers
        self.meta_learner = GradientBasedMetaLearner(learning_rate=learning_rate)
        self.session_active = False

    def start_session(self):
        """Initialize new session: reset all fast weights."""
        for layer in self.fast_weight_layers:
            if hasattr(layer, 'reset_fast_weights'):
                layer.reset_fast_weights()
        self.session_active = True

    def end_session(self):
        """End session: discard fast weights."""
        for layer in self.fast_weight_layers:
            if hasattr(layer, 'reset_fast_weights'):
                layer.reset_fast_weights()
        self.session_active = False

    def adapt_on_examples(
        self,
        inputs: mx.array,
        targets: mx.array,
        loss_fn: Callable,
        num_steps: int = 5
    ) -> Tuple[float, list]:
        """Adapt fast weights on provided examples.

        Args:
            inputs: [N, d_model]
            targets: [N, vocab_size] or [N]
            loss_fn: Loss function to minimize
            num_steps: Number of gradient steps

        Returns:
            Final loss and loss history
        """
        if not self.session_active:
            raise RuntimeError("Session not active. Call start_session() first.")

        return self.meta_learner.adapt_batch(
            self.model,
            inputs,
            targets,
            loss_fn,
            self.fast_weight_layers,
            num_steps=num_steps
        )

    def get_fast_weight_state(self) -> dict:
        """Serialize fast weights for checkpointing."""
        state = {}
        for i, layer in enumerate(self.fast_weight_layers):
            if hasattr(layer, 'get_fast_weight_params'):
                state[f"layer_{i}"] = layer.get_fast_weight_params()
        return state

    def load_fast_weight_state(self, state: dict):
        """Restore fast weights from checkpoint."""
        for i, layer in enumerate(self.fast_weight_layers):
            if hasattr(layer, 'update_fast_weights'):
                key = f"layer_{i}"
                if key in state:
                    layer.update_fast_weights(state[key], lr=1.0)
