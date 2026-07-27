"""
Phase 1: Fair hybrid vs transformer comparison.

Metrics:
- Validation loss vs tokens
- Validation loss vs wall-clock time
- Validation loss vs FLOPs
- Throughput vs sequence length
- Memory footprint
"""

import mlx.core as mx
import mlx.optimizers as optim
from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import numpy as np


@dataclass
class ComparisonConfig:
    """Comparison experiment config."""

    model_name: str  # "hybrid_110m", "transformer_96m", "transformer_110m"
    vocab_size: int = 32768
    seq_len: int = 2048
    batch_size: int = 1
    num_tokens_target: int = 100_000_000  # 100M token budget
    checkpoint_every_tokens: int = 1_000_000
    eval_every_tokens: int = 5_000_000
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    gradient_clip: float = 1.0


@dataclass
class ComparisonMetrics:
    """Metrics snapshot."""

    tokens_seen: int
    step: int
    train_loss: float
    val_loss: float
    wall_clock_sec: float
    throughput_tokens_per_sec: float
    estimated_flops: int
    peak_memory_mb: float


class ComparisonRunner:
    """Fair comparison harness."""

    def __init__(self, config: ComparisonConfig):
        self.config = config
        self.metrics_history: List[ComparisonMetrics] = []
        self.start_time = None

    def run_comparison(self, model, train_data, val_data) -> List[ComparisonMetrics]:
        """
        Run model training with fair comparison metrics.

        Args:
            model: MLX model to train
            train_data: Training dataset iterator
            val_data: Validation dataset iterator

        Returns:
            List of metrics snapshots
        """
        self.start_time = time.time()
        optimizer = optim.Adam(learning_rate=self.config.learning_rate)

        tokens_seen = 0
        step = 0
        batch_iter = iter(train_data)

        while tokens_seen < self.config.num_tokens_target:
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_data)
                batch = next(batch_iter)

            # Training step
            def loss_fn(model):
                logits, _ = model(batch["input_ids"])
                targets = batch["target_ids"]
                loss = mx.mean((logits - targets) ** 2)  # Simplified loss
                return loss

            grad_fn = mx.value_and_grad(model, loss_fn)
            loss_val, grads = grad_fn(model)

            # Gradient clip
            grad_norm = mx.sqrt(mx.sum(mx.array([mx.sum(g ** 2) for g in grads.values()])))
            if grad_norm > self.config.gradient_clip:
                for key in grads:
                    grads[key] = grads[key] * (self.config.gradient_clip / grad_norm)

            optimizer.update(model, grads)

            tokens_in_batch = batch["input_ids"].shape[0] * batch["input_ids"].shape[1]
            tokens_seen += tokens_in_batch
            step += 1

            # Checkpoint evaluation
            if tokens_seen % self.config.eval_every_tokens < tokens_in_batch:
                metrics = self._eval_checkpoint(
                    model, val_data, tokens_seen, step
                )
                self.metrics_history.append(metrics)

        return self.metrics_history

    def _eval_checkpoint(
        self, model, val_data, tokens_seen: int, step: int
    ) -> ComparisonMetrics:
        """Evaluate model at checkpoint."""
        wall_clock = time.time() - self.start_time
        throughput = tokens_seen / wall_clock if wall_clock > 0 else 0

        # Validation loss (simplified)
        val_losses = []
        for batch in val_data:
            logits, _ = model(batch["input_ids"])
            targets = batch["target_ids"]
            loss = mx.mean((logits - targets) ** 2)
            val_losses.append(float(loss))

        val_loss = np.mean(val_losses) if val_losses else 0

        # Estimate FLOPs (simplified)
        # For a forward pass: 2 * params * seq_len per token
        model_params = self._count_params(model)
        flops_estimate = 2 * model_params * self.config.seq_len

        # Memory usage (simplified)
        peak_memory = self._estimate_memory_mb(model)

        return ComparisonMetrics(
            tokens_seen=tokens_seen,
            step=step,
            train_loss=float(loss_val) if 'loss_val' in locals() else 0,
            val_loss=val_loss,
            wall_clock_sec=wall_clock,
            throughput_tokens_per_sec=throughput,
            estimated_flops=flops_estimate,
            peak_memory_mb=peak_memory,
        )

    def _count_params(self, model) -> int:
        """Count model parameters."""
        total = 0

        def count_recursive(obj):
            nonlocal total
            if hasattr(obj, "parameters"):
                for param in obj.parameters():
                    if hasattr(param, "size"):
                        total += param.size
            if hasattr(obj, "__dict__"):
                for v in obj.__dict__.values():
                    if hasattr(v, "parameters") or hasattr(v, "__dict__"):
                        count_recursive(v)

        count_recursive(model)
        return total

    def _estimate_memory_mb(self, model) -> float:
        """Estimate peak memory usage."""
        params = self._count_params(model)
        # Rough estimate: params * 4 bytes (FP32) + activations
        bytes_per_param = 4
        activation_factor = 2.0  # Activations ~2x params
        total_bytes = params * bytes_per_param * (1 + activation_factor)
        return total_bytes / (1024 ** 2)

    def summary(self) -> Dict:
        """Generate comparison summary."""
        if not self.metrics_history:
            return {}

        first = self.metrics_history[0]
        last = self.metrics_history[-1]

        return {
            "model": self.config.model_name,
            "total_tokens": last.tokens_seen,
            "total_steps": last.step,
            "final_val_loss": last.val_loss,
            "best_val_loss": min(m.val_loss for m in self.metrics_history),
            "wall_clock_hours": last.wall_clock_sec / 3600,
            "avg_throughput_tokens_per_sec": last.throughput_tokens_per_sec,
            "peak_memory_mb": last.peak_memory_mb,
        }
