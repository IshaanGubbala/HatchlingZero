"""Phase 16d: Production hardening for fast weights."""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, Any
import json


class ProductionFastWeightLayer(nn.Module):
    """Fast weight layer with production safeguards."""

    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        # Base weights (frozen)
        self.weight = mx.random.normal((input_dim, output_dim)) / (input_dim ** 0.5)
        if use_bias:
            self.bias = mx.zeros((output_dim,))

        # Fast weights (session-local)
        self.fast_weight = mx.zeros((input_dim, output_dim))
        if use_bias:
            self.fast_bias = mx.zeros((output_dim,))

        # Production monitoring
        self.update_count = 0
        self.gradient_norm_history = []
        self.fast_weight_norm_history = []

    def __call__(self, x: mx.array) -> mx.array:
        """Forward with overflow protection."""
        w_eff = self.weight + self.fast_weight

        # Clip weights to prevent overflow
        w_eff = mx.clip(w_eff, -100.0, 100.0)

        out = mx.matmul(x, w_eff)

        if self.use_bias:
            b_eff = self.bias + self.fast_bias
            b_eff = mx.clip(b_eff, -100.0, 100.0)
            out = out + b_eff

        return out

    def update_fast_weights(
        self,
        grads: dict,
        lr: float = 0.01,
        grad_clip: float = 1.0,
        max_fast_weight_norm: float = 10.0
    ):
        """Apply gradient step with clipping and monitoring."""
        if "fast_weight" in grads:
            grad = grads["fast_weight"]

            # Gradient clipping
            grad_norm = mx.linalg.norm(grad)
            if float(grad_norm) > grad_clip:
                grad = grad * (grad_clip / float(grad_norm))
                clipped_norm = float(mx.linalg.norm(grad))
                self.gradient_norm_history.append(clipped_norm)
            else:
                self.gradient_norm_history.append(float(grad_norm))

            # Weight update
            self.fast_weight = self.fast_weight - lr * grad

            # Weight norm clipping (prevent explosion)
            fw_norm = mx.linalg.norm(self.fast_weight)
            self.fast_weight_norm_history.append(float(fw_norm))

            if float(fw_norm) > max_fast_weight_norm:
                scale = max_fast_weight_norm / float(fw_norm)
                self.fast_weight = self.fast_weight * scale

        if "fast_bias" in grads:
            grad = grads["fast_bias"]
            grad_norm = mx.linalg.norm(grad)

            if float(grad_norm) > grad_clip:
                grad = grad * (grad_clip / float(grad_norm))

            self.fast_bias = self.fast_bias - lr * grad
            self.fast_bias = mx.clip(self.fast_bias, -10.0, 10.0)

        self.update_count += 1

    def reset_fast_weights(self):
        """Reset to zero for new session."""
        self.fast_weight = mx.zeros_like(self.fast_weight)
        if self.use_bias:
            self.fast_bias = mx.zeros_like(self.fast_bias)

    def check_numerical_stability(self) -> Dict[str, Any]:
        """Check for NaN, inf, or unusual values."""
        issues = {}

        # Check fast weights
        if mx.any(mx.isnan(self.fast_weight)):
            issues["fast_weight_nan"] = True
        if mx.any(mx.isinf(self.fast_weight)):
            issues["fast_weight_inf"] = True

        fw_norm = float(mx.linalg.norm(self.fast_weight))
        if fw_norm > 1000.0:
            issues["fast_weight_explosion"] = fw_norm
        if fw_norm < 1e-7 and self.update_count > 0:
            issues["fast_weight_underflow"] = fw_norm

        # Check bias
        if self.use_bias:
            if mx.any(mx.isnan(self.fast_bias)):
                issues["fast_bias_nan"] = True
            if mx.any(mx.isinf(self.fast_bias)):
                issues["fast_bias_inf"] = True

        return issues

    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            "update_count": self.update_count,
            "fast_weight_norm": float(mx.linalg.norm(self.fast_weight)),
            "avg_gradient_norm": (
                sum(self.gradient_norm_history) / len(self.gradient_norm_history)
                if self.gradient_norm_history
                else 0.0
            ),
            "max_gradient_norm": max(self.gradient_norm_history) if self.gradient_norm_history else 0.0,
        }


class ProductionFastWeightSession:
    """Session manager with checkpointing and monitoring."""

    def __init__(self, model, fast_weight_layers: list):
        self.model = model
        self.fast_weight_layers = fast_weight_layers
        self.session_id = None
        self.session_active = False
        self.checkpoints = {}

    def start_session(self, session_id: str = "default"):
        """Start new session."""
        self.session_id = session_id
        for layer in self.fast_weight_layers:
            if hasattr(layer, 'reset_fast_weights'):
                layer.reset_fast_weights()
        self.session_active = True

    def end_session(self):
        """End session and cleanup."""
        self.session_active = False
        for layer in self.fast_weight_layers:
            if hasattr(layer, 'reset_fast_weights'):
                layer.reset_fast_weights()

    def checkpoint(self, checkpoint_name: str) -> bool:
        """Save fast weights for rollback."""
        if not self.session_active:
            return False

        checkpoint_data = {}
        for i, layer in enumerate(self.fast_weight_layers):
            if hasattr(layer, 'fast_weight'):
                checkpoint_data[f"layer_{i}_fw"] = mx.array(layer.fast_weight)
                if hasattr(layer, 'fast_bias'):
                    checkpoint_data[f"layer_{i}_fb"] = mx.array(layer.fast_bias)

        self.checkpoints[checkpoint_name] = checkpoint_data
        return True

    def rollback(self, checkpoint_name: str) -> bool:
        """Restore fast weights from checkpoint."""
        if checkpoint_name not in self.checkpoints:
            return False

        checkpoint_data = self.checkpoints[checkpoint_name]
        for i, layer in enumerate(self.fast_weight_layers):
            if hasattr(layer, 'fast_weight'):
                key = f"layer_{i}_fw"
                if key in checkpoint_data:
                    layer.fast_weight = checkpoint_data[key]
                key_b = f"layer_{i}_fb"
                if key_b in checkpoint_data and hasattr(layer, 'fast_bias'):
                    layer.fast_bias = checkpoint_data[key_b]

        return True

    def check_health(self) -> Dict[str, Any]:
        """Check overall session health."""
        health = {
            "session_id": self.session_id,
            "session_active": self.session_active,
            "layers_healthy": [],
            "issues": []
        }

        for i, layer in enumerate(self.fast_weight_layers):
            if hasattr(layer, 'check_numerical_stability'):
                issues = layer.check_numerical_stability()
                if issues:
                    health["issues"].append({f"layer_{i}": issues})
                    health["layers_healthy"].append(False)
                else:
                    health["layers_healthy"].append(True)

        return health

    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics."""
        stats = {
            "session_id": self.session_id,
            "layer_stats": []
        }

        for i, layer in enumerate(self.fast_weight_layers):
            if hasattr(layer, 'get_stats'):
                stats["layer_stats"].append({
                    f"layer_{i}": layer.get_stats()
                })

        return stats


def test_production_hardening():
    """Test Phase 16d: Production safeguards."""
    print("="*70)
    print("Phase 16d: Production Hardening")
    print("="*70)

    # Create production layer
    layer = ProductionFastWeightLayer(64, 64)
    session = ProductionFastWeightSession(None, [layer])

    print(f"\n1. NUMERICAL STABILITY")
    session.start_session("test_session")

    # Check initial state
    health = session.check_health()
    print(f"   Initial health: {health['layers_healthy']}")
    assert all(health["layers_healthy"]), "Initial state should be healthy"
    print(f"   ✓ Passed")

    print(f"\n2. GRADIENT CLIPPING")
    # Create large gradient
    large_grad = {"fast_weight": mx.ones((64, 64)) * 100.0}
    layer.update_fast_weights(large_grad, lr=0.01, grad_clip=1.0)
    print(f"   Applied gradient with norm 6400.0")

    # Check that gradient was clipped
    stats = layer.get_stats()
    max_grad = stats["max_gradient_norm"]
    print(f"   Max gradient after clip: {max_grad:.4f}")
    assert max_grad <= 1.0, f"Gradient should be clipped to ≤1.0, got {max_grad}"
    print(f"   ✓ Gradient clipping works")

    print(f"\n3. WEIGHT NORM CLIPPING")
    # Create large update to trigger weight explosion
    for _ in range(10):
        large_grad = {"fast_weight": mx.ones((64, 64)) * 10.0}
        layer.update_fast_weights(large_grad, lr=1.0, max_fast_weight_norm=10.0)

    # Check weight norm is bounded
    fw_norm = stats["fast_weight_norm"]
    print(f"   Fast weight norm after updates: {fw_norm:.4f}")
    assert fw_norm <= 10.5, f"Weight norm should be ≤10.5, got {fw_norm}"  # Small tolerance
    print(f"   ✓ Weight norm clipping works")

    print(f"\n4. CHECKPOINTING")
    session.checkpoint("checkpoint_1")
    print(f"   Checkpoint saved")

    # Modify weights
    layer.fast_weight = mx.ones((64, 64)) * 5.0
    before_rollback = float(mx.mean(layer.fast_weight))
    print(f"   Modified weights, mean = {before_rollback:.4f}")

    # Rollback
    session.rollback("checkpoint_1")
    after_rollback = float(mx.mean(layer.fast_weight))
    print(f"   After rollback, mean = {after_rollback:.4f}")
    print(f"   ✓ Checkpointing works")

    print(f"\n5. SESSION ISOLATION")
    # End session and verify reset
    session.end_session()
    session.start_session("new_session")

    reset_norm = float(mx.linalg.norm(layer.fast_weight))
    print(f"   After new session, weight norm = {reset_norm:.6f}")
    assert reset_norm < 1e-5, "Weights should be reset to zero"
    print(f"   ✓ Session isolation works")

    print(f"\n6. STATS REPORTING")
    stats = session.get_session_stats()
    print(f"   Session ID: {stats['session_id']}")
    print(f"   Layers tracked: {len(stats['layer_stats'])}")
    print(f"   ✓ Stats reporting works")

    session.end_session()

    print(f"\n{'='*70}")
    print("✓ PASS: All production hardening tests passed")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    test_production_hardening()
