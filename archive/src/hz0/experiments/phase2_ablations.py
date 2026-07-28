"""
Phase 2: Controlled ablations.

Tests:
- Learning rate sweep
- Effective batch size (gradient accumulation)
- Depth vs width trade-offs
- Recurrent state dimensions
- Attention frequency
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import itertools


@dataclass
class AblationConfig:
    """Single ablation experiment."""

    name: str
    var_name: str  # Variable being swept
    values: List[Any]  # Values to test
    fixed_params: Dict[str, Any]  # Fixed parameters

    def generate_experiments(self) -> List[Dict[str, Any]]:
        """Generate experiment configs for sweep."""
        experiments = []
        for val in self.values:
            config = self.fixed_params.copy()
            config[self.var_name] = val
            experiments.append(config)
        return experiments


class AblationSuite:
    """Suite of controlled ablations."""

    @staticmethod
    def learning_rate_sweep() -> AblationConfig:
        """LR sweep: 1.5e-4 to 4.0e-4."""
        return AblationConfig(
            name="learning_rate_sweep",
            var_name="learning_rate",
            values=[1.5e-4, 2.0e-4, 3.0e-4, 4.0e-4],
            fixed_params={
                "model": "hybrid_110m",
                "batch_size": 1,
                "gradient_accumulation": 4,
                "tokens": 50_000_000,
            },
        )

    @staticmethod
    def batch_size_sweep() -> AblationConfig:
        """Effective batch size via accumulation: 1-8."""
        return AblationConfig(
            name="effective_batch_size",
            var_name="gradient_accumulation",
            values=[1, 2, 4, 8],
            fixed_params={
                "model": "hybrid_110m",
                "learning_rate": 2.0e-4,
                "tokens": 50_000_000,
            },
        )

    @staticmethod
    def depth_vs_width() -> AblationConfig:
        """Depth/width trade-offs at ~110M params."""
        return AblationConfig(
            name="depth_vs_width",
            var_name="model_config",
            values=[
                {"layers": 24, "dim": 768},  # Default
                {"layers": 32, "dim": 640},  # Deeper, narrower
                {"layers": 20, "dim": 832},  # Shallower, wider
            ],
            fixed_params={
                "model": "hybrid",
                "learning_rate": 2.0e-4,
                "tokens": 50_000_000,
            },
        )

    @staticmethod
    def recurrent_dimensions() -> AblationConfig:
        """Recurrent state dimensions: Dk × Dv."""
        return AblationConfig(
            name="recurrent_dimensions",
            var_name="head_dims",
            values=[
                {"key_dim": 32, "value_dim": 32},
                {"key_dim": 64, "value_dim": 64},
                {"key_dim": 96, "value_dim": 96},
            ],
            fixed_params={
                "model": "hybrid_110m",
                "learning_rate": 2.0e-4,
                "tokens": 50_000_000,
            },
        )

    @staticmethod
    def attention_frequency() -> AblationConfig:
        """Attention frequency: every N recurrent layers."""
        return AblationConfig(
            name="attention_frequency",
            var_name="attention_every_n",
            values=[2, 3, 4, 6],
            fixed_params={
                "model": "hybrid_110m",
                "learning_rate": 2.0e-4,
                "tokens": 50_000_000,
            },
        )

    @staticmethod
    def all_ablations() -> List[AblationConfig]:
        """Return all ablation suites."""
        return [
            AblationSuite.learning_rate_sweep(),
            AblationSuite.batch_size_sweep(),
            AblationSuite.depth_vs_width(),
            AblationSuite.recurrent_dimensions(),
            AblationSuite.attention_frequency(),
        ]


@dataclass
class AblationResult:
    """Results from one ablation experiment."""

    ablation_name: str
    variable: str
    value: Any
    val_loss: float
    throughput: float
    peak_memory_mb: float
    wall_clock_sec: float

    def loss_per_hour(self) -> float:
        """Validation loss per hour of training."""
        hours = self.wall_clock_sec / 3600 if self.wall_clock_sec > 0 else 1
        return self.val_loss / hours

    def loss_per_token(self) -> float:
        """Loss per token (approx: loss / tokens_seen)."""
        # Rough: 50M tokens, batch=1, seq_len=2048
        tokens = 50_000_000
        return self.val_loss / tokens


class AblationAnalyzer:
    """Analyze ablation results."""

    @staticmethod
    def rank_results(results: List[AblationResult], metric: str = "val_loss") -> List[AblationResult]:
        """Rank results by metric (ascending = better)."""
        return sorted(results, key=lambda r: getattr(r, metric))

    @staticmethod
    def summarize(results: List[AblationResult]) -> Dict[str, Any]:
        """Summarize ablation suite."""
        if not results:
            return {}

        return {
            "total_experiments": len(results),
            "best_val_loss": min(r.val_loss for r in results),
            "best_throughput": max(r.throughput for r in results),
            "best_memory_efficiency": min(r.peak_memory_mb for r in results),
            "best_efficiency_factor": min(r.loss_per_hour() for r in results),
        }
