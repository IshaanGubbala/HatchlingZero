"""Phase 2: Ablation experiments (scaling, optimization, architecture).

Experiments:
- 2.1: Equal tokens per parameter (36M vs 110M)
- 2.2: Learning rate sweep
- 2.3: Batch size / gradient accumulation
- 2.4: Depth vs width
- 2.5: Recurrent state dimensions
- 2.6: Attention frequency
"""

from dataclasses import dataclass
from typing import List, Tuple
import json
from pathlib import Path


@dataclass
class AblationConfig:
    """Single ablation experiment."""

    name: str
    model_size: str  # "36M" | "110M"
    model_dim: int
    num_layers: int
    num_heads: int
    learning_rate: float = 2e-4
    batch_size: int = 2
    gradient_accumulation: int = 4
    seq_length: int = 256
    tokens_target: int = None  # If set, train to this many tokens
    lr_schedule: str = "cosine"  # "cosine" | "linear" | "constant"


class Phase2Experiments:
    """Phase 2 ablation suite."""

    def __init__(self):
        self.experiments: List[AblationConfig] = []
        self._create_experiments()

    def _create_experiments(self):
        """Define all Phase 2 experiments."""

        # Experiment 2.1: Equal tokens per parameter
        print("Experiment 2.1: Equal tokens per parameter")
        self.experiments.extend([
            AblationConfig(
                name="2.1_36M_baseline",
                model_size="36M",
                model_dim=256,
                num_layers=12,
                num_heads=4,
                tokens_target=50_000_000,  # X tokens
            ),
            AblationConfig(
                name="2.1_110M_scaled",
                model_size="110M",
                model_dim=768,
                num_layers=24,
                num_heads=12,
                tokens_target=150_000_000,  # ~3.06X tokens
            ),
        ])

        # Experiment 2.2: Learning rate sweep
        print("Experiment 2.2: Learning rate sweep")
        for lr in [1.0e-4, 1.5e-4, 2.0e-4, 3.0e-4, 4.0e-4]:
            self.experiments.append(
                AblationConfig(
                    name=f"2.2_lr_{lr:.1e}_110M",
                    model_size="110M",
                    model_dim=768,
                    num_layers=24,
                    num_heads=12,
                    learning_rate=lr,
                    tokens_target=25_000_000,  # Short run for sweep
                )
            )

        # Experiment 2.3: Batch size / gradient accumulation
        print("Experiment 2.3: Effective batch size")
        for grad_accum in [1, 2, 4, 8]:
            self.experiments.append(
                AblationConfig(
                    name=f"2.3_accum_{grad_accum}_110M",
                    model_size="110M",
                    model_dim=768,
                    num_layers=24,
                    num_heads=12,
                    gradient_accumulation=grad_accum,
                    tokens_target=25_000_000,
                )
            )

        # Experiment 2.4: Depth vs width
        print("Experiment 2.4: Depth vs width")
        self.experiments.extend([
            AblationConfig(
                name="2.4_deep_narrow_110M",
                model_size="110M",
                model_dim=512,
                num_layers=32,  # Deeper
                num_heads=8,
                tokens_target=50_000_000,
            ),
            AblationConfig(
                name="2.4_shallow_wide_110M",
                model_size="110M",
                model_dim=1024,
                num_layers=12,  # Shallower
                num_heads=16,
                tokens_target=50_000_000,
            ),
        ])

        # Experiment 2.5: Recurrent state dimensions
        print("Experiment 2.5: Recurrent state dimensions")
        for dk, dv in [(32, 32), (64, 64), (96, 96)]:
            # Note: This requires model changes to expose head_dim
            self.experiments.append(
                AblationConfig(
                    name=f"2.5_dk{dk}_dv{dv}_110M",
                    model_size="110M",
                    model_dim=768,
                    num_layers=24,
                    num_heads=12,  # Will be adjusted for dk/dv
                    tokens_target=25_000_000,
                )
            )

        # Experiment 2.6: Attention frequency
        print("Experiment 2.6: Attention frequency")
        for attn_every in [2, 3, 4, 6]:
            self.experiments.append(
                AblationConfig(
                    name=f"2.6_attn_every_{attn_every}_110M",
                    model_size="110M",
                    model_dim=768,
                    num_layers=24,
                    num_heads=12,
                    tokens_target=25_000_000,
                )
            )

    def save_experiments(self, output_path: str = "configs/phase2_ablations.json"):
        """Save experiment configs to JSON."""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        configs = [
            {
                "name": exp.name,
                "model_size": exp.model_size,
                "model_dim": exp.model_dim,
                "num_layers": exp.num_layers,
                "num_heads": exp.num_heads,
                "learning_rate": exp.learning_rate,
                "batch_size": exp.batch_size,
                "gradient_accumulation": exp.gradient_accumulation,
                "seq_length": exp.seq_length,
                "tokens_target": exp.tokens_target,
                "lr_schedule": exp.lr_schedule,
            }
            for exp in self.experiments
        ]

        with open(output_path, "w") as f:
            json.dump(configs, f, indent=2)

        print(f"\n✓ Saved {len(self.experiments)} experiments to {output_path}")

    def print_summary(self):
        """Print experiment summary."""
        print("\n" + "="*70)
        print("Phase 2: Ablation Experiments Summary")
        print("="*70)

        by_type = {}
        for exp in self.experiments:
            exp_type = exp.name.split("_")[0]
            if exp_type not in by_type:
                by_type[exp_type] = []
            by_type[exp_type].append(exp)

        for exp_type in sorted(by_type.keys()):
            exps = by_type[exp_type]
            print(f"\n{exp_type}: {len(exps)} experiments")
            for exp in exps:
                print(f"  {exp.name:<40} | LR={exp.learning_rate:.1e} | Accum={exp.gradient_accumulation}")

        total = len(self.experiments)
        print(f"\n{'='*70}")
        print(f"Total: {total} experiments")
        print(f"Estimated total tokens: {sum(e.tokens_target or 25_000_000 for e in self.experiments) / 1e9:.1f}B")
        print(f"Estimated time (1 tok/s): {sum(e.tokens_target or 25_000_000 for e in self.experiments) / 3600:.0f} hours")
        print("="*70)


def main():
    """Generate Phase 2 ablations."""
    ablations = Phase2Experiments()
    ablations.print_summary()
    ablations.save_experiments()

    print("\n✓ Phase 2 experiments ready for execution")
    print("Next: Run experiments in parallel on available hardware")


if __name__ == "__main__":
    main()
