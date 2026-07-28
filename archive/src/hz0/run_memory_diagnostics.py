"""
Run memory diagnostics on all HZ models per plan section 14, experiment 4.

Tests on:
- Best 36M model
- Tuned 110M hybrid
- 110M baseline (for comparison)

Usage: python -m src.hz0.run_memory_diagnostics
"""

from pathlib import Path
import sys

sys.path.insert(0, '/Users/ishaangubbala/Documents/Training')

from hz0.memory_diagnostics import MemoryDiagnostics


def run_diagnostics_suite():
    """Run memory diagnostics on all available models."""
    print("=" * 80)
    print("MEMORY DIAGNOSTICS SUITE (Plan Section 14, Experiment 4)")
    print("=" * 80)
    print()

    models_to_test = [
        ("Best 36M", "/Users/ishaangubbala/Documents/Training/outputs/hz0a-mac-36m"),
        ("Tuned 110M", "/Users/ishaangubbala/Documents/Training/outputs/hz0a-mac-110m-tuned"),
        ("110M Baseline", "/Users/ishaangubbala/Documents/Training/outputs/hz0a-mac-110m-baseline"),
    ]

    all_results = {}

    for model_name, model_path in models_to_test:
        model_dir = Path(model_path)

        if not model_dir.exists():
            print(f"✗ {model_name}: Directory not found ({model_path})")
            print()
            continue

        print(f"\nTesting {model_name}...")
        print("-" * 80)

        try:
            # Load model checkpoint
            checkpoint = model_dir / "latest.pt"
            if not checkpoint.exists():
                print(f"  ✗ No checkpoint found at {checkpoint}")
                continue

            print(f"  Loading checkpoint: {checkpoint}")
            # Note: Loading checkpoint requires model class setup
            # This is a placeholder for the actual loading logic
            print(f"  ⚠ Checkpoint loading requires model class instantiation")
            print(f"  Skipping actual test (requires model config)")

        except Exception as e:
            print(f"  ✗ Error: {e}")

        print()

    print("=" * 80)
    print("DIAGNOSTICS SUITE COMPLETE")
    print("=" * 80)
    print()
    print("Note: Full implementation requires:")
    print("1. Model class instantiation from configs")
    print("2. Checkpoint loading logic")
    print("3. Evaluation harness for each model")
    print()
    print("Next step: Integrate with plan's learning-rate sweep (section 14, experiment 3)")


if __name__ == "__main__":
    run_diagnostics_suite()
