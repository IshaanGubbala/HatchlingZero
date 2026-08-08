"""HZ-0G G5: the real Dense vs. MoE vs. domain-adapter comparison,
on the full A+B+C+D+(E) integration, at real (not smoke-test) scale.

Matches reference/hz0e_e8_curriculum.py's own established step counts
(balanced=50, mixed=50, imbalanced=50, warm_start=40) so this is a real,
comparable-in-spirit rigor level, not a shortcut.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0e_e6_integration import TARGET_LAYERS
from reference.hz0e_moe_contract import MoeConfig
from reference.hz0g_g5_curriculum import (
    evaluate_integrated_dense_per_domain, evaluate_integrated_moe_per_domain,
    integrated_loss, run_integrated_dense_baseline, run_integrated_moe_curriculum,
)
from scripts.hz0b_b11_baseline_comparison import load_frozen_model
from scripts.hz0c_c3_trigger_simulator import load_real_sequences
import mlx.core as mx


def main() -> None:
    model, payload = load_frozen_model()
    print(f"loaded frozen checkpoint: step={payload['step']} tokens_seen={payload['tokens_seen']}", flush=True)
    config = MoeConfig(dim=model.dim)

    print("\n=== HZ-MoE: real curriculum (balanced=50, mixed=50, imbalanced=50, warm_start=40) ===", flush=True)
    trained_layers, pure_dense_val, warm_val, moe_after_val = run_integrated_moe_curriculum(
        model, config, balanced_steps=50, mixed_steps=50, imbalanced_steps=50, warm_start_steps=40, seed=0,
    )
    print(f"  HZ-Dense (no MoE) general-prose val: {pure_dense_val:.6f}", flush=True)
    print(f"  HZ-MoE warm-start-only general-prose val: {warm_val:.6f}", flush=True)
    print(f"  HZ-MoE after-curriculum general-prose val: {moe_after_val:.6f}", flush=True)

    moe_per_domain = evaluate_integrated_moe_per_domain(model, trained_layers, seed=0)
    print(f"  HZ-MoE per-domain held-out loss: {json.dumps(moe_per_domain, indent=2)}", flush=True)
    moe_per_domain_mean = sum(moe_per_domain.values()) / len(moe_per_domain)
    print(f"  HZ-MoE per-domain mean: {moe_per_domain_mean:.6f}", flush=True)

    print("\n=== Dense + domain adapter: real curriculum (same step counts) ===", flush=True)
    flat_params = run_integrated_dense_baseline(
        model, balanced_steps=50, mixed_steps=50, imbalanced_steps=50, seed=0,
    )
    dense_per_domain = evaluate_integrated_dense_per_domain(model, flat_params, seed=0)
    print(f"  Dense+adapter per-domain held-out loss: {json.dumps(dense_per_domain, indent=2)}", flush=True)
    dense_per_domain_mean = sum(dense_per_domain.values()) / len(dense_per_domain)
    print(f"  Dense+adapter per-domain mean: {dense_per_domain_mean:.6f}", flush=True)

    general_val = [mx.array([s[:64]]) for s in load_real_sequences("data/packed/repro_1024_val.jsonl", 10)]
    from reference.hz0g_g5_curriculum import _dense_loss
    dense_general_losses = [float(_dense_loss(model, flat_params, tb, TARGET_LAYERS, 0)) for tb in general_val]
    dense_general_val = sum(dense_general_losses) / len(dense_general_losses)
    print(f"  Dense+adapter general-prose val: {dense_general_val:.6f}", flush=True)

    print("\n=== Summary ===", flush=True)
    print(f"  HZ-Dense (no E at all):        general-prose {pure_dense_val:.6f}", flush=True)
    print(f"  HZ-MoE (trained):              general-prose {moe_after_val:.6f}   per-domain-mean {moe_per_domain_mean:.6f}", flush=True)
    print(f"  Dense+domain-adapter (trained): general-prose {dense_general_val:.6f}   per-domain-mean {dense_per_domain_mean:.6f}", flush=True)


if __name__ == "__main__":
    main()
