#!/usr/bin/env python3
"""Real width/quality/FLOP frontier for exact BDH: is the 32x latent width
multiplier actually load-bearing, or is it inherited from the paper and
never validated at this project's scale?

Why this experiment, and why it is NOT the thing that already failed
twice: the cost-breakdown profile (`scripts/hz0h_bdh_cost_breakdown_profile.py`)
found that BDH's three wide projections (encoder, encoder_v, decoder) are
**82.8% of per-level FLOPs** at 27.6% each, while attention -- despite
running at the expanded 2048 width -- is only 17.2%. Those three
projections scale LINEARLY with `mlp_internal_dim_multiplier`, so halving
it removes roughly 41% of total FLOPs directly, with no router, no
gather/scatter, and no runtime sparsity decision.

That distinction matters given this project's real track record. Every
prior attempt to cut BDH's cost by exploiting its ~5% activation sparsity
at RUNTIME lost: Phase 4's BlockBDH router (real speedup, quality
collapse, 3 failed fix attempts) and this session's dynamic block routing
(slower AND worse). The one structural cost reduction that WON --
PackedBlockBDH -- won precisely because it removes width statically,
offline, with no per-forward decision. This experiment is in that same
proven-working family: change the architecture's real dimensions up
front, then train and measure honestly. It adds no mechanism at all.

Method: identical recipe to the established curriculum comparison
(`scripts/hz0h_factorized_curriculum_full_comparison.py`, whose own
functions are IMPORTED here rather than re-implemented so the recipe
cannot silently drift) -- real 25M-token byte corpus, the canonical
training-only 2->4->6->8 recurrent-depth curriculum, one arm at a time,
same optimizer/schedule/seed. Results are therefore directly comparable
to that comparison's own real numbers (dense BDH 1.3848, matched
Transformer 1.5141).

Real, disclosed: lowering the multiplier lowers BOTH FLOPs and parameter
count -- these are not parameter-matched arms, and that is deliberate and
stated rather than hidden. The real question being asked is "does BDH
need this much width to keep its quality win," not "is width free."
A narrower BDH that still beats the Transformer would be a strictly
better result (fewer FLOPs AND fewer parameters); a narrower BDH that
loses tells us the width is genuinely load-bearing and the FLOP gap is
structural.

The curriculum is mandatory here, not optional: FactorizedBDH's own
short, no-curriculum probe found a quality "edge" that reversed
completely once trained properly
(`docs/restart/hz0h_factorized_curriculum_full_comparison_results.md`).
Any width verdict from a non-curriculum run would be untrustworthy for
exactly that reason.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0a_matched_transformer import MatchedTransformerConfig, MatchedTransformerLM
from reference.hz0h_bdh_torch import BDH, BDHConfig
from scripts.hz0h_bdh_cost_breakdown_profile import analytic_flops, matched_transformer_flops
from scripts.hz0h_factorized_curriculum_full_comparison import parse_stages, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--validation-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=25_000_000)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--validation-batch-size", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-embd", type=int, default=512)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--multipliers", default="32,16,8,4",
                        help="Real latent width multipliers to sweep. 32 is the canonical/baseline value; "
                             "4 is the most interesting arm -- it puts BDH within 1.85x of the matched "
                             "Transformer's forward FLOPs at only 3.41M parameters (vs the Transformer's ~25M).")
    parser.add_argument("--skip-transformer", action="store_true",
                        help="Skip the Transformer arm (e.g. when reusing the established 1.5141 number).")
    parser.add_argument("--curriculum-stages", default="6250000:2,12500000:4,18750000:6,25000000:8")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("this comparison requires real CUDA hardware")
    device = torch.device("cuda")
    dtype = torch.bfloat16
    stages = parse_stages(args.curriculum_stages)
    multipliers = [int(value) for value in args.multipliers.split(",")]

    # `train` from the imported harness reads its settings off one namespace;
    # build it here so every arm provably shares the identical recipe.
    def arm_args(multiplier: int) -> SimpleNamespace:
        return SimpleNamespace(
            data=args.data, validation_data=args.validation_data, target_tokens=args.target_tokens,
            batch_size=args.batch_size, sequence_length=args.sequence_length,
            validation_batch_size=args.validation_batch_size, eval_every=args.eval_every,
            eval_batches=args.eval_batches, warmup_steps=args.warmup_steps,
            learning_rate=args.learning_rate, seed=args.seed, n_embd=args.n_embd,
            n_layer=args.n_layer, n_head=args.n_head, mlp_internal_dim_multiplier=multiplier,
        )

    arms = {}
    for multiplier in multipliers:
        torch.manual_seed(args.seed)
        config = BDHConfig(
            n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
            mlp_internal_dim_multiplier=multiplier, vocab_size=256, dropout=0.0,
        )
        model = BDH(config).to(device=device, dtype=dtype)
        model.attn.freqs = model.attn.freqs.to(torch.float32)
        name = f"bdh_mult_{multiplier}"
        result = train(name, model, "dense", arm_args(multiplier), stages)

        latent_width = args.n_embd * multiplier // args.n_head
        level_flops = analytic_flops(
            args.batch_size, args.sequence_length, args.n_embd, args.n_head, latent_width,
        )
        result["latent_width_per_head"] = latent_width
        result["mlp_internal_dim_multiplier"] = multiplier
        result["forward_flops_at_full_depth"] = sum(level_flops.values()) * args.n_layer
        arms[name] = result
        del model
        torch.cuda.empty_cache()

    if not args.skip_transformer:
        torch.manual_seed(args.seed)
        transformer_config = MatchedTransformerConfig({
            "vocab_size": 256, "d_model": args.n_embd, "num_layers": 6, "num_heads": 4,
            "head_dim": args.n_embd // 4, "d_ff": 2048, "use_rope": True,
        })
        transformer = MatchedTransformerLM(transformer_config).to(device=device, dtype=dtype)
        result = train("matched_transformer", transformer, "transformer", arm_args(args.n_head), stages)
        result["forward_flops_at_full_depth"] = matched_transformer_flops(
            args.batch_size, args.sequence_length, args.n_embd, 6, 4, args.n_embd // 4, 2048,
        )["whole_model"]
        arms["matched_transformer"] = result
        del transformer
        torch.cuda.empty_cache()

    baseline = arms.get(f"bdh_mult_{multipliers[0]}")
    for name, arm in arms.items():
        if baseline and arm.get("forward_flops_at_full_depth"):
            arm["flops_vs_widest_bdh"] = arm["forward_flops_at_full_depth"] / baseline["forward_flops_at_full_depth"]
            arm["validation_loss_minus_widest_bdh"] = arm["best_validation_loss"] - baseline["best_validation_loss"]

    report = {
        "device": "cuda",
        "hardware": torch.cuda.get_device_name(device),
        "dtype": "bfloat16",
        "curriculum_stages": stages,
        "target_tokens": args.target_tokens,
        "multipliers_swept": multipliers,
        "reference_numbers_from_established_curriculum_comparison": {
            "dense_bdh_mult_32": 1.3848,
            "matched_transformer": 1.5141,
            "source": "docs/restart/hz0h_factorized_curriculum_full_comparison_results.md",
        },
        "arms": arms,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
