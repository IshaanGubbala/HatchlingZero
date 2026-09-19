"""Real parameter sizing for a chat_only HZLanguageModel target: <300M
total params, <50% amortized-active params per token under block
recurrence (2026-09-16 AMX-execution-geometry pass).

"Amortized active" definition (stated explicitly, not the strict per-
forward-pass MoE-style definition): local_mixer + lm_rq/rk/rv + lm_head +
token_embed run on EVERY token (100% active); ws (HZCQReasoningWorkspace)
+ mem (HZCQPersistentMemory) run once per block_size=K tokens, so their
real parameter count is divided by K to get an average per-token active
count. This is a genuine, measured consequence of the just-verified block
recurrence (Experiment 3/4), not a hypothetical -- ws/mem's GEMMs
literally do not execute for 1-of-every-K tokens. It is NOT the same as
per-forward-pass conditional compute (MoE): every token that DOES execute
a block boundary still uses the FULL ws/mem weight matrices, dense, no
routing/sparsity. That's a separate, larger architecture change, not
attempted here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.hz_language_model_torch import HZLanguageModel
from hatchling_world.language.byte_tokenizer import ByteTokenizer

ALWAYS_ACTIVE_MODULES = {"local_mixer", "lm_rq", "lm_rk", "lm_rv", "lm_head", "token_embed"}
AMORTIZED_MODULES = {"ws", "mem"}  # divided by K


def size_config(d_model, memory_slots, workspace_slots, block_mixer_heads, vocab_size, k_values):
    torch.manual_seed(0)
    model = HZLanguageModel(vocab_size=vocab_size, d_model=d_model, memory_slots=memory_slots,
                            workspace_slots=workspace_slots, n_rounds_l1=2,
                            block_mixer_heads=block_mixer_heads, chat_only=True)
    by_module = {}
    for name, p in model.named_parameters():
        top = name.split(".")[0]
        by_module[top] = by_module.get(top, 0) + p.numel()
    total = sum(by_module.values())
    always_active = sum(v for k, v in by_module.items() if k in ALWAYS_ACTIVE_MODULES)
    amortized_source = sum(v for k, v in by_module.items() if k in AMORTIZED_MODULES)
    unaccounted = total - always_active - amortized_source
    assert unaccounted == 0, f"unaccounted params in chat_only breakdown: {by_module}"

    rows = []
    for k in k_values:
        active_per_token = always_active + amortized_source / k
        rows.append({"K": k, "active_per_token": active_per_token,
                     "active_fraction": active_per_token / total})
    return {"d_model": d_model, "total_params": total, "by_module": by_module,
           "always_active": always_active, "amortized_source": amortized_source, "k_sweep": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model-values", type=int, nargs="+", default=[1536, 2048, 2560, 3072, 3584])
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--block-mixer-heads", type=int, default=8)
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32, 64])
    args = parser.parse_args()

    tok = ByteTokenizer()
    print(f"vocab_size={tok.vocab_size}, memory_slots={args.memory_slots}, "
         f"workspace_slots={args.workspace_slots}, block_mixer_heads={args.block_mixer_heads}\n")

    for d in args.d_model_values:
        result = size_config(d, args.memory_slots, args.workspace_slots, args.block_mixer_heads,
                             tok.vocab_size, args.k_values)
        total = result["total_params"]
        under_300m = "OK" if total < 300_000_000 else "OVER"
        print(f"d_model={d:5d}  total={total:>13,}  [{under_300m} <300M]")
        for row in result["k_sweep"]:
            fraction = row["active_fraction"]
            gate = "OK" if fraction < 0.5 else "over"
            print(f"    K={row['K']:3d}  active/token={row['active_per_token']:>13,.0f}  "
                 f"fraction={fraction:.1%}  [{gate} <50%]")
        print()


if __name__ == "__main__":
    main()
