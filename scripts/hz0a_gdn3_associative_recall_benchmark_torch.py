"""Torch/CUDA port of `scripts/hz0a_gdn3_associative_recall_benchmark.py`
-- runs on cpu/mps/cuda, so this can execute on the RTX 3060 (MLX cannot).
Same task, same fair-comparison discipline, same parameter-matched mixers.

Also adds `--seeds` for multi-seed replication -- the explicit next step
named in `docs/restart/hz0a_gdn3_associative_recall_results.md` ("multi-
seed replication... before any retrain decision, not simply trusting this
one corrected run either") after the first single-seed MLX result flipped
once two testing confounds were fixed.
"""
from __future__ import annotations

import argparse
import random
import statistics

import torch
from torch import nn

from reference.hz0a_gdn3_tiny_lm_torch import TinyGDNLMTorch

VOCAB_SIZE = 512
NUM_KEYS, NUM_VALUES = 8, 8
DISTRACTOR_LOW, DISTRACTOR_HIGH = 100, 500
QUERY_TOKEN = 30
SEQ_LEN, BATCH_SIZE = 48, 32
STEPS, LR = 3000, 3e-4


def make_batch(rng: random.Random, batch_size: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for _ in range(batch_size):
        current_value = {}
        events = []
        keys_order = list(range(NUM_KEYS))
        rng.shuffle(keys_order)
        for key in keys_order:
            value = rng.randrange(NUM_VALUES)
            current_value[key] = value
            events.append((10 + key, 20 + value))
        for _ in range(rng.randint(2, 4)):
            key = rng.randrange(NUM_KEYS)
            value = rng.randrange(NUM_VALUES)
            current_value[key] = value
            events.append((10 + key, 20 + value))
        rng.shuffle(events)

        row = []
        for key_tok, value_tok in events:
            row.append(rng.randint(DISTRACTOR_LOW, DISTRACTOR_HIGH))
            row.append(key_tok)
            row.append(value_tok)
        query_key = rng.randrange(NUM_KEYS)
        assert len(row) + 2 <= SEQ_LEN - 1, "events + query overflow SEQ_LEN"
        row.append(QUERY_TOKEN)
        row.append(10 + query_key)
        row += [rng.randint(DISTRACTOR_LOW, DISTRACTOR_HIGH) for _ in range(SEQ_LEN - 1 - len(row))]
        row.append(20 + current_value[query_key])
        rows.append(row)
    tokens = torch.tensor(rows, dtype=torch.long, device=device)
    return tokens, tokens[:, -1]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def run(use_candidate: bool, seed: int, device: torch.device, steps: int, *, dim: int = 64, layers: int = 4, heads: int = 4, d_ff: int = 128, log_every: int = 0, label: str = "") -> float:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    model = TinyGDNLMTorch(VOCAB_SIZE, dim, layers, heads, d_ff, use_candidate).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    eval_tokens, eval_targets = make_batch(random.Random(seed + 1), 256, device)

    for step in range(steps):
        tokens, _ = make_batch(rng, BATCH_SIZE, device)
        logits, _ = model(tokens)
        final_logits = logits[:, -2, :]
        targets = tokens[:, -1]
        loss = nn.functional.cross_entropy(final_logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            with torch.no_grad():
                eval_logits, _ = model(eval_tokens)
                eval_acc = (eval_logits[:, -2, :].argmax(dim=-1) == eval_targets).float().mean().item()
            print(f"[{label}] step {step:5d}  train_loss {loss.item():.4f}  eval_acc {eval_acc:.4f}")

    with torch.no_grad():
        logits, _ = model(eval_tokens)
        predicted = logits[:, -2, :].argmax(dim=-1)
        return (predicted == eval_targets).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=[999, 1000, 1001])
    parser.add_argument("--dim", type=int, default=64, help="model width -- both arms always get matched parameter counts regardless of this value")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=0, help="print periodic train_loss/eval_acc during each run; 0 disables")
    parser.add_argument("--compile-step", action="store_true", help="torch.compile each mixer's whole-chunk recurrence loop (the same validated, exact -- 0.0 diff -- technique from scripts/hz0a_torch_stage2_runner.py's own --compile-step: reference/hz0a_torch_model.py's GDN2Mixer._seq_fn and reference/hz0a_gdn3_candidate_mixer_torch.py's GDN3CandidateMixerTorch._seq_fn). Applied identically to BOTH arms of the comparison -- this speeds up wall-clock time, it does not change the experiment's controlled variables (LR, batch size, optimizer, dtype all stay exactly as originally specified), so it does not confound comparability with prior seeds/results.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    if args.compile_step:
        from reference.hz0a_torch_model import GDN2Mixer, _gdn2_sequential
        from reference.hz0a_gdn3_candidate_mixer_torch import GDN3CandidateMixerTorch, _gdn3_sequential
        GDN2Mixer._seq_fn = staticmethod(torch.compile(_gdn2_sequential))
        GDN3CandidateMixerTorch._seq_fn = staticmethod(torch.compile(_gdn3_sequential))
    print(f"device={device} steps={args.steps} seeds={args.seeds} dim={args.dim} layers={args.layers} heads={args.heads} d_ff={args.d_ff} compile_step={args.compile_step}")

    current_accuracies, candidate_accuracies = [], []
    for seed in args.seeds:
        acc_current = run(use_candidate=False, seed=seed, device=device, steps=args.steps, dim=args.dim, layers=args.layers, heads=args.heads, d_ff=args.d_ff, log_every=args.log_every, label=f"current seed={seed}")
        acc_candidate = run(use_candidate=True, seed=seed, device=device, steps=args.steps, dim=args.dim, layers=args.layers, heads=args.heads, d_ff=args.d_ff, log_every=args.log_every, label=f"candidate seed={seed}")
        current_accuracies.append(acc_current)
        candidate_accuracies.append(acc_candidate)
        print(f"seed={seed}  current={acc_current:.4f}  candidate={acc_candidate:.4f}  diff={acc_candidate-acc_current:+.4f}")

    print("\n=== summary across seeds ===")
    print(f"current GDN2:    mean={statistics.mean(current_accuracies):.4f}  stdev={statistics.pstdev(current_accuracies):.4f}  values={[round(v,4) for v in current_accuracies]}")
    print(f"GDN-3 candidate: mean={statistics.mean(candidate_accuracies):.4f}  stdev={statistics.pstdev(candidate_accuracies):.4f}  values={[round(v,4) for v in candidate_accuracies]}")
    diff = statistics.mean(candidate_accuracies) - statistics.mean(current_accuracies)
    print(f"mean difference (candidate - current): {diff:+.4f}")
    wins = sum(1 for c, g in zip(current_accuracies, candidate_accuracies) if g > c)
    print(f"candidate wins in {wins}/{len(args.seeds)} seeds")


if __name__ == "__main__":
    main()
