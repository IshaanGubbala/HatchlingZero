"""Experiment 3/4's missing axis: real corpus quality comparison between
lm_forward (per-token H) and lm_forward_blocked at several K, on the
REAL 25M-byte corpus (data/packed/hz0h_bytes_25m_train.jsonl) already
used by scripts/hz_world_training_run1_stage_a_corpus.py.

Reuses that script's CorpusPool/corpus_eval exactly (real, already-
validated data loading) -- only the train step and model forward path
change (lm_forward vs lm_forward_blocked at a given K).

This is a real, if small-scale (few thousand steps, small d_model),
train run -- not a systems-only benchmark. Same random init, same data,
same step count, same optimizer/lr across arms, for a fair comparison.
Local CPU/MPS scale (small model), not the eventual <300M-param target --
this answers "does block recurrence cost quality at all, and how much,
as a function of K" before committing to a larger run.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hz_world_training_run1_stage_a_corpus import CorpusPool  # noqa: E402
from hz2_patch_prepare_data import BinaryCorpusPool  # noqa: E402
from hz_bpe_prepare_data import BinaryTokenCorpusPool  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hz_training_metrics import TrainingMetricsLogger  # noqa: E402


def train_step(model, opt, pool, rng, vocab_size, block_size, device, batch_size=1):
    ids_batch = [pool.sample(rng) for _ in range(batch_size)]
    token_ids = torch.tensor(ids_batch, device=device)
    if block_size is None:
        logits = model.lm_forward(token_ids)
    else:
        logits = model.lm_forward_blocked(token_ids, block_size=block_size)
    target = token_ids[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        acc = (logits.argmax(-1) == target).float().mean().item()
    return loss.item(), acc


@torch.no_grad()
def eval_loss_and_acc(model, val_pool, rng, vocab_size, n_windows, block_size, device, batch_size=1):
    total_loss, correct, total = 0.0, 0, 0
    n_batches = max(1, n_windows // batch_size)
    for _ in range(n_batches):
        ids_batch = [val_pool.sample(rng) for _ in range(batch_size)]
        token_ids = torch.tensor(ids_batch, device=device)
        if block_size is None:
            logits = model.lm_forward(token_ids)
        else:
            logits = model.lm_forward_blocked(token_ids, block_size=block_size)
        target = token_ids[:, 1:]
        total_loss += F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1),
                                      reduction="sum").item()
        correct += int((logits.argmax(-1) == target).float().sum().item())
        total += target.numel()
    return total_loss / total, correct / total


def lr_at_step(step, total_steps, peak_lr, warmup_fraction=0.05, min_lr_fraction=0.1):
    """5% warmup (linear) then cosine decay to min_lr_fraction * peak_lr,
    matching the convention already stated in
    plans/HZ_Mixed_Depth_Pilot_2026-09-15.md's data-training section
    ("5% warmup, cosine decay to 0.1x peak") -- applied here since the
    first long run (12,000 steps, flat lr=3e-4) showed a noisy, plateaued
    loss curve (2.55-2.68 for ~10,000 steps) that a proper schedule is
    the standard, well-evidenced first fix for before assuming the
    architecture itself has hit a real ceiling."""
    import math
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_fraction + (1 - min_lr_fraction) * cosine)


def save_model_checkpoint(model, opt, args, label, step, path):
    """Real gap this fixes: the first long (3.48hr) committed run had NO
    checkpoint-saving anywhere in this script -- only the loss/accuracy
    history was recorded, so that run's trained weights were unrecoverable
    once the process exited. Saves everything a fresh HZLanguageModel
    needs to be reconstructed exactly (constructor kwargs + state_dict),
    not just raw tensors -- see load_model_checkpoint in
    scripts/hz_generate_sample.py. Also saves optimizer state (added
    2026-09-17 for the GPU-migration resume case below) -- checkpoints
    saved BEFORE this change won't have it, so resuming from one of those
    restarts AdamW's moment estimates fresh, a real, disclosed gap."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "config": {"vocab_size": model.vocab_size, "d_model": args.d_model, "memory_slots": args.memory_slots,
                  "workspace_slots": args.workspace_slots, "n_rounds_l1": 2,
                  "block_mixer_heads": args.block_mixer_heads, "chat_only": args.chat_only},
        "tokenizer_dir": str(args.tokenizer_dir) if getattr(args, "tokenizer_dir", None) else None,
        "corpus_format": args.corpus_format,
        "label": label, "step": step,
    }, path)


def run_arm(label, block_size, args, vocab_size, corpus_pool, val_pool, device):
    torch.manual_seed(args.seed)
    model = HZLanguageModel(vocab_size=vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                            workspace_slots=args.workspace_slots, n_rounds_l1=2,
                            block_mixer_heads=args.block_mixer_heads, chat_only=args.chat_only).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(args.seed + 1)
    eval_rng = random.Random(args.seed + 2)

    start_step = 0
    if args.resume_from:
        print(f"resuming from checkpoint: {args.resume_from}", flush=True)
        payload = torch.load(args.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(payload["state_dict"])
        start_step = payload["step"] + 1
        if "optimizer_state_dict" in payload:
            opt.load_state_dict(payload["optimizer_state_dict"])
            print("  loaded optimizer state (AdamW moments preserved)", flush=True)
        else:
            print("  WARNING: checkpoint has no optimizer_state_dict (older format) -- "
                 "AdamW moment estimates restart fresh from this step, a real, disclosed gap", flush=True)
        print(f"  resuming at step {start_step} (checkpoint was saved at step {payload['step']})", flush=True)

    metrics_logger = None
    landscape_batch = None
    if args.metrics_out:
        metrics_logger = TrainingMetricsLogger(model, args.metrics_out,
                                               landscape_every_evals=args.landscape_every_evals)
        landscape_rng = random.Random(args.seed + 3)
        landscape_ids = torch.tensor([val_pool.sample(landscape_rng) for _ in range(8)], device=device)
        landscape_target = landscape_ids[:, 1:]

        def eval_batch_fn():
            with torch.no_grad():
                if block_size is None:
                    logits = model.lm_forward(landscape_ids)
                else:
                    logits = model.lm_forward_blocked(landscape_ids, block_size=block_size)
                return F.cross_entropy(logits.reshape(-1, vocab_size), landscape_target.reshape(-1)).item()
        landscape_batch = eval_batch_fn

    history = []
    t0 = time.time()
    for step in range(start_step, args.steps):
        if args.lr_schedule:
            current_lr = lr_at_step(step, args.steps, args.lr)
            for group in opt.param_groups:
                group["lr"] = current_lr
        else:
            current_lr = args.lr
        loss, acc = train_step(model, opt, corpus_pool, rng, vocab_size, block_size, device, args.batch_size)
        if step % args.eval_every == 0 or step == args.steps - 1:
            val_loss, val_acc = eval_loss_and_acc(model, val_pool, eval_rng, vocab_size,
                                                   args.eval_windows, block_size, device, args.batch_size)
            elapsed = time.time() - t0
            print(f"[{label}] step {step:4d}  lr={current_lr:.2e}  train_loss={loss:.4f} train_acc={acc:.4f}  "
                 f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}  elapsed={elapsed:.1f}s", flush=True)
            history.append({"step": step, "lr": current_lr, "train_loss": loss, "train_acc": acc,
                           "val_loss": val_loss, "val_acc": val_acc, "elapsed_seconds": elapsed})
            if metrics_logger is not None:
                metrics_logger.log(step=step, train_loss=loss, train_acc=acc, val_loss=val_loss,
                                   val_acc=val_acc, lr=current_lr, elapsed=elapsed, model=model,
                                   eval_batch_fn=landscape_batch)
        if args.checkpoint_out and (step % args.checkpoint_every == 0 or step == args.steps - 1) and step > 0:
            save_model_checkpoint(model, opt, args, label, step, args.checkpoint_out)
            print(f"  saved checkpoint at step {step} to {args.checkpoint_out}", flush=True)
    return {"label": label, "block_size": block_size, "history": history,
           "final_val_loss": history[-1]["val_loss"], "final_val_acc": history[-1]["val_acc"],
           "total_seconds": history[-1]["elapsed_seconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--block-mixer-heads", type=int, default=4)
    parser.add_argument("--chat-only", action="store_true",
                        help="Skip L1-L6 Nursery-curriculum heads (see HZLanguageModel's chat_only flag).")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Windows per training step -- real minibatching (previously always 1), added "
                             "for the long-commit run: reduces single-window gradient noise and better "
                             "utilizes the GPU on a wide (d_model=3072) model.")
    parser.add_argument("--lr-schedule", action="store_true",
                        help="5%% linear warmup + cosine decay to 0.1x peak lr (see lr_at_step), instead of "
                             "a flat lr -- added after the first long run's noisy, plateaued loss curve.")
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-windows", type=int, default=50)
    parser.add_argument("--checkpoint-out", type=Path, default=None,
                        help="Save model checkpoint (state_dict + reconstruction config) periodically and "
                             "at the end -- real gap fixed 2026-09-17: the earlier 3.48hr committed run had "
                             "no checkpoint saving at all, so its trained weights were unrecoverable once "
                             "the process exited.")
    parser.add_argument("--checkpoint-every", type=int, default=1000,
                        help="Save --checkpoint-out every this many steps (in addition to at the end).")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--corpus-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--corpus-val-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--corpus-max-lines", type=int, default=20000)
    parser.add_argument("--corpus-val-max-lines", type=int, default=2000)
    parser.add_argument("--corpus-format", choices=["jsonl", "binary", "binary_bpe"], default="jsonl",
                        help="jsonl = hz_world_training_run1_stage_a_corpus.CorpusPool (one int-list per "
                             "line). binary = hz2_patch_prepare_data.BinaryCorpusPool (fixed-length raw "
                             "uint8, memmap'd). binary_bpe = hz_bpe_prepare_data.BinaryTokenCorpusPool "
                             "(fixed-length uint16 BPE token ids, memmap'd -- requires --vocab-size to match "
                             "the tokenizer's actual_vocab_size from its manifest.json).")
    parser.add_argument("--corpus-sequence-length", type=int, default=256,
                        help="Required for --corpus-format binary/binary_bpe (bytes for binary, tokens for "
                             "binary_bpe; jsonl format doesn't need this).")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="Required for --corpus-format binary_bpe -- the tokenizer's actual vocab size "
                             "(see manifest.json's vocab_size_actual from hz_bpe_prepare_data.py). Ignored "
                             "for jsonl/binary, which use ByteTokenizer's fixed 260-value vocab.")
    parser.add_argument("--metrics-out", type=Path, default=None,
                        help="Live-training instrumentation for the artifact dashboard (2026-09-17): "
                             "appends one JSONL record per eval step with weight-matrix heatmaps and a "
                             "loss-landscape/optimizer-trajectory projection (see hz_training_metrics.py). "
                             "None (default) disables this -- it is not free (landscape evals add real "
                             "forward passes periodically).")
    parser.add_argument("--landscape-every-evals", type=int, default=10,
                        help="Compute the (more expensive) loss-landscape grid every this many eval steps; "
                             "the cheap weight-norm/heatmap/trajectory point is still logged every eval step.")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Resume from a checkpoint saved by --checkpoint-out (added 2026-09-17 for the "
                             "GPU-migration case: pause on one pod's low-VRAM card, resume on another with "
                             "more headroom). Continues the step counter and LR schedule from where the "
                             "checkpoint left off; restores optimizer state too if the checkpoint has it "
                             "(checkpoints saved before this flag existed won't -- AdamW moments restart "
                             "fresh in that case, a real, disclosed gap, not a silent one).")
    parser.add_argument("--tokenizer-dir", type=Path, default=None,
                        help="BPE tokenizer directory (vocab.json/merges.txt), recorded into the checkpoint "
                             "so hz_generate_sample.py can load the matching tokenizer. Only meaningful for "
                             "--corpus-format binary_bpe.")
    parser.add_argument("--k-values", type=int, nargs="+", default=[8, 16, 32],
                        help="Block sizes to compare against per-token (K=1-equivalent) lm_forward.")
    parser.add_argument("--out", type=Path, default=Path("results/local/hz_block_recurrence_quality.json"))
    parser.add_argument("--skip-per-token", action="store_true",
                        help="Skip the per_token (K=1-equivalent-frequency, no local_mixer control) baseline "
                             "arm -- at large d_model this is prohibitively slow and, once the K-vs-per_token "
                             "question is already answered (see the earlier controlled comparison), not needed "
                             "for a pure screening run of a specific target config.")
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    print(f"device={args.device}")

    if args.corpus_format == "binary_bpe":
        if args.vocab_size is None:
            raise SystemExit("--vocab-size is required for --corpus-format binary_bpe "
                             "(read it from the corpus manifest.json's vocab_size_actual)")
        vocab_size = args.vocab_size
        print(f"loading real corpus (binary_bpe, memmap): {args.corpus_data}...")
        corpus_pool = BinaryTokenCorpusPool(args.corpus_data, args.corpus_sequence_length)
        val_pool = BinaryTokenCorpusPool(args.corpus_val_data, args.corpus_sequence_length)
        print(f"loaded {corpus_pool.n_sequences} train sequences, {val_pool.n_sequences} val sequences, "
             f"sequence length={args.corpus_sequence_length} tokens, vocab_size={vocab_size}")
    elif args.corpus_format == "binary":
        vocab_size = ByteTokenizer().vocab_size
        print(f"loading real corpus (binary, memmap): {args.corpus_data}...")
        corpus_pool = BinaryCorpusPool(args.corpus_data, args.corpus_sequence_length)
        val_pool = BinaryCorpusPool(args.corpus_val_data, args.corpus_sequence_length)
        print(f"loaded {corpus_pool.n_sequences} train sequences, {val_pool.n_sequences} val sequences, "
             f"sequence length={args.corpus_sequence_length} bytes")
    else:
        vocab_size = ByteTokenizer().vocab_size
        print(f"loading real corpus (jsonl): {args.corpus_data} (max {args.corpus_max_lines} windows)...")
        corpus_pool = CorpusPool(args.corpus_data, args.corpus_max_lines)
        val_pool = CorpusPool(args.corpus_val_data, args.corpus_val_max_lines)
        print(f"loaded {len(corpus_pool.windows)} train windows, {len(val_pool.windows)} val windows, "
             f"window length={len(corpus_pool.windows[0])} bytes")

    results = [] if args.skip_per_token else [run_arm("per_token", None, args, vocab_size, corpus_pool, val_pool, args.device)]
    for k in args.k_values:
        results.append(run_arm(f"K={k}", k, args, vocab_size, corpus_pool, val_pool, args.device))

    print("\n=== summary ===")
    baseline = results[0]
    for r in results:
        delta = r["final_val_loss"] - baseline["final_val_loss"]
        speedup = baseline["total_seconds"] / r["total_seconds"]
        print(f"{r['label']:>10}: final_val_loss={r['final_val_loss']:.4f} "
             f"(delta vs per_token: {delta:+.4f})  final_val_acc={r['final_val_acc']:.4f}  "
             f"total_time={r['total_seconds']:.1f}s  speedup={speedup:.2f}x")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"config": vars(args) | {"corpus_data": str(args.corpus_data),
                                                             "corpus_val_data": str(args.corpus_val_data),
                                                             "out": str(args.out)},
                                    "results": results}, indent=2, default=str) + "\n")
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
