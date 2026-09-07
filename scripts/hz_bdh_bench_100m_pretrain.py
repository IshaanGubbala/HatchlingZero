#!/usr/bin/env python3
"""HZ-Bench-100M, combined_best BDH variant (plans/Hatchling world.md
section 0.8 update, user-directed: "use combined_best BDH"). Real
motivation: Stage 0 with `HZLanguageModel` (the Hatchling-World-native
HZCQ persistent-memory/reasoning-workspace architecture) measured only
202 corpus tokens/sec and 22.64GB peak VRAM on a real 48GB L40S at a
mere ~300-token context -- a real, severe, measured systems bottleneck
from HZ's per-token sequential recurrence. `combined_best` BDH
(`reference/hz0h_bdh_combined_best_torch.py`) is a DIFFERENT, real,
already-validated architecture from this session's earlier BDH-
reconstruction work: batched, parallel-attention-over-the-sequence
recurrence (not token-by-token stepping), and an EARLIER real local
measurement (`results/local/combined_best_comparison.log`, a small
~3.8M-param model) found 100,433-120,957 tok/s -- 500-600x
HZLanguageModel's measured rate. This script re-tests that architecture
at real ~100M scale on real streamed, decontaminated corpus data.

Real, disclosed scope reduction: `combined_best` BDH has NO persistent
`S`/reasoning `H` structures (that is HZCQ's own extension, not part
of upstream/combined_best BDH) -- so the Knowledge/Nursery/Chat
channels built for `HZLanguageModel` cannot be reused here as-is. This
run is CORPUS-ONLY (FineWeb-Edu + Wikipedia, decontaminated), matching
Stage 0's own systems-test purpose: does combined_best solve the
throughput/memory problem at scale, at all, before deciding whether
and how to wire persistent-memory rehearsal channels into a BDH-based
lineage later.

Training uses `combined_bdh_forward(model, jump=None, idx,
real_prefix_iterations=n_layer, num_jumps=0, targets=target)` --
EXACTLY the call `scripts/hz0h_bdh_combined_best_comparison.py`'s own
`train_bdh(..., use_softmax_scaled=True)` uses, full real depth, no
jump-operator shortcut (that is an INFERENCE-time optimization,
distilled AFTER a real base model exists -- not needed for training).
Raw-byte tokenization (vocab_size=256), matching combined_best's own
established convention throughout this codebase -- not this session's
`ByteTokenizer` (which adds BOS/EOS/PAD on top of the 256 raw bytes).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.hz0h_bdh_combined_best_torch import combined_bdh_forward  # noqa: E402
from reference.hz0h_bdh_torch import BDH, BDHConfig  # noqa: E402
from hatchling_world.knowledge.decontamination import build_benchmark_ngram_hashes, DEFAULT_BENCHMARK_TASKS  # noqa: E402
from hatchling_world.knowledge.web_corpus_stream import WebCorpusMixture  # noqa: E402

PAD_BYTE = 0


def make_batch(mixture: WebCorpusMixture, batch_size: int, sequence_length: int, device: str) -> torch.Tensor:
    rows = []
    for _ in range(batch_size):
        _, text = next(mixture)
        raw = text.encode("utf-8", errors="ignore")[:sequence_length + 1]
        if len(raw) < sequence_length + 1:
            raw = raw + bytes([PAD_BYTE]) * (sequence_length + 1 - len(raw))
        rows.append(list(raw))
    return torch.tensor(rows, dtype=torch.long, device=device)


def corpus_eval_loss(model: BDH, n_layer: int, held_out_batches: list) -> float:
    losses = []
    with torch.no_grad():
        for data in held_out_batches:
            idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
            _, loss = combined_bdh_forward(model, None, idx, real_prefix_iterations=n_layer,
                                            num_jumps=0, targets=target)
            losses.append(loss.item())
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-embd", type=int, default=1440, help="from hz_bdh_bench_size_solver.py's ~100M solve")
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--mult", type=int, default=16, help="mlp_internal_dim_multiplier, combined_best recipe")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--total-steps", type=int, default=500, help="Stage 0 default: systems test")
    parser.add_argument("--eval-every", type=int, default=0, help="0 = never (Stage 0 systems-test mode)")
    parser.add_argument("--n-held-out-batches", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--fineweb-weight", type=float, default=0.9)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000)
    parser.add_argument("--benchmark-tasks", type=str, default=",".join(DEFAULT_BENCHMARK_TASKS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/local/hz_bdh_bench_100m"))
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_bdh_bench_100m_results.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[hz-bdh-bench] device={device}", flush=True)

    torch.manual_seed(args.seed)
    config = BDHConfig(n_layer=args.n_layer, n_embd=args.n_embd, n_head=args.n_head,
                        mlp_internal_dim_multiplier=args.mult, vocab_size=256, dropout=0.0)
    model = BDH(config).to(device=device, dtype=torch.float32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[hz-bdh-bench] FRESH combined_best BDH: n_embd={args.n_embd} n_layer={args.n_layer} "
          f"n_head={args.n_head} mult={args.mult} n_params={n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("[hz-bdh-bench] building benchmark decontamination hashes...", flush=True)
    benchmark_tasks = args.benchmark_tasks.split(",")
    benchmark_hashes = build_benchmark_ngram_hashes(benchmark_tasks)
    print(f"[hz-bdh-bench] {len(benchmark_hashes)} benchmark n-gram hashes across {benchmark_tasks}", flush=True)

    corpus_mix = WebCorpusMixture(benchmark_hashes=benchmark_hashes, fineweb_weight=args.fineweb_weight,
                                   seed=args.seed, chunk_chars=args.sequence_length + 1,
                                   shuffle_buffer_size=args.shuffle_buffer_size)
    held_out_mix = WebCorpusMixture(benchmark_hashes=benchmark_hashes, fineweb_weight=args.fineweb_weight,
                                     seed=args.seed + 777, chunk_chars=args.sequence_length + 1,
                                     shuffle_buffer_size=args.shuffle_buffer_size)
    held_out_batches = [make_batch(held_out_mix, args.batch_size, args.sequence_length, device)
                         for _ in range(args.n_held_out_batches)]
    print(f"[hz-bdh-bench] captured {len(held_out_batches)} frozen held-out batches "
          f"(seed+777, never trained on)", flush=True)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    corpus_tokens_seen = 0
    eval_points = []
    t0 = time.time()

    for step in range(args.total_steps):
        data = make_batch(corpus_mix, args.batch_size, args.sequence_length, device)
        idx, target = data[:, :-1].contiguous(), data[:, 1:].contiguous()
        opt.zero_grad(set_to_none=True)
        _, loss = combined_bdh_forward(model, None, idx, real_prefix_iterations=args.n_layer,
                                        num_jumps=0, targets=target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        corpus_tokens_seen += args.batch_size * args.sequence_length

        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / elapsed
            tokens_per_sec = corpus_tokens_seen / elapsed
            vram = (f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB" if device == "cuda" else "n/a")
            print(f"[hz-bdh-bench] step={step+1}/{args.total_steps} ({elapsed:.1f}s, "
                  f"{steps_per_sec:.2f} steps/sec, {tokens_per_sec:.0f} corpus tok/sec) "
                  f"loss={loss.item():.4f} corpus_tokens_seen={corpus_tokens_seen:,} peak_vram={vram} "
                  f"decon_rejected={corpus_mix.stats['rejected_contaminated']} "
                  f"corpus_sources={corpus_mix.stats['fineweb_docs']}fw/{corpus_mix.stats['wiki_docs']}wiki",
                  flush=True)

        if args.eval_every and (step + 1) % args.eval_every == 0:
            ckpt_path = args.checkpoint_dir / f"step_{step+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            held_out_loss = corpus_eval_loss(model, args.n_layer, held_out_batches)
            estimated_flops = 6 * n_params * corpus_tokens_seen
            elapsed = time.time() - t0
            print(f"[hz-bdh-bench] EVAL @ step={step+1}: held_out_loss={held_out_loss:.4f} "
                  f"corpus_tokens_seen={corpus_tokens_seen:,} estimated_flops={estimated_flops:.3e}", flush=True)
            eval_points.append({
                "step": step + 1, "held_out_loss": held_out_loss, "corpus_tokens_seen": corpus_tokens_seen,
                "estimated_flops": estimated_flops, "steps_per_sec": (step + 1) / elapsed,
                "tokens_per_sec": corpus_tokens_seen / elapsed,
                "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None),
            })
            with open(args.results_file, "w") as f:
                json.dump({"n_params": n_params, "n_embd": args.n_embd, "n_layer": args.n_layer,
                            "n_head": args.n_head, "mult": args.mult, "device": device,
                            "eval_points": eval_points}, f, indent=2, default=str)

    total_time = time.time() - t0
    vram = (f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB" if device == "cuda" else "n/a")
    print(f"\n[hz-bdh-bench] DONE: {args.total_steps} steps in {total_time:.1f}s "
          f"({args.total_steps/total_time:.2f} steps/sec, {corpus_tokens_seen/total_time:.0f} corpus tok/sec, "
          f"peak_vram={vram}, corpus_tokens_seen={corpus_tokens_seen:,}, "
          f"estimated_flops={6*n_params*corpus_tokens_seen:.3e})", flush=True)
    final_ckpt = args.checkpoint_dir / "final.pt"
    torch.save(model.state_dict(), final_ckpt)
    print(f"[hz-bdh-bench] final checkpoint: {final_ckpt}", flush=True)

    if not args.eval_every:
        with open(args.results_file, "w") as f:
            json.dump({"n_params": n_params, "n_embd": args.n_embd, "n_layer": args.n_layer,
                        "n_head": args.n_head, "mult": args.mult, "device": device,
                        "mode": "stage0_systems_test", "total_steps": args.total_steps,
                        "total_seconds": total_time, "steps_per_sec": args.total_steps / total_time,
                        "corpus_tokens_seen": corpus_tokens_seen, "tokens_per_sec": corpus_tokens_seen / total_time,
                        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None),
                        "decon_rejected": corpus_mix.stats["rejected_contaminated"]}, f, indent=2, default=str)
        print(f"[hz-bdh-bench] wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
