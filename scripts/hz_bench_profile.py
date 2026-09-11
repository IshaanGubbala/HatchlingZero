"""Profile the real HZLanguageModel corpus-training step (plan doc
section 0.12, HZCQ-BR proposal, step 1: "profile current HZ on CUDA --
real kernel-launch/GEMM/round timing breakdown, establishes where the 86
tok/sec is actually going before changing anything").

Real, disclosed reason this is a separate script rather than a
--profile flag on hz_bench_100m_pretrain.py: the two things worth
measuring are genuinely different experiments --
  (a) pure model compute (forward+backward on a fixed-length token
      sequence, no network) -- isolates the T*R serial-recurrence cost
      the HZCQ-BR proposal is actually about, via torch.profiler's
      kernel/op breakdown.
  (b) real corpus_train_step wall-clock, split into "time spent waiting
      on next(corpus_mix)" (network I/O) vs "time spent in the model"
      (compute) -- because the 86 tok/sec figure from section 0.11
      divides real corpus tokens by TOTAL wall time, and total wall time
      includes real HF Parquet fetches (especially with
      --shuffle-buffer-size 0's strictly-sequential reads, no readahead
      overlap with compute). If I/O turns out to dominate, closing the
      81x gap needs an overlapped data-loader BEFORE any architecture
      change -- a real, cheap thing to rule in/out first rather than
      assume the whole gap is T*R serial depth.
Both parts share the exact same model construction and corpus_train_step
codepath as hz_bench_100m_pretrain.py (imported, not reimplemented) so
the numbers are directly comparable to that script's own logged
corpus tok/sec.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.profiler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hz_bench_100m_pretrain import autocast_context, corpus_train_step, make_optimizer  # noqa: E402
from hatchling_world.knowledge.decontamination import build_benchmark_ngram_hashes, DEFAULT_BENCHMARK_TASKS  # noqa: E402
from hatchling_world.knowledge.web_corpus_stream import WebCorpusMixture  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402


def profile_pure_compute(model, tok, opt, args, device: str) -> dict:
    """(a) Isolate model compute cost: fixed-length synthetic token
    sequence, no network, no decontamination -- just corpus_train_step's
    forward+backward on real ~chunk_chars-length input."""
    torch.manual_seed(0)
    token_ids = torch.randint(0, tok.vocab_size, (1, args.chunk_chars))
    text = None  # unused path below reimplements corpus_train_step's body directly on token_ids

    def _step():
        opt.zero_grad(set_to_none=True)
        with autocast_context(args.dtype, device):
            logits = model.lm_forward(token_ids, gradient_checkpointing=args.gradient_checkpointing)
            target = token_ids[:, 1:]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, tok.vocab_size), target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()

    # warmup (first call pays one-time CUDA context/allocator costs)
    _step()

    n_iters = args.compute_iters
    t0 = time.time()
    for _ in range(n_iters):
        _step()
    elapsed = time.time() - t0
    tokens_per_iter = args.chunk_chars - 1  # target is token_ids[:, 1:]
    result = {
        "n_iters": n_iters,
        "chunk_chars": args.chunk_chars,
        "n_rounds_l1": args.n_rounds_l1,
        "elapsed_sec": elapsed,
        "sec_per_iter": elapsed / n_iters,
        "compute_tok_per_sec": (tokens_per_iter * n_iters) / elapsed,
    }

    # Real bug, found 2026-09-11: profiling the FULL chunk_chars-length
    # sequence (e.g. 2000) means torch.profiler records one event per
    # tiny op INSIDE EVERY per-token checkpointed step -- tens of
    # thousands of events for one profiled iteration. key_averages()'s
    # own aggregation over that many events is not memory-efficient and
    # reliably blew up to 8GB+ RSS within seconds (reproduced both on a
    # RunPod pod and locally on this Mac -- not pod-specific flakiness).
    # The operator-cost BREAKDOWN doesn't need the full sequence length
    # to be representative (the same op types/shapes repeat every token
    # step); profile a short synthetic sequence instead, completely
    # separate from the real-length timing loop above.
    profile_token_ids = torch.randint(0, tok.vocab_size, (1, args.profile_chunk_chars))

    def _step_profile():
        opt.zero_grad(set_to_none=True)
        with autocast_context(args.dtype, device):
            logits = model.lm_forward(profile_token_ids, gradient_checkpointing=args.gradient_checkpointing)
            target = profile_token_ids[:, 1:]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, tok.vocab_size), target.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device == "cuda":
            torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=False) as prof:
        for _ in range(args.profile_iters):
            _step_profile()
    sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_key, row_limit=25)
    n_events = len(prof.key_averages())
    result["n_distinct_ops"] = n_events
    result["profiler_table"] = table
    return result


def profile_real_io_vs_compute(model, tok, opt, args, device: str) -> dict:
    """(b) Real corpus_mix draws: split wall time into fetch (next())
    vs compute (corpus_train_step's forward+backward on the real text
    that came back). Uses --shuffle-buffer-size 0, matching the section
    0.11 fix -- this measures the ACTUAL current bottleneck shape, not
    a hypothetical one."""
    benchmark_hashes = build_benchmark_ngram_hashes(DEFAULT_BENCHMARK_TASKS)
    corpus_mix = WebCorpusMixture(benchmark_hashes=benchmark_hashes, fineweb_weight=0.9, seed=0,
                                   chunk_chars=args.chunk_chars, shuffle_buffer_size=0)
    fetch_times, compute_times, token_counts = [], [], []
    for i in range(args.io_iters):
        t0 = time.time()
        _, text = next(corpus_mix)
        t_fetch = time.time() - t0

        t1 = time.time()
        _, _, n_tok = corpus_train_step(model, opt, tok, text, args.gradient_checkpointing, args.dtype, device)
        if device == "cuda":
            torch.cuda.synchronize()
        t_compute = time.time() - t1

        fetch_times.append(t_fetch)
        compute_times.append(t_compute)
        token_counts.append(n_tok)
        print(f"[hz-bench-profile] io iter {i+1}/{args.io_iters} fetch={t_fetch:.3f}s compute={t_compute:.3f}s "
              f"n_tok={n_tok}", flush=True)

    total_fetch = sum(fetch_times)
    total_compute = sum(compute_times)
    total_tok = sum(token_counts)
    total_wall = total_fetch + total_compute
    return {
        "io_iters": args.io_iters,
        "total_fetch_sec": total_fetch,
        "total_compute_sec": total_compute,
        "total_wall_sec": total_wall,
        "fetch_fraction": total_fetch / total_wall if total_wall > 0 else None,
        "compute_fraction": total_compute / total_wall if total_wall > 0 else None,
        "total_tokens": total_tok,
        "end_to_end_tok_per_sec": total_tok / total_wall if total_wall > 0 else None,
        "compute_only_tok_per_sec": total_tok / total_compute if total_compute > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=2336)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--workspace-slots", type=int, default=64)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--optimizer", choices=["adamw", "adam8bit"], default="adamw")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--chunk-chars", type=int, default=2000)
    parser.add_argument("--compute-iters", type=int, default=10)
    parser.add_argument("--profile-chunk-chars", type=int, default=64,
                         help="separate, short sequence length for the torch.profiler op-breakdown block -- "
                              "NOT the same as --chunk-chars, which drives the real-length timing loop. "
                              "Profiling the full-length sequence generates tens of thousands of events "
                              "and reliably OOMs on key_averages() aggregation (see the code comment).")
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--io-iters", type=int, default=10)
    parser.add_argument("--skip-io", action="store_true", help="only run part (a), skip real network fetches")
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_bench_profile.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[hz-bench-profile] device={device}", flush=True)
    torch.set_default_device(device)
    torch.manual_seed(0)

    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[hz-bench-profile] model n_params={n_params:,} d_model={args.d_model} "
          f"n_rounds_l1={args.n_rounds_l1} gradient_checkpointing={args.gradient_checkpointing} "
          f"dtype={args.dtype}", flush=True)
    opt = make_optimizer(model.parameters(), args.optimizer, args.lr, device)

    print("[hz-bench-profile] === part (a): pure compute profiling (no network) ===", flush=True)
    compute_result = profile_pure_compute(model, tok, opt, args, device)
    print(f"[hz-bench-profile] compute: {compute_result['n_iters']} iters, "
          f"{compute_result['sec_per_iter']:.3f} sec/iter, "
          f"{compute_result['compute_tok_per_sec']:.2f} compute-only tok/sec, "
          f"{compute_result['n_distinct_ops']} distinct profiler ops", flush=True)
    print(compute_result["profiler_table"], flush=True)

    io_result = None
    if not args.skip_io:
        print("[hz-bench-profile] === part (b): real corpus fetch vs compute split ===", flush=True)
        io_result = profile_real_io_vs_compute(model, tok, opt, args, device)
        print(f"[hz-bench-profile] io/compute split: fetch={io_result['fetch_fraction']:.1%} "
              f"compute={io_result['compute_fraction']:.1%} "
              f"end_to_end_tok_per_sec={io_result['end_to_end_tok_per_sec']:.2f} "
              f"compute_only_tok_per_sec={io_result['compute_only_tok_per_sec']:.2f}", flush=True)

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "device": device,
        "n_params": n_params,
        "d_model": args.d_model,
        "n_rounds_l1": args.n_rounds_l1,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dtype": args.dtype,
        "compute": {k: v for k, v in compute_result.items() if k != "profiler_table"},
        "io_vs_compute": io_result,
    }
    with open(args.results_file, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[hz-bench-profile] wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
