#!/usr/bin/env python3
"""HZ-Bench-100M pretraining (plans/Hatchling world.md section 0.8,
user-directed pivot from 5M-scale diagnostics to real benchmark
scaling). Trains a single HZ model through a real, STATIC (not
adaptive-mastery) mixture, matching the user's explicit percentages:

  85% real corpus LM   -- FineWeb-Edu sample-10BT (90%) + Wikipedia (10%),
                           streamed, benchmark-decontaminated
   5% Knowledge         -- real SQuAD factual QA (SQuAD's own source
                           passages ARE Wikipedia articles, so this is a
                           real, disclosed choice of "Wikipedia/factual
                           emphasis" reusing this session's already-
                           validated mechanism, not a new one)
   5% Nursery rehearsal -- L0-L6 + Library, combined into one macro-
                           channel (uniform random sub-task per call),
                           not 12 near-equal channels anymore
   5% Chat/instruction  -- real Dolly-15k instruction/response pairs

This is a genuine departure from the Nursery-era adaptive per-subskill
scheduler (`need = max(1-min(mastery))`): pretraining at this scale
uses a fixed mixture ratio, matching standard practice (Chinchilla/
Qwen-style corpus mixtures), not a diagnostic-curriculum scheduler.
Hatchling World's persistent, no-reset, single-theta lineage discipline
is kept -- only what it feeds the model changes.

Real, disclosed mechanics:
  - Benchmark decontamination (`hatchling_world.knowledge.
    decontamination`) is applied to every corpus document before it is
    used, per the user's explicit mandatory requirement.
  - "Tokens seen" (the metric that matters once comparing against
    Qwen-style reported token budgets, per the user's own point) is
    tracked as CORPUS tokens only, matching how pretraining token
    counts are conventionally reported -- Knowledge/Nursery/Chat are
    rehearsal streams, not pretraining corpus.
  - Estimated training FLOPs uses the standard 6*N*tokens
    approximation (Kaplan/Chinchilla convention) -- a real, disclosed
    approximation, not a measured profiler number.
  - Every eval checkpoint emits: held-out corpus loss (on a FROZEN set
    of chunks captured once at a different seed offset, never trained
    on), the full lm-eval harness on the 6 Milestone-1 target tasks,
    real generation samples, tokens/sec, peak VRAM (CUDA only), and
    cumulative corpus tokens -- matching the user's explicit checklist,
    no more judging the mainline by bespoke probes alone.
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

import hz_nursery_train as nt  # noqa: E402
from hz_world_training_run1_stage_b_library import library_train_step  # noqa: E402
from hz_world_training_run1_stage_b_knowledge import knowledge_train_step  # noqa: E402
from hz_chat_micro_sft_train import train_step as chat_train_step, sample_generations  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.language.tokenizer import NOVEL_LABELS  # noqa: E402
from hatchling_world.knowledge.chat_data import build_chat_split  # noqa: E402
from hatchling_world.knowledge.decontamination import build_benchmark_ngram_hashes, DEFAULT_BENCHMARK_TASKS  # noqa: E402
from hatchling_world.knowledge.web_corpus_stream import WebCorpusMixture  # noqa: E402


def corpus_train_step(model, opt, tok, text: str):
    token_ids = torch.tensor([tok.encode(text)])
    logits = model.lm_forward(token_ids)
    target = token_ids[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size), target.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        acc = (logits.argmax(-1) == target).float().mean().item()
    return loss.item(), acc, len(token_ids[0])


def corpus_eval_loss(model, tok, held_out_chunks: list) -> float:
    losses = []
    with torch.no_grad():
        for text in held_out_chunks:
            token_ids = torch.tensor([tok.encode(text)])
            logits = model.lm_forward(token_ids)
            target = token_ids[:, 1:]
            losses.append(F.cross_entropy(logits.reshape(-1, tok.vocab_size), target.reshape(-1)).item())
    return sum(losses) / len(losses)


def nursery_rehearsal_step(model, opt, tok, rngs: dict, args) -> tuple:
    sub_task = rngs["_picker"].choice(
        ["L0", "L1", "L2", "L3", "L4-logic", "L4-counting", "L5", "L6", "Library"])
    if sub_task == "L0":
        return nt.l0_train_step(model, opt, tok, rngs["L0"], args.l0_batch_size)
    if sub_task == "L1":
        return nt.l1_train_step(model, opt, tok, rngs["L1"], args.l_n_objects)
    if sub_task == "L2":
        return nt.l2_train_step(model, opt, tok, rngs["L2"], args.l_n_objects)
    if sub_task == "L3":
        return nt.l3_train_step(model, opt, tok, rngs["L3"], args.l_n_objects)
    if sub_task == "L4-logic":
        return nt.l4_logic_train_step(model, opt, tok, rngs["L4-logic"], args.l_n_objects)
    if sub_task == "L4-counting":
        return nt.l4_counting_train_step(model, opt, tok, rngs["L4-counting"], args.l_n_objects)
    if sub_task == "L5":
        return nt.l5_train_step(model, opt, tok, rngs["L5"], args.l_n_objects)
    if sub_task == "L6":
        return nt.l6_train_step(model, opt, tok, rngs["L6"], args.l6_n_sentences)
    return library_train_step(model, opt, tok, rngs["Library"], args.library_n_facts)


def run_benchmark_suite(checkpoint_path: Path, d_model: int, memory_slots: int, workspace_slots: int,
                         n_rounds_l1: int, tasks: list, limit: int) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import hz_lm_eval_adapter  # noqa: F401  (registers "hz")
    from lm_eval import simple_evaluate
    model_args = (f"checkpoint={checkpoint_path},d_model={d_model},memory_slots={memory_slots},"
                  f"workspace_slots={workspace_slots},n_rounds_l1={n_rounds_l1}")
    results = simple_evaluate(model="hz", model_args=model_args, tasks=tasks, limit=limit,
                               log_samples=False, random_seed=0, numpy_random_seed=0, torch_random_seed=0)
    return {task: metrics for task, metrics in results["results"].items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=2336, help="from hz_bench_size_solver.py's ~100M solve")
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--workspace-slots", type=int, default=64)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--total-steps", type=int, default=500, help="Stage 0 default: systems test, not real training")
    parser.add_argument("--eval-every", type=int, default=0, help="0 = never (Stage 0 systems-test mode)")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--corpus-share", type=float, default=0.85)
    parser.add_argument("--knowledge-share", type=float, default=0.05)
    parser.add_argument("--nursery-share", type=float, default=0.05)
    parser.add_argument("--chat-share", type=float, default=0.05)
    parser.add_argument("--fineweb-weight", type=float, default=0.9)
    parser.add_argument("--chunk-chars", type=int, default=2000)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000,
                         help="streaming shuffle buffer per source; use a small value (e.g. 50) for local smoke tests")
    parser.add_argument("--l0-batch-size", type=int, default=16)
    parser.add_argument("--l-n-objects", type=int, default=4)
    parser.add_argument("--l6-n-sentences", type=int, default=3)
    parser.add_argument("--library-n-facts", type=int, default=20)
    parser.add_argument("--chat-max-train", type=int, default=1741)
    parser.add_argument("--n-held-out-corpus-chunks", type=int, default=20)
    parser.add_argument("--benchmark-tasks", type=str, default=",".join(DEFAULT_BENCHMARK_TASKS))
    parser.add_argument("--benchmark-limit", type=int, default=50)
    parser.add_argument("--n-generation-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/local/hz_bench_100m"))
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_bench_100m_results.json"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[hz-bench-100m] device={device}", flush=True)
    # Real, disclosed reason for this rather than threading a device param
    # through every train_step helper: those helpers (Nursery L0-L6,
    # Knowledge, Chat, Library) were all built and validated during a
    # CPU-only Mac session and construct tensors with bare `torch.tensor(...)`
    # -- no device kwarg anywhere. `set_default_device` makes every such
    # call land on the right device without touching that code.
    torch.set_default_device(device)

    torch.manual_seed(args.seed)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[hz-bench-100m] FRESH model: d_model={args.d_model} n_params={n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print("[hz-bench-100m] building benchmark decontamination hashes...", flush=True)
    benchmark_tasks = args.benchmark_tasks.split(",")
    benchmark_hashes = build_benchmark_ngram_hashes(benchmark_tasks)
    print(f"[hz-bench-100m] {len(benchmark_hashes)} benchmark n-gram hashes across {benchmark_tasks}", flush=True)

    corpus_mix = WebCorpusMixture(benchmark_hashes=benchmark_hashes, fineweb_weight=args.fineweb_weight,
                                   seed=args.seed, chunk_chars=args.chunk_chars,
                                   shuffle_buffer_size=args.shuffle_buffer_size)
    held_out_mix = WebCorpusMixture(benchmark_hashes=benchmark_hashes, fineweb_weight=args.fineweb_weight,
                                     seed=args.seed + 777, chunk_chars=args.chunk_chars,
                                     shuffle_buffer_size=args.shuffle_buffer_size)
    held_out_chunks = [next(held_out_mix)[1] for _ in range(args.n_held_out_corpus_chunks)]
    print(f"[hz-bench-100m] captured {len(held_out_chunks)} frozen held-out corpus chunks "
          f"(seed+777, never trained on)", flush=True)

    chat_train, chat_held = build_chat_split(seed=args.seed, max_train=args.chat_max_train, max_held_out=50)
    print(f"[hz-bench-100m] real data: chat {len(chat_train)} train / {len(chat_held)} held-out", flush=True)

    chat_rng = random.Random(args.seed + 52)
    knowledge_rng = random.Random(args.seed + 51)
    nursery_rngs = {s: random.Random(args.seed + i) for i, s in
                    enumerate(["L0", "L1", "L2", "L3", "L4-logic", "L4-counting", "L5", "L6", "Library"])}
    nursery_rngs["_picker"] = random.Random(args.seed + 900)
    mixture_rng = random.Random(args.seed + 999)

    channels = ["Corpus", "Knowledge", "Nursery", "Chat"]
    weights = [args.corpus_share, args.knowledge_share, args.nursery_share, args.chat_share]
    assert abs(sum(weights) - 1.0) < 1e-6, f"mixture shares must sum to 1.0, got {sum(weights)}"

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    corpus_tokens_seen = 0
    step_calls = {c: 0 for c in channels}
    corpus_source_calls = {"fineweb_edu": 0, "wikipedia": 0}
    eval_points = []

    t0 = time.time()
    for step in range(args.total_steps):
        chosen = mixture_rng.choices(channels, weights=weights, k=1)[0]
        step_calls[chosen] += 1
        if chosen == "Corpus":
            source, text = next(corpus_mix)
            corpus_source_calls[source] += 1
            loss, acc, n_tok = corpus_train_step(model, opt, tok, text)
            corpus_tokens_seen += n_tok
        elif chosen == "Knowledge":
            knowledge_train_step(model, opt, tok, knowledge_rng)
        elif chosen == "Nursery":
            nursery_rehearsal_step(model, opt, tok, nursery_rngs, args)
        else:
            chat_train_step(model, opt, tok, chat_rng.choice(chat_train))

        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / elapsed
            tokens_per_sec = corpus_tokens_seen / elapsed
            vram = (f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB" if device == "cuda" else "n/a")
            print(f"[hz-bench-100m] step={step+1}/{args.total_steps} ({elapsed:.1f}s, "
                  f"{steps_per_sec:.2f} steps/sec, {tokens_per_sec:.0f} corpus tok/sec) "
                  f"last={chosen} corpus_tokens_seen={corpus_tokens_seen:,} peak_vram={vram} "
                  f"calls={step_calls} corpus_sources={corpus_source_calls} "
                  f"decon_rejected={corpus_mix.stats['rejected_contaminated']}", flush=True)

        if args.eval_every and (step + 1) % args.eval_every == 0:
            ckpt_path = args.checkpoint_dir / f"step_{step+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            held_out_loss = corpus_eval_loss(model, tok, held_out_chunks)
            estimated_flops = 6 * n_params * corpus_tokens_seen
            print(f"\n[hz-bench-100m] ===== EVAL @ step={step+1} =====", flush=True)
            print(f"[hz-bench-100m] held_out_corpus_loss={held_out_loss:.4f} "
                  f"corpus_tokens_seen={corpus_tokens_seen:,} estimated_flops={estimated_flops:.3e}", flush=True)
            print("[hz-bench-100m] running benchmark suite...", flush=True)
            bench_scores = run_benchmark_suite(ckpt_path, args.d_model, args.memory_slots, args.workspace_slots,
                                                args.n_rounds_l1, benchmark_tasks, args.benchmark_limit)
            for task, metrics in bench_scores.items():
                print(f"[hz-bench-100m] {task}: {metrics}", flush=True)
            samples = sample_generations(model, tok, chat_held, n=args.n_generation_samples)
            for s in samples:
                print(f"[hz-bench-100m] Q: {s['instruction']}", flush=True)
                print(f"[hz-bench-100m]   generated: {s['generated']!r}", flush=True)
            elapsed = time.time() - t0
            eval_points.append({
                "step": step + 1, "corpus_tokens_seen": corpus_tokens_seen,
                "estimated_flops": estimated_flops, "held_out_corpus_loss": held_out_loss,
                "benchmark_scores": bench_scores, "samples": samples,
                "steps_per_sec": (step + 1) / elapsed, "tokens_per_sec": corpus_tokens_seen / elapsed,
                "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None),
                "calls": dict(step_calls), "corpus_source_calls": dict(corpus_source_calls),
                "decon_rejected": corpus_mix.stats["rejected_contaminated"],
            })
            with open(args.results_file, "w") as f:
                json.dump({"n_params": n_params, "d_model": args.d_model, "device": device,
                            "eval_points": eval_points}, f, indent=2, default=str)
            print(f"[hz-bench-100m] wrote {args.results_file}\n", flush=True)

    total_time = time.time() - t0
    vram = (f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB" if device == "cuda" else "n/a")
    print(f"\n[hz-bench-100m] DONE: {args.total_steps} steps in {total_time:.1f}s "
          f"({args.total_steps/total_time:.2f} steps/sec, {corpus_tokens_seen/total_time:.0f} corpus tok/sec, "
          f"peak_vram={vram}, corpus_tokens_seen={corpus_tokens_seen:,}, "
          f"estimated_flops={6*n_params*corpus_tokens_seen:.3e})", flush=True)
    final_ckpt = args.checkpoint_dir / "final.pt"
    torch.save(model.state_dict(), final_ckpt)
    print(f"[hz-bench-100m] final checkpoint: {final_ckpt}", flush=True)

    if not args.eval_every:
        with open(args.results_file, "w") as f:
            json.dump({"n_params": n_params, "d_model": args.d_model, "device": device,
                        "mode": "stage0_systems_test", "total_steps": args.total_steps,
                        "total_seconds": total_time, "steps_per_sec": args.total_steps / total_time,
                        "corpus_tokens_seen": corpus_tokens_seen, "tokens_per_sec": corpus_tokens_seen / total_time,
                        "peak_vram_gb": (torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None),
                        "calls": step_calls, "corpus_source_calls": corpus_source_calls,
                        "decon_rejected": corpus_mix.stats["rejected_contaminated"]}, f, indent=2, default=str)
        print(f"[hz-bench-100m] wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
