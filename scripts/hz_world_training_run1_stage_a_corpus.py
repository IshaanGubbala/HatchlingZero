#!/usr/bin/env python3
"""Hatchling World Training Run 1, Stage A + real corpus channel (plans/
Hatchling world.md section 0.7, step 2 of the sequenced plan). Extends
`hz_world_training_run1_stage_a_interleaved.py`'s already-validated
interleaved scheduler with ONE more entry, "Corpus" -- real byte-level
text from `data/packed/hz0h_bytes_25m_train.jsonl` (already pre-chunked
into fixed 256-byte windows, byte value == token id, matching
`ByteTokenizer` exactly, no remapping) -- introduced FIRST, since corpus
learning is the foundational theta-building channel per 0.7's own
`training corpus -> theta` framing. Uses `ByteTokenizer` throughout (not
`NurseryTokenizer`) so real corpus text and every Nursery/School stage
share one embedding space, per step (1)'s verified result.

Real, disclosed expectation going in (not hidden): step (1) found L6
does NOT transfer to byte-level (flat at chance, likely compounding the
already-known multi-fact S capacity limit) -- L6 is kept in this run
for completeness/honesty, not expected to suddenly work here.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hz_nursery_train as nt  # noqa: E402
import hz_world_training_run1_stage_a as run1  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.language.tokenizer import NOVEL_LABELS  # noqa: E402


class CorpusPool:
    """Real byte-level text, loaded once. A bounded prefix of the real
    334,377-window training file (not the whole 383MB) -- a real,
    working-set-sized sample for this controlled experiment, not a
    production-scale corpus run."""

    def __init__(self, path: Path, max_lines: int):
        self.windows = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                self.windows.append(json.loads(line))
        if not self.windows:
            raise ValueError(f"no windows loaded from {path}")

    def sample(self, rng: random.Random):
        return rng.choice(self.windows)


def corpus_train_step(model, opt, pool: CorpusPool, rng: random.Random, vocab_size: int):
    ids = pool.sample(rng)
    token_ids = torch.tensor([ids])  # (1, T) -- fixed-length real window, no padding needed
    logits = model.lm_forward(token_ids)
    target = token_ids[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        acc = (logits.argmax(-1) == target).float().mean().item()
    return loss.item(), acc


def corpus_eval(model, val_pool: CorpusPool, rng: random.Random, vocab_size: int, n_windows: int):
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(n_windows):
            ids = val_pool.sample(rng)
            token_ids = torch.tensor([ids])
            logits = model.lm_forward(token_ids)
            target = token_ids[:, 1:]
            correct += int((logits.argmax(-1) == target).float().sum().item())
            total += target.numel()
    return correct / total


def make_stage_step_fns(model, opt, tok, args, rngs, corpus_pool):
    return {
        "Corpus": lambda: corpus_train_step(model, opt, corpus_pool, rngs["Corpus"], tok.vocab_size),
        "L0": lambda: nt.l0_train_step(model, opt, tok, rngs["L0"], args.l0_batch_size),
        "L1": lambda: nt.l1_train_step(model, opt, tok, rngs["L1"], args.l1_n_objects),
        "L2": lambda: nt.l2_train_step(model, opt, tok, rngs["L2"], args.l2_n_objects),
        "L3": lambda: nt.l3_train_step(model, opt, tok, rngs["L3"], args.l3_n_objects),
        "L4-logic": lambda: nt.l4_logic_train_step(model, opt, tok, rngs["L4-logic"], args.l4_n_objects),
        "L4-counting": lambda: nt.l4_counting_train_step(model, opt, tok, rngs["L4-counting"], args.l4_n_objects),
        "L5": lambda: nt.l5_train_step(model, opt, tok, rngs["L5"], args.l5_n_objects),
        "L6": lambda: nt.l6_train_step(model, opt, tok, rngs["L6"], args.l6_n_sentences),
    }


PHASE_BUDGET_ARG = {
    "Corpus": "corpus_steps", "L0": "l0_steps", "L1": "l1_steps", "L2": "l2_steps", "L3": "l3_steps",
    "L4-logic": "l4_logic_steps", "L4-counting": "l4_counting_steps", "L5": "l5_steps", "L6": "l6_steps",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--corpus-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--corpus-val-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--corpus-max-lines", type=int, default=20000)
    parser.add_argument("--corpus-val-max-lines", type=int, default=2000)
    parser.add_argument("--corpus-steps", type=int, default=2000)
    parser.add_argument("--corpus-eval-windows", type=int, default=100)
    parser.add_argument("--l0-steps", type=int, default=2000)
    parser.add_argument("--l0-batch-size", type=int, default=16)
    parser.add_argument("--l1-steps", type=int, default=1500)
    parser.add_argument("--l1-n-objects", type=int, default=4)
    parser.add_argument("--l2-steps", type=int, default=1500)
    parser.add_argument("--l2-n-objects", type=int, default=4)
    parser.add_argument("--l3-steps", type=int, default=2000)
    parser.add_argument("--l3-n-objects", type=int, default=4)
    parser.add_argument("--l4-logic-steps", type=int, default=2000)
    parser.add_argument("--l4-counting-steps", type=int, default=2000)
    parser.add_argument("--l4-n-objects", type=int, default=4)
    parser.add_argument("--l5-steps", type=int, default=2000)
    parser.add_argument("--l5-n-objects", type=int, default=4)
    parser.add_argument("--l6-steps", type=int, default=2000)
    parser.add_argument("--l6-n-sentences", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=150)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/local/hz_world_run1_stage_a_corpus"))
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_world_run1_stage_a_corpus_retention.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[stage-a-corpus] PERSISTENT model (byte-level, vocab_size={tok.vocab_size}) created once: "
          f"n_params={n_params}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"[stage-a-corpus] loading real corpus: {args.corpus_data} (max {args.corpus_max_lines} windows)...", flush=True)
    corpus_pool = CorpusPool(args.corpus_data, args.corpus_max_lines)
    corpus_val_pool = CorpusPool(args.corpus_val_data, args.corpus_val_max_lines)
    print(f"[stage-a-corpus] loaded {len(corpus_pool.windows)} train windows, "
          f"{len(corpus_val_pool.windows)} val windows, window length={len(corpus_pool.windows[0])} bytes", flush=True)

    stage_order = ["Corpus", "L0", "L1", "L2", "L3", "L4-logic", "L4-counting", "L5", "L6"]
    rngs = {
        "Corpus": random.Random(args.seed), "L0": random.Random(args.seed + 1), "L1": random.Random(args.seed + 2),
        "L2": random.Random(args.seed + 3), "L3": random.Random(args.seed + 4),
        "L4-logic": random.Random(args.seed + 5), "L4-counting": random.Random(args.seed + 6),
        "L5": random.Random(args.seed + 7), "L6": random.Random(args.seed + 8),
    }
    corpus_eval_rng = random.Random(args.seed + nt.TEST_SEED_OFFSET)
    schedule_rng = random.Random(args.seed + 999)
    step_fns = make_stage_step_fns(model, opt, tok, args, rngs, corpus_pool)
    retention_fns = run1.make_retention_fns(tok, args)
    retention_fns["Corpus"] = lambda m: {"held_out_next_byte_acc":
        corpus_eval(m, corpus_val_pool, corpus_eval_rng, tok.vocab_size, args.corpus_eval_windows)}
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    retention_matrix = []
    introduced = []
    per_stage_calls = {s: 0 for s in stage_order}
    t_start = time.time()
    for new_stage in stage_order:
        introduced.append(new_stage)
        phase_steps = getattr(args, PHASE_BUDGET_ARG[new_stage])
        t0 = time.time()
        print(f"\n[stage-a-corpus] ===== phase introducing {new_stage}: {phase_steps} steps, "
              f"sampled uniformly among {introduced} =====", flush=True)
        for step in range(phase_steps):
            chosen = schedule_rng.choice(introduced)
            step_fns[chosen]()
            per_stage_calls[chosen] += 1
            if (step + 1) % args.log_every == 0:
                print(f"[stage-a-corpus][{new_stage} phase] step={step+1}/{phase_steps} "
                      f"last_sampled={chosen} calls_so_far={per_stage_calls}", flush=True)
        print(f"[stage-a-corpus] {new_stage} phase done in {time.time()-t0:.0f}s. "
              f"Retention check on all {len(introduced)} introduced stages...", flush=True)

        scores = {s: retention_fns[s](model) for s in introduced}
        retention_matrix.append({"after_stage": new_stage, "scores": scores})
        print(f"[stage-a-corpus] retention after {new_stage}: " +
              " | ".join(f"{s}={scores[s]}" for s in introduced), flush=True)

        ckpt_path = args.checkpoint_dir / f"after_{new_stage.replace('-', '_')}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[stage-a-corpus] checkpoint saved: {ckpt_path}", flush=True)

    total_time = time.time() - t_start
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump({"stage_order": stage_order, "retention_matrix": retention_matrix,
                    "n_params": n_params, "total_seconds": total_time,
                    "per_stage_calls": per_stage_calls}, f, indent=2)
    print(f"\n[stage-a-corpus] DONE in {total_time:.0f}s. Wrote {args.results_file}", flush=True)
    print(f"[stage-a-corpus] total train calls per stage: {per_stage_calls}", flush=True)


if __name__ == "__main__":
    main()
