#!/usr/bin/env python3
"""Hatchling World Training Run 1, Stage A (plans/Hatchling world.md
section 0.6): ONE persistent, continuously-trained model moving through
L0-L6 in sequence -- NOT the discard-per-experiment pattern every
script this session used (init model -> train on task -> measure ->
throw away). Same model, same optimizer state, carried across every
stage boundary, checkpointed at each one.

The real, new thing this unlocks: after finishing each stage, ALL
previously-completed stages are RE-EVALUATED on the same model (a real
retention/forgetting matrix), using the exact same fixed eval seeds
every time so repeated checks are directly comparable. Isolated
per-task scripts this session could never answer "does learning L4
counting damage L1 grounding" -- this can.

Reuses every train_step/eval function from scripts/hz_nursery_train.py
unchanged (that module is a real, already-validated library of per-
stage training/eval logic, not being rewritten here).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hz_nursery_train as nt  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.tokenizer import NurseryTokenizer, NOVEL_LABELS  # noqa: E402

TEST_SEED_OFFSET = nt.TEST_SEED_OFFSET


def make_retention_fns(tok, args):
    """One eval callable per stage, each with its OWN fixed seed
    (matching hz_nursery_train.py's own per-stage seed offsets) so
    repeated retention checks across checkpoints use the identical
    held-out episode sequence -- a fair, reproducible comparison, not
    fresh noise each time."""
    def l0(model):
        rng = random.Random(args.seed + TEST_SEED_OFFSET)
        held_out_ids = nt.l0_batch(tok, rng, 200)
        with torch.no_grad():
            logits = model.lm_forward(held_out_ids)
            target = held_out_ids[:, 1:]
            mask = (target != tok.pad_id)
            pred = logits.argmax(-1)
            acc = (((pred == target) & mask).float().sum() / mask.float().sum().clamp_min(1)).item()
        return {"next_token_acc": acc}

    def l1(model):
        rng = random.Random(args.seed + 2 + TEST_SEED_OFFSET)
        return {"held_out_acc": nt.l1_eval(model, tok, rng, args.l1_n_objects, args.eval_episodes)}

    def l2(model):
        rng = random.Random(args.seed + 3 + TEST_SEED_OFFSET)
        sel_acc, cons_acc = nt.l2_eval(model, tok, rng, args.l2_n_objects, args.eval_episodes)
        return {"sel_acc": sel_acc, "cons_acc": cons_acc}

    def l3(model):
        seen_rng = random.Random(args.seed + 4 + TEST_SEED_OFFSET)
        unseen_rng = random.Random(args.seed + 4 + 2 * TEST_SEED_OFFSET)
        return {"seen_combo_acc": nt.l3_eval(model, tok, seen_rng, args.l3_n_objects, args.eval_episodes, split="train"),
                "unseen_combo_acc": nt.l3_eval(model, tok, unseen_rng, args.l3_n_objects, args.eval_episodes, split="test")}

    def l4_logic(model):
        seen_rng = random.Random(args.seed + 5 + TEST_SEED_OFFSET)
        unseen_rng = random.Random(args.seed + 5 + 2 * TEST_SEED_OFFSET)
        return {"seen_combo_acc": nt.l4_logic_eval(model, tok, seen_rng, args.l4_n_objects, args.eval_episodes, split="train"),
                "unseen_combo_acc": nt.l4_logic_eval(model, tok, unseen_rng, args.l4_n_objects, args.eval_episodes, split="test")}

    def l4_counting(model):
        rng = random.Random(args.seed + 6 + TEST_SEED_OFFSET)
        return {"held_out_acc": nt.l4_counting_eval(model, tok, rng, args.l4_n_objects, args.eval_episodes)}

    def l5(model):
        rng = random.Random(args.seed + 7 + TEST_SEED_OFFSET)
        return {"held_out_acc": nt.l5_eval(model, tok, rng, args.l5_n_objects, args.eval_episodes)}

    def l6(model):
        rng = random.Random(args.seed + 8 + TEST_SEED_OFFSET)
        acc, by_query = nt.l6_eval(model, tok, rng, args.l6_n_sentences, args.eval_episodes)
        return {"held_out_acc": acc}

    return {"L0": l0, "L1": l1, "L2": l2, "L3": l3, "L4-logic": l4_logic,
            "L4-counting": l4_counting, "L5": l5, "L6": l6}


def run_stage_training(name, model, opt, tok, args):
    """Train one stage for its configured step budget, printing live
    progress. Returns nothing -- state lives entirely in model/opt,
    exactly the point (no fresh model per stage)."""
    if name == "L0":
        train_rng = random.Random(args.seed + 1)
        for step in range(args.l0_steps):
            loss, acc = nt.l0_train_step(model, opt, tok, train_rng, args.l0_batch_size)
            if (step + 1) % args.log_every == 0:
                print(f"[stage-a][L0] step={step+1}/{args.l0_steps} loss={loss:.4f} acc={acc:.3f}", flush=True)
    elif name == "L1":
        train_rng = random.Random(args.seed + 2)
        for step in range(args.l1_steps):
            loss, acc = nt.l1_train_step(model, opt, tok, train_rng, args.l1_n_objects)
            if (step + 1) % args.log_every == 0:
                print(f"[stage-a][L1] step={step+1}/{args.l1_steps} loss={loss:.4f} acc={acc:.3f}", flush=True)
    elif name == "L2":
        train_rng = random.Random(args.seed + 3)
        for step in range(args.l2_steps):
            loss, sel_acc, cons_acc = nt.l2_train_step(model, opt, tok, train_rng, args.l2_n_objects)
            if (step + 1) % args.log_every == 0:
                print(f"[stage-a][L2] step={step+1}/{args.l2_steps} loss={loss:.4f} sel_acc={sel_acc:.3f} cons_acc={cons_acc:.3f}", flush=True)
    elif name == "L3":
        train_rng = random.Random(args.seed + 4)
        for step in range(args.l3_steps):
            loss, acc = nt.l3_train_step(model, opt, tok, train_rng, args.l3_n_objects)
            if (step + 1) % args.log_every == 0:
                print(f"[stage-a][L3] step={step+1}/{args.l3_steps} loss={loss:.4f} acc={acc:.3f}", flush=True)
    elif name == "L4-logic":
        train_rng = random.Random(args.seed + 5)
        for step in range(args.l4_logic_steps):
            loss, acc = nt.l4_logic_train_step(model, opt, tok, train_rng, args.l4_n_objects)
            if (step + 1) % args.log_every == 0:
                print(f"[stage-a][L4-logic] step={step+1}/{args.l4_logic_steps} loss={loss:.4f} acc={acc:.3f}", flush=True)
    elif name == "L4-counting":
        train_rng = random.Random(args.seed + 6)
        for step in range(args.l4_counting_steps):
            loss, acc = nt.l4_counting_train_step(model, opt, tok, train_rng, args.l4_n_objects)
            if (step + 1) % args.log_every == 0:
                print(f"[stage-a][L4-counting] step={step+1}/{args.l4_counting_steps} loss={loss:.4f} acc={acc:.3f}", flush=True)
    elif name == "L5":
        train_rng = random.Random(args.seed + 7)
        for step in range(args.l5_steps):
            loss, acc = nt.l5_train_step(model, opt, tok, train_rng, args.l5_n_objects)
            if (step + 1) % args.log_every == 0:
                print(f"[stage-a][L5] step={step+1}/{args.l5_steps} loss={loss:.4f} acc={acc:.3f}", flush=True)
    elif name == "L6":
        train_rng = random.Random(args.seed + 8)
        for step in range(args.l6_steps):
            loss, acc = nt.l6_train_step(model, opt, tok, train_rng, args.l6_n_sentences)
            if (step + 1) % args.log_every == 0:
                print(f"[stage-a][L6] step={step+1}/{args.l6_steps} loss={loss:.4f} acc={acc:.3f}", flush=True)
    else:
        raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
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
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/local/hz_world_run1_stage_a"))
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_world_run1_stage_a_retention.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[stage-a] PERSISTENT model created once: vocab_size={tok.vocab_size} n_params={n_params}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    retention_fns = make_retention_fns(tok, args)
    stage_order = ["L0", "L1", "L2", "L3", "L4-logic", "L4-counting", "L5", "L6"]
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    retention_matrix = []  # list of {"after_stage": name, "scores": {stage: metrics}}
    completed = []
    t_start = time.time()
    for stage in stage_order:
        t0 = time.time()
        print(f"\n[stage-a] ===== training {stage} (persistent model, {n_params} params, "
              f"{len(completed)} prior stages already learned) =====", flush=True)
        run_stage_training(stage, model, opt, tok, args)
        completed.append(stage)
        print(f"[stage-a] {stage} training done in {time.time()-t0:.0f}s. Retention check on all "
              f"{len(completed)} completed stages...", flush=True)

        scores = {}
        for s in completed:
            scores[s] = retention_fns[s](model)
        retention_matrix.append({"after_stage": stage, "scores": scores})
        print(f"[stage-a] retention after {stage}: " +
              " | ".join(f"{s}={scores[s]}" for s in completed), flush=True)

        ckpt_path = args.checkpoint_dir / f"after_{stage.replace('-', '_')}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"[stage-a] checkpoint saved: {ckpt_path}", flush=True)

    total_time = time.time() - t_start
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump({"stage_order": stage_order, "retention_matrix": retention_matrix,
                    "n_params": n_params, "total_seconds": total_time}, f, indent=2)
    print(f"\n[stage-a] DONE in {total_time:.0f}s. Wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
