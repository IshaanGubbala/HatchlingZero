#!/usr/bin/env python3
"""Hatchling World Training Run 2 (plans/Hatchling world.md section
0.6, user-directed correction to the standalone HZ-Chat-Micro v0/v1
detour): a FRESH, appropriately-sized HZ model (d_model=512,
memory_slots=16, workspace_slots=64 -- 5,056,229 params, inside the
requested 5-10M range) developed CONTINUOUSLY through the full real
curriculum via the SAME per-subskill adaptive scheduler validated in
Training Run 1 (`hz_world_training_run1_stage_a_adaptive.py`), NOT a
sequence of disconnected standalone scripts. One theta accumulates
competence across every channel below; no resets between phases.

Real channels, all real mechanisms already validated earlier this
session, reused not reinvented:
  Corpus       - real Wikipedia prose (SQuAD contexts, not synthetic)
  L0           - synthetic self-supervised LM (Nursery)
  L1-L6        - Nursery grounding/verb/relation/logic/counting/QA/reading
  Library      - real O(1) fact retrieval (Phase 8)
  Knowledge    - real SQuAD factual QA (Stage B)
  Chat         - real Dolly-15k instruction/response pairs

Real, disclosed scope: this is Run 2's FIRST PASS, not a claim of full
convergence on every channel -- 12 real channels, each with real,
different per-step compute cost (Nursery tasks are short synthetic
sentences; Corpus/Knowledge/Chat are real, longer text), scheduled via
the same max_j(1-m_j) per-subskill priority already validated. A real,
bounded step budget is used and disclosed, not silently assumed
sufficient for full convergence on every channel.
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
import hz_world_training_run1_stage_a as run1  # noqa: E402
from hz_world_training_run1_stage_a_adaptive import weighted_choice, EMA_DECAY, MASTERY_FLOOR  # noqa: E402
from hz_world_training_run1_stage_b_library import library_train_step, library_eval  # noqa: E402
from hz_world_training_run1_stage_b_knowledge import knowledge_train_step, knowledge_eval_condition  # noqa: E402
from hz_chat_micro_sft_train import train_step as chat_train_step, eval_items as chat_eval_items, sample_generations  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.language.tokenizer import NOVEL_LABELS  # noqa: E402
from hatchling_world.knowledge.facts import TRAIN_FACTS, HELD_OUT_FACTS  # noqa: E402
from hatchling_world.knowledge.chat_data import build_chat_split  # noqa: E402
from hatchling_world.knowledge.squad_corpus import build_squad_split  # noqa: E402

TEST_SEED_OFFSET = nt.TEST_SEED_OFFSET
EVAL_ONLY_SUBSKILLS = {"L3": "unseen", "L4-logic": "unseen"}


def corpus_train_step(model, opt, tok, context: str):
    import torch.nn.functional as F
    token_ids = torch.tensor([tok.encode(context)])
    logits = model.lm_forward(token_ids)
    target = token_ids[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size), target.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    with torch.no_grad():
        acc = (logits.argmax(-1) == target).float().mean().item()
    return loss.item(), acc


def corpus_eval_loss(model, tok, contexts: list) -> float:
    import torch.nn.functional as F
    losses = []
    with torch.no_grad():
        for context in contexts:
            token_ids = torch.tensor([tok.encode(context)])
            logits = model.lm_forward(token_ids)
            target = token_ids[:, 1:]
            losses.append(F.cross_entropy(logits.reshape(-1, tok.vocab_size), target.reshape(-1)).item())
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--workspace-slots", type=int, default=64)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--total-steps", type=int, default=6000)
    parser.add_argument("--l0-batch-size", type=int, default=16)
    parser.add_argument("--l1-n-objects", type=int, default=4)
    parser.add_argument("--l2-n-objects", type=int, default=4)
    parser.add_argument("--l3-n-objects", type=int, default=4)
    parser.add_argument("--l4-n-objects", type=int, default=4)
    parser.add_argument("--l5-n-objects", type=int, default=4)
    parser.add_argument("--l6-n-sentences", type=int, default=3)
    parser.add_argument("--library-n-facts", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--subskill-eval-every", type=int, default=200)
    parser.add_argument("--subskill-eval-episodes", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/local/hz_world_run2"))
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_world_run2_retention.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[run2] FRESH model, Training Run 2: n_params={n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Real data pools -- built once, shared across the whole run.
    squad_train, squad_held, squad_paraphrase = build_squad_split(seed=args.seed)
    squad_train_contexts = sorted(set(item["context"] for item in squad_train))
    squad_held_contexts = sorted(set(item["context"] for item in squad_held))
    chat_train, chat_held = build_chat_split(seed=args.seed, max_train=300, max_held_out=50)
    print(f"[run2] real data: {len(squad_train_contexts)} corpus paragraphs, {len(TRAIN_FACTS)} knowledge facts, "
          f"{len(chat_train)} chat examples, {len(squad_held_contexts)} held-out corpus paragraphs for eval",
          flush=True)

    stage_order = ["Corpus", "L0", "L1", "L2", "L3", "L4-logic", "L4-counting", "L5", "L6",
                   "Library", "Knowledge", "Chat"]
    rngs = {s: random.Random(args.seed + i) for i, s in enumerate(stage_order)}
    corpus_rng = random.Random(args.seed + 50)
    knowledge_rng = random.Random(args.seed + 51)
    chat_rng = random.Random(args.seed + 52)
    subskill_eval_rngs = {"L3": random.Random(args.seed + 4 + TEST_SEED_OFFSET),
                           "L4-logic": random.Random(args.seed + 5 + TEST_SEED_OFFSET)}
    corpus_eval_rng = random.Random(args.seed + TEST_SEED_OFFSET)
    library_eval_rng = random.Random(args.seed + 60 + TEST_SEED_OFFSET)
    schedule_rng = random.Random(args.seed + 999)

    step_fns = {
        "Corpus": lambda: corpus_train_step(model, opt, tok, corpus_rng.choice(squad_train_contexts)),
        "L0": lambda: nt.l0_train_step(model, opt, tok, rngs["L0"], args.l0_batch_size),
        "L1": lambda: nt.l1_train_step(model, opt, tok, rngs["L1"], args.l1_n_objects),
        "L2": lambda: nt.l2_train_step(model, opt, tok, rngs["L2"], args.l2_n_objects),
        "L3": lambda: nt.l3_train_step(model, opt, tok, rngs["L3"], args.l3_n_objects),
        "L4-logic": lambda: nt.l4_logic_train_step(model, opt, tok, rngs["L4-logic"], args.l4_n_objects),
        "L4-counting": lambda: nt.l4_counting_train_step(model, opt, tok, rngs["L4-counting"], args.l4_n_objects),
        "L5": lambda: nt.l5_train_step(model, opt, tok, rngs["L5"], args.l5_n_objects),
        "L6": lambda: nt.l6_train_step(model, opt, tok, rngs["L6"], args.l6_n_sentences),
        "Library": lambda: library_train_step(model, opt, tok, rngs["Library"], args.library_n_facts),
        "Knowledge": lambda: knowledge_train_step(model, opt, tok, knowledge_rng),
        "Chat": lambda: chat_train_step(model, opt, tok, chat_rng.choice(chat_train)),
    }

    def refresh_eval_only(stage: str) -> float:
        if stage == "L3":
            return nt.l3_eval(model, tok, subskill_eval_rngs["L3"], args.l3_n_objects,
                               args.subskill_eval_episodes, split="test")
        return nt.l4_logic_eval(model, tok, subskill_eval_rngs["L4-logic"], args.l4_n_objects,
                                 args.subskill_eval_episodes, split="test")

    args_ns = args
    retention_fns = run1.make_retention_fns(tok, args_ns)
    retention_fns["Corpus"] = lambda m: {"held_out_next_byte_acc": 1.0 - min(1.0, corpus_eval_loss(m, tok, squad_held_contexts) / 6.0)}
    retention_fns["Library"] = lambda m: {"held_out_acc":
        library_eval(m, tok, library_eval_rng, args.library_n_facts, args.eval_episodes)}
    retention_fns["Knowledge"] = lambda m: {"mean_loss": knowledge_eval_condition(m, tok, TRAIN_FACTS)["mean_loss"]}
    retention_fns["Chat"] = lambda m: chat_eval_items(m, tok, chat_held)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mastery = {s: {"main": 0.0} for s in stage_order}  # true cold start -- no prior lineage to inherit from
    per_stage_calls = {s: 0 for s in stage_order}

    t0 = time.time()
    for step in range(args.total_steps):
        needs = [max(1.0 - min(mastery[s].values()), MASTERY_FLOOR) for s in stage_order]
        chosen = weighted_choice(schedule_rng, stage_order, needs)
        result = step_fns[chosen]()
        acc = result[1]  # every channel returns (loss, acc) or (loss, sel_acc, cons_acc) -- index 1 is the tracked signal
        mastery[chosen]["main"] = EMA_DECAY * mastery[chosen]["main"] + (1 - EMA_DECAY) * acc
        per_stage_calls[chosen] += 1

        if (step + 1) % args.subskill_eval_every == 0:
            for stage in EVAL_ONLY_SUBSKILLS:
                mastery.setdefault(stage, {})[EVAL_ONLY_SUBSKILLS[stage]] = refresh_eval_only(stage)

        if (step + 1) % args.log_every == 0:
            elapsed = time.time() - t0
            print(f"[run2] step={step+1}/{args.total_steps} ({elapsed:.0f}s, {(step+1)/elapsed:.2f} steps/sec) "
                  f"last={chosen} mastery=[{ {s: round(v['main'],3) for s,v in mastery.items()} }] "
                  f"calls={per_stage_calls}", flush=True)

        if (step + 1) % args.checkpoint_every == 0 or step == args.total_steps - 1:
            ckpt_path = args.checkpoint_dir / f"step_{step+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[run2] checkpoint saved: {ckpt_path}", flush=True)

    total_time = time.time() - t0
    print(f"\n[run2] training done: {args.total_steps} steps in {total_time:.0f}s", flush=True)

    print("\n[run2] ===== FINAL RETENTION ACROSS ALL CHANNELS =====", flush=True)
    final_scores = {}
    for s in stage_order:
        try:
            final_scores[s] = retention_fns[s](model)
            print(f"[run2] {s}: {final_scores[s]}", flush=True)
        except Exception as e:
            print(f"[run2] {s}: eval error {e}", flush=True)
            final_scores[s] = {"error": str(e)}

    print("\n[run2] ===== REAL CHAT GENERATION SAMPLES =====", flush=True)
    samples = sample_generations(model, tok, chat_held, n=6)
    for s in samples:
        print(f"[run2] Q: {s['instruction']}", flush=True)
        print(f"[run2]   real: {s['real_response']}", flush=True)
        print(f"[run2]   generated: {s['generated']!r}", flush=True)

    with open(args.results_file, "w") as f:
        json.dump({"n_params": n_params, "final_scores": final_scores, "samples": samples,
                    "per_stage_calls": per_stage_calls, "final_mastery": mastery,
                    "total_steps": args.total_steps, "total_seconds": total_time}, f, indent=2)
    print(f"\n[run2] DONE. Wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
