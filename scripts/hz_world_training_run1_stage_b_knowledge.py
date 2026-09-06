#!/usr/bin/env python3
"""Hatchling World Training Run 1, Stage B's real knowledge-acquisition
test (plans/Hatchling world.md section 0.6, user-directed): does a real
fact, trained via the ordinary next-byte LM objective, survive into
theta and generalize past its exact trained wording -- not O(1)
retrieval, not another colored-object task.

real fact (TRAIN_FACTS) -> theta -> later held-out completion, with
four real, disclosed controls (`hatchling_world/knowledge/facts.py`):
SEEN (trained prompt), PARAPHRASE (same fact, different wording, never
trained), UNSEEN (a different real fact, same kind, never trained),
WRONG (a trained prompt paired with another fact's real completion).
Measured via `HZLanguageModel.lm_forward` teacher-forced completion
loss -- a genuine autoregressive next-byte predictor, the exact same
cloze-likelihood methodology real LM knowledge probes use. No new
model code.

Continues from C_11 (`results/local/hz_world_run1_stage_b_library_
subskill/after_Library.pt`) -- the persistent lineage, not a fresh
model -- with "Knowledge" as an 11th per-subskill-scheduled channel
alongside the existing 10, so this tests acquisition AND non-forgetting
in the same run, not knowledge in isolation.
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
from hz_world_training_run1_stage_a_corpus import CorpusPool, corpus_train_step, corpus_eval  # noqa: E402
from hz_world_training_run1_stage_a_adaptive import weighted_choice, EMA_DECAY, MASTERY_FLOOR  # noqa: E402
from hz_world_training_run1_stage_b_library import library_train_step, library_eval  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.language.tokenizer import NOVEL_LABELS  # noqa: E402
from hatchling_world.knowledge import TRAIN_FACTS, HELD_OUT_FACTS, PARAPHRASE_PROBES, WRONG_COMPLETION_PROBES  # noqa: E402

TEST_SEED_OFFSET = nt.TEST_SEED_OFFSET
EVAL_ONLY_SUBSKILLS = {"L3": "unseen", "L4-logic": "unseen"}


def knowledge_tensors(tok, prompt: str, completion: str):
    prompt_ids = tok.encode(prompt, add_bos=True, add_eos=False)
    full_ids = tok.encode(prompt + completion, add_bos=True, add_eos=True)
    return torch.tensor([full_ids]), len(prompt_ids)


def knowledge_loss_and_acc(model, tok, prompt: str, completion: str, backward: bool):
    full_ids, prompt_len = knowledge_tensors(tok, prompt, completion)
    logits = model.lm_forward(full_ids)  # (1, T-1, V)
    target = full_ids[:, 1:]
    T_minus_1 = target.shape[1]
    positions = torch.arange(T_minus_1)
    mask = (positions + 1) >= prompt_len  # loss only over the COMPLETION's byte positions
    loss_per_pos = F.cross_entropy(logits.reshape(-1, tok.vocab_size), target.reshape(-1),
                                    reduction="none").reshape(1, T_minus_1)
    masked_loss = (loss_per_pos * mask.float()).sum() / mask.float().sum().clamp_min(1)
    with torch.no_grad():
        pred = logits.argmax(-1)
        acc = (((pred == target) & mask).float().sum() / mask.float().sum().clamp_min(1)).item()
    if backward:
        return masked_loss, acc
    return masked_loss.item(), acc


def knowledge_train_step(model, opt, tok, rng):
    prompt, completion = rng.choice(TRAIN_FACTS)
    loss, acc = knowledge_loss_and_acc(model, tok, prompt, completion, backward=True)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item(), acc


def knowledge_eval_condition(model, tok, probes: list) -> dict:
    with torch.no_grad():
        losses, accs = [], []
        for prompt, completion in probes:
            loss, acc = knowledge_loss_and_acc(model, tok, prompt, completion, backward=False)
            losses.append(loss); accs.append(acc)
    return {"mean_loss": sum(losses) / len(losses), "mean_byte_acc": sum(accs) / len(accs)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                         default=Path("results/local/hz_world_run1_stage_b_library_subskill/after_Library.pt"))
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--corpus-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_train.jsonl"))
    parser.add_argument("--corpus-val-data", type=Path, default=Path("data/packed/hz0h_bytes_25m_val.jsonl"))
    parser.add_argument("--corpus-max-lines", type=int, default=20000)
    parser.add_argument("--corpus-val-max-lines", type=int, default=2000)
    parser.add_argument("--corpus-eval-windows", type=int, default=100)
    parser.add_argument("--library-n-facts", type=int, default=20)
    parser.add_argument("--knowledge-steps", type=int, default=2000)
    parser.add_argument("--l0-batch-size", type=int, default=16)
    parser.add_argument("--l1-n-objects", type=int, default=4)
    parser.add_argument("--l2-n-objects", type=int, default=4)
    parser.add_argument("--l3-n-objects", type=int, default=4)
    parser.add_argument("--l4-n-objects", type=int, default=4)
    parser.add_argument("--l5-n-objects", type=int, default=4)
    parser.add_argument("--l6-n-sentences", type=int, default=3)
    parser.add_argument("--eval-episodes", type=int, default=150)
    parser.add_argument("--subskill-eval-every", type=int, default=100)
    parser.add_argument("--subskill-eval-episodes", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("results/local/hz_world_run1_stage_b_knowledge"))
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_world_run1_stage_b_knowledge_retention.json"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = ByteTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    print(f"[stage-b-knowledge] loading checkpoint (C_11): {args.checkpoint}", flush=True)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[stage-b-knowledge] CONTINUING the persistent trainee: n_params={n_params}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    corpus_pool = CorpusPool(args.corpus_data, args.corpus_max_lines)
    corpus_val_pool = CorpusPool(args.corpus_val_data, args.corpus_val_max_lines)
    print(f"[stage-b-knowledge] loaded {len(corpus_pool.windows)} corpus train windows", flush=True)

    old_stages = ["Corpus", "L0", "L1", "L2", "L3", "L4-logic", "L4-counting", "L5", "L6", "Library"]
    all_stages = old_stages + ["Knowledge"]
    rngs = {
        "Corpus": random.Random(args.seed), "L0": random.Random(args.seed + 1), "L1": random.Random(args.seed + 2),
        "L2": random.Random(args.seed + 3), "L3": random.Random(args.seed + 4),
        "L4-logic": random.Random(args.seed + 5), "L4-counting": random.Random(args.seed + 6),
        "L5": random.Random(args.seed + 7), "L6": random.Random(args.seed + 8),
        "Library": random.Random(args.seed + 10), "Knowledge": random.Random(args.seed + 11),
    }
    subskill_eval_rngs = {"L3": random.Random(args.seed + 4 + TEST_SEED_OFFSET),
                           "L4-logic": random.Random(args.seed + 5 + TEST_SEED_OFFSET)}
    corpus_eval_rng = random.Random(args.seed + TEST_SEED_OFFSET)
    library_eval_rng = random.Random(args.seed + 10 + TEST_SEED_OFFSET)
    schedule_rng = random.Random(args.seed + 999)

    step_fns = {
        "Corpus": lambda: corpus_train_step(model, opt, corpus_pool, rngs["Corpus"], tok.vocab_size),
        "L0": lambda: nt.l0_train_step(model, opt, tok, rngs["L0"], args.l0_batch_size),
        "L1": lambda: nt.l1_train_step(model, opt, tok, rngs["L1"], args.l1_n_objects),
        "L2": lambda: nt.l2_train_step(model, opt, tok, rngs["L2"], args.l2_n_objects),
        "L3": lambda: nt.l3_train_step(model, opt, tok, rngs["L3"], args.l3_n_objects),
        "L4-logic": lambda: nt.l4_logic_train_step(model, opt, tok, rngs["L4-logic"], args.l4_n_objects),
        "L4-counting": lambda: nt.l4_counting_train_step(model, opt, tok, rngs["L4-counting"], args.l4_n_objects),
        "L5": lambda: nt.l5_train_step(model, opt, tok, rngs["L5"], args.l5_n_objects),
        "L6": lambda: nt.l6_train_step(model, opt, tok, rngs["L6"], args.l6_n_sentences),
        "Library": lambda: library_train_step(model, opt, tok, rngs["Library"], args.library_n_facts),
        "Knowledge": lambda: knowledge_train_step(model, opt, tok, rngs["Knowledge"]),
    }

    def refresh_eval_only(stage: str) -> float:
        if stage == "L3":
            return nt.l3_eval(model, tok, subskill_eval_rngs["L3"], args.l3_n_objects,
                               args.subskill_eval_episodes, split="test")
        return nt.l4_logic_eval(model, tok, subskill_eval_rngs["L4-logic"], args.l4_n_objects,
                                 args.subskill_eval_episodes, split="test")

    retention_fns = run1.make_retention_fns(tok, args)
    retention_fns["Corpus"] = lambda m: {"held_out_next_byte_acc":
        corpus_eval(m, corpus_val_pool, corpus_eval_rng, tok.vocab_size, args.corpus_eval_windows)}
    retention_fns["Library"] = lambda m: {"held_out_acc":
        library_eval(m, tok, library_eval_rng, args.library_n_facts, args.eval_episodes)}
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def knowledge_report(m):
        return {"SEEN": knowledge_eval_condition(m, tok, TRAIN_FACTS),
                "PARAPHRASE": knowledge_eval_condition(m, tok, PARAPHRASE_PROBES),
                "UNSEEN": knowledge_eval_condition(m, tok, HELD_OUT_FACTS),
                "WRONG": knowledge_eval_condition(m, tok, WRONG_COMPLETION_PROBES)}

    print("\n[stage-b-knowledge] ===== BEFORE Knowledge: retention on the inherited lineage (C_11) =====", flush=True)
    before_scores = {s: retention_fns[s](model) for s in old_stages}
    for s in old_stages:
        print(f"[stage-b-knowledge] before Knowledge, {s}={before_scores[s]}", flush=True)
    before_knowledge = knowledge_report(model)
    print(f"[stage-b-knowledge] before Knowledge, knowledge probes (untrained baseline)={before_knowledge}", flush=True)

    def initial_mastery(stage: str) -> dict:
        s = before_scores[stage]
        if stage == "Corpus":
            return {"main": s["held_out_next_byte_acc"]}
        if stage == "L2":
            return {"sel": s["sel_acc"], "cons": s["cons_acc"]}
        if stage in ("L3", "L4-logic"):
            return {"seen": s["seen_combo_acc"], "unseen": s["unseen_combo_acc"]}
        acc_key = "next_token_acc" if stage == "L0" else "held_out_acc"
        return {"main": s[acc_key]}

    mastery = {s: initial_mastery(s) for s in old_stages}
    mastery["Knowledge"] = {"main": 0.0}  # brand new, max priority
    print(f"[stage-b-knowledge] initial per-subskill mastery: {mastery}", flush=True)

    print(f"\n[stage-b-knowledge] ===== Knowledge phase: {args.knowledge_steps} steps, PER-SUBSKILL "
          f"mastery-weighted among all {len(all_stages)} stages =====", flush=True)
    per_stage_calls = {s: 0 for s in all_stages}
    t0 = time.time()
    for step in range(args.knowledge_steps):
        needs = [max(1.0 - min(mastery[s].values()), MASTERY_FLOOR) for s in all_stages]
        chosen = weighted_choice(schedule_rng, all_stages, needs)
        result = step_fns[chosen]()
        if chosen == "L2":
            _, sel_acc, cons_acc = result
            mastery["L2"]["sel"] = EMA_DECAY * mastery["L2"]["sel"] + (1 - EMA_DECAY) * sel_acc
            mastery["L2"]["cons"] = EMA_DECAY * mastery["L2"]["cons"] + (1 - EMA_DECAY) * cons_acc
        elif chosen in ("L3", "L4-logic"):
            _, acc = result
            mastery[chosen]["seen"] = EMA_DECAY * mastery[chosen]["seen"] + (1 - EMA_DECAY) * acc
        else:
            _, acc = result
            mastery[chosen]["main"] = EMA_DECAY * mastery[chosen]["main"] + (1 - EMA_DECAY) * acc
        per_stage_calls[chosen] += 1

        if (step + 1) % args.subskill_eval_every == 0:
            for stage, sub in EVAL_ONLY_SUBSKILLS.items():
                fresh = refresh_eval_only(stage)
                mastery[stage][sub] = EMA_DECAY * mastery[stage][sub] + (1 - EMA_DECAY) * fresh

        if (step + 1) % args.log_every == 0:
            print(f"[stage-b-knowledge] step={step+1}/{args.knowledge_steps} last_sampled={chosen} "
                  f"mastery=[{ {s: mastery[s] for s in all_stages} }] calls_so_far={per_stage_calls}", flush=True)
    print(f"[stage-b-knowledge] Knowledge phase done in {time.time()-t0:.0f}s.", flush=True)

    print("\n[stage-b-knowledge] ===== AFTER Knowledge: retention on ALL 10 prior stages =====", flush=True)
    after_scores = {s: retention_fns[s](model) for s in old_stages}
    for s in old_stages:
        print(f"[stage-b-knowledge] after Knowledge, {s}={after_scores[s]}", flush=True)

    after_knowledge = knowledge_report(model)
    print(f"\n[stage-b-knowledge] === KNOWLEDGE PROBE RESULT ===")
    for cond, res in after_knowledge.items():
        print(f"[stage-b-knowledge] {cond}: {res}", flush=True)

    ckpt_path = args.checkpoint_dir / "after_Knowledge.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[stage-b-knowledge] checkpoint saved (C_12): {ckpt_path}", flush=True)

    with open(args.results_file, "w") as f:
        json.dump({"before_scores": before_scores, "after_scores": after_scores,
                    "before_knowledge": before_knowledge, "after_knowledge": after_knowledge,
                    "per_stage_calls": per_stage_calls, "final_mastery": mastery,
                    "n_params": n_params}, f, indent=2)
    print(f"\n[stage-b-knowledge] DONE. Wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
