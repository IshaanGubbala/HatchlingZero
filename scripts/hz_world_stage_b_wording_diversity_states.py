#!/usr/bin/env python3
"""Real, decisive follow-up to the exposure-scaling fork (user-directed):
does WORDING DIVERSITY, not more repetition of one template, fix the
paraphrase-generalization collapse found there? That experiment showed
more exposure to a SINGLE template drives discrimination up
(Delta_truth: +0.028 -> +0.989) while driving paraphrase generalization
down (Delta_para: -0.276 -> -0.833) -- real overfitting to the exact
trained byte sequence, not a simple capacity/undertraining story.

Minimal, controlled test: two arms, IDENTICAL total updates (E=30 full
passes over the same 40 US-state-capital facts = 1,200 steps each, the
exposure level the prior experiment already showed a strong, clean
Delta_truth signal at), both starting from the same C_13 checkpoint.

  SINGLE-TEMPLATE arm: every update uses the same one canonical
    template (matching every previous Knowledge run).
  MULTI-TEMPLATE arm: 5 genuinely different sentence templates,
    assigned one per full pass (6 passes each, 5*6=30 passes total) --
    same total 1,200 updates, just distributed across 5 wordings
    instead of repeating one.

A SIXTH template, held out from training in BOTH arms, is used ONLY for
the paraphrase probe -- the model is never tested on a wording it
trained on, in either arm (stricter than the previous experiment's
paraphrase probe, which reused a single fixed paraphrase template that
was itself never varied).

Pre-committed success bar, stated before running: Delta_truth >= 0.3
AND Delta_para > 0 for the multi-template arm. If met without
destroying discrimination, wording diversity is the real fix and the
Stage B recipe becomes "many facts x many phrasings/fact," not "many
repetitions of one phrasing." If Delta_para stays <= 0 even here,
capacity/representation becomes the next real suspect, not before.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hz_world_training_run1_stage_b_knowledge import knowledge_loss_and_acc, knowledge_eval_condition  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.byte_tokenizer import ByteTokenizer  # noqa: E402
from hatchling_world.language.tokenizer import NOVEL_LABELS  # noqa: E402
from hatchling_world.knowledge.facts_v2 import US_STATE_CAPITALS  # noqa: E402

# 5 genuinely different training templates (prompt, completion=capital
# name {v}) plus one held-out-from-training template used ONLY for the
# paraphrase probe -- 6 real, distinct wordings of the same fact.
TRAIN_TEMPLATES = [
    "the capital of the state of {k} is ",
    "{k} has its state capital located in the city of ",
    "the government of the state of {k} is based in ",
    "if you travel to the capital of {k} you would arrive in ",
    "the seat of state government for {k} is the city of ",
]
HELD_OUT_TEMPLATE = "which city serves as the capital of the state of {k}? the answer is "

HELD_OUT_FRAC = 0.20


def build_split(seed: int):
    keys = sorted(US_STATE_CAPITALS.keys())
    random.Random(seed).shuffle(keys)
    n_held = max(1, round(len(keys) * HELD_OUT_FRAC))
    held_keys, train_keys = keys[:n_held], keys[n_held:]
    return train_keys, held_keys


def make_probes(train_keys, held_keys):
    canonical = TRAIN_TEMPLATES[0]
    seen = [(canonical.format(k=k), US_STATE_CAPITALS[k]) for k in train_keys]
    unseen = [(canonical.format(k=k), US_STATE_CAPITALS[k]) for k in held_keys]
    paraphrase = [(HELD_OUT_TEMPLATE.format(k=k), US_STATE_CAPITALS[k]) for k in train_keys]
    shuffled = train_keys[:]
    random.Random(1).shuffle(shuffled)
    wrong = []
    for i, k in enumerate(train_keys):
        wrong_k = shuffled[(i + 1) % len(shuffled)]
        if wrong_k == k:
            continue
        wrong.append((canonical.format(k=k), US_STATE_CAPITALS[wrong_k]))
    return seen, unseen, paraphrase, wrong


def train_step(model, opt, tok, prompt, completion):
    loss, _ = knowledge_loss_and_acc(model, tok, prompt, completion, backward=True)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


def train_single_template(base_state_dict, tok, train_keys, exposures, seed, args):
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    model.load_state_dict(copy.deepcopy(base_state_dict))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(seed)
    tmpl = TRAIN_TEMPLATES[0]
    for _ in range(exposures):
        order = train_keys[:]
        rng.shuffle(order)
        for k in order:
            train_step(model, opt, tok, tmpl.format(k=k), US_STATE_CAPITALS[k])
    return model


def train_multi_template(base_state_dict, tok, train_keys, exposures, seed, args):
    """Same TOTAL updates as the single-template arm (exposures full
    passes over the same facts) -- one template per pass, cycling
    through all 5, so total exposures split evenly across wordings
    rather than the arm simply seeing more data overall."""
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    model.load_state_dict(copy.deepcopy(base_state_dict))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(seed)
    n_templates = len(TRAIN_TEMPLATES)
    for pass_idx in range(exposures):
        tmpl = TRAIN_TEMPLATES[pass_idx % n_templates]
        order = train_keys[:]
        rng.shuffle(order)
        for k in order:
            train_step(model, opt, tok, tmpl.format(k=k), US_STATE_CAPITALS[k])
    return model


def evaluate(model, tok, seen, unseen, paraphrase, wrong):
    model.eval()
    scores = {
        "SEEN": knowledge_eval_condition(model, tok, seen),
        "WRONG": knowledge_eval_condition(model, tok, wrong),
        "PARAPHRASE": knowledge_eval_condition(model, tok, paraphrase),
        "UNSEEN": knowledge_eval_condition(model, tok, unseen),
    }
    d_truth = scores["WRONG"]["mean_loss"] - scores["SEEN"]["mean_loss"]
    d_para = scores["UNSEEN"]["mean_loss"] - scores["PARAPHRASE"]["mean_loss"]
    return scores, d_truth, d_para


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                         default=Path("results/local/hz_world_run1_stage_b_knowledge_v2/after_Knowledge_v2.pt"))
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--exposures", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-file", type=Path,
                         default=Path("results/local/hz_world_stage_b_wording_diversity_states.json"))
    args = parser.parse_args()

    tok = ByteTokenizer()
    print(f"[wording-diversity] loading checkpoint (C_13): {args.checkpoint}", flush=True)
    base_state_dict = torch.load(args.checkpoint, map_location="cpu")

    train_keys, held_keys = build_split(seed=args.seed)
    seen, unseen, paraphrase, wrong = make_probes(train_keys, held_keys)
    print(f"[wording-diversity] {len(train_keys)} train states, {len(held_keys)} held-out states, "
          f"{len(TRAIN_TEMPLATES)} training templates, 1 held-out-from-training template for paraphrase eval",
          flush=True)
    print(f"[wording-diversity] both arms: {args.exposures} exposures/fact = "
          f"{args.exposures * len(train_keys)} total updates (IDENTICAL)", flush=True)

    base_model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                                  workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                                  n_qa_labels=len(NOVEL_LABELS))
    base_model.load_state_dict(base_state_dict)
    baseline_scores, baseline_d_truth, baseline_d_para = evaluate(base_model, tok, seen, unseen, paraphrase, wrong)
    print(f"\n[wording-diversity] BASELINE (C_13, untouched): delta_truth={baseline_d_truth:+.3f} "
          f"delta_para={baseline_d_para:+.3f}", flush=True)

    print(f"\n[wording-diversity] ===== SINGLE-TEMPLATE arm ({args.exposures} exposures, "
          f"template: '{TRAIN_TEMPLATES[0]}') =====", flush=True)
    single_model = train_single_template(base_state_dict, tok, train_keys, args.exposures, args.seed + 100, args)
    single_scores, single_d_truth, single_d_para = evaluate(single_model, tok, seen, unseen, paraphrase, wrong)
    print(f"[wording-diversity] SINGLE-TEMPLATE: {single_scores}", flush=True)
    print(f"[wording-diversity] SINGLE-TEMPLATE: delta_truth={single_d_truth:+.3f} delta_para={single_d_para:+.3f}",
          flush=True)

    print(f"\n[wording-diversity] ===== MULTI-TEMPLATE arm ({args.exposures} exposures across "
          f"{len(TRAIN_TEMPLATES)} templates, identical total updates) =====", flush=True)
    multi_model = train_multi_template(base_state_dict, tok, train_keys, args.exposures, args.seed + 200, args)
    multi_scores, multi_d_truth, multi_d_para = evaluate(multi_model, tok, seen, unseen, paraphrase, wrong)
    print(f"[wording-diversity] MULTI-TEMPLATE: {multi_scores}", flush=True)
    print(f"[wording-diversity] MULTI-TEMPLATE: delta_truth={multi_d_truth:+.3f} delta_para={multi_d_para:+.3f}",
          flush=True)

    print("\n[wording-diversity] === SUMMARY ===")
    print(f"  baseline (C_13):  delta_truth={baseline_d_truth:+.3f}  delta_para={baseline_d_para:+.3f}")
    print(f"  single-template:  delta_truth={single_d_truth:+.3f}  delta_para={single_d_para:+.3f}")
    print(f"  multi-template:   delta_truth={multi_d_truth:+.3f}  delta_para={multi_d_para:+.3f}")

    passed = multi_d_truth >= 0.3 and multi_d_para > 0
    print(f"\n[wording-diversity] === PRE-COMMITTED VERDICT ===")
    print(f"multi-template delta_truth={multi_d_truth:+.3f} (need >=0.3)  "
          f"delta_para={multi_d_para:+.3f} (need >0)")
    if passed:
        print("[wording-diversity] PASS -- wording diversity fixes paraphrase generalization without "
              "destroying discrimination. Stage B recipe: many facts x many phrasings/fact.")
    else:
        print("[wording-diversity] FAIL -- wording diversity alone does not clear the bar. "
              "Capacity/representation becomes the next real suspect.")

    with open(args.results_file, "w") as f:
        json.dump({
            "baseline": {"scores": baseline_scores, "delta_truth": baseline_d_truth, "delta_para": baseline_d_para},
            "single_template": {"scores": single_scores, "delta_truth": single_d_truth, "delta_para": single_d_para},
            "multi_template": {"scores": multi_scores, "delta_truth": multi_d_truth, "delta_para": multi_d_para},
            "passed": passed,
        }, f, indent=2)
    print(f"\n[wording-diversity] DONE. Wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
