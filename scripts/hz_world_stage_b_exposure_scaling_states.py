#!/usr/bin/env python3
"""Real, decisive fork experiment (user-directed): does the 102-fact
scale-up's failure on states/elements/planets (plans/Hatchling
world.md section 0.6) reflect UNDERTRAINING (the 102-fact run gave
each fact only ~13.7 exposures, vs the successful 12-fact proof's ~31 --
a real confound) or a genuine capacity/representation limit?

Minimal, controlled test: US state capitals (a NEW domain relative to
the original 12-fact proof, chosen -- per the user's own reasoning --
because its answers are distinctive strings, not a tiny shared number/
ordinal vocabulary like elements/planets). Four INDEPENDENT arms, each
starting fresh from the SAME checkpoint (C_13, `results/local/
hz_world_run1_stage_b_knowledge_v2/after_Knowledge_v2.pt`), trained on
ONLY the states category's 40 train facts for EXACTLY E full shuffled
passes (deterministic exposures/fact, not expected-value random
sampling), E in {5, 15, 30, 60}. Same facts, same evaluation, same
architecture across all four arms -- exposure count is the only
variable.

Real, pre-registered success/failure reading, stated before running:
  Delta_truth(E) = L_wrong(E) - L_correct(E)
  Delta_para(E)  = L_unseen(E) - L_paraphrase(E)
If both trend positive and clearly separate from ~0 as E increases:
undertraining was the real cause -- HZ can acquire this domain given
enough exposure. If both stay near zero or negative even at E=60: a
real capacity/representation limit, not a training-budget problem --
the honest next move would be scaling model capacity, not more
scheduler tuning.
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
from hatchling_world.knowledge.facts_v2 import CATEGORIES  # noqa: E402


def build_states_split(seed: int):
    table, train_tmpl, para_tmpl, held_out_frac = CATEGORIES["states"]
    keys = sorted(table.keys())
    random.Random(seed).shuffle(keys)
    n_held = max(1, round(len(keys) * held_out_frac))
    held_keys, train_keys = keys[:n_held], keys[n_held:]

    train_facts = [(train_tmpl.format(k=k, v=table[k]), table[k]) for k in train_keys]
    paraphrase_probes = [(para_tmpl.format(k=k, v=table[k]), k) for k in train_keys]
    held_out_facts = [(train_tmpl.format(k=k, v=table[k]), table[k]) for k in held_keys]
    wrong_probes = []
    shuffled = train_keys[:]
    random.Random(seed + 1).shuffle(shuffled)
    for i, k in enumerate(train_keys):
        wrong_k = shuffled[(i + 1) % len(shuffled)]
        if wrong_k == k:
            continue
        wrong_probes.append((train_tmpl.format(k=k, v=table[k]), table[wrong_k]))
    return train_facts, paraphrase_probes, held_out_facts, wrong_probes


def train_one_arm(base_state_dict, tok, train_facts, exposures: int, seed: int, args):
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    model.load_state_dict(copy.deepcopy(base_state_dict))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(seed)

    for epoch in range(exposures):
        order = train_facts[:]
        rng.shuffle(order)
        for prompt, completion in order:
            loss, _ = knowledge_loss_and_acc(model, tok, prompt, completion, backward=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                         default=Path("results/local/hz_world_run1_stage_b_knowledge_v2/after_Knowledge_v2.pt"))
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--exposures", type=int, nargs="+", default=[5, 15, 30, 60])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-file", type=Path,
                         default=Path("results/local/hz_world_stage_b_exposure_scaling_states.json"))
    args = parser.parse_args()

    tok = ByteTokenizer()
    print(f"[exposure-scaling] loading checkpoint (C_13): {args.checkpoint}", flush=True)
    base_state_dict = torch.load(args.checkpoint, map_location="cpu")

    train_facts, paraphrase_probes, held_out_facts, wrong_probes = build_states_split(seed=args.seed)
    print(f"[exposure-scaling] states split: {len(train_facts)} train, {len(paraphrase_probes)} paraphrase, "
          f"{len(held_out_facts)} held-out, {len(wrong_probes)} wrong-completion", flush=True)

    # Real baseline: the C_13 checkpoint itself, untouched, before any
    # of these arms train further -- same reference point for all arms.
    base_model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                                  workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                                  n_qa_labels=len(NOVEL_LABELS))
    base_model.load_state_dict(base_state_dict)
    base_model.eval()
    baseline = {
        "SEEN": knowledge_eval_condition(base_model, tok, train_facts),
        "WRONG": knowledge_eval_condition(base_model, tok, wrong_probes),
        "PARAPHRASE": knowledge_eval_condition(base_model, tok, paraphrase_probes),
        "UNSEEN": knowledge_eval_condition(base_model, tok, held_out_facts),
    }
    print(f"[exposure-scaling] E=0 (C_13 baseline, untouched): {baseline}", flush=True)
    d_truth0 = baseline["WRONG"]["mean_loss"] - baseline["SEEN"]["mean_loss"]
    d_para0 = baseline["UNSEEN"]["mean_loss"] - baseline["PARAPHRASE"]["mean_loss"]
    print(f"[exposure-scaling] E=0: delta_truth={d_truth0:+.3f} delta_para={d_para0:+.3f}", flush=True)

    results = {"E=0": {"scores": baseline, "delta_truth": d_truth0, "delta_para": d_para0}}
    for E in args.exposures:
        print(f"\n[exposure-scaling] ===== arm E={E} (exactly {E} shuffled passes over "
              f"{len(train_facts)} train facts = {E * len(train_facts)} steps) =====", flush=True)
        model = train_one_arm(base_state_dict, tok, train_facts, E, seed=args.seed + E, args=args)
        model.eval()
        scores = {
            "SEEN": knowledge_eval_condition(model, tok, train_facts),
            "WRONG": knowledge_eval_condition(model, tok, wrong_probes),
            "PARAPHRASE": knowledge_eval_condition(model, tok, paraphrase_probes),
            "UNSEEN": knowledge_eval_condition(model, tok, held_out_facts),
        }
        d_truth = scores["WRONG"]["mean_loss"] - scores["SEEN"]["mean_loss"]
        d_para = scores["UNSEEN"]["mean_loss"] - scores["PARAPHRASE"]["mean_loss"]
        print(f"[exposure-scaling] E={E}: {scores}", flush=True)
        print(f"[exposure-scaling] E={E}: delta_truth={d_truth:+.3f} (>0 means correct preferred) "
              f"delta_para={d_para:+.3f} (>0 means paraphrase generalizes)", flush=True)
        results[f"E={E}"] = {"scores": scores, "delta_truth": d_truth, "delta_para": d_para}

    print("\n[exposure-scaling] === SUMMARY (delta_truth, delta_para vs exposure) ===")
    for key, r in results.items():
        print(f"  {key}: delta_truth={r['delta_truth']:+.3f}  delta_para={r['delta_para']:+.3f}")
    trend_truth = [results[f"E={e}"]["delta_truth"] for e in args.exposures]
    trend_para = [results[f"E={e}"]["delta_para"] for e in args.exposures]
    both_positive_at_max = trend_truth[-1] > 0 and trend_para[-1] > 0
    monotonic_ish = trend_truth[-1] > trend_truth[0] and trend_para[-1] > trend_para[0]
    print(f"\n[exposure-scaling] VERDICT: both deltas positive at max exposure (E={args.exposures[-1]})? "
          f"{both_positive_at_max}. Trending up from E={args.exposures[0]} to E={args.exposures[-1]}? {monotonic_ish}.")
    if both_positive_at_max and monotonic_ish:
        print("[exposure-scaling] -> UNDERTRAINING was the real cause; HZ can acquire this domain given enough exposure.")
    else:
        print("[exposure-scaling] -> Does NOT clearly resolve with more exposure alone; real capacity/representation question remains open.")

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[exposure-scaling] DONE. Wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
