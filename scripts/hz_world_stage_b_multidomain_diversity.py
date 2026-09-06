#!/usr/bin/env python3
"""Real, decisive test (user-directed): does the wording-diversity +
matched-repetition Stage B recipe -- proven on US state capitals
(`scripts/hz_world_stage_b_wording_diversity_states.py`: delta_truth
+0.410, delta_para +3.508, both far clearing the pre-committed bar) --
generalize ACROSS knowledge domains, or was it specific to the states
domain?

Applies the identical recipe to the full 102-fact, 4-category dataset
(`hatchling_world.knowledge.facts_v2.CATEGORIES`: world capitals,
US state capitals, chemical elements, planets), using per-category
templates (`hatchling_world.knowledge.templates_v3`, 5 training
wordings + 1 held-out-from-training wording per category, same
discipline as the states experiment). One model, one training run, all
four domains trained together, each (fact, template) pair getting
EXACTLY 30 exposures (matched repetition, per the proven recipe) --
102 facts x 5 templates x 30 exposures = 15,300 total updates.

Real, pre-registered, PER-CATEGORY success bar (not just aggregate,
per the user's explicit instruction): Delta_truth > 0 AND
Delta_para > 0 for EACH of the four categories independently. If all
four pass: the recipe generalizes across domains, and the next real
step is a factual-prose corpus + model scaling. If elements/planets
still fail despite matched diversity + repetition: real evidence for a
representation/capacity limit, not undertraining or curriculum design.

Continues from C_13 (the persistent Stage B lineage's own checkpoint,
same starting point as every Knowledge experiment this thread) as a
focused, standalone fork -- same methodology as the states-only
experiments, not re-merged into the full 11-stage scheduler this run
(matches the user's own "nothing else" scoping).
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
from hatchling_world.knowledge.templates_v3 import TEMPLATES_BY_CATEGORY  # noqa: E402


def build_all_splits(seed: int):
    """Real per-category entity-level train/held-out split, same
    discipline as facts_v2.build_knowledge_v2 -- whole entities
    reserved, never seen in ANY wording."""
    splits = {}
    for cat, (table, _train_tmpl, _para_tmpl, held_out_frac) in CATEGORIES.items():
        keys = sorted(table.keys())
        random.Random(seed).shuffle(keys)
        n_held = max(1, round(len(keys) * held_out_frac))
        splits[cat] = {"held": keys[:n_held], "train": keys[n_held:], "table": table}
    return splits


def make_all_probes(splits):
    """Per-category (SEEN, WRONG, PARAPHRASE, UNSEEN) probes, using
    template[0] as the canonical wording (matching SEEN/WRONG/UNSEEN)
    and the held-out-from-training wording for PARAPHRASE."""
    probes = {}
    for cat, split in splits.items():
        table = split["table"]
        canonical = TEMPLATES_BY_CATEGORY[cat]["train"][0]
        held_out_tmpl = TEMPLATES_BY_CATEGORY[cat]["held_out"]
        train_keys, held_keys = split["train"], split["held"]

        seen = [(canonical.format(k=k), table[k]) for k in train_keys]
        unseen = [(canonical.format(k=k), table[k]) for k in held_keys]
        paraphrase = [(held_out_tmpl.format(k=k), table[k]) for k in train_keys]

        shuffled = train_keys[:]
        random.Random(1).shuffle(shuffled)
        wrong = []
        for i, k in enumerate(train_keys):
            wrong_k = shuffled[(i + 1) % len(shuffled)]
            if wrong_k == k:
                continue
            wrong.append((canonical.format(k=k), table[wrong_k]))
        probes[cat] = {"SEEN": seen, "WRONG": wrong, "PARAPHRASE": paraphrase, "UNSEEN": unseen}
    return probes


def train_step(model, opt, tok, prompt, completion):
    loss, _ = knowledge_loss_and_acc(model, tok, prompt, completion, backward=True)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


def train_multidomain(base_state_dict, tok, splits, exposures_per_fact_per_template, seed, args):
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    model.load_state_dict(copy.deepcopy(base_state_dict))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(seed)

    # Flat list of (category, key) across ALL four domains.
    all_facts = [(cat, k) for cat, split in splits.items() for k in split["train"]]
    n_templates = 5
    total_updates = 0
    for epoch in range(exposures_per_fact_per_template):
        for template_idx in range(n_templates):
            order = all_facts[:]
            rng.shuffle(order)
            for cat, k in order:
                tmpl = TEMPLATES_BY_CATEGORY[cat]["train"][template_idx]
                table = splits[cat]["table"]
                train_step(model, opt, tok, tmpl.format(k=k), table[k])
                total_updates += 1
        if (epoch + 1) % max(1, exposures_per_fact_per_template // 10) == 0:
            print(f"[multidomain-diversity] epoch {epoch+1}/{exposures_per_fact_per_template} "
                  f"({total_updates} updates so far)", flush=True)
    return model, total_updates


def evaluate_all(model, tok, probes):
    results = {}
    for cat, cp in probes.items():
        scores = {name: knowledge_eval_condition(model, tok, plist) for name, plist in cp.items()}
        d_truth = scores["WRONG"]["mean_loss"] - scores["SEEN"]["mean_loss"]
        d_para = scores["UNSEEN"]["mean_loss"] - scores["PARAPHRASE"]["mean_loss"]
        results[cat] = {"scores": scores, "delta_truth": d_truth, "delta_para": d_para}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                         default=Path("results/local/hz_world_run1_stage_b_knowledge_v2/after_Knowledge_v2.pt"))
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--exposures-per-template", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-out", type=Path,
                         default=Path("results/local/hz_world_stage_b_multidomain_diversity/after_diversity.pt"))
    parser.add_argument("--results-file", type=Path,
                         default=Path("results/local/hz_world_stage_b_multidomain_diversity.json"))
    args = parser.parse_args()

    tok = ByteTokenizer()
    print(f"[multidomain-diversity] loading checkpoint (C_13): {args.checkpoint}", flush=True)
    base_state_dict = torch.load(args.checkpoint, map_location="cpu")

    splits = build_all_splits(seed=args.seed)
    for cat, s in splits.items():
        print(f"[multidomain-diversity] {cat}: {len(s['train'])} train, {len(s['held'])} held-out", flush=True)
    probes = make_all_probes(splits)

    n_facts = sum(len(s["train"]) for s in splits.values())
    total_updates = n_facts * 5 * args.exposures_per_template
    print(f"[multidomain-diversity] {n_facts} total train facts x 5 templates x "
          f"{args.exposures_per_template} exposures/fact/template = {total_updates} total updates", flush=True)

    base_model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                                  workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                                  n_qa_labels=len(NOVEL_LABELS))
    base_model.load_state_dict(base_state_dict)
    baseline = evaluate_all(base_model, tok, probes)
    print("\n[multidomain-diversity] ===== BASELINE (C_13, untouched) =====", flush=True)
    for cat, r in baseline.items():
        print(f"[multidomain-diversity] baseline {cat}: delta_truth={r['delta_truth']:+.3f} "
              f"delta_para={r['delta_para']:+.3f}", flush=True)

    print(f"\n[multidomain-diversity] ===== training: {n_facts} facts x 5 templates x "
          f"{args.exposures_per_template} exposures =====", flush=True)
    model, actual_updates = train_multidomain(base_state_dict, tok, splits, args.exposures_per_template,
                                               args.seed + 300, args)
    print(f"[multidomain-diversity] training done, {actual_updates} total updates applied", flush=True)

    after = evaluate_all(model, tok, probes)
    print("\n[multidomain-diversity] ===== AFTER (per-category results) =====", flush=True)
    all_passed = True
    for cat, r in after.items():
        passed = r["delta_truth"] > 0 and r["delta_para"] > 0
        all_passed = all_passed and passed
        print(f"[multidomain-diversity] {cat}: {r['scores']}", flush=True)
        print(f"[multidomain-diversity] {cat}: delta_truth={r['delta_truth']:+.3f} "
              f"delta_para={r['delta_para']:+.3f}  {'PASS' if passed else 'FAIL'}", flush=True)

    print("\n[multidomain-diversity] === SUMMARY (baseline -> after, per category) ===")
    for cat in CATEGORIES:
        b, a = baseline[cat], after[cat]
        print(f"  {cat}: delta_truth {b['delta_truth']:+.3f} -> {a['delta_truth']:+.3f}   "
              f"delta_para {b['delta_para']:+.3f} -> {a['delta_para']:+.3f}")

    print(f"\n[multidomain-diversity] === PRE-COMMITTED VERDICT (per-category, not aggregate) ===")
    print(f"ALL FOUR categories pass (delta_truth>0 AND delta_para>0)? {all_passed}")
    if all_passed:
        print("[multidomain-diversity] PASS -- the Stage B recipe generalizes across domains. "
              "Next real step: factual-prose corpus + HZ-Micro scaling.")
    else:
        failing = [cat for cat in CATEGORIES if not (after[cat]["delta_truth"] > 0 and after[cat]["delta_para"] > 0)]
        print(f"[multidomain-diversity] FAIL -- {failing} did not clear the bar despite matched diversity + "
              f"repetition. Real evidence for a representation/capacity limit in those domains, not "
              f"undertraining or curriculum design.")

    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.checkpoint_out)
    print(f"\n[multidomain-diversity] checkpoint saved: {args.checkpoint_out}", flush=True)

    with open(args.results_file, "w") as f:
        json.dump({"baseline": baseline, "after": after, "all_passed": all_passed,
                    "total_updates": actual_updates}, f, indent=2)
    print(f"[multidomain-diversity] DONE. Wrote {args.results_file}", flush=True)


if __name__ == "__main__":
    main()
