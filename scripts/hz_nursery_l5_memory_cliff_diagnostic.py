#!/usr/bin/env python3
"""Diagnosing L5's sharp 2->3-fact memory cliff (plans/Hatchling
world.md: n_facts=2 -> ~50-52%, n_facts=3+ -> exactly chance,
regardless of distractor count), explicit user request 2026-09-05:
"don't immediately enlarge S and declare victory. First determine why
its nominal slots aren't actually functioning like independent facts."
Three diagnostics, run in the user's own specified order, PAPER-0
style -- find where the information disappears before redesigning:

  1. Memory-slot sweep (M_S in {4,8,12,16} -- HZCQPersistentMemoryConfig
     hard-validates memory_slots into [4,16], "the plan's stated 4-16
     range," so 32 isn't reachable within the validated architecture;
     disclosed as a real ceiling on how far this specific diagnostic
     can push, not a choice) on the exact n_facts=3, n_distractors=0
     config that showed the cliff. If the cliff moves with M_S, this is
     straightforward capacity; if it doesn't move at all, M_S isn't the
     bottleneck.

  2. Slot-diversity probe: after teaching K facts (K=1,2,3), measure
     mean pairwise cosine similarity and effective rank (participation
     ratio) across S's M_S slot vectors. If slots increasingly
     converge toward each other as more facts are taught, "8 nominal
     slots" doesn't mean "8 independent memories" -- the gate/write
     mechanism may be overwriting rather than allocating.

  3. Fact-decoding probe ON S ITSELF: after teaching 3 facts (S frozen,
     backbone frozen), train a small linear probe per taught-fact
     query_idx to predict that fact's label DIRECTLY from S, bypassing
     H/qa_head/ws.run entirely. High probe accuracy despite low
     end-to-end recall means the information IS still in S (a
     retrieval/readout failure in H); low probe accuracy means the
     information is genuinely gone from S (a real storage failure).
     This is the one experiment that actually separates the two.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hz_nursery_train as nt  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.tokenizer import NurseryTokenizer, NOVEL_LABELS  # noqa: E402
from hatchling_world.language.nursery_generator import generate_l5_stress_episode  # noqa: E402


def episode_tensors(tok, ep):
    sequence_ids_list = [torch.tensor([tok.encode(s)]) for s in ep["sequence"]]
    question_ids = torch.tensor([tok.encode(ep["question"])])
    answer_idx = torch.tensor([ep["answer_idx"]])
    return sequence_ids_list, question_ids, answer_idx


# ---- Part 1: memory-slot sweep ----

def train_and_eval_slot_config(memory_slots, tok, args):
    torch.manual_seed(args.seed)
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_rng = random.Random(args.seed + 1)
    eval_rng = random.Random(args.seed + 1 + nt.TEST_SEED_OFFSET)

    for step in range(args.sweep_steps):
        ep = generate_l5_stress_episode(train_rng, n_facts=args.n_facts, n_distractors=args.n_distractors)
        sequence_ids_list, question_ids, answer_idx = episode_tensors(tok, ep)
        logits = model.stress_recall_forward(sequence_ids_list, question_ids)
        loss = F.cross_entropy(logits, answer_idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % args.eval_every == 0:
            correct = 0
            with torch.no_grad():
                for _ in range(args.eval_episodes):
                    e = generate_l5_stress_episode(eval_rng, n_facts=args.n_facts, n_distractors=args.n_distractors)
                    seq_ids, q_ids, a_idx = episode_tensors(tok, e)
                    lg = model.stress_recall_forward(seq_ids, q_ids)
                    correct += int((lg.argmax(-1) == a_idx).item())
            acc = correct / args.eval_episodes
            print(f"[cliff][slots={memory_slots}] step={step+1}/{args.sweep_steps} held_out_acc={acc:.3f}", flush=True)

    correct = 0
    with torch.no_grad():
        for _ in range(args.eval_episodes * 2):
            e = generate_l5_stress_episode(eval_rng, n_facts=args.n_facts, n_distractors=args.n_distractors)
            seq_ids, q_ids, a_idx = episode_tensors(tok, e)
            lg = model.stress_recall_forward(seq_ids, q_ids)
            correct += int((lg.argmax(-1) == a_idx).item())
    final_acc = correct / (args.eval_episodes * 2)
    return model, final_acc


def part1_slot_sweep(tok, args):
    print("\n[cliff] === PART 1: memory-slot sweep (M_S) ===", flush=True)
    results = {}
    for memory_slots in args.slot_sweep:
        print(f"[cliff] ---- M_S={memory_slots} ----", flush=True)
        _, acc = train_and_eval_slot_config(memory_slots, tok, args)
        results[memory_slots] = acc
        print(f"[cliff] M_S={memory_slots} FINAL held_out_acc={acc:.3f}", flush=True)
    return results


# ---- Part 2: slot-diversity probe ----

def cosine_sim_matrix(S: torch.Tensor) -> torch.Tensor:
    """S: (M_S, D). Returns (M_S, M_S) pairwise cosine similarity."""
    normed = F.normalize(S, dim=-1)
    return normed @ normed.T


def participation_ratio(S: torch.Tensor) -> float:
    """Effective rank via participation ratio: (sum sv)^2 / sum(sv^2).
    Ranges from 1 (all variance in one direction -- fully collapsed) to
    M_S (variance spread evenly across all slots -- fully diverse)."""
    sv = torch.linalg.svdvals(S)
    return float((sv.sum() ** 2) / (sv ** 2).sum())


def part2_slot_diversity(model, tok, args):
    print("\n[cliff] === PART 2: slot-diversity probe (does teaching more facts collapse S's slots?) ===", flush=True)
    rng = random.Random(args.seed + 500)
    results = {}
    for k in [1, 2, 3]:
        sims, ranks = [], []
        for _ in range(args.probe_episodes):
            ep = generate_l5_stress_episode(rng, n_facts=k, n_distractors=0)
            with torch.no_grad():
                S = model.mem.init_state(1)
                for sentence in ep["sequence"]:
                    ids = torch.tensor([tok.encode(sentence)])
                    for t in range(ids.shape[1]):
                        S = model.mem.update(S, model.token_embed(ids[:, t]).unsqueeze(1))
                S0 = S[0]  # (M_S, D)
                sim = cosine_sim_matrix(S0)
                off_diag = sim[~torch.eye(sim.shape[0], dtype=torch.bool)]
                sims.append(off_diag.mean().item())
                ranks.append(participation_ratio(S0))
        mean_sim = sum(sims) / len(sims)
        mean_rank = sum(ranks) / len(ranks)
        print(f"[cliff][diversity] n_facts={k} mean_pairwise_cosine_sim={mean_sim:.4f} "
              f"mean_participation_ratio={mean_rank:.3f} (max possible = {model.mem.S_init.shape[0]})", flush=True)
        results[k] = {"mean_pairwise_cosine_sim": mean_sim, "mean_participation_ratio": mean_rank}
    return results


# ---- Part 3: fact-decoding probe directly on S ----

class FactProbe(nn.Module):
    """Tiny linear probe: pooled S -> label logits. Backbone (mem/ws)
    stays completely frozen -- this probe's own accuracy is the only
    thing being measured, so any signal it finds was already present
    in S, not created by additional backbone training."""

    def __init__(self, d_model: int, n_labels: int):
        super().__init__()
        self.head = nn.Linear(d_model, n_labels)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        return self.head(S.mean(dim=1))


def part3_fact_decoding_probe(model, tok, args):
    print("\n[cliff] === PART 3: fact-decoding probe directly on S (storage vs retrieval) ===", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)

    n_facts = 3
    results = {}
    for query_idx in range(n_facts):
        probe = FactProbe(args.d_model, len(NOVEL_LABELS))
        opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)
        train_rng = random.Random(args.seed + 600 + query_idx)
        eval_rng = random.Random(args.seed + 600 + query_idx + nt.TEST_SEED_OFFSET)

        def build_S_and_label(rng):
            ep = generate_l5_stress_episode(rng, n_facts=n_facts, n_distractors=0)
            with torch.no_grad():
                S = model.mem.init_state(1)
                for sentence in ep["sequence"]:
                    ids = torch.tensor([tok.encode(sentence)])
                    for t in range(ids.shape[1]):
                        S = model.mem.update(S, model.token_embed(ids[:, t]).unsqueeze(1))
            label_idx = torch.tensor([NOVEL_LABELS.index(ep["labels"][query_idx])])
            return S, label_idx

        for step in range(args.probe_steps):
            S, label_idx = build_S_and_label(train_rng)
            logits = probe(S)
            loss = F.cross_entropy(logits, label_idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        correct = 0
        with torch.no_grad():
            for _ in range(args.probe_eval_episodes):
                S, label_idx = build_S_and_label(eval_rng)
                logits = probe(S)
                correct += int((logits.argmax(-1) == label_idx).item())
        acc = correct / args.probe_eval_episodes
        chance = 1.0 / len(NOVEL_LABELS)
        print(f"[cliff][probe] query_idx={query_idx} (fact taught {'1st' if query_idx==0 else '2nd' if query_idx==1 else '3rd'}) "
              f"probe_acc={acc:.3f} (chance={chance:.3f})", flush=True)
        results[query_idx] = acc

    for p in model.parameters():
        p.requires_grad_(True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-facts", type=int, default=3, help="the cliff config: n_facts=3, n_distractors=0")
    parser.add_argument("--n-distractors", type=int, default=0)
    parser.add_argument("--slot-sweep", type=int, nargs="+", default=[4, 8, 12, 16])
    parser.add_argument("--sweep-steps", type=int, default=2500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--probe-episodes", type=int, default=100)
    parser.add_argument("--probe-steps", type=int, default=1500)
    parser.add_argument("--probe-eval-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_nursery_l5_memory_cliff_diagnostic.json"))
    args = parser.parse_args()

    tok = NurseryTokenizer()

    part1_results = part1_slot_sweep(tok, args)

    print("\n[cliff] training the default-M_S=8 model once more for parts 2/3 "
          "(same architecture the original cliff was found with)...", flush=True)
    model_m8, m8_acc = train_and_eval_slot_config(8, tok, args)
    print(f"[cliff] M_S=8 reference model FINAL held_out_acc={m8_acc:.3f} (n_facts={args.n_facts})", flush=True)

    part2_results = part2_slot_diversity(model_m8, tok, args)
    part3_results = part3_fact_decoding_probe(model_m8, tok, args)

    all_results = {
        "config": {"n_facts": args.n_facts, "n_distractors": args.n_distractors},
        "part1_slot_sweep": part1_results,
        "part2_slot_diversity": part2_results,
        "part3_fact_decoding_probe": part3_results,
    }
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[cliff] wrote {args.results_file}", flush=True)

    print("\n[cliff] === SUMMARY ===")
    print("Part 1 (slot sweep):", part1_results)
    print("Part 2 (slot diversity by n_facts):", part2_results)
    print("Part 3 (fact-decoding probe by query_idx):", part3_results)


if __name__ == "__main__":
    main()
