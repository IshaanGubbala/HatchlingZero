#!/usr/bin/env python3
"""Gate-inspection diagnostic, follow-up to hz_nursery_l5_memory_cliff_
diagnostic.py's Part 3 finding (real storage/overwrite failure in
mem.update's gated write, not H's retrieval). Explicit user request,
2026-09-05: "inspect the gate value g (mem._gate) across successive
fact-teaching updates -- does it stay uniformly high regardless of
whether the new content is genuinely novel, which would directly
explain why teaching fact 2 overwrites fact 1 instead of being written
alongside it." No changes to mem.update or the gate are made here --
this calls the SAME submodules (q_proj/k_proj/v_proj/write_proj/
ln_read/ln_state/_gate) that update() already uses internally, just
also captures g, which update()'s own return value discards. Read-only
introspection, not a modification.

Real structural detail worth knowing before interpreting results:
HZCQPersistentMemory's gate is a "protected zero init" (gate_w2 starts
at exactly zero, gate_b2 set so sigmoid(gate_b2) = g_init = 0.58) --
the gate STARTS completely content-insensitive by design (g_logit =
gate_b2, a constant, until gate_w1/gate_w2 move away from zero during
training) and must learn to become content-sensitive. If training
never pressures it to, it can remain close to that near-constant
value regardless of what's actually being written.

Two real experiments:
  A. Per-fact-boundary gate trace: mean gate value and how much S
     actually changes (cosine similarity pre/post) at the end of each
     of the 3 taught facts, averaged over many held-out episodes.
  B. Repeat-vs-novel controlled test: does the gate write LESS when
     the incoming content is a word-for-word REPEAT of something
     already in S (redundant, should need no update) vs a genuinely
     NEW fact (should need real allocation)? If the gate is truly
     content-blind, these should not differ.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hz_nursery_train as nt  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.tokenizer import NurseryTokenizer, NOVEL_LABELS  # noqa: E402
from hatchling_world.language.nursery_generator import generate_l5_stress_episode  # noqa: E402


def update_with_gate_capture(mem, S_prev: torch.Tensor, demo_hidden: torch.Tensor):
    """Exact copy of HZCQPersistentMemory.update's own math (same
    submodules, same call order) -- the only difference is returning g
    alongside S_new, which update() computes internally but discards."""
    Q = mem.q_proj(S_prev)
    K = mem.k_proj(demo_hidden)
    V = mem.v_proj(demo_hidden)
    scale = 1.0 / math.sqrt(Q.size(-1))
    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale
    attn = F.softmax(scores, dim=-1)
    read = torch.matmul(attn, V)
    delta_S = mem.ln_read(mem.write_proj(read))
    g = mem._gate(S_prev, delta_S)
    S_new = mem.ln_state(S_prev + g * delta_S)
    return S_new, g


def train_reference_model(tok, args):
    """Same recipe as the cliff diagnostic's M_S=8 reference model."""
    torch.manual_seed(args.seed)
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=8,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_rng = random.Random(args.seed + 1)
    for step in range(args.train_steps):
        ep = generate_l5_stress_episode(train_rng, n_facts=3, n_distractors=0)
        sequence_ids_list = [torch.tensor([tok.encode(s)]) for s in ep["sequence"]]
        question_ids = torch.tensor([tok.encode(ep["question"])])
        answer_idx = torch.tensor([ep["answer_idx"]])
        logits = model.stress_recall_forward(sequence_ids_list, question_ids)
        loss = F.cross_entropy(logits, answer_idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"[gate-diag] pretrain step={step+1}/{args.train_steps} loss={loss.item():.3f}", flush=True)
    return model


def ingest_sentence(model, tok, S, sentence):
    """Walk one sentence's tokens through mem.update (gate-capturing
    version), returning the final S and the list of per-token gate
    tensors (each (1, M_S, 1))."""
    ids = torch.tensor([tok.encode(sentence)])
    gates = []
    with torch.no_grad():
        for t in range(ids.shape[1]):
            x = model.token_embed(ids[:, t]).unsqueeze(1)
            S, g = update_with_gate_capture(model.mem, S, x)
            gates.append(g)
    return S, gates


def part_a_fact_boundary_trace(model, tok, args):
    print("\n[gate-diag] === PART A: gate value + S-change at each fact boundary ===", flush=True)
    rng = random.Random(args.seed + 700)
    per_fact_gate, per_fact_cos_change = {1: [], 2: [], 3: []}, {1: [], 2: [], 3: []}

    for _ in range(args.trace_episodes):
        ep = generate_l5_stress_episode(rng, n_facts=3, n_distractors=0)
        S = model.mem.init_state(1)
        for k, sentence in enumerate(ep["sequence"], start=1):
            S_before = S.clone()
            S, gates = ingest_sentence(model, tok, S, sentence)
            last_g_mean = gates[-1].mean().item()
            cos_change = F.cosine_similarity(S_before[0], S[0], dim=-1).mean().item()
            per_fact_gate[k].append(last_g_mean)
            per_fact_cos_change[k].append(cos_change)

    results = {}
    for k in [1, 2, 3]:
        mean_g = sum(per_fact_gate[k]) / len(per_fact_gate[k])
        mean_cos = sum(per_fact_cos_change[k]) / len(per_fact_cos_change[k])
        print(f"[gate-diag][fact {k}] mean_gate_at_last_token={mean_g:.4f} "
              f"mean_cos_sim(S_before, S_after)={mean_cos:.4f} (1.0 = S barely changed, "
              f"-1.0..0 = S was substantially overwritten/rotated away from its prior content)", flush=True)
        results[k] = {"mean_gate_at_last_token": mean_g, "mean_cos_sim_pre_post": mean_cos}
    return results


def part_b_repeat_vs_novel(model, tok, args):
    print("\n[gate-diag] === PART B: repeat vs novel content -- is the gate content-sensitive at all? ===", flush=True)
    rng = random.Random(args.seed + 800)
    repeat_gates, novel_gates = [], []

    for _ in range(args.trace_episodes):
        ep1 = generate_l5_stress_episode(rng, n_facts=1, n_distractors=0)
        ep2 = generate_l5_stress_episode(rng, n_facts=1, n_distractors=0)
        fact1_sentence = ep1["sequence"][0]
        fact2_sentence = ep2["sequence"][0]

        S0 = model.mem.init_state(1)
        S1, _ = ingest_sentence(model, tok, S0, fact1_sentence)

        _, repeat_g = ingest_sentence(model, tok, S1, fact1_sentence)
        repeat_gates.append(repeat_g[-1].mean().item())

        _, novel_g = ingest_sentence(model, tok, S1, fact2_sentence)
        novel_gates.append(novel_g[-1].mean().item())

    mean_repeat = sum(repeat_gates) / len(repeat_gates)
    mean_novel = sum(novel_gates) / len(novel_gates)
    print(f"[gate-diag] mean_gate(REPEAT already-known fact)={mean_repeat:.4f}", flush=True)
    print(f"[gate-diag] mean_gate(NOVEL new fact)={mean_novel:.4f}", flush=True)
    print(f"[gate-diag] difference (novel - repeat) = {mean_novel - mean_repeat:+.4f} "
          f"-- near 0 means the gate is NOT distinguishing redundant from new content", flush=True)
    return {"mean_gate_repeat": mean_repeat, "mean_gate_novel": mean_novel, "difference": mean_novel - mean_repeat}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-steps", type=int, default=2500)
    parser.add_argument("--trace-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_nursery_l5_gate_diagnostic.json"))
    args = parser.parse_args()

    tok = NurseryTokenizer()
    print("[gate-diag] training reference model (M_S=8, n_facts=3, same recipe as cliff diagnostic)...", flush=True)
    model = train_reference_model(tok, args)

    part_a = part_a_fact_boundary_trace(model, tok, args)
    part_b = part_b_repeat_vs_novel(model, tok, args)

    all_results = {"part_a_fact_boundary": part_a, "part_b_repeat_vs_novel": part_b}
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[gate-diag] wrote {args.results_file}", flush=True)

    print("\n[gate-diag] === SUMMARY ===")
    print("Part A (gate + S-change per fact boundary):", part_a)
    print("Part B (repeat vs novel gate):", part_b)


if __name__ == "__main__":
    main()
