#!/usr/bin/env python3
"""Phase 8 (Library), plans/Hatchling world.md section 10. Real
motivation: this session's L5-stress diagnostic thread fully
root-caused a sharp capacity cliff in S (a content-blind write gate
that overwrites rather than allocates -- three separate fixes tried,
all failed to move recall past ~35% even at n_facts=3). The Library is
the plan's own proposed answer: an external, unbounded fact store
queried via READ(query) instead of writing every fact into S.

The real test here: does OFFLOADING retrieval onto an external,
O(1)-cost lookup (hatchling_world.library.library_read) -- so the
model only ever needs to hold the CURRENT query's retrieved answer in
S, never the whole library -- restore near-100% recall at library
sizes far beyond where S-only storage collapsed (n_facts=3+)?

Mechanism: reuses HZLanguageModel.qa_forward EXACTLY as validated for
L5's single-fact case (100% held-out, this session's very first Nursery
result) -- no new model code. The only new step is environment-side:
build a library of n_facts (color, label) pairs, READ() the one fact
relevant to the query, and feed ONLY that retrieved fact as the "teach"
sentence. From the model's perspective this IS the L5 task, regardless
of library size -- that's the whole point.

Real comparison baked in: prints the L5-stress capacity-cliff numbers
already measured this session (hz_nursery_l5_memory_stress.py) directly
alongside the Library results at matching n_facts, so the "S caps at
~2 facts; Library does not" claim is a real side-by-side number, not
an assertion.
"""
from __future__ import annotations

import argparse
import json
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
from hatchling_world.library import generate_library_episode, library_read  # noqa: E402

# Real numbers already measured this session, plans/Hatchling world.md
# Phase 0's L5 memory-stress writeup (n_distractors=0 rows) -- printed
# alongside the Library results below for a direct, honest comparison.
S_ONLY_CAPACITY_CLIFF = {1: 1.000, 2: 0.525, 3: 0.240, 4: 0.265}


def library_qa_forward(model, tok, teach_sentence, question_sentence):
    """Reuses HZLanguageModel.qa_forward's exact S-ingestion + readout
    mechanism, but with `read_null_x` instead of a real object set --
    the Library task is pure language (a retrieved label, not an
    object to identify), matching L5-stress/L6's own "no object
    grounding" precedent. No changes to the production model class;
    same non-invasive external-function pattern used throughout this
    session's diagnostic scripts."""
    B = 1
    S = model.mem.init_state(B)
    teach_ids = torch.tensor([tok.encode(teach_sentence)])
    for t in range(teach_ids.shape[1]):
        S = model.mem.update(S, model.token_embed(teach_ids[:, t]).unsqueeze(1))
    question_ids = torch.tensor([tok.encode(question_sentence)])
    for t in range(question_ids.shape[1]):
        S = model.mem.update(S, model.token_embed(question_ids[:, t]).unsqueeze(1))

    x_null = model.read_null_x.expand(B, 1, model.D)
    H = model.ws.run(B, S, x_null, n_rounds=model.n_rounds_l1)
    q = model.qa_rq(H).mean(dim=1, keepdim=True)
    scores = torch.matmul(q, model.qa_rk(H).transpose(-1, -2)) / (model.D ** 0.5)
    read = torch.matmul(F.softmax(scores, dim=-1), H).mean(dim=1)
    return model.qa_head(read)


def library_episode_to_tensors(tok, ep):
    """Retrieves the one relevant fact via the real READ(query) action,
    then packages it exactly like an L5 single-fact episode -- teach
    sentence built from the RETRIEVED fact only, never the whole
    library."""
    retrieved_label = library_read(ep["fact_table"], ep["query_color"])
    teach = f"the {ep['query_color']} object is called {retrieved_label}"
    label_idx = torch.tensor([NOVEL_LABELS.index(ep["answer_label"])])
    return teach, ep["question"], label_idx


def train_and_eval(n_facts, tok, args):
    torch.manual_seed(args.seed)
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=8,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train_rng = random.Random(args.seed + 1)
    eval_rng = random.Random(args.seed + 1 + nt.TEST_SEED_OFFSET)

    for step in range(args.steps):
        ep = generate_library_episode(train_rng, n_facts=n_facts)
        teach, question, label_idx = library_episode_to_tensors(tok, ep)
        logits = library_qa_forward(model, tok, teach, question)
        loss = F.cross_entropy(logits, label_idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % args.eval_every == 0:
            correct = 0
            with torch.no_grad():
                for _ in range(args.eval_episodes):
                    e = generate_library_episode(eval_rng, n_facts=n_facts)
                    teach, question, label_idx = library_episode_to_tensors(tok, e)
                    lg = library_qa_forward(model, tok, teach, question)
                    correct += int((lg.argmax(-1) == label_idx).item())
            print(f"[library][n_facts={n_facts}] step={step+1}/{args.steps} "
                  f"held_out_acc={correct/args.eval_episodes:.3f}", flush=True)

    correct = 0
    with torch.no_grad():
        for _ in range(args.eval_episodes * 2):
            e = generate_library_episode(eval_rng, n_facts=n_facts)
            teach, question, label_idx = library_episode_to_tensors(tok, e)
            lg = library_qa_forward(model, tok, teach, question)
            correct += int((lg.argmax(-1) == label_idx).item())
    return correct / (args.eval_episodes * 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--eval-every", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--library-sizes", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results-file", type=Path, default=Path("results/local/hz_library_read_test.json"))
    args = parser.parse_args()

    tok = NurseryTokenizer()
    results = {}
    for n_facts in args.library_sizes:
        print(f"\n[library] ==== n_facts (library size) = {n_facts} ====", flush=True)
        acc = train_and_eval(n_facts, tok, args)
        results[n_facts] = acc
        print(f"[library] n_facts={n_facts} FINAL held_out_acc={acc:.3f} "
              f"(READ() cost is O(1) regardless of library size)", flush=True)

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[library] wrote {args.results_file}", flush=True)

    print("\n[library] === SUMMARY: Library (external READ) vs S-only storage (this session) ===")
    print(f"{'n_facts':>8} {'library_acc':>12} {'S_only_acc':>12}")
    for n_facts, acc in results.items():
        s_only = S_ONLY_CAPACITY_CLIFF.get(n_facts, "n/a (never tested at this size)")
        s_only_str = f"{s_only:.3f}" if isinstance(s_only, float) else s_only
        print(f"{n_facts:>8} {acc:>12.3f} {s_only_str:>12}")


if __name__ == "__main__":
    main()
