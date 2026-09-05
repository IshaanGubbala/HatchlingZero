#!/usr/bin/env python3
"""Real, continuously-running Language Nursery training loop that feeds
scripts/hz_world_live_view.py's shared state file, same mechanism as
scripts/hz_world_rollout_demo.py uses for room-navigation -- so
plans/Hatchling world.md's L0-L3 Nursery stages can be watched live in
a browser, not just read off a training-log tail.

Runs the SAME model (reference/hz_language_model_torch.py's
HZLanguageModel) and the SAME generators
(hatchling_world/language/nursery_generator.py) as
scripts/hz_nursery_train.py -- this file does not reimplement any
training logic, it imports hz_nursery_train's tensor-packing helpers
directly and just adds live snapshotting + a real-time cadence.

Real, disclosed simplification vs. hz_nursery_train.py's reported
numbers: for watchability this script evaluates on a SMALL held-out
batch every `--eval-every` steps (default n=40), not the 200-episode
batches used for the numbers written into plans/Hatchling world.md.
The metrics shown here are real (genuine held-out forward passes, no
mocking), just noisier -- treat the live numbers as "is it moving in
the right direction," not as a replacement for the paper-grade runs.

Runs every landed stage in curriculum order -- L0, L1, L2, L3, L4's two
halves (logic-AND, counting-verification), L5, the L5 memory-stress
test, L6, and School-0's two halves (arithmetic, rule/logic) -- in one
process, so a viewer watching from the start sees the model progress
through the whole curriculum the way the plan itself sequences it.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import hz_nursery_train`

import hz_nursery_train as nt  # noqa: E402  -- reuse its tensor-packing + train-step helpers verbatim
import hz_school0_train as sc0  # noqa: E402
from reference.hz_language_model_torch import HZLanguageModel  # noqa: E402
from hatchling_world.language.tokenizer import NurseryTokenizer  # noqa: E402
from hatchling_world.language.nursery_generator import (  # noqa: E402
    generate_l0_sentence, generate_l1_grounding_episode, generate_l2_verb_episode,
    generate_l3_relation_episode, generate_l4_logic_and_episode, generate_l4_counting_episode,
    generate_l5_qa_episode, generate_l5_stress_episode, generate_l6_reading_episode,
)
from hatchling_world.language.tokenizer import NOVEL_LABELS, SIZES, NUMBERS  # noqa: E402
from hatchling_world.school.generator import generate_arithmetic_episode, generate_rule_episode  # noqa: E402

HISTORY_LEN = 60
L2_COPY_BASELINE = 0.8045  # real, measured (see plans/Hatchling world.md's L2 writeup) -- "copy pre-state, ignore verb"


def write_snapshot(path: Path, data: dict) -> None:
    """Atomic write -- server must never read a half-written file."""
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent) or ".", prefix=".tmp_nursery_state_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


class HistoryTracker:
    """Rolling per-metric history, matching the room demo's HISTORY_LEN
    convention so the viewer's chart code can treat both schemas the
    same way."""

    def __init__(self):
        self.history: dict[str, list[float]] = {}

    def push(self, **metrics: float) -> None:
        for name, value in metrics.items():
            self.history.setdefault(name, []).append(value)
            self.history[name] = self.history[name][-HISTORY_LEN:]

    def snapshot(self) -> dict:
        return dict(self.history)


def base_snapshot(stage: str, stage_idx: int, n_stages: int, step: int, total_steps: int,
                   instruction: str, objects: list, target_idx, pred_idx,
                   metrics_current: dict, metrics_chance: dict, metrics_baseline: dict,
                   history: dict, verb=None, consequence_true=None, consequence_pred=None,
                   tokens=None, matching_indices=None, verify_true=None, verify_pred=None,
                   passage=None, recall_true=None, recall_pred=None) -> dict:
    return {
        "kind": "nursery",
        "stage": stage,
        "stage_idx": stage_idx,
        "n_stages": n_stages,
        "step": step,
        "total_steps": total_steps,
        "instruction": instruction,
        "objects": objects,
        "target_idx": target_idx,
        "pred_idx": pred_idx,
        "verb": verb,
        "consequence_true": consequence_true,
        "consequence_pred": consequence_pred,
        "tokens": tokens,
        # L4-counting only: which objects satisfy the queried property
        # (for visual counting alongside the verdict), and the model's
        # verify-true-or-false prediction vs the real answer -- there is
        # no single "target object" for a whole-set verification task.
        "matching_indices": matching_indices,
        "verify_true": verify_true,
        "verify_pred": verify_pred,
        # L5/L5-stress/L6/School-0: prior turns shown above the main
        # instruction/question (teach sentence(s), a read passage, a
        # taught rule), and the model's predicted LABEL/WORD vs the
        # real answer -- generic across every label space (novel
        # labels, sizes, number words), unlike verify_true/pred which
        # is specifically the L4-counting true/false case.
        "passage": passage,
        "recall_true": recall_true,
        "recall_pred": recall_pred,
        "metrics": {
            "current": metrics_current,
            "chance": metrics_chance,
            "baseline": metrics_baseline,
            "history": history,
        },
    }


def run_l0(model, opt, tok, args, state_file, tracker, stage_idx, n_stages):
    train_rng = random.Random(args.seed + 1)
    eval_rng = random.Random(args.seed + 1 + nt.TEST_SEED_OFFSET)
    for step in range(args.l0_steps):
        nt.l0_train_step(model, opt, tok, train_rng, args.l0_batch_size)

        if (step + 1) % args.eval_every == 0:
            held_out_ids = nt.l0_batch(tok, eval_rng, args.demo_eval_n)
            with torch.no_grad():
                logits = model.lm_forward(held_out_ids)
                target = held_out_ids[:, 1:]
                mask = (target != tok.pad_id)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, tok.vocab_size), target.reshape(-1), reduction="none")
                loss = (loss * mask.reshape(-1).float()).sum() / mask.float().sum().clamp_min(1)
                ppl = torch.exp(loss).item()
            tracker.push(held_out_ppl=ppl)

        # live token-by-token teacher-forced demo on one fresh held-out sentence
        demo_ids = nt.l0_batch(tok, eval_rng, 1)
        with torch.no_grad():
            logits = model.lm_forward(demo_ids)
            pred = logits.argmax(-1)[0].tolist()
        true_words = [tok.id_to_word[i] for i in demo_ids[0, 1:].tolist() if i != tok.pad_id]
        pred_words = [tok.id_to_word[i] for i in pred[:len(true_words)]]
        tokens = [{"word": t, "correct": (t == p)} for t, p in zip(true_words, pred_words)]

        write_snapshot(state_file, base_snapshot(
            stage="L0", stage_idx=stage_idx, n_stages=n_stages, step=step + 1, total_steps=args.l0_steps,
            instruction=" ".join(true_words), objects=[], target_idx=None, pred_idx=None,
            metrics_current={"held_out_ppl": (tracker.history.get("held_out_ppl") or [float(tok.vocab_size)])[-1]},
            metrics_chance={"held_out_ppl": float(tok.vocab_size)}, metrics_baseline={},
            history=tracker.snapshot(), tokens=tokens))
        time.sleep(args.step_delay)


def run_l1(model, opt, tok, args, state_file, tracker, stage_idx, n_stages):
    train_rng = random.Random(args.seed + 2)
    eval_rng = random.Random(args.seed + 2 + nt.TEST_SEED_OFFSET)
    for step in range(args.l1_steps):
        nt.l1_train_step(model, opt, tok, train_rng, args.l1_n_objects)

        if (step + 1) % args.eval_every == 0:
            acc = nt.l1_eval(model, tok, eval_rng, args.l1_n_objects, args.demo_eval_n)
            tracker.push(held_out_acc=acc)

        ep = generate_l1_grounding_episode(eval_rng, n_objects=args.l1_n_objects)
        instr_ids, type_idx, color_idx, size_idx, pos_idx, target = nt.l1_episode_tensors(tok, ep)
        with torch.no_grad():
            logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
            pred_idx = int(logits.argmax(-1).item())

        write_snapshot(state_file, base_snapshot(
            stage="L1", stage_idx=stage_idx, n_stages=n_stages, step=step + 1, total_steps=args.l1_steps,
            instruction=ep["instruction"], objects=ep["objects"], target_idx=ep["target_idx"], pred_idx=pred_idx,
            metrics_current={"held_out_acc": (tracker.history.get("held_out_acc") or [1.0 / args.l1_n_objects])[-1]},
            metrics_chance={"held_out_acc": 1.0 / args.l1_n_objects}, metrics_baseline={},
            history=tracker.snapshot()))
        time.sleep(args.step_delay)


def run_l2(model, opt, tok, args, state_file, tracker, stage_idx, n_stages):
    train_rng = random.Random(args.seed + 3)
    eval_rng = random.Random(args.seed + 3 + nt.TEST_SEED_OFFSET)
    for step in range(args.l2_steps):
        nt.l2_train_step(model, opt, tok, train_rng, args.l2_n_objects)

        if (step + 1) % args.eval_every == 0:
            sel_acc, cons_acc = nt.l2_eval(model, tok, eval_rng, args.l2_n_objects, args.demo_eval_n)
            tracker.push(held_out_sel_acc=sel_acc, held_out_cons_acc=cons_acc)

        ep = generate_l2_verb_episode(eval_rng, n_objects=args.l2_n_objects)
        (instr_ids, type_idx, color_idx, size_idx, pos_idx, held, opened,
         target, cons_target) = nt.l2_episode_tensors(tok, ep)
        with torch.no_grad():
            sel_logits, cons_logits = model.verb_forward(
                instr_ids, type_idx, color_idx, size_idx, pos_idx, held, opened)
            pred_idx = int(sel_logits.argmax(-1).item())
            cons_pred = (cons_logits[0] > 0).tolist()

        write_snapshot(state_file, base_snapshot(
            stage="L2", stage_idx=stage_idx, n_stages=n_stages, step=step + 1, total_steps=args.l2_steps,
            instruction=ep["instruction"], objects=ep["objects"], target_idx=ep["target_idx"], pred_idx=pred_idx,
            verb=ep["verb"],
            consequence_true={"position_right": ep["position_after"] == "right",
                               "held": ep["held_after"], "opened": ep["opened_after"]},
            consequence_pred={"position_right": bool(cons_pred[0]),
                               "held": bool(cons_pred[1]), "opened": bool(cons_pred[2])},
            metrics_current={
                "held_out_sel_acc": (tracker.history.get("held_out_sel_acc") or [1.0 / args.l2_n_objects])[-1],
                "held_out_cons_acc": (tracker.history.get("held_out_cons_acc") or [0.5])[-1],
            },
            metrics_chance={"held_out_sel_acc": 1.0 / args.l2_n_objects, "held_out_cons_acc": 0.5},
            metrics_baseline={"held_out_cons_acc": L2_COPY_BASELINE},
            history=tracker.snapshot()))
        time.sleep(args.step_delay)


def run_l3(model, opt, tok, args, state_file, tracker, stage_idx, n_stages):
    train_rng = random.Random(args.seed + 4)
    eval_seen_rng = random.Random(args.seed + 4 + nt.TEST_SEED_OFFSET)
    eval_unseen_rng = random.Random(args.seed + 4 + 2 * nt.TEST_SEED_OFFSET)
    for step in range(args.l3_steps):
        nt.l3_train_step(model, opt, tok, train_rng, args.l3_n_objects)

        if (step + 1) % args.eval_every == 0:
            seen_acc = nt.l3_eval(model, tok, eval_seen_rng, args.l3_n_objects, args.demo_eval_n, split="train")
            unseen_acc = nt.l3_eval(model, tok, eval_unseen_rng, args.l3_n_objects, args.demo_eval_n, split="test")
            tracker.push(held_out_seen_combo_acc=seen_acc, held_out_unseen_combo_acc=unseen_acc)

        # alternate which split the LIVE example is drawn from so a
        # viewer sees both the easy (seen-combo) and hard (unseen-combo)
        # case, not just whichever is more flattering
        split = "test" if step % 2 == 0 else "train"
        ep = generate_l3_relation_episode(eval_unseen_rng, n_objects=args.l3_n_objects, split=split)
        instr_ids, type_idx, color_idx, size_idx, pos_idx, target = nt.l1_episode_tensors(tok, ep)
        with torch.no_grad():
            logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
            pred_idx = int(logits.argmax(-1).item())

        write_snapshot(state_file, base_snapshot(
            stage=f"L3 ({'UNSEEN combo' if split == 'test' else 'seen combo'})",
            stage_idx=stage_idx, n_stages=n_stages, step=step + 1, total_steps=args.l3_steps,
            instruction=ep["instruction"], objects=ep["objects"], target_idx=ep["target_idx"], pred_idx=pred_idx,
            metrics_current={
                "held_out_seen_combo_acc": (tracker.history.get("held_out_seen_combo_acc") or [1.0 / args.l3_n_objects])[-1],
                "held_out_unseen_combo_acc": (tracker.history.get("held_out_unseen_combo_acc") or [1.0 / args.l3_n_objects])[-1],
            },
            metrics_chance={"held_out_seen_combo_acc": 1.0 / args.l3_n_objects,
                             "held_out_unseen_combo_acc": 1.0 / args.l3_n_objects},
            metrics_baseline={},
            history=tracker.snapshot()))
        time.sleep(args.step_delay)


def run_l4_logic(model, opt, tok, args, state_file, tracker, stage_idx, n_stages):
    train_rng = random.Random(args.seed + 5)
    eval_seen_rng = random.Random(args.seed + 5 + nt.TEST_SEED_OFFSET)
    eval_unseen_rng = random.Random(args.seed + 5 + 2 * nt.TEST_SEED_OFFSET)
    for step in range(args.l4_logic_steps):
        nt.l4_logic_train_step(model, opt, tok, train_rng, args.l4_n_objects)

        if (step + 1) % args.eval_every == 0:
            seen_acc = nt.l4_logic_eval(model, tok, eval_seen_rng, args.l4_n_objects, args.demo_eval_n, split="train")
            unseen_acc = nt.l4_logic_eval(model, tok, eval_unseen_rng, args.l4_n_objects, args.demo_eval_n, split="test")
            tracker.push(held_out_seen_combo_acc=seen_acc, held_out_unseen_combo_acc=unseen_acc)

        split = "test" if step % 2 == 0 else "train"
        ep = generate_l4_logic_and_episode(eval_unseen_rng, n_objects=args.l4_n_objects, split=split)
        instr_ids, type_idx, color_idx, size_idx, pos_idx, target = nt.l1_episode_tensors(tok, ep)
        with torch.no_grad():
            logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
            pred_idx = int(logits.argmax(-1).item())

        write_snapshot(state_file, base_snapshot(
            stage=f"L4-logic ({'UNSEEN combo' if split == 'test' else 'seen combo'})",
            stage_idx=stage_idx, n_stages=n_stages, step=step + 1, total_steps=args.l4_logic_steps,
            instruction=ep["instruction"], objects=ep["objects"], target_idx=ep["target_idx"], pred_idx=pred_idx,
            metrics_current={
                "held_out_seen_combo_acc": (tracker.history.get("held_out_seen_combo_acc") or [1.0 / args.l4_n_objects])[-1],
                "held_out_unseen_combo_acc": (tracker.history.get("held_out_unseen_combo_acc") or [1.0 / args.l4_n_objects])[-1],
            },
            metrics_chance={"held_out_seen_combo_acc": 1.0 / args.l4_n_objects,
                             "held_out_unseen_combo_acc": 1.0 / args.l4_n_objects},
            metrics_baseline={},
            history=tracker.snapshot()))
        time.sleep(args.step_delay)


def run_l4_counting(model, opt, tok, args, state_file, tracker, stage_idx, n_stages):
    train_rng = random.Random(args.seed + 6)
    eval_rng = random.Random(args.seed + 6 + nt.TEST_SEED_OFFSET)
    for step in range(args.l4_counting_steps):
        nt.l4_counting_train_step(model, opt, tok, train_rng, args.l4_n_objects)

        if (step + 1) % args.eval_every == 0:
            held_out_acc = nt.l4_counting_eval(model, tok, eval_rng, args.l4_n_objects, args.demo_eval_n)
            tracker.push(held_out_acc=held_out_acc)

        ep = generate_l4_counting_episode(eval_rng, n_objects=args.l4_n_objects)
        instr_ids, type_idx, color_idx, size_idx, pos_idx, label = nt.l4_counting_tensors(tok, ep)
        with torch.no_grad():
            logit = model.verify_count_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
            pred = bool((logit > 0).item())

        matching = [i for i, o in enumerate(ep["objects"]) if o[ep["property_kind"]] == ep["value"]]

        write_snapshot(state_file, base_snapshot(
            stage="L4-counting", stage_idx=stage_idx, n_stages=n_stages,
            step=step + 1, total_steps=args.l4_counting_steps,
            instruction=ep["instruction"], objects=ep["objects"], target_idx=None, pred_idx=None,
            matching_indices=matching, verify_true=bool(ep["label"]), verify_pred=pred,
            metrics_current={"held_out_acc": (tracker.history.get("held_out_acc") or [0.5])[-1]},
            metrics_chance={"held_out_acc": 0.5}, metrics_baseline={},
            history=tracker.snapshot()))
        time.sleep(args.step_delay)


def run_l5(model, opt, tok, args, state_file, tracker, stage_idx, n_stages):
    train_rng = random.Random(args.seed + 7)
    eval_rng = random.Random(args.seed + 7 + nt.TEST_SEED_OFFSET)
    for step in range(args.l5_steps):
        nt.l5_train_step(model, opt, tok, train_rng, args.l5_n_objects)

        if (step + 1) % args.eval_every == 0:
            held_out_acc = nt.l5_eval(model, tok, eval_rng, args.l5_n_objects, args.demo_eval_n)
            tracker.push(held_out_acc=held_out_acc)

        ep = generate_l5_qa_episode(eval_rng, n_objects=args.l5_n_objects)
        teach_ids, question_ids, type_idx, color_idx, size_idx, pos_idx, label_idx = nt.l5_episode_tensors(tok, ep)
        with torch.no_grad():
            logits = model.qa_forward(teach_ids, question_ids, type_idx, color_idx, size_idx, pos_idx)
            pred_idx = int(logits.argmax(-1).item())

        write_snapshot(state_file, base_snapshot(
            stage="L5", stage_idx=stage_idx, n_stages=n_stages, step=step + 1, total_steps=args.l5_steps,
            instruction=ep["question"], objects=ep["objects"], target_idx=ep["target_idx"], pred_idx=None,
            passage=[ep["teach"]], recall_true=ep["label"], recall_pred=NOVEL_LABELS[pred_idx],
            metrics_current={"held_out_acc": (tracker.history.get("held_out_acc") or [1.0 / len(NOVEL_LABELS)])[-1]},
            metrics_chance={"held_out_acc": 1.0 / len(NOVEL_LABELS)}, metrics_baseline={},
            history=tracker.snapshot()))
        time.sleep(args.step_delay)


def run_l5_stress(model, opt, tok, args, state_file, tracker, stage_idx, n_stages):
    train_rng = random.Random(args.seed + 8)
    eval_rng = random.Random(args.seed + 8 + nt.TEST_SEED_OFFSET)
    n_facts, n_distractors = args.l5_stress_facts, args.l5_stress_distractors
    for step in range(args.l5_stress_steps):
        ep = generate_l5_stress_episode(train_rng, n_facts=n_facts, n_distractors=n_distractors)
        sequence_ids_list = [torch.tensor([tok.encode(s)]) for s in ep["sequence"]]
        question_ids = torch.tensor([tok.encode(ep["question"])])
        answer_idx = torch.tensor([ep["answer_idx"]])
        logits = model.stress_recall_forward(sequence_ids_list, question_ids)
        loss = torch.nn.functional.cross_entropy(logits, answer_idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (step + 1) % args.eval_every == 0:
            correct = 0
            with torch.no_grad():
                for _ in range(args.demo_eval_n):
                    e = generate_l5_stress_episode(eval_rng, n_facts=n_facts, n_distractors=n_distractors)
                    seq_ids = [torch.tensor([tok.encode(s)]) for s in e["sequence"]]
                    q_ids = torch.tensor([tok.encode(e["question"])])
                    lg = model.stress_recall_forward(seq_ids, q_ids)
                    correct += int((lg.argmax(-1).item() == e["answer_idx"]))
            tracker.push(held_out_acc=correct / args.demo_eval_n)

        with torch.no_grad():
            pred_idx = int(model.stress_recall_forward(sequence_ids_list, question_ids).argmax(-1).item())

        write_snapshot(state_file, base_snapshot(
            stage=f"L5-stress (facts={n_facts}, distractors={n_distractors})",
            stage_idx=stage_idx, n_stages=n_stages, step=step + 1, total_steps=args.l5_stress_steps,
            instruction=ep["question"], objects=[], target_idx=None, pred_idx=None,
            passage=ep["sequence"], recall_true=ep["answer"], recall_pred=NOVEL_LABELS[pred_idx],
            metrics_current={"held_out_acc": (tracker.history.get("held_out_acc") or [1.0 / len(NOVEL_LABELS)])[-1]},
            metrics_chance={"held_out_acc": 1.0 / len(NOVEL_LABELS)}, metrics_baseline={},
            history=tracker.snapshot()))
        time.sleep(args.step_delay)


def run_l6(model, opt, tok, args, state_file, tracker, stage_idx, n_stages):
    train_rng = random.Random(args.seed + 9)
    eval_rng = random.Random(args.seed + 9 + nt.TEST_SEED_OFFSET)
    for step in range(args.l6_steps):
        nt.l6_train_step(model, opt, tok, train_rng, args.l6_n_sentences)

        if (step + 1) % args.eval_every == 0:
            held_out_acc, _ = nt.l6_eval(model, tok, eval_rng, args.l6_n_sentences, args.demo_eval_n)
            tracker.push(held_out_acc=held_out_acc)

        ep = generate_l6_reading_episode(eval_rng, n_sentences=args.l6_n_sentences)
        sentence_ids_list, question_ids, answer_idx = nt.l6_episode_tensors(tok, ep)
        with torch.no_grad():
            logits = model.read_forward(sentence_ids_list, question_ids)
            pred_idx = int(logits.argmax(-1).item())

        write_snapshot(state_file, base_snapshot(
            stage="L6", stage_idx=stage_idx, n_stages=n_stages, step=step + 1, total_steps=args.l6_steps,
            instruction=ep["question"], objects=[], target_idx=None, pred_idx=None,
            passage=ep["sentences"], recall_true=ep["answer"], recall_pred=SIZES[pred_idx],
            metrics_current={"held_out_acc": (tracker.history.get("held_out_acc") or [0.5])[-1]},
            metrics_chance={"held_out_acc": 0.5}, metrics_baseline={},
            history=tracker.snapshot()))
        time.sleep(args.step_delay)


def run_school0_arith(model, opt, tok, args, state_file, tracker, stage_idx, n_stages):
    train_rng = random.Random(args.seed + 10)
    eval_seen_rng = random.Random(args.seed + 10 + nt.TEST_SEED_OFFSET)
    eval_unseen_rng = random.Random(args.seed + 10 + 2 * nt.TEST_SEED_OFFSET)
    for step in range(args.school_arith_steps):
        sc0.arith_train_step(model, opt, tok, train_rng)

        if (step + 1) % args.eval_every == 0:
            seen = sc0.arith_eval(model, tok, eval_seen_rng, args.demo_eval_n, split="train")
            unseen = sc0.arith_eval(model, tok, eval_unseen_rng, args.demo_eval_n, split="test")
            tracker.push(held_out_seen_pair_acc=seen, held_out_unseen_pair_acc=unseen)

        split = "test" if step % 2 == 0 else "train"
        ep = generate_arithmetic_episode(eval_unseen_rng, split=split)
        ids = torch.tensor([tok.encode(ep["instruction"])])
        with torch.no_grad():
            pred_idx = int(model.arithmetic_forward(ids).argmax(-1).item())

        write_snapshot(state_file, base_snapshot(
            stage=f"School0-arith ({'UNSEEN pair' if split == 'test' else 'seen pair'})",
            stage_idx=stage_idx, n_stages=n_stages, step=step + 1, total_steps=args.school_arith_steps,
            instruction=ep["instruction"], objects=[], target_idx=None, pred_idx=None,
            recall_true=NUMBERS[ep["sum_idx"]], recall_pred=NUMBERS[pred_idx],
            metrics_current={
                "held_out_seen_pair_acc": (tracker.history.get("held_out_seen_pair_acc") or [1.0 / len(NUMBERS)])[-1],
                "held_out_unseen_pair_acc": (tracker.history.get("held_out_unseen_pair_acc") or [1.0 / len(NUMBERS)])[-1],
            },
            metrics_chance={"held_out_seen_pair_acc": 1.0 / len(NUMBERS), "held_out_unseen_pair_acc": 1.0 / len(NUMBERS)},
            metrics_baseline={}, history=tracker.snapshot()))
        time.sleep(args.step_delay)


def run_school0_rule(model, opt, tok, args, state_file, tracker, stage_idx, n_stages):
    train_rng = random.Random(args.seed + 11)
    eval_rng = random.Random(args.seed + 11 + nt.TEST_SEED_OFFSET)
    for step in range(args.school_rule_steps):
        sc0.rule_train_step(model, opt, tok, train_rng)

        if (step + 1) % args.eval_every == 0:
            held_out_acc = sc0.rule_eval(model, tok, eval_rng, args.demo_eval_n)
            tracker.push(held_out_acc=held_out_acc)

        ep = generate_rule_episode(eval_rng)
        rule_ids = torch.tensor([tok.encode(ep["rule"])])
        question_ids = torch.tensor([tok.encode(ep["question"])])
        with torch.no_grad():
            pred_idx = int(model.rule_forward(rule_ids, question_ids).argmax(-1).item())

        write_snapshot(state_file, base_snapshot(
            stage="School0-rule", stage_idx=stage_idx, n_stages=n_stages,
            step=step + 1, total_steps=args.school_rule_steps,
            instruction=ep["question"], objects=[], target_idx=None, pred_idx=None,
            passage=[ep["rule"]], recall_true=ep["conclusion_size"], recall_pred=SIZES[pred_idx],
            metrics_current={"held_out_acc": (tracker.history.get("held_out_acc") or [0.5])[-1]},
            metrics_chance={"held_out_acc": 0.5}, metrics_baseline={}, history=tracker.snapshot()))
        time.sleep(args.step_delay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", type=Path, default=Path("/tmp/hz_world_live_state.json"))
    parser.add_argument("--step-delay", type=float, default=0.08, help="seconds between snapshots, for watchability")
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--demo-eval-n", type=int, default=40,
                         help="held-out episodes per eval point -- SMALL vs hz_nursery_train.py's 200, "
                              "for live responsiveness; see module docstring")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--memory-slots", type=int, default=8)
    parser.add_argument("--workspace-slots", type=int, default=32)
    parser.add_argument("--n-rounds-l1", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--l0-steps", type=int, default=800)
    parser.add_argument("--l0-batch-size", type=int, default=16)
    parser.add_argument("--l1-steps", type=int, default=800)
    parser.add_argument("--l1-n-objects", type=int, default=4)
    parser.add_argument("--l2-steps", type=int, default=800)
    parser.add_argument("--l2-n-objects", type=int, default=4)
    parser.add_argument("--l3-steps", type=int, default=800)
    parser.add_argument("--l3-n-objects", type=int, default=4)
    parser.add_argument("--l4-logic-steps", type=int, default=800)
    parser.add_argument("--l4-counting-steps", type=int, default=800)
    parser.add_argument("--l4-n-objects", type=int, default=4)
    parser.add_argument("--l5-steps", type=int, default=600)
    parser.add_argument("--l5-n-objects", type=int, default=4)
    parser.add_argument("--l5-stress-steps", type=int, default=1200)
    parser.add_argument("--l5-stress-facts", type=int, default=3)
    parser.add_argument("--l5-stress-distractors", type=int, default=2)
    parser.add_argument("--l6-steps", type=int, default=1200)
    parser.add_argument("--l6-n-sentences", type=int, default=3)
    parser.add_argument("--school-arith-steps", type=int, default=1200)
    parser.add_argument("--school-rule-steps", type=int, default=600)
    parser.add_argument("--loop", action="store_true", help="repeat the whole L0-School0 curriculum forever")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=args.d_model, memory_slots=args.memory_slots,
                             workspace_slots=args.workspace_slots, n_rounds_l1=args.n_rounds_l1,
                             n_qa_labels=len(NOVEL_LABELS), n_read_labels=len(SIZES), n_arith_labels=len(NUMBERS))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"[hz_nursery_live_demo] vocab_size={tok.vocab_size} n_params="
          f"{sum(p.numel() for p in model.parameters())}, writing live state to {args.state_file}", flush=True)

    stages = [
        ("L0", run_l0), ("L1", run_l1), ("L2", run_l2), ("L3", run_l3),
        ("L4-logic", run_l4_logic), ("L4-counting", run_l4_counting),
        ("L5", run_l5), ("L5-stress", run_l5_stress), ("L6", run_l6),
        ("School0-arith", run_school0_arith), ("School0-rule", run_school0_rule),
    ]
    run = 0
    while True:
        run += 1
        for stage_idx, (name, fn) in enumerate(stages):
            tracker = HistoryTracker()
            print(f"[hz_nursery_live_demo] run={run} starting stage {name}", flush=True)
            fn(model, opt, tok, args, args.state_file, tracker, stage_idx, len(stages))
        if not args.loop:
            break
    print("[hz_nursery_live_demo] curriculum complete", flush=True)


if __name__ == "__main__":
    main()
