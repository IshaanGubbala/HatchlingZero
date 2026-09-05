"""Real structural + behavioral tests for School-0 (plans/Hatchling
world.md section 8.2): arithmetic with a held-out operand-pair split,
and conditional-rule application (a real deduction test, distinct from
L5's fact recall)."""
from __future__ import annotations

import random

import torch

from hatchling_world.school.generator import (
    generate_arithmetic_episode, generate_rule_episode, ARITH_TRAIN_PAIRS, ARITH_HELD_OUT_PAIRS,
)
from hatchling_world.language.tokenizer import NurseryTokenizer, NUMBERS, SIZES
from reference.hz_language_model_torch import HZLanguageModel


def test_arith_train_and_held_out_pairs_are_disjoint():
    assert set(ARITH_TRAIN_PAIRS).isdisjoint(ARITH_HELD_OUT_PAIRS)
    assert len(ARITH_TRAIN_PAIRS) + len(ARITH_HELD_OUT_PAIRS) == 25
    rng = random.Random(0)
    for _ in range(50):
        ep = generate_arithmetic_episode(rng, split="test")
        assert (ep["a"], ep["b"]) in ARITH_HELD_OUT_PAIRS
    for _ in range(50):
        ep = generate_arithmetic_episode(rng, split="train")
        assert (ep["a"], ep["b"]) in ARITH_TRAIN_PAIRS


def test_arithmetic_episode_sum_is_correct_and_in_vocabulary():
    rng = random.Random(1)
    for _ in range(50):
        ep = generate_arithmetic_episode(rng, split="train")
        assert ep["sum"] == ep["a"] + ep["b"]
        assert NUMBERS[ep["sum_idx"]] == NUMBERS[ep["a"] + ep["b"]]
        assert ep["sum"] < len(NUMBERS)


def test_rule_episode_answer_never_stated_directly():
    rng = random.Random(2)
    for _ in range(50):
        ep = generate_rule_episode(rng)
        assert ep["conclusion_size"] not in ep["question"]
        assert ep["premise_color"] in ep["rule"] and ep["premise_color"] in ep["question"]
        assert ep["conclusion_size"] in ep["rule"]
        assert SIZES[ep["answer_idx"]] == ep["conclusion_size"]


def test_arithmetic_forward_shapes_and_gradients():
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=32, memory_slots=8, workspace_slots=16,
                             n_rounds_l1=4, n_arith_labels=len(NUMBERS))
    rng = random.Random(3)
    ep = generate_arithmetic_episode(rng, split="train")
    ids = torch.tensor([tok.encode(ep["instruction"])])
    logits = model.arithmetic_forward(ids)
    assert logits.shape == (1, len(NUMBERS))
    loss = logits.sum()
    loss.backward()
    assert model.arithmetic_head.weight.grad is not None
    assert model.mem.q_proj.weight.grad is not None


def test_rule_forward_shapes_and_gradients_reuses_read_head():
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=32, memory_slots=8, workspace_slots=16,
                             n_rounds_l1=4, n_read_labels=len(SIZES))
    rng = random.Random(4)
    ep = generate_rule_episode(rng)
    rule_ids = torch.tensor([tok.encode(ep["rule"])])
    question_ids = torch.tensor([tok.encode(ep["question"])])
    logits = model.rule_forward(rule_ids, question_ids)
    assert logits.shape == (1, len(SIZES))
    loss = logits.sum()
    loss.backward()
    assert model.read_head.weight.grad is not None, "rule_forward must route through the shared read_head"
