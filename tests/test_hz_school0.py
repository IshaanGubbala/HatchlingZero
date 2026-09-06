"""Real structural + behavioral tests for School-0 (plans/Hatchling
world.md section 8.2): arithmetic with a held-out operand-pair split,
and conditional-rule application (a real deduction test, distinct from
L5's fact recall)."""
from __future__ import annotations

import random

import torch

from hatchling_world.school.generator import (
    generate_arithmetic_episode, generate_rule_episode, generate_cs_program_episode,
    generate_physics_episode, generate_physics_fixed_identity_episode, PHYSICS_IDENTITY_LABELS,
    generate_value_retrieval_episode, generate_entity_select_episode,
    ARITH_TRAIN_PAIRS, ARITH_HELD_OUT_PAIRS,
)
from hatchling_world.language.tokenizer import NurseryTokenizer, NUMBERS, SIZES, COLORS
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


def test_cs_program_episode_sum_is_correct():
    rng = random.Random(5)
    for _ in range(50):
        ep = generate_cs_program_episode(rng)
        assert ep["sum"] == ep["x"] + ep["y"]
        assert ep["sum"] < len(NUMBERS)
        assert NUMBERS[ep["x"]] in ep["program"][0] and "x" in ep["program"][0]
        assert NUMBERS[ep["y"]] in ep["program"][1] and "y" in ep["program"][1]


def test_cs_program_forward_shapes_and_gradients_reuses_arithmetic_head():
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=32, memory_slots=8, workspace_slots=16,
                             n_rounds_l1=4, n_arith_labels=len(NUMBERS))
    rng = random.Random(6)
    ep = generate_cs_program_episode(rng)
    statement_ids_list = [torch.tensor([tok.encode(s)]) for s in ep["program"]]
    question_ids = torch.tensor([tok.encode(ep["question"])])
    logits = model.cs_program_forward(statement_ids_list, question_ids)
    assert logits.shape == (1, len(NUMBERS))
    loss = logits.sum()
    loss.backward()
    assert model.arithmetic_head.weight.grad is not None, "cs_program_forward must route through the shared arithmetic_head"
    assert model.mem.q_proj.weight.grad is not None


def test_physics_episode_answer_is_the_large_object_and_colors_are_distinct():
    rng = random.Random(7)
    for _ in range(50):
        ep = generate_physics_episode(rng)
        assert ep["large_color"] != ep["small_color"]
        assert ep["answer_color"] == ep["large_color"]
        assert COLORS[ep["answer_idx"]] == ep["large_color"]
        assert ep["large_color"] in ep["question"] and ep["small_color"] in ep["question"]


def test_physics_forward_shapes_and_gradients_reuses_read_head():
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=32, memory_slots=8, workspace_slots=16,
                             n_rounds_l1=4, n_read_labels=len(COLORS))
    rng = random.Random(8)
    ep = generate_physics_episode(rng)
    teach_ids = torch.tensor([tok.encode(ep["teach"])])
    scenario_ids = torch.tensor([tok.encode(ep["scenario"])])
    question_ids = torch.tensor([tok.encode(ep["question"])])
    logits = model.physics_forward(teach_ids, scenario_ids, question_ids)
    assert logits.shape == (1, len(COLORS))
    loss = logits.sum()
    loss.backward()
    assert model.read_head.weight.grad is not None, "physics_forward must route through the shared read_head"
    assert model.mem.q_proj.weight.grad is not None


def test_physics_fixed_identity_episode_answer_is_the_large_id_and_ids_are_distinct():
    rng = random.Random(9)
    for _ in range(50):
        ep = generate_physics_fixed_identity_episode(rng)
        assert ep["large_id"] != ep["small_id"]
        assert {ep["large_id"], ep["small_id"]} == set(PHYSICS_IDENTITY_LABELS)
        assert ep["answer_id"] == ep["large_id"]
        assert PHYSICS_IDENTITY_LABELS[ep["answer_idx"]] == ep["large_id"]
        assert ep["large_id"] in ep["question"] and ep["small_id"] in ep["question"]


def test_physics_forward_works_unchanged_on_fixed_identity_variant():
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=32, memory_slots=8, workspace_slots=16,
                             n_rounds_l1=4, n_read_labels=len(PHYSICS_IDENTITY_LABELS))
    rng = random.Random(10)
    ep = generate_physics_fixed_identity_episode(rng)
    teach_ids = torch.tensor([tok.encode(ep["teach"])])
    scenario_ids = torch.tensor([tok.encode(ep["scenario"])])
    question_ids = torch.tensor([tok.encode(ep["question"])])
    logits = model.physics_forward(teach_ids, scenario_ids, question_ids)
    assert logits.shape == (1, len(PHYSICS_IDENTITY_LABELS))
    logits.sum().backward()
    assert model.read_head.weight.grad is not None


def test_value_retrieval_episode_answer_matches_the_asked_variable():
    rng = random.Random(11)
    for _ in range(50):
        ep = generate_value_retrieval_episode(rng)
        if ep["question"] == "what is x":
            assert ep["answer"] == ep["x"]
        else:
            assert ep["question"] == "what is y"
            assert ep["answer"] == ep["y"]
        assert ep["answer"] == ep["answer_idx"]


def test_entity_select_episode_answer_points_to_the_asked_property():
    rng = random.Random(12)
    for _ in range(50):
        ep = generate_entity_select_episode(rng)
        assert {ep["prop_x"], ep["prop_y"]} == {"widget", "gadget"}
        assert ep["prop_x"] != ep["prop_y"]
        expected = "x" if ep["prop_x"] == ep["asked_prop"] else "y"
        assert ep["answer_id"] == expected
        assert PHYSICS_IDENTITY_LABELS[ep["answer_idx"]] == ep["answer_id"]


def test_entity_select_forward_shapes_and_gradients_reuses_read_head():
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=32, memory_slots=8, workspace_slots=16,
                             n_rounds_l1=4, n_read_labels=len(PHYSICS_IDENTITY_LABELS))
    rng = random.Random(13)
    ep = generate_entity_select_episode(rng)
    statement_ids_list = [torch.tensor([tok.encode(s)]) for s in ep["program"]]
    question_ids = torch.tensor([tok.encode(ep["question"])])
    logits = model.entity_select_forward(statement_ids_list, question_ids)
    assert logits.shape == (1, len(PHYSICS_IDENTITY_LABELS))
    loss = logits.sum()
    loss.backward()
    assert model.read_head.weight.grad is not None, "entity_select_forward must route through the shared read_head"
    assert model.mem.q_proj.weight.grad is not None
