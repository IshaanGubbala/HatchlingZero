"""Real structural + behavioral tests for the Language Nursery L0/L1
stages, plans/Hatchling world.md section 5 and the rescue ladder's
"is the language-model loss actually learning" check (section 1.3)."""
from __future__ import annotations

import random

import torch

from hatchling_world.language.nursery_generator import (
    apply_verb, generate_l0_sentence, generate_l1_grounding_episode, generate_l2_verb_episode,
    generate_l3_relation_episode, HELD_OUT_COMBOS, TRAIN_COMBOS,
)
from hatchling_world.language.tokenizer import COLORS, NOUNS, NurseryTokenizer, POSITIONS, SIZES
from reference.hz_language_model_torch import HZLanguageModel


def test_tokenizer_roundtrip():
    tok = NurseryTokenizer()
    ids = tok.encode("the red ball moves")
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids) == "the red ball moves"


def test_l0_sentences_are_valid_vocabulary():
    tok = NurseryTokenizer()
    rng = random.Random(0)
    for _ in range(50):
        s = generate_l0_sentence(rng)
        ids = tok.encode(s)
        assert tok.unk_id not in ids, f"L0 generator produced an out-of-vocabulary word: {s}"


def test_l1_episode_has_exactly_one_matching_object():
    rng = random.Random(0)
    for _ in range(50):
        ep = generate_l1_grounding_episode(rng, n_objects=4)
        target_color = ep["objects"][ep["target_idx"]]["color"]
        matches = [i for i, o in enumerate(ep["objects"]) if o["color"] == target_color]
        assert matches == [ep["target_idx"]], "instruction must uniquely identify exactly one object"


def test_lm_forward_shapes_and_gradients():
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=32, memory_slots=8, workspace_slots=16, n_rounds_l1=4)
    ids = torch.tensor([tok.encode("the red ball moves")])
    logits = model.lm_forward(ids)
    assert logits.shape == (1, ids.shape[1] - 1, tok.vocab_size)
    loss = logits.sum()
    loss.backward()
    assert model.token_embed.weight.grad is not None
    assert torch.isfinite(model.token_embed.weight.grad).all()


def test_ground_forward_shapes_and_gradients():
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=32, memory_slots=8, workspace_slots=16, n_rounds_l1=4)
    rng = random.Random(1)
    ep = generate_l1_grounding_episode(rng, n_objects=4)
    instr_ids = torch.tensor([tok.encode(ep["instruction"])])
    type_idx = torch.tensor([[NOUNS.index(o["type"]) for o in ep["objects"]]])
    color_idx = torch.tensor([[COLORS.index(o["color"]) for o in ep["objects"]]])
    size_idx = torch.tensor([[SIZES.index(o["size"]) for o in ep["objects"]]])
    pos_idx = torch.tensor([[POSITIONS.index(o["position"]) for o in ep["objects"]]])
    logits = model.ground_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx)
    assert logits.shape == (1, 4)
    loss = logits.sum()
    loss.backward()
    assert model.object_encoder.weight.grad is not None
    assert torch.isfinite(model.object_encoder.weight.grad).all()


def test_apply_verb_changes_exactly_the_relevant_attribute():
    """The whole point of L2: each verb changes ONE attribute and
    leaves the others exactly as they were -- a sharp, checkable
    definition of "verb meaning" as a state transition."""
    held, opened, pos = apply_verb("push", held=False, opened=False, position="left")
    assert (held, opened, pos) == (False, False, "right")
    held, opened, pos = apply_verb("pickup", held=False, opened=True, position="left")
    assert (held, opened, pos) == (True, True, "left")
    held, opened, pos = apply_verb("drop", held=True, opened=False, position="right")
    assert (held, opened, pos) == (False, False, "right")
    held, opened, pos = apply_verb("open", held=True, opened=False, position="right")
    assert (held, opened, pos) == (True, True, "right")
    held, opened, pos = apply_verb("close", held=False, opened=True, position="left")
    assert (held, opened, pos) == (False, False, "left")


def test_l2_episode_has_exactly_one_matching_object_and_consistent_consequence():
    rng = random.Random(0)
    for _ in range(50):
        ep = generate_l2_verb_episode(rng, n_objects=4)
        target_color = ep["objects"][ep["target_idx"]]["color"]
        matches = [i for i, o in enumerate(ep["objects"]) if o["color"] == target_color]
        assert matches == [ep["target_idx"]]
        target = ep["objects"][ep["target_idx"]]
        held_after, opened_after, position_after = apply_verb(
            ep["verb"], target["held"], target["opened"], target["position"])
        assert (held_after, opened_after, position_after) == (
            ep["held_after"], ep["opened_after"], ep["position_after"])


def test_verb_forward_shapes_and_gradients():
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=32, memory_slots=8, workspace_slots=16, n_rounds_l1=4)
    rng = random.Random(2)
    ep = generate_l2_verb_episode(rng, n_objects=4)
    instr_ids = torch.tensor([tok.encode(ep["instruction"])])
    type_idx = torch.tensor([[NOUNS.index(o["type"]) for o in ep["objects"]]])
    color_idx = torch.tensor([[COLORS.index(o["color"]) for o in ep["objects"]]])
    size_idx = torch.tensor([[SIZES.index(o["size"]) for o in ep["objects"]]])
    pos_idx = torch.tensor([[POSITIONS.index(o["position"]) for o in ep["objects"]]])
    held = torch.tensor([[float(o["held"]) for o in ep["objects"]]])
    opened = torch.tensor([[float(o["opened"]) for o in ep["objects"]]])
    sel_logits, cons_logits = model.verb_forward(instr_ids, type_idx, color_idx, size_idx, pos_idx, held, opened)
    assert sel_logits.shape == (1, 4)
    assert cons_logits.shape == (1, 3)
    loss = sel_logits.sum() + cons_logits.sum()
    loss.backward()
    assert model.object_state_encoder.weight.grad is not None
    assert torch.isfinite(model.object_state_encoder.weight.grad).all()
    assert model.consequence_head.weight.grad is not None


def test_l3_episode_has_exactly_one_matching_object_and_needs_both_properties():
    rng = random.Random(0)
    for split in ("train", "test"):
        for _ in range(50):
            ep = generate_l3_relation_episode(rng, n_objects=4, split=split)
            target = ep["objects"][ep["target_idx"]]
            matches = [i for i, o in enumerate(ep["objects"])
                       if o["size"] == target["size"] and o["color"] == target["color"]]
            assert matches == [ep["target_idx"]], "instruction must uniquely identify exactly one object"
            same_color = [o for i, o in enumerate(ep["objects"]) if i != ep["target_idx"] and o["color"] == target["color"]]
            same_size = [o for i, o in enumerate(ep["objects"]) if i != ep["target_idx"] and o["size"] == target["size"]]
            assert same_color, "color alone must collide with a decoy -- composition must be necessary"
            assert same_size, "size alone must collide with a decoy -- composition must be necessary"


def test_l3_train_and_held_out_combos_are_disjoint():
    assert set(HELD_OUT_COMBOS).isdisjoint(TRAIN_COMBOS)
    rng = random.Random(1)
    for _ in range(50):
        ep = generate_l3_relation_episode(rng, n_objects=4, split="test")
        target = ep["objects"][ep["target_idx"]]
        assert (target["size"], target["color"]) in HELD_OUT_COMBOS
    for _ in range(50):
        ep = generate_l3_relation_episode(rng, n_objects=4, split="train")
        target = ep["objects"][ep["target_idx"]]
        assert (target["size"], target["color"]) in TRAIN_COMBOS


def test_model_uses_default_ln_recurrence_and_d_over_2_value_write():
    """Real check that the Nursery model follows the plan's own KEEP
    list -- no new recurrence experiments here either."""
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=32, memory_slots=8, workspace_slots=16, n_rounds_l1=4)
    assert model.ws.config.value_dim == model.D // 2
    assert model.ws.config.identity_biased is False
    assert model.ws.config.bounded_residual is False
    assert model.ws.config.bounded_accumulating is False
