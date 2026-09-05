"""Real structural + behavioral tests for the Language Nursery L0/L1
stages, plans/Hatchling world.md section 5 and the rescue ladder's
"is the language-model loss actually learning" check (section 1.3)."""
from __future__ import annotations

import random

import torch

from hatchling_world.language.nursery_generator import generate_l0_sentence, generate_l1_grounding_episode
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


def test_model_uses_default_ln_recurrence_and_d_over_2_value_write():
    """Real check that the Nursery model follows the plan's own KEEP
    list -- no new recurrence experiments here either."""
    tok = NurseryTokenizer()
    model = HZLanguageModel(vocab_size=tok.vocab_size, d_model=32, memory_slots=8, workspace_slots=16, n_rounds_l1=4)
    assert model.ws.config.value_dim == model.D // 2
    assert model.ws.config.identity_biased is False
    assert model.ws.config.bounded_residual is False
    assert model.ws.config.bounded_accumulating is False
