"""Regression tests for `reference/hz0b_b9_finetune.py`'s block-unfreezing
utilities -- deterministic, uses a tiny synthetic `HZ0AMlxModel` (not the
real 301M checkpoint) so this runs fast and needs no local checkpoint
file. The real, checkpoint-dependent B9 Stage 1 result lives in
`scripts/hz0b_b9_stage1_finetune_probe.py`.
"""
import mlx.core as mx
from mlx.utils import tree_flatten

from reference.hz0a_mlx_model import HZ0AMlxModel
from reference.hz0b_b9_finetune import apply_block_params, block_param_count, block_params_dict


def make_tiny_model():
    return HZ0AMlxModel(vocab_size=32, dim=16, layers=3, heads=2, d_ff=32, attention_indices=(1,))


def test_block_params_dict_round_trips_through_apply():
    model = make_tiny_model()
    original = block_params_dict(model, 2)
    apply_block_params(model, 2, original)  # no-op: writing the same values back
    after = block_params_dict(model, 2)
    assert set(original.keys()) == set(after.keys())
    for key in original:
        assert bool(mx.array_equal(original[key], after[key]))


def test_apply_block_params_actually_changes_only_the_targeted_block():
    model = make_tiny_model()
    before_block2 = {k: mx.array(v) for k, v in block_params_dict(model, 2).items()}
    before_block0 = {k: mx.array(v) for k, v in block_params_dict(model, 0).items()}

    modified = {k: v + 1.0 for k, v in before_block2.items()}
    apply_block_params(model, 2, modified)

    after_block2 = block_params_dict(model, 2)
    after_block0 = block_params_dict(model, 0)
    for key in before_block2:
        assert not bool(mx.array_equal(before_block2[key], after_block2[key])), "targeted block must actually change"
    for key in before_block0:
        assert bool(mx.array_equal(before_block0[key], after_block0[key])), "untouched block must stay exactly the same"


def test_apply_block_params_respects_key_prefix_and_ignores_other_keys():
    model = make_tiny_model()
    before = {k: mx.array(v) for k, v in block_params_dict(model, 1).items()}
    combined = {**{k: v + 5.0 for k, v in before.items()}, "controller.some_unrelated_param": mx.zeros((4,))}
    apply_block_params(model, 1, combined)  # must ignore the "controller." key entirely, no crash
    after = block_params_dict(model, 1)
    for key in before:
        assert not bool(mx.array_equal(before[key], after[key]))


def test_block_param_count_matches_manual_flatten():
    model = make_tiny_model()
    expected = sum(v.size for _, v in tree_flatten(model.blocks[1].parameters()))
    assert block_param_count(model, 1) == expected
