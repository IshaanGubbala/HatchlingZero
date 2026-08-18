"""Correctness gates for persistent, physically packed BlockBDH."""
from __future__ import annotations

import pytest
import torch

from reference.hz0h_bdh_blocksparse_torch import bdh_blocksparse_forward
from reference.hz0h_bdh_compiled_blocks_torch import (
    PackedBlockBDH,
    calibrate_compiled_block_layout,
    weighted_reverse_cuthill_mckee,
)
from reference.hz0h_bdh_torch import BDH, BDHConfig


def _config() -> BDHConfig:
    return BDHConfig(
        n_layer=2,
        n_embd=32,
        n_head=4,
        mlp_internal_dim_multiplier=8,
        vocab_size=41,
        dropout=0.0,
    )


def _column_indices(blocks: torch.Tensor, block_size: int) -> torch.Tensor:
    offsets = torch.arange(block_size)
    return (blocks[:, None] * block_size + offsets).reshape(-1)


def test_packed_forward_loss_and_all_gradients_match_fixed_mask_oracle():
    torch.manual_seed(17)
    source = BDH(_config()).train()
    # Deliberately permute blocks to verify physical order and RoPE pairs.
    blocks = torch.tensor([3, 0, 2])
    block_size = 8
    packed = PackedBlockBDH(source, blocks, block_size=block_size).train()
    idx = torch.randint(0, source.config.vocab_size, (2, 7))
    targets = torch.randint(0, source.config.vocab_size, idx.shape)

    oracle_logits, oracle_loss = bdh_blocksparse_forward(
        source, idx, blocks, block_size, targets
    )
    packed_logits, packed_loss = packed(idx, targets)

    torch.testing.assert_close(packed_logits, oracle_logits, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(packed_loss, oracle_loss, rtol=2e-6, atol=2e-6)
    oracle_loss.backward()
    packed_loss.backward()

    columns = _column_indices(blocks, block_size)
    width = source.config.n_embd * source.config.mlp_internal_dim_multiplier // source.config.n_head
    torch.testing.assert_close(
        packed.encoder.grad, source.encoder.grad.index_select(2, columns), rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        packed.encoder_v.grad, source.encoder_v.grad.index_select(2, columns), rtol=2e-5, atol=2e-6
    )
    oracle_decoder_grad = source.decoder.grad.reshape(source.config.n_head, width, source.config.n_embd)
    torch.testing.assert_close(
        packed.decoder.grad,
        oracle_decoder_grad.index_select(1, columns).reshape_as(packed.decoder),
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(packed.embed.weight.grad, source.embed.weight.grad, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(packed.lm_head.grad, source.lm_head.grad, rtol=2e-5, atol=2e-6)


def test_packed_model_reduces_trainable_parameters_and_has_no_runtime_gather(monkeypatch):
    torch.manual_seed(19)
    source = BDH(_config()).eval()
    packed = PackedBlockBDH(source, torch.tensor([1, 3]), block_size=8).eval()
    assert sum(p.numel() for p in packed.parameters()) < sum(p.numel() for p in source.parameters())

    def fail_index_select(*_args, **_kwargs):
        raise AssertionError("index_select must not execute in the packed forward hot path")

    monkeypatch.setattr(torch.Tensor, "index_select", fail_index_select)
    idx = torch.randint(0, source.config.vocab_size, (2, 5))
    logits, loss = packed(idx, idx)
    assert torch.isfinite(logits).all()
    assert loss is not None and torch.isfinite(loss)


def test_state_dict_round_trip_preserves_layout_and_output():
    torch.manual_seed(23)
    source = BDH(_config()).eval()
    blocks = torch.tensor([3, 1])
    first = PackedBlockBDH(source, blocks, block_size=8).eval()
    second = PackedBlockBDH(source, blocks, block_size=8).eval()
    second.load_state_dict(first.state_dict())
    idx = torch.randint(0, source.config.vocab_size, (2, 6))
    torch.testing.assert_close(first(idx)[0], second(idx)[0], rtol=0, atol=0)
    assert torch.equal(first.packed_block_indices, blocks)
    assert torch.equal(second.packed_block_indices, blocks)


def test_calibration_is_deterministic_pair_safe_and_selects_requested_count():
    torch.manual_seed(29)
    model = BDH(_config()).eval()
    batches = [torch.randint(0, model.config.vocab_size, (2, 6)) for _ in range(2)]
    first = calibrate_compiled_block_layout(
        model, batches, block_size=8, active_fraction=0.5
    )
    second = calibrate_compiled_block_layout(
        model, batches, block_size=8, active_fraction=0.5
    )
    assert first == second
    assert first.block_size % 2 == 0
    assert len(first.block_indices) == 4
    assert len(set(first.block_indices)) == 4
    assert len(first.importance) == 8
    packed = PackedBlockBDH.from_layout(model, first)
    assert packed.config.mlp_internal_dim_multiplier == 4


def test_weighted_rcm_is_deterministic_permutation():
    graph = torch.tensor(
        [[0.0, 4.0, 0.1, 0.0], [4.0, 0.0, 3.0, 0.1], [0.1, 3.0, 0.0, 2.0], [0.0, 0.1, 2.0, 0.0]]
    )
    first = weighted_reverse_cuthill_mckee(graph)
    second = weighted_reverse_cuthill_mckee(graph)
    assert torch.equal(first, second)
    assert sorted(first.tolist()) == list(range(4))


@pytest.mark.parametrize(
    ("blocks", "block_size", "message"),
    [
        (torch.tensor([0]), 3, "positive even"),
        (torch.tensor([], dtype=torch.long), 8, "non-empty"),
        (torch.tensor([1, 1]), 8, "unique"),
        (torch.tensor([8]), 8, "outside"),
    ],
)
def test_invalid_layouts_are_rejected(blocks, block_size, message):
    model = BDH(_config())
    with pytest.raises(ValueError, match=message):
        PackedBlockBDH(model, blocks, block_size=block_size)


def test_calibration_rejects_empty_batches_and_bad_fraction():
    model = BDH(_config())
    with pytest.raises(ValueError, match="at least one"):
        calibrate_compiled_block_layout(model, [], block_size=8, active_fraction=0.5)
    with pytest.raises(ValueError, match="active_fraction"):
        calibrate_compiled_block_layout(model, [], block_size=8, active_fraction=0.0)
