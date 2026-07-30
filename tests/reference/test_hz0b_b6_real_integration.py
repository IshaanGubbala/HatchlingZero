"""HZ-0B B6 real-integration regression test: the free, deterministic half
of B6's exit gate (empty memory behaves exactly like no memory) checked
against the ACTUAL frozen HZ-0A hybrid checkpoint and REAL held-out data,
not synthetic hidden states. Skips if the checkpoint isn't present (it
lives under the gitignored `outputs/` directory, so a fresh clone or CI
environment without that checkpoint should skip cleanly, not fail).

The other half of B6's exit gate (a trained read-only memory path
measurably improving a memory-specific task) is a real training
experiment with meaningfully long runtime and inherent randomness --
that lives in `scripts/hz0b_b6_real_integration_probe.py` as a
documented, run-on-demand result (see
`docs/restart/hz0b_b6_real_integration_results.md`), not a fast
deterministic unit test.
"""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_unflatten

from reference.hz0a_mlx_model import HZ0AMlxModel
from reference.hz0b_b6_hz0a_integration import forward
from reference.hz0b_memory_simulator import reset as memory_reset
from reference.hz0b_readonly_integration import init_readonly_integration

CHECKPOINT = Path("outputs/hz0a_stage2_100m_hybrid_seed7/native_metal_checkpoint_best_full_holdout")
VAL_DATA = Path("data/packed/repro_256_val.jsonl")

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "state.json").exists() or not VAL_DATA.exists(),
    reason="frozen HZ-0A checkpoint / real held-out data not present locally (both gitignored under outputs/ and data/)",
)


def _load_model() -> HZ0AMlxModel:
    payload = json.loads((CHECKPOINT / "state.json").read_text())
    model = HZ0AMlxModel(24576, 768, 31, 12, 2304, (4, 9, 14, 19, 24, 29), native_metal=True)
    model_arrays = [(item["key"], mx.load(str(CHECKPOINT / item["file"]))) for item in payload["arrays"] if item["group"] == "model"]
    model.update(tree_unflatten(model_arrays))
    mx.eval(model.parameters())
    return model


def test_empty_memory_exactly_matches_no_memory_on_real_frozen_checkpoint():
    model = _load_model()
    lines = VAL_DATA.open().readlines()[:8]
    tokens = mx.array([json.loads(line)[:256] for line in lines], dtype=mx.int32)

    logits_no_memory, _ = forward(model, tokens)
    mx.eval(logits_no_memory)

    params = init_readonly_integration(768, 32, 32, seed=42)
    memory = memory_reset(batch_size=tokens.shape[0], num_slots=8, key_dim=32, value_dim=32)
    logits_empty_memory, _ = forward(model, tokens, memory_params=params, memory_state=memory)
    mx.eval(logits_empty_memory)

    assert bool(mx.array_equal(logits_no_memory, logits_empty_memory)), "empty memory must be bit-identical to no memory, even against the real frozen checkpoint"

    loss_no_memory = float(mx.mean(nn.losses.cross_entropy(logits_no_memory[:, :-1].astype(mx.float32), tokens[:, 1:])))
    loss_empty_memory = float(mx.mean(nn.losses.cross_entropy(logits_empty_memory[:, :-1].astype(mx.float32), tokens[:, 1:])))
    assert loss_no_memory == loss_empty_memory
