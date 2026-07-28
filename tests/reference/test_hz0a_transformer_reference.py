from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.hz0a_gdn2_reference import cross_entropy_loss
from reference.hz0a_transformer_reference import TinyTransformerModel


ROOT = Path(__file__).resolve().parents[2]


def test_matched_transformer_count_is_reproducible() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/hz0a_transformer_param_count.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert payload["count_matches_config"] is True
    assert payload["parameter_count_computed"] == 301179928
    assert payload["absolute_parameter_difference"] == 1816


def test_tiny_transformer_is_deterministic_and_has_shared_lm_loss() -> None:
    model_a = TinyTransformerModel.init(rng_seed=33, vocab_size=32, d_model=16, num_layers=3, num_heads=4, d_ff=32)
    model_b = TinyTransformerModel.init(rng_seed=33, vocab_size=32, d_model=16, num_layers=3, num_heads=4, d_ff=32)
    token_ids = np.array([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=np.int64)
    targets = np.array([[2, 3, 4, 5], [3, 2, 1, 0]], dtype=np.int64)

    logits_a = model_a(token_ids)
    logits_b = model_b(token_ids)
    np.testing.assert_array_equal(logits_a, logits_b)
    assert model_a.loss(token_ids, targets) == cross_entropy_loss(logits_a, targets)
    assert np.isfinite(logits_a).all()
