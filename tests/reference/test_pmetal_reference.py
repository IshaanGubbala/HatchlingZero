from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reference.hz0a_gdn2_reference import gdn2_scan, init_state  # noqa: E402
from restart.hz0a_pmetal.python.pmetal_reference import (  # noqa: E402
    Gdn2ForwardInputs,
    gdn2_forward,
)


def test_pmetal_style_forward_matches_numpy_oracle() -> None:
    rng = np.random.default_rng(42)
    q = rng.normal(size=(2, 5, 3, 4)).astype(np.float32)
    k = rng.normal(size=(2, 5, 3, 4)).astype(np.float32)
    v = rng.normal(size=(2, 5, 3, 6)).astype(np.float32)
    decay = rng.normal(size=(2, 5, 3, 4)).astype(np.float32)
    erase = rng.normal(size=(2, 5, 3, 4)).astype(np.float32)
    write = rng.normal(size=(2, 5, 3, 6)).astype(np.float32)
    state = init_state(batch_size=2, num_heads=3, d_v=6, d_k=4)

    expected_out, expected_state = gdn2_scan(q, k, v, decay, erase, write, initial_state=state)
    actual = gdn2_forward(
        Gdn2ForwardInputs(
            q=q,
            k=k,
            v=v,
            decay_logits=decay,
            erase_logits=erase,
            write_logits=write,
            initial_state=state,
        )
    )

    np.testing.assert_allclose(actual.outputs, expected_out, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual.final_state, expected_state, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual.backward_cache.q, q)
    np.testing.assert_allclose(actual.backward_cache.initial_state, state)
