from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reference.hz0a_gdn2_reference import gdn2_scan


@dataclass
class Gdn2ForwardInputs:
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    decay_logits: np.ndarray
    erase_logits: np.ndarray
    write_logits: np.ndarray
    initial_state: np.ndarray


@dataclass
class Gdn2ForwardCache:
    decay_logits: np.ndarray
    erase_logits: np.ndarray
    write_logits: np.ndarray
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    initial_state: np.ndarray


@dataclass
class Gdn2ForwardResult:
    outputs: np.ndarray
    final_state: np.ndarray
    backward_cache: Gdn2ForwardCache


def gdn2_forward(inputs: Gdn2ForwardInputs) -> Gdn2ForwardResult:
    outputs, final_state = gdn2_scan(
        inputs.q,
        inputs.k,
        inputs.v,
        inputs.decay_logits,
        inputs.erase_logits,
        inputs.write_logits,
        initial_state=inputs.initial_state,
    )
    cache = Gdn2ForwardCache(
        decay_logits=inputs.decay_logits.copy(),
        erase_logits=inputs.erase_logits.copy(),
        write_logits=inputs.write_logits.copy(),
        q=inputs.q.copy(),
        k=inputs.k.copy(),
        v=inputs.v.copy(),
        initial_state=inputs.initial_state.copy(),
    )
    return Gdn2ForwardResult(outputs=outputs, final_state=final_state, backward_cache=cache)
