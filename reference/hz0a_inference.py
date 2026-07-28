from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass

import numpy as np

from reference.hz0a_gdn2_reference import TinyHZ0AModel


@dataclass
class DecodeResult:
    logits: np.ndarray
    states: list[np.ndarray | None]


def _require_recurrent(model: TinyHZ0AModel) -> None:
    if any(block.is_attention for block in model.blocks):
        raise ValueError("tokenwise recurrent decode requires a model with no attention blocks")


def prefill(model: TinyHZ0AModel, token_ids: np.ndarray) -> DecodeResult:
    _require_recurrent(model)
    logits, states = model(token_ids)
    return DecodeResult(logits=logits, states=states)


def decode_tokenwise(model: TinyHZ0AModel, token_ids: np.ndarray, states: list[np.ndarray | None] | None = None) -> DecodeResult:
    _require_recurrent(model)
    current_states = model.init_states(token_ids.shape[0]) if states is None else states
    outputs = []
    for index in range(token_ids.shape[1]):
        logits, current_states = model(token_ids[:, index:index + 1], current_states)
        outputs.append(logits)
    return DecodeResult(logits=np.concatenate(outputs, axis=1), states=current_states)


def reset_states(model: TinyHZ0AModel, batch_size: int) -> list[np.ndarray | None]:
    _require_recurrent(model)
    return model.init_states(batch_size)


def serialize_states(states: list[np.ndarray | None]) -> list[str | None]:
    encoded: list[str | None] = []
    for state in states:
        if state is None:
            encoded.append(None)
            continue
        buffer = io.BytesIO()
        np.save(buffer, state, allow_pickle=False)
        encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
    return encoded


def deserialize_states(serialized: list[str | None]) -> list[np.ndarray | None]:
    states: list[np.ndarray | None] = []
    for item in serialized:
        if item is None:
            states.append(None)
            continue
        states.append(np.load(io.BytesIO(base64.b64decode(item)), allow_pickle=False))
    return states


def benchmark(model: TinyHZ0AModel, token_ids: np.ndarray) -> dict[str, float | int]:
    start = time.perf_counter()
    prefill_result = prefill(model, token_ids)
    prefill_seconds = time.perf_counter() - start
    start = time.perf_counter()
    decode_result = decode_tokenwise(model, token_ids)
    decode_seconds = time.perf_counter() - start
    return {
        "batch_size": token_ids.shape[0],
        "sequence_length": token_ids.shape[1],
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "prefill_tokens_per_second": float(token_ids.size / prefill_seconds),
        "decode_tokens_per_second": float(token_ids.size / decode_seconds),
        "max_logit_difference": float(np.max(np.abs(prefill_result.logits - decode_result.logits))),
    }
