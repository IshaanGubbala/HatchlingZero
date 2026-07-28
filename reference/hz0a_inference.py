from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass

import numpy as np

from reference.hz0a_gdn2_reference import TinyHZ0AModel
from reference.hz0a_gdn2_reference import CausalSelfAttention, softmax


@dataclass
class DecodeResult:
    logits: np.ndarray
    states: list[np.ndarray | None]


@dataclass
class AttentionKVCache:
    key: np.ndarray
    value: np.ndarray


def _normalize_projection(projection: np.ndarray) -> np.ndarray:
    scale = np.maximum(np.max(np.abs(projection), axis=-1, keepdims=True), 1.0)
    normalized = projection / scale
    normalized /= np.sqrt(np.mean(np.square(normalized), axis=-1, keepdims=True) + 1e-6)
    return np.clip(np.nan_to_num(normalized, nan=0.0, posinf=8.0, neginf=-8.0), -8.0, 8.0)


def attention_decode_step(
    attention: CausalSelfAttention,
    x: np.ndarray,
    cache: AttentionKVCache | None = None,
) -> tuple[np.ndarray, AttentionKVCache]:
    """Decode one causal-attention token using an append-only KV cache."""
    bsz, steps, dim = x.shape
    if steps != 1:
        raise ValueError("attention_decode_step expects exactly one token")
    head_dim = dim // attention.num_heads
    q = attention.q_proj(x).astype(np.float64).reshape(bsz, 1, attention.num_heads, head_dim).transpose(0, 2, 1, 3)
    k = attention.k_proj(x).astype(np.float64).reshape(bsz, 1, attention.num_heads, head_dim).transpose(0, 2, 1, 3)
    v = attention.v_proj(x).astype(np.float64).reshape(bsz, 1, attention.num_heads, head_dim).transpose(0, 2, 1, 3)
    q, k, v = _normalize_projection(q), _normalize_projection(k), _normalize_projection(v)
    if cache is not None:
        k = np.concatenate((cache.key, k), axis=2)
        v = np.concatenate((cache.value, v), axis=2)
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) / np.sqrt(head_dim)
    weights = softmax(scores, axis=-1)
    out = np.matmul(weights, v).transpose(0, 2, 1, 3).reshape(bsz, 1, dim)
    return attention.out_proj(out.astype(np.float32)), AttentionKVCache(key=k, value=v)


def serialize_attention_cache(cache: AttentionKVCache) -> str:
    buffer = io.BytesIO()
    np.savez(buffer, key=cache.key, value=cache.value)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def deserialize_attention_cache(serialized: str) -> AttentionKVCache:
    payload = np.load(io.BytesIO(base64.b64decode(serialized)), allow_pickle=False)
    return AttentionKVCache(key=payload["key"], value=payload["value"])


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
