"""HZ-0B Phase B4: fair, non-HZ-0B baselines, evaluated on the same
synthetic tasks as B2's simulator.

Per the plan: "This prevents the project from crediting HZ-0B for
improvements that come merely from more capacity." Every baseline here is
deliberately simpler than reference/hz0b_memory_simulator.py in some
specific, named way, so a future comparison (B5+) can attribute any HZ-0B
advantage to its actual mechanism (content-addressable slots with
protection/reinforcement semantics) rather than to raw parameter count,
unbounded context, or an external retrieval system doing the real work.

Each baseline exposes the same minimal common interface: reset(),
read(query) -> readout, write(key, value) -> new_state. Not every baseline
can do everything HZ-0B can (e.g. NoMemoryBaseline literally cannot
recall anything by construction) -- that limitation is the point, not a
bug to fix.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx


# ---- 1. No memory: a pure control. Must fail every recall task by
# construction -- if it doesn't, the evaluation harness itself is broken. ----

@dataclass(frozen=True)
class NoMemoryState:
    pass


def no_memory_reset() -> NoMemoryState:
    return NoMemoryState()


def no_memory_write(state: NoMemoryState, key: mx.array, value: mx.array) -> NoMemoryState:
    return state  # writes are discarded, always


def no_memory_read(state: NoMemoryState, query: mx.array) -> mx.array:
    return mx.zeros_like(query)  # nothing was ever stored


# ---- 2. Larger recurrent state: raw capacity, no explicit addressing.
# A single big accumulator vector that sums (decayed) writes -- shows
# that MORE STATE without slot structure doesn't give exact recall. ----

@dataclass(frozen=True)
class LargeRecurrentState:
    accumulator: mx.array  # [batch, dim] -- dim intentionally >> a single key/value, "larger" than HZ-0B's per-slot width
    decay: float = 0.98


def large_recurrent_reset(batch_size: int, dim: int) -> LargeRecurrentState:
    return LargeRecurrentState(accumulator=mx.zeros((batch_size, dim)))


def large_recurrent_write(state: LargeRecurrentState, key: mx.array, value: mx.array) -> LargeRecurrentState:
    # No addressing at all: every write is superimposed on the same
    # accumulator, decayed each step -- this is the mechanism, not a
    # simplification of it. Blending is intentionally content-blind.
    return replace(state, accumulator=state.accumulator * state.decay + value)


def large_recurrent_read(state: LargeRecurrentState, query: mx.array) -> mx.array:
    return state.accumulator  # query is ignored -- there is no addressing to condition on


# ---- 3. Longer context: an unbounded list of every (key, value) ever
# written, read via full differentiable attention over all of them. This
# is close to an upper bound on what pure context length buys, with none
# of HZ-0B's capacity bound, protection, or forgetting. ----

@dataclass(frozen=True)
class LongContextState:
    keys: mx.array    # [batch, seen_so_far, key_dim]
    values: mx.array  # [batch, seen_so_far, value_dim]


def long_context_reset(batch_size: int, key_dim: int, value_dim: int) -> LongContextState:
    return LongContextState(keys=mx.zeros((batch_size, 0, key_dim)), values=mx.zeros((batch_size, 0, value_dim)))


def long_context_write(state: LongContextState, key: mx.array, value: mx.array) -> LongContextState:
    return LongContextState(keys=mx.concatenate([state.keys, key[:, None, :]], axis=1), values=mx.concatenate([state.values, value[:, None, :]], axis=1))


def long_context_read(state: LongContextState, query: mx.array) -> mx.array:
    if state.keys.shape[1] == 0:
        return mx.zeros_like(query if query.shape[-1] == state.values.shape[-1] else state.values[:, 0, :] if state.values.shape[1] else query)
    scores = mx.sum(state.keys * query[:, None, :], axis=-1) / mx.sqrt(mx.array(float(query.shape[-1])))
    weights = mx.softmax(scores, axis=-1)
    return mx.sum(state.values * weights[:, :, None], axis=1)


# ---- 4. Simple key-value cache: exact-match hash lookup only. No
# similarity fallback, no interference handling, no protection -- the
# "obvious" baseline HZ-0B needs to beat on noisy-key/similar-key tasks. ----

@dataclass(frozen=True)
class SimpleKVCacheState:
    table: dict  # python dict, keyed by a hashable tuple(key.tolist()) -- deliberately NOT a tensor op, this is the simplest possible thing


def simple_kv_cache_reset() -> SimpleKVCacheState:
    return SimpleKVCacheState(table={})


def _key_hash(key: mx.array, batch_index: int) -> tuple:
    return tuple(round(float(x), 4) for x in key[batch_index].tolist())


def simple_kv_cache_write(state: SimpleKVCacheState, key: mx.array, value: mx.array) -> SimpleKVCacheState:
    new_table = dict(state.table)
    new_table[_key_hash(key, 0)] = value[0]  # last write for an exact-matching key always wins, no protection concept exists
    return SimpleKVCacheState(table=new_table)


def simple_kv_cache_read(state: SimpleKVCacheState, query: mx.array) -> mx.array:
    hit = state.table.get(_key_hash(query, 0))
    return (hit if hit is not None else mx.zeros_like(query))[None, :]


# ---- 5. External vector retrieval: unbounded store, hard (non-
# differentiable) top-1 nearest-neighbor lookup -- unlike long-context's
# soft attention, this is explicitly a retrieval system standing in for
# the memory, not part of a differentiable model. ----

@dataclass(frozen=True)
class ExternalRetrievalState:
    keys: mx.array
    values: mx.array


def external_retrieval_reset(batch_size: int, key_dim: int, value_dim: int) -> ExternalRetrievalState:
    return ExternalRetrievalState(keys=mx.zeros((batch_size, 0, key_dim)), values=mx.zeros((batch_size, 0, value_dim)))


def external_retrieval_write(state: ExternalRetrievalState, key: mx.array, value: mx.array) -> ExternalRetrievalState:
    return ExternalRetrievalState(keys=mx.concatenate([state.keys, key[:, None, :]], axis=1), values=mx.concatenate([state.values, value[:, None, :]], axis=1))


def external_retrieval_read(state: ExternalRetrievalState, query: mx.array) -> mx.array:
    if state.keys.shape[1] == 0:
        return mx.zeros_like(query)
    scores = mx.sum(state.keys * query[:, None, :], axis=-1)
    best = mx.argmax(scores, axis=-1)
    return mx.take_along_axis(state.values, best[:, None, None], axis=1)[:, 0, :]


# ---- 6. Equal-parameter feed-forward adapter: same parameter budget as
# HZ-0B's learned projections (query/key/value/gate, roughly 4 * dim^2),
# but NO memory state at all -- a pure, data-independent (per-call)
# nonlinear transform of the current input. Tests whether an HZ-0B
# advantage (measured later, once there's something to compare) comes
# from the memory mechanism or just from having extra learned parameters
# to push the hidden state through. ----

@dataclass(frozen=True)
class FeedForwardAdapterParams:
    w1: mx.array
    b1: mx.array
    w2: mx.array
    b2: mx.array


def feed_forward_adapter_init(dim: int, hidden_dim: int, seed: int = 0) -> FeedForwardAdapterParams:
    key = mx.random.key(seed)
    key1, key2 = mx.random.split(key)
    scale = (2.0 / dim) ** 0.5
    return FeedForwardAdapterParams(
        w1=mx.random.normal((dim, hidden_dim), key=key1) * scale,
        b1=mx.zeros((hidden_dim,)),
        w2=mx.random.normal((hidden_dim, dim), key=key2) * scale,
        b2=mx.zeros((dim,)),
    )


def feed_forward_adapter_read(params: FeedForwardAdapterParams, query: mx.array) -> mx.array:
    # No state, no write() to speak of -- "read" here just means "the
    # adapter's output given the current hidden state," since there is
    # nothing it could have stored from an earlier write.
    hidden = mx.maximum(query @ params.w1 + params.b1, mx.array(0.0))
    return hidden @ params.w2 + params.b2


def feed_forward_adapter_param_count(dim: int, hidden_dim: int) -> int:
    return dim * hidden_dim + hidden_dim + hidden_dim * dim + dim
