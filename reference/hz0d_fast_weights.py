"""HZ-0D D1: the fast-weight state and lifecycle contract.

Per `docs/restart/hz0d_d1_contract.md` (read that doc for the reasoning
behind every choice below; this module is its real, tested
implementation, not a parallel design).

`W_effective = W_base + A_fast @ B_fast` -- a low-rank session-local
delta applied to the anchor-attention output projection
(`reference/hz0a_mlx_model.py::CausalAttention.out`) at HZ-0C's existing
6 anchor layers. `W_base` is a plain input array here, never a tracked
parameter of anything in this module -- there is no code path by which
`mx.grad` over `(A_fast, B_fast)` could reach it.

Every lifecycle operation here is independently tested
(`tests/reference/test_hz0d_fast_weights.py`), including a real
finite-difference check on the gradient -- directly required by
`docs/restart/hz0d_history_audit.md`'s finding that a prior "gradient-
based" fast-weight mechanism was never actually gradient descent (it
computed a perturbed loss and then never used it). That mistake is not
repeated here: `update_fast_weights` takes real, externally-computed
gradients (e.g. from `mx.value_and_grad`), never approximates them
itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class FastWeightConfig:
    """Static, session-independent shape/policy parameters -- never
    itself session state (see `FastWeightState` for the part that
    changes per session)."""

    dim: int = 768            # matches the frozen HZ-0A/HZ-0C backbone's hidden size
    rank: int = 16            # A_fast/B_fast inner dimension; see contract doc section 2
    num_layers: int = 6       # matches len(ATTENTION_INDICES) = (4, 9, 14, 19, 24, 29)
    decay_rate: float = 1.0   # multiplicative per-factor decay; 1.0 = no-op (contract section 4)
    max_delta_norm: float = 1.0    # Frobenius-norm bound on the REALIZED delta A@B per layer (section 5)
    max_updates_per_session: int = 50  # session-level policy cap; enforced by callers, not this module
    init_seed: int = 0        # deterministic seed for the asymmetric init below (section 2 addendum)
    init_scale: float = 0.02  # std of the small-random half of the asymmetric init


@dataclass(frozen=True)
class FastWeightState:
    """Session-local fast-weight state. Immutable (frozen dataclass,
    matching `reference/hz0b_memory_simulator.py::MemoryState`'s own
    convention) -- every lifecycle function returns a NEW state rather
    than mutating one in place, so a caller can never accidentally hold
    a stale alias to "the" current state."""

    a_fast: mx.array         # [num_layers, dim, rank]
    b_fast: mx.array         # [num_layers, rank, dim]
    update_count: mx.array   # scalar int32, total update_fast_weights calls this session


def init_fast_weights(config: FastWeightConfig) -> FastWeightState:
    """Asymmetric init: `b_fast` is exactly zero, `a_fast` is small
    random noise -- so `A_fast @ B_fast` is still EXACTLY zero for every
    layer at session start (the D6 exit gate's "inactive fast weights
    reproduce HZ-0C behavior" requirement, satisfied by construction),
    but the GRADIENT is not.

    An earlier version of this function zero-initialized BOTH factors,
    which turned out to be a real bug caught by
    `tests/reference/test_hz0d_fast_weights.py::test_real_gradient_step_strictly_reduces_loss`:
    for `delta = A @ B`, `dL/dA = upstream @ B.T` and `dL/dB = A.T @
    upstream` -- if BOTH `A` and `B` start at zero, BOTH gradients are
    exactly zero (each one's formula multiplies by the OTHER factor),
    a dead saddle point no gradient step can ever leave. Standard LoRA
    initialization avoids this by zeroing only ONE factor; done the same
    way here, not discovered independently -- but verified independently,
    which is the actual point per `docs/restart/hz0d_history_audit.md`."""
    key = mx.random.key(config.init_seed)
    return FastWeightState(
        a_fast=mx.random.normal((config.num_layers, config.dim, config.rank), key=key) * config.init_scale,
        b_fast=mx.zeros((config.num_layers, config.rank, config.dim)),
        update_count=mx.array(0, dtype=mx.int32),
    )


def effective_delta(state: FastWeightState, layer_index: int) -> mx.array:
    """The realized `[dim, dim]` weight delta for one layer."""
    return state.a_fast[layer_index] @ state.b_fast[layer_index]


def apply_fast_linear(x: mx.array, base_weight: mx.array, base_bias: mx.array, state: FastWeightState, layer_index: int) -> mx.array:
    """`x @ (base_weight + A_fast @ B_fast).T + base_bias` -- matches
    `nn.Linear`'s own `[out_features, in_features]` weight convention
    (same convention as every other manual-matmul caller in this
    project, e.g. `reference/hz0c_surprise_trigger.py::masked_anchor_attention`).
    `base_weight`/`base_bias` are plain arrays, not parameters of this
    module or function -- `mx.grad` over `(state.a_fast, state.b_fast)`
    has no path to them."""
    delta = effective_delta(state, layer_index)
    return x @ (base_weight + delta).T + base_bias


def update_fast_weights(state: FastWeightState, layer_index: int, grad_a: mx.array, grad_b: mx.array, *, lr: float, config: FastWeightConfig) -> FastWeightState:
    """One real gradient-descent step on layer `layer_index`'s factors,
    using REAL, externally-computed gradients (e.g. from
    `mx.value_and_grad` over a real loss) -- never approximated inside
    this function. This is the exact discipline
    `docs/restart/hz0d_history_audit.md` found missing in the prior
    implementation.

    Clips the REALIZED delta's Frobenius norm (not the factors
    individually) to `config.max_delta_norm` after the step -- see
    contract doc section 5 for why the delta, not the factors, is what
    gets bounded."""
    updated_a_layer = state.a_fast[layer_index] - lr * grad_a
    updated_b_layer = state.b_fast[layer_index] - lr * grad_b
    delta = updated_a_layer @ updated_b_layer
    delta_norm = mx.sqrt(mx.sum(delta * delta))
    scale = mx.minimum(mx.array(1.0), config.max_delta_norm / (delta_norm + 1e-8))
    # Scale is applied to the FACTORS (splitting the scale evenly across
    # both, via its square root) so the clipped delta's norm is exactly
    # `scale * delta_norm`, matching a direct clip on the delta itself,
    # while keeping the state representation strictly low-rank (never
    # materializing and reclipping a dense [dim, dim] delta into new
    # factors, which would silently destroy the rank bound).
    factor_scale = mx.sqrt(scale)
    clipped_a_layer = updated_a_layer * factor_scale
    clipped_b_layer = updated_b_layer * factor_scale
    new_a = mx.where(
        mx.arange(state.a_fast.shape[0])[:, None, None] == layer_index,
        mx.broadcast_to(clipped_a_layer[None, :, :], state.a_fast.shape),
        state.a_fast,
    )
    new_b = mx.where(
        mx.arange(state.b_fast.shape[0])[:, None, None] == layer_index,
        mx.broadcast_to(clipped_b_layer[None, :, :], state.b_fast.shape),
        state.b_fast,
    )
    return FastWeightState(a_fast=new_a, b_fast=new_b, update_count=state.update_count + 1)


def decay_fast_weights(state: FastWeightState, decay_rate: float) -> FastWeightState:
    """Multiplicative decay of BOTH factors -- the effective delta
    `A @ B` therefore decays as roughly `decay_rate**2` per call, not
    `decay_rate`; documented explicitly in the contract doc since this
    is easy to miscount. `decay_rate=1.0` is an exact no-op (tested via
    `mx.array_equal`, not approximate closeness)."""
    return FastWeightState(
        a_fast=state.a_fast * decay_rate,
        b_fast=state.b_fast * decay_rate,
        update_count=state.update_count,
    )


def snapshot(state: FastWeightState) -> dict[str, mx.array]:
    """A named checkpoint is the CALLER's responsibility (e.g. a dict of
    `{name: snapshot(state)}`) -- this function only produces one
    checkpoint's contents, matching the archived implementation's
    `checkpoint(name)` design pattern (carried forward as a pattern per
    `docs/restart/hz0d_history_audit.md`, not its code)."""
    return {"a_fast": mx.array(state.a_fast), "b_fast": mx.array(state.b_fast), "update_count": mx.array(state.update_count)}


def rollback(checkpoint: dict[str, mx.array]) -> FastWeightState:
    """Exact restore from a `snapshot()` result."""
    return FastWeightState(
        a_fast=mx.array(checkpoint["a_fast"]),
        b_fast=mx.array(checkpoint["b_fast"]),
        update_count=mx.array(checkpoint["update_count"]),
    )


def reset_fast_weights(config: FastWeightConfig) -> FastWeightState:
    """Returns to the EXACT same zero state `init_fast_weights` would
    produce for this config -- a session boundary is indistinguishable
    from a fresh session, by construction (both call the same function)."""
    return init_fast_weights(config)


def serialize(state: FastWeightState) -> dict:
    """JSON-safe checkpoint dict, matching this project's own
    established `payload["arrays"]`-style convention (HZ-0A/B/C
    checkpoints) rather than a new, bespoke format."""
    return {
        "a_fast": state.a_fast.tolist(),
        "b_fast": state.b_fast.tolist(),
        "update_count": int(state.update_count),
    }


def deserialize(data: dict, config: FastWeightConfig) -> FastWeightState:
    """Inverse of `serialize`. `config` is required (not embedded in the
    payload) since layer/dim/rank shape is a static contract property,
    not session state -- matching this module's own
    `FastWeightConfig`/`FastWeightState` split."""
    del config  # shape is implied by the payload arrays themselves; kept for API symmetry with callers that validate against a known config
    return FastWeightState(
        a_fast=mx.array(data["a_fast"], dtype=mx.float32),
        b_fast=mx.array(data["b_fast"], dtype=mx.float32),
        update_count=mx.array(data["update_count"], dtype=mx.int32),
    )


def fast_state_memory_bytes(config: FastWeightConfig) -> int:
    """Audited maximum fast-state memory for one session, in bytes --
    computed directly from the config's real shapes (not a separate,
    hand-maintained estimate that could drift out of sync). See contract
    doc section 10 for the worked example at the default config."""
    element_count = 2 * config.num_layers * config.dim * config.rank
    return element_count * 4  # float32
