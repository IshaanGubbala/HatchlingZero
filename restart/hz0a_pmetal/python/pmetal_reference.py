from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reference.hz0a_gdn2_reference import HZ0ABlock, TinyHZ0AModel, cross_entropy_loss, gdn2_scan


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


@dataclass
class Gdn2BackwardResult:
    gradients: dict[str, np.ndarray]


def _sigmoid64(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value.astype(np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def gdn2_backward(
    grad_outputs: np.ndarray,
    grad_final_state: np.ndarray,
    backward_cache: Gdn2ForwardCache,
) -> Gdn2BackwardResult:
    """Explicit reverse scan matching the A3 recurrence derivation."""
    q = backward_cache.q
    k = backward_cache.k
    v = backward_cache.v
    decay_logits = backward_cache.decay_logits
    erase_logits = backward_cache.erase_logits
    write_logits = backward_cache.write_logits
    batch, steps, heads, d_k = q.shape

    states = [backward_cache.initial_state.astype(np.float64)]
    decays: list[np.ndarray] = []
    erases: list[np.ndarray] = []
    writes: list[np.ndarray] = []
    for t in range(steps):
        decay = _sigmoid64(decay_logits[:, t])
        erase = _sigmoid64(erase_logits[:, t])
        write = _sigmoid64(write_logits[:, t])
        decays.append(decay)
        erases.append(erase)
        writes.append(write)
        previous = states[-1]
        next_state = (
            decay[:, :, None, :] * (1.0 - erase[:, :, None, :]) * previous
            + write[:, :, :, None] * v[:, t].astype(np.float64)[:, :, :, None] * k[:, t].astype(np.float64)[:, :, None, :]
        )
        states.append(next_state)

    gradients = {
        "q": np.zeros_like(q, dtype=np.float64),
        "k": np.zeros_like(k, dtype=np.float64),
        "v": np.zeros_like(v, dtype=np.float64),
        "decay_logits": np.zeros_like(decay_logits, dtype=np.float64),
        "erase_logits": np.zeros_like(erase_logits, dtype=np.float64),
        "write_logits": np.zeros_like(write_logits, dtype=np.float64),
        "initial_state": np.zeros_like(backward_cache.initial_state, dtype=np.float64),
    }
    grad_state = grad_final_state.astype(np.float64).copy()
    for t in reversed(range(steps)):
        state_t = states[t + 1]
        state_prev = states[t]
        q_t = q[:, t].astype(np.float64)
        k_t = k[:, t].astype(np.float64)
        v_t = v[:, t].astype(np.float64)
        decay, erase, write = decays[t], erases[t], writes[t]
        grad_y = grad_outputs[:, t].astype(np.float64)
        gradients["q"][:, t] = np.einsum("bhv,bhvk->bhk", grad_y, state_t)
        grad_state_total = grad_state + grad_y[:, :, :, None] * q_t[:, :, None, :]
        a = decay[:, :, None, :] * (1.0 - erase[:, :, None, :])
        grad_prev = grad_state_total * a
        grad_a = np.sum(grad_state_total * state_prev, axis=2)
        grad_decay = grad_a * (1.0 - erase)
        grad_erase = grad_a * (-decay)
        outer = v_t[:, :, :, None] * k_t[:, :, None, :]
        grad_write = np.sum(grad_state_total * outer, axis=3)
        grad_outer = grad_state_total * write[:, :, :, None]
        gradients["v"][:, t] = np.sum(grad_outer * k_t[:, :, None, :], axis=3)
        gradients["k"][:, t] = np.sum(grad_outer * v_t[:, :, :, None], axis=2)
        gradients["decay_logits"][:, t] = grad_decay * decay * (1.0 - decay)
        gradients["erase_logits"][:, t] = grad_erase * erase * (1.0 - erase)
        gradients["write_logits"][:, t] = grad_write * write * (1.0 - write)
        grad_state = grad_prev
    gradients["initial_state"] = grad_state
    return Gdn2BackwardResult(gradients=gradients)


def gdn2_backward_chunked(
    inputs: Gdn2ForwardInputs,
    grad_outputs: np.ndarray,
    grad_final_state: np.ndarray,
    *,
    chunk_size: int,
) -> Gdn2BackwardResult:
    """Checkpoint states per chunk, then reverse chunks with state cotangents."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    caches: list[Gdn2ForwardCache] = []
    state = inputs.initial_state.copy()
    for start in range(0, inputs.q.shape[1], chunk_size):
        end = min(start + chunk_size, inputs.q.shape[1])
        result = gdn2_forward(Gdn2ForwardInputs(
            q=inputs.q[:, start:end],
            k=inputs.k[:, start:end],
            v=inputs.v[:, start:end],
            decay_logits=inputs.decay_logits[:, start:end],
            erase_logits=inputs.erase_logits[:, start:end],
            write_logits=inputs.write_logits[:, start:end],
            initial_state=state,
        ))
        caches.append(result.backward_cache)
        state = result.final_state

    chunk_gradients: list[dict[str, np.ndarray]] = []
    state_gradient = grad_final_state
    for index in reversed(range(len(caches))):
        start = index * chunk_size
        end = start + caches[index].q.shape[1]
        result = gdn2_backward(grad_outputs[:, start:end], state_gradient, caches[index])
        chunk_gradients.append(result.gradients)
        state_gradient = result.gradients["initial_state"]
    chunk_gradients.reverse()
    gradients = {
        name: np.concatenate([chunk[name] for chunk in chunk_gradients], axis=1)
        if name != "initial_state"
        else state_gradient
        for name in chunk_gradients[0]
    }
    return Gdn2BackwardResult(gradients=gradients)


@dataclass
class BlockForwardInputs:
    block: HZ0ABlock
    x: np.ndarray
    state: np.ndarray | None


@dataclass
class BlockForwardCache:
    x: np.ndarray
    state: np.ndarray | None
    is_attention: bool


@dataclass
class BlockForwardResult:
    outputs: np.ndarray
    final_state: np.ndarray | None
    backward_cache: BlockForwardCache


@dataclass
class TinyModelForwardInputs:
    model: TinyHZ0AModel
    token_ids: np.ndarray
    targets: np.ndarray | None = None


@dataclass
class TinyModelForwardResult:
    logits: np.ndarray
    states: list[np.ndarray | None]
    loss: float | None


@dataclass
class AdamWState:
    step: int
    first_moment: np.ndarray
    second_moment: np.ndarray


@dataclass
class AdamWUpdateResult:
    parameters: np.ndarray
    state: AdamWState
    update_norm: float


def adamw_step(
    parameters: np.ndarray,
    gradients: np.ndarray,
    state: AdamWState | None = None,
    *,
    learning_rate: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 0.0,
) -> AdamWUpdateResult:
    """Deterministic NumPy AdamW contract for PMetal optimizer parity."""
    if parameters.shape != gradients.shape:
        raise ValueError("parameters and gradients must have the same shape")
    if not np.isfinite(parameters).all() or not np.isfinite(gradients).all():
        raise FloatingPointError("AdamW refuses non-finite parameters or gradients")
    if state is None:
        state = AdamWState(
            step=0,
            first_moment=np.zeros_like(parameters, dtype=np.float64),
            second_moment=np.zeros_like(parameters, dtype=np.float64),
        )
    if state.first_moment.shape != parameters.shape or state.second_moment.shape != parameters.shape:
        raise ValueError("AdamW state shape does not match parameters")

    step = state.step + 1
    grad64 = gradients.astype(np.float64)
    m = beta1 * state.first_moment + (1.0 - beta1) * grad64
    v = beta2 * state.second_moment + (1.0 - beta2) * np.square(grad64)
    m_hat = m / (1.0 - beta1**step)
    v_hat = v / (1.0 - beta2**step)
    update = learning_rate * (m_hat / (np.sqrt(v_hat) + epsilon) + weight_decay * parameters.astype(np.float64))
    next_parameters = parameters.astype(np.float64) - update
    if not np.isfinite(next_parameters).all():
        raise FloatingPointError("AdamW produced non-finite parameters")
    return AdamWUpdateResult(
        parameters=next_parameters.astype(parameters.dtype),
        state=AdamWState(step=step, first_moment=m, second_moment=v),
        update_norm=float(np.linalg.norm(update)),
    )


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


def block_forward(inputs: BlockForwardInputs) -> BlockForwardResult:
    outputs, final_state = inputs.block(inputs.x, inputs.state)
    return BlockForwardResult(
        outputs=outputs,
        final_state=final_state,
        backward_cache=BlockForwardCache(
            x=inputs.x.copy(),
            state=None if inputs.state is None else inputs.state.copy(),
            is_attention=inputs.block.is_attention,
        ),
    )


def tiny_model_forward(inputs: TinyModelForwardInputs) -> TinyModelForwardResult:
    logits, states = inputs.model(inputs.token_ids)
    loss = None
    if inputs.targets is not None:
        loss = cross_entropy_loss(logits, inputs.targets)
    return TinyModelForwardResult(logits=logits, states=states, loss=loss)
