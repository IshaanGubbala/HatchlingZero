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
