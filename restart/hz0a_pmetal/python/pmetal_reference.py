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
