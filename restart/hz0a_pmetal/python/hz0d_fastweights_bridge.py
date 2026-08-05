"""HZ-0D Phase D9 Python <-> Rust bridge.

Loads `libhz0d_pmetal_fastweights_bridge` (built from
`restart/hz0a_pmetal/crates/hz0d-pmetal-fastweights-bridge`) via
`ctypes` and wraps it in a numpy-backed functional API, matching
`python/hz0b_memory_bridge.py`'s own established convention in this
workspace (same `ctypes`-cdylib approach, no PyO3/maturin).

Unlike the Python reference (`reference/hz0d_fast_weights.py`), every
state here carries an explicit `sessions` (batch) dimension -- D9's own
"batched deterministic sessions" requirement, verified independent and
deterministic in `restart/hz0a_pmetal/crates/hz0d-pmetal-fastweights/tests/correctness.rs`.
`reset` takes a caller-supplied `a_fast_init` array (generate it with
real MLX randomness, e.g. `reference/hz0d_fast_weights.py::init_fast_weights`,
then pass its `a_fast` here) rather than generating randomness itself --
see the Rust crate's own module docstring for why.
"""
from __future__ import annotations

import ctypes
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_LIB_NAMES = {
    "Darwin": "libhz0d_pmetal_fastweights_bridge.dylib",
    "Linux": "libhz0d_pmetal_fastweights_bridge.so",
    "Windows": "hz0d_pmetal_fastweights_bridge.dll",
}


def _find_library() -> Path:
    lib_name = _LIB_NAMES.get(platform.system(), _LIB_NAMES["Linux"])
    workspace_root = Path(__file__).resolve().parents[1]
    for profile in ("release", "debug"):
        candidate = workspace_root / "target" / profile / lib_name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{lib_name} not found under {workspace_root}/target/{{release,debug}}. "
        "Build it first: cargo build --release -p hz0d-pmetal-fastweights-bridge"
    )


class _CFastWeightStateIn(ctypes.Structure):
    _fields_ = [
        ("sessions", ctypes.c_size_t),
        ("num_layers", ctypes.c_size_t),
        ("dim", ctypes.c_size_t),
        ("rank", ctypes.c_size_t),
        ("a_fast", ctypes.POINTER(ctypes.c_float)),
        ("b_fast", ctypes.POINTER(ctypes.c_float)),
        ("update_count", ctypes.POINTER(ctypes.c_int32)),
    ]


class _CFastWeightStateOut(ctypes.Structure):
    _fields_ = [
        ("a_fast", ctypes.POINTER(ctypes.c_float)),
        ("b_fast", ctypes.POINTER(ctypes.c_float)),
        ("update_count", ctypes.POINTER(ctypes.c_int32)),
    ]


def _load() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_find_library()))
    f32p = ctypes.POINTER(ctypes.c_float)
    i32p = ctypes.POINTER(ctypes.c_int32)

    lib.hz0d_reset.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, f32p, ctypes.POINTER(_CFastWeightStateOut)]
    lib.hz0d_reset.restype = None

    lib.hz0d_apply.argtypes = [ctypes.POINTER(_CFastWeightStateIn), f32p, f32p, f32p, ctypes.c_size_t, f32p]
    lib.hz0d_apply.restype = None

    lib.hz0d_update.argtypes = [ctypes.POINTER(_CFastWeightStateIn), ctypes.c_size_t, ctypes.c_size_t, f32p, f32p, ctypes.c_float, ctypes.c_float, ctypes.POINTER(_CFastWeightStateOut)]
    lib.hz0d_update.restype = None

    lib.hz0d_decay.argtypes = [ctypes.POINTER(_CFastWeightStateIn), ctypes.c_float, ctypes.POINTER(_CFastWeightStateOut)]
    lib.hz0d_decay.restype = None

    lib.hz0d_effective_delta.argtypes = [ctypes.POINTER(_CFastWeightStateIn), ctypes.c_size_t, ctypes.c_size_t, f32p]
    lib.hz0d_effective_delta.restype = None
    return lib


_LIB = None


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _load()
    return _LIB


def _f32p(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _i32p(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _c32(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float32)


@dataclass(frozen=True)
class FastWeightState:
    sessions: int
    num_layers: int
    dim: int
    rank: int
    a_fast: np.ndarray      # [sessions, num_layers, dim, rank] float32
    b_fast: np.ndarray      # [sessions, num_layers, rank, dim] float32
    update_count: np.ndarray  # [sessions] int32

    def _as_c_in(self) -> _CFastWeightStateIn:
        return _CFastWeightStateIn(
            sessions=self.sessions, num_layers=self.num_layers, dim=self.dim, rank=self.rank,
            a_fast=_f32p(self.a_fast), b_fast=_f32p(self.b_fast), update_count=_i32p(self.update_count),
        )

    def _new_out_buffers(self):
        return {
            "a_fast": np.zeros(self.sessions * self.num_layers * self.dim * self.rank, dtype=np.float32),
            "b_fast": np.zeros(self.sessions * self.num_layers * self.rank * self.dim, dtype=np.float32),
            "update_count": np.zeros(self.sessions, dtype=np.int32),
        }

    def _out_to_state(self, buffers: dict) -> "FastWeightState":
        return FastWeightState(
            sessions=self.sessions, num_layers=self.num_layers, dim=self.dim, rank=self.rank,
            a_fast=buffers["a_fast"].reshape(self.sessions, self.num_layers, self.dim, self.rank),
            b_fast=buffers["b_fast"].reshape(self.sessions, self.num_layers, self.rank, self.dim),
            update_count=buffers["update_count"],
        )


def reset(sessions: int, num_layers: int, dim: int, rank: int, a_fast_init: np.ndarray) -> FastWeightState:
    """`a_fast_init`: `[sessions, num_layers, dim, rank]` (or already
    flat), real MLX-generated randomness handed in by the caller (see
    module docstring). `b_fast` is exactly zero."""
    a_flat = _c32(np.asarray(a_fast_init)).reshape(-1)
    out_buffers = {
        "a_fast": np.zeros(sessions * num_layers * dim * rank, dtype=np.float32),
        "b_fast": np.zeros(sessions * num_layers * rank * dim, dtype=np.float32),
        "update_count": np.zeros(sessions, dtype=np.int32),
    }
    out = _CFastWeightStateOut(a_fast=_f32p(out_buffers["a_fast"]), b_fast=_f32p(out_buffers["b_fast"]), update_count=_i32p(out_buffers["update_count"]))
    _lib().hz0d_reset(sessions, num_layers, dim, rank, _f32p(a_flat), ctypes.byref(out))
    return FastWeightState(
        sessions=sessions, num_layers=num_layers, dim=dim, rank=rank,
        a_fast=out_buffers["a_fast"].reshape(sessions, num_layers, dim, rank),
        b_fast=out_buffers["b_fast"].reshape(sessions, num_layers, rank, dim),
        update_count=out_buffers["update_count"],
    )


def apply(state: FastWeightState, x: np.ndarray, base_weight: np.ndarray, base_bias: np.ndarray, layer: int) -> np.ndarray:
    """`x`: `[sessions, dim]`. `base_weight`/`base_bias`: the SAME frozen
    backbone weights shared by every session. Returns `[sessions, dim]`."""
    x = _c32(x).reshape(-1)
    base_weight = _c32(base_weight).reshape(-1)
    base_bias = _c32(base_bias).reshape(-1)
    y = np.zeros(state.sessions * state.dim, dtype=np.float32)
    _lib().hz0d_apply(ctypes.byref(state._as_c_in()), _f32p(x), _f32p(base_weight), _f32p(base_bias), layer, _f32p(y))
    return y.reshape(state.sessions, state.dim)


def update(state: FastWeightState, session: int, layer: int, grad_a: np.ndarray, grad_b: np.ndarray, *, lr: float, max_delta_norm: float) -> FastWeightState:
    out_buffers = state._new_out_buffers()
    out = _CFastWeightStateOut(a_fast=_f32p(out_buffers["a_fast"]), b_fast=_f32p(out_buffers["b_fast"]), update_count=_i32p(out_buffers["update_count"]))
    _lib().hz0d_update(ctypes.byref(state._as_c_in()), session, layer, _f32p(_c32(grad_a).reshape(-1)), _f32p(_c32(grad_b).reshape(-1)), lr, max_delta_norm, ctypes.byref(out))
    return state._out_to_state(out_buffers)


def decay(state: FastWeightState, decay_rate: float) -> FastWeightState:
    out_buffers = state._new_out_buffers()
    out = _CFastWeightStateOut(a_fast=_f32p(out_buffers["a_fast"]), b_fast=_f32p(out_buffers["b_fast"]), update_count=_i32p(out_buffers["update_count"]))
    _lib().hz0d_decay(ctypes.byref(state._as_c_in()), decay_rate, ctypes.byref(out))
    return state._out_to_state(out_buffers)


def effective_delta(state: FastWeightState, session: int, layer: int) -> np.ndarray:
    out = np.zeros(state.dim * state.dim, dtype=np.float32)
    _lib().hz0d_effective_delta(ctypes.byref(state._as_c_in()), session, layer, _f32p(out))
    return out.reshape(state.dim, state.dim)


def snapshot(state: FastWeightState) -> FastWeightState:
    """A snapshot is just the state itself (numpy arrays, already
    plain/copyable) -- matching `hz0b_memory_bridge.py`'s own reasoning
    for not exposing a separate serialize/restore FFI call."""
    return FastWeightState(
        sessions=state.sessions, num_layers=state.num_layers, dim=state.dim, rank=state.rank,
        a_fast=state.a_fast.copy(), b_fast=state.b_fast.copy(), update_count=state.update_count.copy(),
    )


def rollback(checkpoint: FastWeightState) -> FastWeightState:
    return snapshot(checkpoint)
