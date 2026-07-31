"""HZ-0B Phase B10 Python <-> Rust bridge.

Loads `libhz0b_pmetal_memory_bridge` (built from
`restart/hz0a_pmetal/crates/hz0b-pmetal-memory-bridge`) via `ctypes` and
wraps it in a numpy-backed functional API that mirrors
`reference/hz0b_memory_simulator.py`'s own signatures exactly (same
function names, same argument order, same immutable-state style) so a
caller can swap one import for the other with no other code changes.

No PyO3/maturin dependency -- this workspace has none, so a plain
`extern "C"` cdylib loaded via ctypes was chosen over introducing a new
build toolchain. See `docs/restart/hz0b_b10_pmetal_design.md`.

Every buffer this module hands to Rust is numpy-owned and pre-sized from
`batch`/`num_slots`/`key_dim`/`value_dim`, which the caller already knows
-- no allocation crosses the FFI boundary in either direction.
"""
from __future__ import annotations

import ctypes
import platform
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

SOURCE_SUPERVISED = 0
SOURCE_LATENT = 1

_LIB_NAMES = {
    "Darwin": "libhz0b_pmetal_memory_bridge.dylib",
    "Linux": "libhz0b_pmetal_memory_bridge.so",
    "Windows": "hz0b_pmetal_memory_bridge.dll",
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
        "Build it first: cargo build --release -p hz0b-pmetal-memory-bridge"
    )


class _CMemoryStateIn(ctypes.Structure):
    _fields_ = [
        ("batch", ctypes.c_size_t),
        ("num_slots", ctypes.c_size_t),
        ("key_dim", ctypes.c_size_t),
        ("value_dim", ctypes.c_size_t),
        ("keys", ctypes.POINTER(ctypes.c_float)),
        ("values", ctypes.POINTER(ctypes.c_float)),
        ("confidence", ctypes.POINTER(ctypes.c_float)),
        ("age", ctypes.POINTER(ctypes.c_int32)),
        ("protection", ctypes.POINTER(ctypes.c_float)),
        ("write_count", ctypes.POINTER(ctypes.c_int32)),
        ("last_write_step", ctypes.POINTER(ctypes.c_int32)),
        ("write_source", ctypes.POINTER(ctypes.c_int32)),
    ]


class _CMemoryStateOut(ctypes.Structure):
    _fields_ = [
        ("keys", ctypes.POINTER(ctypes.c_float)),
        ("values", ctypes.POINTER(ctypes.c_float)),
        ("confidence", ctypes.POINTER(ctypes.c_float)),
        ("age", ctypes.POINTER(ctypes.c_int32)),
        ("protection", ctypes.POINTER(ctypes.c_float)),
        ("write_count", ctypes.POINTER(ctypes.c_int32)),
        ("last_write_step", ctypes.POINTER(ctypes.c_int32)),
        ("write_source", ctypes.POINTER(ctypes.c_int32)),
    ]


def _load() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_find_library()))
    f32p = ctypes.POINTER(ctypes.c_float)
    i32p = ctypes.POINTER(ctypes.c_int32)

    lib.hz0b_reset.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.POINTER(_CMemoryStateOut)]
    lib.hz0b_reset.restype = None

    lib.hz0b_read.argtypes = [ctypes.POINTER(_CMemoryStateIn), f32p, i32p, ctypes.c_uint8, ctypes.c_uint8, f32p, f32p]
    lib.hz0b_read.restype = None

    lib.hz0b_write.argtypes = [ctypes.POINTER(_CMemoryStateIn), f32p, f32p, f32p, ctypes.c_int32, ctypes.c_int32, i32p, ctypes.POINTER(_CMemoryStateOut), i32p, ctypes.POINTER(ctypes.c_uint8)]
    lib.hz0b_write.restype = None

    lib.hz0b_reinforce.argtypes = [ctypes.POINTER(_CMemoryStateIn), i32p, ctypes.POINTER(_CMemoryStateOut)]
    lib.hz0b_reinforce.restype = None

    lib.hz0b_update.argtypes = [ctypes.POINTER(_CMemoryStateIn), i32p, f32p, ctypes.POINTER(_CMemoryStateOut)]
    lib.hz0b_update.restype = None

    lib.hz0b_protect.argtypes = [ctypes.POINTER(_CMemoryStateIn), i32p, f32p, ctypes.POINTER(_CMemoryStateOut)]
    lib.hz0b_protect.restype = None

    lib.hz0b_forget_or_decay.argtypes = [ctypes.POINTER(_CMemoryStateIn), ctypes.c_float, ctypes.POINTER(_CMemoryStateOut)]
    lib.hz0b_forget_or_decay.restype = None

    lib.hz0b_delete.argtypes = [ctypes.POINTER(_CMemoryStateIn), i32p, ctypes.POINTER(_CMemoryStateOut)]
    lib.hz0b_delete.restype = None
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


@dataclass(frozen=True)
class MemoryState:
    batch: int
    num_slots: int
    key_dim: int
    value_dim: int
    keys: np.ndarray            # [batch, num_slots, key_dim] float32
    values: np.ndarray          # [batch, num_slots, value_dim] float32
    confidence: np.ndarray      # [batch, num_slots] float32
    age: np.ndarray             # [batch, num_slots] int32
    protection: np.ndarray      # [batch, num_slots] float32
    write_count: np.ndarray     # [batch, num_slots] int32
    last_write_step: np.ndarray  # [batch, num_slots] int32
    write_source: np.ndarray    # [batch, num_slots] int32

    def _as_c_in(self) -> _CMemoryStateIn:
        return _CMemoryStateIn(
            batch=self.batch, num_slots=self.num_slots, key_dim=self.key_dim, value_dim=self.value_dim,
            keys=_f32p(self.keys), values=_f32p(self.values), confidence=_f32p(self.confidence),
            age=_i32p(self.age), protection=_f32p(self.protection), write_count=_i32p(self.write_count),
            last_write_step=_i32p(self.last_write_step), write_source=_i32p(self.write_source),
        )

    def _new_out_buffers(self):
        n = self.batch * self.num_slots
        return {
            "keys": np.zeros(n * self.key_dim, dtype=np.float32),
            "values": np.zeros(n * self.value_dim, dtype=np.float32),
            "confidence": np.zeros(n, dtype=np.float32),
            "age": np.zeros(n, dtype=np.int32),
            "protection": np.zeros(n, dtype=np.float32),
            "write_count": np.zeros(n, dtype=np.int32),
            "last_write_step": np.zeros(n, dtype=np.int32),
            "write_source": np.zeros(n, dtype=np.int32),
        }

    def _out_to_state(self, buffers: dict) -> "MemoryState":
        return MemoryState(
            batch=self.batch, num_slots=self.num_slots, key_dim=self.key_dim, value_dim=self.value_dim,
            keys=buffers["keys"].reshape(self.batch, self.num_slots, self.key_dim),
            values=buffers["values"].reshape(self.batch, self.num_slots, self.value_dim),
            confidence=buffers["confidence"].reshape(self.batch, self.num_slots),
            age=buffers["age"].reshape(self.batch, self.num_slots),
            protection=buffers["protection"].reshape(self.batch, self.num_slots),
            write_count=buffers["write_count"].reshape(self.batch, self.num_slots),
            last_write_step=buffers["last_write_step"].reshape(self.batch, self.num_slots),
            write_source=buffers["write_source"].reshape(self.batch, self.num_slots),
        )


def reset(batch_size: int, num_slots: int, key_dim: int, value_dim: int) -> MemoryState:
    n = batch_size * num_slots
    buffers = {
        "keys": np.zeros(n * key_dim, dtype=np.float32), "values": np.zeros(n * value_dim, dtype=np.float32),
        "confidence": np.zeros(n, dtype=np.float32), "age": np.zeros(n, dtype=np.int32),
        "protection": np.zeros(n, dtype=np.float32), "write_count": np.zeros(n, dtype=np.int32),
        "last_write_step": np.zeros(n, dtype=np.int32), "write_source": np.zeros(n, dtype=np.int32),
    }
    out = _CMemoryStateOut(
        keys=_f32p(buffers["keys"]), values=_f32p(buffers["values"]), confidence=_f32p(buffers["confidence"]),
        age=_i32p(buffers["age"]), protection=_f32p(buffers["protection"]), write_count=_i32p(buffers["write_count"]),
        last_write_step=_i32p(buffers["last_write_step"]), write_source=_i32p(buffers["write_source"]),
    )
    _lib().hz0b_reset(batch_size, num_slots, key_dim, value_dim, ctypes.byref(out))
    return MemoryState(
        batch=batch_size, num_slots=num_slots, key_dim=key_dim, value_dim=value_dim,
        keys=buffers["keys"].reshape(batch_size, num_slots, key_dim),
        values=buffers["values"].reshape(batch_size, num_slots, value_dim),
        confidence=buffers["confidence"].reshape(batch_size, num_slots),
        age=buffers["age"].reshape(batch_size, num_slots),
        protection=buffers["protection"].reshape(batch_size, num_slots),
        write_count=buffers["write_count"].reshape(batch_size, num_slots),
        last_write_step=buffers["last_write_step"].reshape(batch_size, num_slots),
        write_source=buffers["write_source"].reshape(batch_size, num_slots),
    )


def _c32(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float32)


def _ci32(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.int32)


def read(state: MemoryState, query: np.ndarray, *, slot_idx: np.ndarray | None = None, hard: bool = False, confidence_weighted: bool = True):
    query = _c32(query)
    readout = np.zeros(state.batch * state.value_dim, dtype=np.float32)
    weights = np.zeros(state.batch * state.num_slots, dtype=np.float32)
    idx_ptr = _i32p(_ci32(slot_idx)) if slot_idx is not None else None
    _lib().hz0b_read(ctypes.byref(state._as_c_in()), _f32p(query), idx_ptr, 1 if hard else 0, 1 if confidence_weighted else 0, _f32p(readout), _f32p(weights))
    return readout.reshape(state.batch, state.value_dim), weights.reshape(state.batch, state.num_slots)


def write(state: MemoryState, key: np.ndarray, value: np.ndarray, strength: np.ndarray, *, step: int, source: int = SOURCE_LATENT, slot_idx: np.ndarray | None = None):
    key, value, strength = _c32(key), _c32(value), _c32(strength)
    out_buffers = state._new_out_buffers()
    out = _CMemoryStateOut(**{k: (_f32p(v) if v.dtype == np.float32 else _i32p(v)) for k, v in out_buffers.items()})
    out_slot = np.zeros(state.batch, dtype=np.int32)
    out_rejected = np.zeros(state.batch, dtype=np.uint8)
    idx_ptr = _i32p(_ci32(slot_idx)) if slot_idx is not None else None
    _lib().hz0b_write(ctypes.byref(state._as_c_in()), _f32p(key), _f32p(value), _f32p(strength), step, source, idx_ptr, ctypes.byref(out), _i32p(out_slot), out_rejected.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)))
    return state._out_to_state(out_buffers), out_slot, out_rejected.astype(bool)


def reinforce(state: MemoryState, slot_idx: np.ndarray) -> MemoryState:
    out_buffers = state._new_out_buffers()
    out = _CMemoryStateOut(**{k: (_f32p(v) if v.dtype == np.float32 else _i32p(v)) for k, v in out_buffers.items()})
    _lib().hz0b_reinforce(ctypes.byref(state._as_c_in()), _i32p(_ci32(slot_idx)), ctypes.byref(out))
    return state._out_to_state(out_buffers)


def update(state: MemoryState, slot_idx: np.ndarray, new_value: np.ndarray) -> MemoryState:
    out_buffers = state._new_out_buffers()
    out = _CMemoryStateOut(**{k: (_f32p(v) if v.dtype == np.float32 else _i32p(v)) for k, v in out_buffers.items()})
    _lib().hz0b_update(ctypes.byref(state._as_c_in()), _i32p(_ci32(slot_idx)), _f32p(_c32(new_value)), ctypes.byref(out))
    return state._out_to_state(out_buffers)


def protect(state: MemoryState, slot_idx: np.ndarray, strength: np.ndarray) -> MemoryState:
    out_buffers = state._new_out_buffers()
    out = _CMemoryStateOut(**{k: (_f32p(v) if v.dtype == np.float32 else _i32p(v)) for k, v in out_buffers.items()})
    _lib().hz0b_protect(ctypes.byref(state._as_c_in()), _i32p(_ci32(slot_idx)), _f32p(_c32(strength)), ctypes.byref(out))
    return state._out_to_state(out_buffers)


def forget_or_decay(state: MemoryState, *, decay_rate: float = 0.9) -> MemoryState:
    out_buffers = state._new_out_buffers()
    out = _CMemoryStateOut(**{k: (_f32p(v) if v.dtype == np.float32 else _i32p(v)) for k, v in out_buffers.items()})
    _lib().hz0b_forget_or_decay(ctypes.byref(state._as_c_in()), decay_rate, ctypes.byref(out))
    return state._out_to_state(out_buffers)


def delete(state: MemoryState, slot_idx: np.ndarray) -> MemoryState:
    out_buffers = state._new_out_buffers()
    out = _CMemoryStateOut(**{k: (_f32p(v) if v.dtype == np.float32 else _i32p(v)) for k, v in out_buffers.items()})
    _lib().hz0b_delete(ctypes.byref(state._as_c_in()), _i32p(_ci32(slot_idx)), ctypes.byref(out))
    return state._out_to_state(out_buffers)
