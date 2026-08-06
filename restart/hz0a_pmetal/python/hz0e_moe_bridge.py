"""HZ-0E E9 Python <-> Rust/Metal bridge.

Loads `libhz0e_pmetal_moe_bridge` (built from
`restart/hz0a_pmetal/crates/hz0e-pmetal-moe-bridge`) via `ctypes` and
wraps it in a numpy-backed API for the REAL Metal MoE forward kernel
(`hz0a-pmetal-gpu::MetalMoeSwiGlu::forward_logits` -- real top-1
routing, real Metal expert/fallback SwiGLU dispatch, real gate
scaling). No PyO3/maturin dependency, matching
`python/hz0b_memory_bridge.py`/`python/hz0d_fastweights_bridge.py`'s
own established convention in this workspace.

Unlike those two bridges, the underlying Metal kernel wrapper is
expensive to CREATE (compiles the shader, builds a compute pipeline)
but cheap to REUSE -- `MoeKernel` is a real create-once/call-many
handle (a Python context manager wrapping the Rust `Hz0eMoeHandle`),
not a stateless per-call function. A real model integration should
create ONE `MoeKernel` and reuse it across every layer/token-batch
call, not recreate it per call.
"""
from __future__ import annotations

import ctypes
import platform
from pathlib import Path

import numpy as np

_LIB_NAMES = {
    "Darwin": "libhz0e_pmetal_moe_bridge.dylib",
    "Linux": "libhz0e_pmetal_moe_bridge.so",
    "Windows": "hz0e_pmetal_moe_bridge.dll",
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
        "Build it first: cargo build --release -p hz0e-pmetal-moe-bridge"
    )


def _load() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_find_library()))
    f32p = ctypes.POINTER(ctypes.c_float)
    err_pp = ctypes.POINTER(ctypes.c_char_p)

    lib.hz0e_moe_new.argtypes = [err_pp]
    lib.hz0e_moe_new.restype = ctypes.c_void_p

    lib.hz0e_moe_free.argtypes = [ctypes.c_void_p]
    lib.hz0e_moe_free.restype = None

    lib.hz0e_moe_free_error.argtypes = [ctypes.c_char_p]
    lib.hz0e_moe_free_error.restype = None

    lib.hz0e_moe_forward_logits.argtypes = [
        ctypes.c_void_p, f32p, f32p, f32p, f32p, f32p, f32p,
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_float, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
        f32p, err_pp,
    ]
    lib.hz0e_moe_forward_logits.restype = ctypes.c_int32
    return lib


_LIB = None


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        _LIB = _load()
    return _LIB


def _f32p(arr: np.ndarray):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _c32(arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr, dtype=np.float32)


class PMetalMoeError(RuntimeError):
    pass


class MoeKernel:
    """Create-once, call-many handle around the real Metal MoE forward
    kernel. Use as a context manager (`with MoeKernel() as kernel:`) or
    call `.close()` explicitly -- the underlying Metal pipeline is real
    GPU state that must be freed exactly once."""

    def __init__(self) -> None:
        err = ctypes.c_char_p()
        handle = _lib().hz0e_moe_new(ctypes.byref(err))
        if not handle:
            message = err.value.decode("utf-8", "replace") if err.value else "hz0e_moe_new failed"
            if err.value:
                _lib().hz0e_moe_free_error(err)
            raise PMetalMoeError(message)
        self._handle = handle

    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib().hz0e_moe_free(self._handle)
            self._handle = None

    def __enter__(self) -> "MoeKernel":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def forward_logits(
        self, router_logits: np.ndarray, x: np.ndarray,
        expert_weights: np.ndarray, expert_biases: np.ndarray,
        fallback_weights: np.ndarray, fallback_biases: np.ndarray,
        *, experts: int, capacity_factor: float, dim: int, expert_d_ff: int, fallback_d_ff: int,
    ) -> np.ndarray:
        """`router_logits`: `[tokens, experts]`. `x`: `[tokens, dim]`.
        `expert_weights`: `[experts, 3*expert_d_ff*dim]` (gate/up/down
        concatenated per expert, matching `reference/hz0e_moe_contract.py`'s
        own `[expert_d_ff, dim]`/`[expert_d_ff, dim]`/`[dim, expert_d_ff]`
        row-major layout flattened). `expert_biases`:
        `[experts, 2*expert_d_ff+dim]`. `fallback_weights`:
        `[3*fallback_d_ff*dim]`, `fallback_biases`: `[2*fallback_d_ff+dim]`.
        Returns `[tokens, dim]`, the real router-gated MoE output."""
        if self._handle is None:
            raise PMetalMoeError("forward_logits called on a closed MoeKernel")
        tokens = x.shape[0]
        router_logits = _c32(router_logits).reshape(-1)
        x = _c32(x).reshape(-1)
        expert_weights = _c32(expert_weights).reshape(-1)
        expert_biases = _c32(expert_biases).reshape(-1)
        fallback_weights = _c32(fallback_weights).reshape(-1)
        fallback_biases = _c32(fallback_biases).reshape(-1)

        # Real, REQUIRED safety check: the C ABI passes raw pointers with
        # NO length information -- the Rust side reads exactly
        # `tokens*experts`/`experts*3*expert_d_ff*dim`/etc floats
        # starting at each pointer, regardless of how large the actual
        # buffer the caller allocated is. A caller passing a too-small
        # array would NOT raise a clean error on the Rust side; it would
        # read past the end of the buffer (undefined behavior). Found
        # directly while writing this bridge's own test suite -- caught
        # BEFORE any real model integration could hit it, not after.
        # Every buffer's length is validated here, on the Python side,
        # BEFORE any pointer crosses the FFI boundary.
        expected = {
            "router_logits": tokens * experts,
            "x": tokens * dim,
            "expert_weights": experts * 3 * expert_d_ff * dim,
            "expert_biases": experts * (2 * expert_d_ff + dim),
            "fallback_weights": 3 * fallback_d_ff * dim,
            "fallback_biases": 2 * fallback_d_ff + dim,
        }
        actual = {
            "router_logits": router_logits.size, "x": x.size,
            "expert_weights": expert_weights.size, "expert_biases": expert_biases.size,
            "fallback_weights": fallback_weights.size, "fallback_biases": fallback_biases.size,
        }
        for name, expected_size in expected.items():
            if actual[name] != expected_size:
                raise PMetalMoeError(f"{name}: expected {expected_size} elements, got {actual[name]}")

        output = np.zeros(tokens * dim, dtype=np.float32)
        err = ctypes.c_char_p()
        status = _lib().hz0e_moe_forward_logits(
            self._handle, _f32p(router_logits), _f32p(x),
            _f32p(expert_weights), _f32p(expert_biases), _f32p(fallback_weights), _f32p(fallback_biases),
            tokens, experts, capacity_factor, dim, expert_d_ff, fallback_d_ff,
            _f32p(output), ctypes.byref(err),
        )
        if status != 0:
            message = err.value.decode("utf-8", "replace") if err.value else "hz0e_moe_forward_logits failed"
            if err.value:
                _lib().hz0e_moe_free_error(err)
            raise PMetalMoeError(message)
        return output.reshape(tokens, dim)
