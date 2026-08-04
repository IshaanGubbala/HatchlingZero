"""HZ-0C C8: the real Python<->Rust integration point, closing the
"model-level integration" gap `docs/restart/hz0c_c9_end_to_end_report_results.md`
disclosed (PMetal's kernels were correctness-proven in isolation, via
Rust-vs-Rust and Rust-vs-Python-fixture tests, but nothing in the actual
Python model/eval path ever called them). No Python<->Rust FFI mechanism
existed anywhere in this repo before this module -- `ctypes` against a
real compiled `cdylib` (`restart/hz0a_pmetal/crates/hz0a-pmetal-bridge`),
not a new heavyweight build dependency (no PyO3/maturin), matching this
project's dependency-conscious convention throughout the Rust side.

`conditional_attention_forward`/`_backward` below are dependency-free
NumPy<->ctypes marshaling around
`hz0c_conditional_attention_forward`/`_backward`
(`restart/hz0a_pmetal/crates/hz0a-pmetal-bridge/src/lib.rs`), which
themselves wrap the SAME `conditional_anchor_attention_f32`/
`_backward_f32` CPU reference already verified against the real Python
`masked_anchor_attention` in
`restart/hz0a_pmetal/crates/hz0a-pmetal-kernel/tests/parity_with_python_reference.rs`
-- so correctness here rests on that existing, real cross-language
fixture check, not a new, separate claim.

Binary-trigger-only contract, same as the underlying kernel (see
`conditional_anchor_attention_f32`'s own doc comment in
`hz0a-pmetal-kernel/src/lib.rs`): a non-binary trigger is out of scope
and not checked here.
"""
from __future__ import annotations

import ctypes
import platform
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TARGET_DIR = _REPO_ROOT / "restart" / "hz0a_pmetal" / "target"
_LIB_NAMES = {
    "Darwin": "libhz0a_pmetal_bridge.dylib",
    "Linux": "libhz0a_pmetal_bridge.so",
}


class PMetalBridgeUnavailable(RuntimeError):
    """Raised when the compiled cdylib cannot be found or loaded -- e.g. a
    fresh checkout before `cargo build --release -p hz0a-pmetal-bridge`
    has been run. Callers should treat this the same way the rest of this
    project treats a missing frozen checkpoint: a real, expected "not
    built here" condition, not a bug to work around silently."""


def _library_path() -> Path:
    lib_name = _LIB_NAMES.get(platform.system())
    if lib_name is None:
        raise PMetalBridgeUnavailable(f"no PMetal bridge library name known for platform {platform.system()!r}")
    for profile in ("release", "debug"):
        candidate = _TARGET_DIR / profile / lib_name
        if candidate.exists():
            return candidate
    raise PMetalBridgeUnavailable(
        f"{lib_name} not found under {_TARGET_DIR} -- run "
        "`cargo build --release -p hz0a-pmetal-bridge --manifest-path restart/hz0a_pmetal/Cargo.toml` first"
    )


_LIB = None


def _lib() -> ctypes.CDLL:
    global _LIB
    if _LIB is None:
        loaded = ctypes.CDLL(str(_library_path()))
        loaded.hz0c_conditional_attention_forward.restype = ctypes.c_int32
        loaded.hz0c_conditional_attention_backward.restype = ctypes.c_int32
        loaded.hz0c_metal_conditional_attention_create.restype = ctypes.c_void_p
        loaded.hz0c_metal_conditional_attention_destroy.argtypes = [ctypes.c_void_p]
        loaded.hz0c_metal_conditional_attention_forward.restype = ctypes.c_int32
        _LIB = loaded
    return _LIB


def _as_f32(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array, dtype=np.float32)
    return contiguous


def _ptr(array: np.ndarray) -> ctypes.POINTER(ctypes.c_float):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _validate_shapes(*, dim: int, heads: int, x_shape, trigger_shape, qkv_w_shape, qkv_b_shape, out_w_shape, out_b_shape=None) -> None:
    """The Rust FFI functions TRUST the caller's declared batch/seq/dim/heads
    and read exactly that many elements from each raw pointer -- they have
    no way to independently check a buffer's real length (a bare `*const
    f32` carries no length). Passing an undersized array would be
    undefined behavior (an out-of-bounds read, likely a segfault), not a
    clean error, at the FFI boundary. This validates EVERY array's shape
    in safe Python before any pointer ever crosses into Rust, so a
    mismatch raises a normal Python exception instead."""
    if dim <= 0 or heads <= 0 or dim % heads != 0:
        raise ValueError(f"dim ({dim}) must be positive and divisible by heads ({heads})")
    if len(x_shape) != 3 or x_shape[2] != dim:
        raise ValueError(f"x shape {x_shape} inconsistent with dim={dim} (expected [batch, seq, {dim}])")
    batch, seq = x_shape[0], x_shape[1]
    if tuple(trigger_shape) != (batch, seq):
        raise ValueError(f"trigger shape {trigger_shape} must be {(batch, seq)}")
    if tuple(qkv_w_shape) != (3 * dim, dim):
        raise ValueError(f"qkv_w shape {qkv_w_shape} must be {(3 * dim, dim)}")
    if tuple(qkv_b_shape) != (3 * dim,):
        raise ValueError(f"qkv_b shape {qkv_b_shape} must be {(3 * dim,)}")
    if tuple(out_w_shape) != (dim, dim):
        raise ValueError(f"out_w shape {out_w_shape} must be {(dim, dim)}")
    if out_b_shape is not None and tuple(out_b_shape) != (dim,):
        raise ValueError(f"out_b shape {out_b_shape} must be {(dim,)}")


def conditional_attention_forward(x: np.ndarray, trigger: np.ndarray, *, qkv_w: np.ndarray, qkv_b: np.ndarray, out_w: np.ndarray, out_b: np.ndarray, heads: int) -> np.ndarray:
    """x: [batch, seq, dim]. trigger: [batch, seq], BINARY (0.0/1.0) only.
    qkv_w: [3*dim, dim] (nn.Linear.weight layout, matching
    `reference/hz0c_surprise_trigger.py::masked_anchor_attention`'s own
    convention). Returns [batch, seq, dim], float32."""
    _validate_shapes(dim=x.shape[-1], heads=heads, x_shape=x.shape, trigger_shape=trigger.shape, qkv_w_shape=qkv_w.shape, qkv_b_shape=qkv_b.shape, out_w_shape=out_w.shape, out_b_shape=out_b.shape)
    batch, seq, dim = x.shape
    x_f32 = _as_f32(x.reshape(-1))
    qkv_w_f32 = _as_f32(qkv_w.reshape(-1))
    qkv_b_f32 = _as_f32(qkv_b.reshape(-1))
    out_w_f32 = _as_f32(out_w.reshape(-1))
    out_b_f32 = _as_f32(out_b.reshape(-1))
    trigger_f32 = _as_f32(trigger.reshape(-1))
    output = np.zeros(batch * seq * dim, dtype=np.float32)
    status = _lib().hz0c_conditional_attention_forward(
        ctypes.c_int64(batch), ctypes.c_int64(seq), ctypes.c_int64(dim), ctypes.c_int64(heads),
        _ptr(x_f32), _ptr(qkv_w_f32), _ptr(qkv_b_f32), _ptr(out_w_f32), _ptr(out_b_f32), _ptr(trigger_f32),
        _ptr(output),
    )
    if status != 0:
        raise RuntimeError(f"hz0c_conditional_attention_forward failed with status {status} (likely a shape mismatch)")
    return output.reshape(batch, seq, dim)


def conditional_attention_backward(x: np.ndarray, trigger: np.ndarray, grad_output: np.ndarray, *, qkv_w: np.ndarray, qkv_b: np.ndarray, out_w: np.ndarray, heads: int) -> dict[str, np.ndarray]:
    """Backward counterpart. Returns a dict with grad_x [batch,seq,dim],
    grad_qkv_weight [3*dim,dim], grad_qkv_bias [3*dim], grad_out_weight
    [dim,dim], grad_out_bias [dim], all float32."""
    _validate_shapes(dim=x.shape[-1], heads=heads, x_shape=x.shape, trigger_shape=trigger.shape, qkv_w_shape=qkv_w.shape, qkv_b_shape=qkv_b.shape, out_w_shape=out_w.shape)
    if grad_output.shape != x.shape:
        raise ValueError(f"grad_output shape {grad_output.shape} must match x shape {x.shape}")
    batch, seq, dim = x.shape
    x_f32 = _as_f32(x.reshape(-1))
    qkv_w_f32 = _as_f32(qkv_w.reshape(-1))
    qkv_b_f32 = _as_f32(qkv_b.reshape(-1))
    out_w_f32 = _as_f32(out_w.reshape(-1))
    trigger_f32 = _as_f32(trigger.reshape(-1))
    grad_output_f32 = _as_f32(grad_output.reshape(-1))
    grad_x = np.zeros(batch * seq * dim, dtype=np.float32)
    grad_qkv_weight = np.zeros(3 * dim * dim, dtype=np.float32)
    grad_qkv_bias = np.zeros(3 * dim, dtype=np.float32)
    grad_out_weight = np.zeros(dim * dim, dtype=np.float32)
    grad_out_bias = np.zeros(dim, dtype=np.float32)
    status = _lib().hz0c_conditional_attention_backward(
        ctypes.c_int64(batch), ctypes.c_int64(seq), ctypes.c_int64(dim), ctypes.c_int64(heads),
        _ptr(x_f32), _ptr(qkv_w_f32), _ptr(qkv_b_f32), _ptr(out_w_f32), _ptr(trigger_f32), _ptr(grad_output_f32),
        _ptr(grad_x), _ptr(grad_qkv_weight), _ptr(grad_qkv_bias), _ptr(grad_out_weight), _ptr(grad_out_bias),
    )
    if status != 0:
        raise RuntimeError(f"hz0c_conditional_attention_backward failed with status {status} (likely a shape mismatch)")
    return {
        "grad_x": grad_x.reshape(batch, seq, dim),
        "grad_qkv_weight": grad_qkv_weight.reshape(3 * dim, dim),
        "grad_qkv_bias": grad_qkv_bias,
        "grad_out_weight": grad_out_weight.reshape(dim, dim),
        "grad_out_bias": grad_out_bias,
    }


class MetalConditionalAttention:
    """Real Metal GPU dispatch for conditional anchor attention, through a
    reusable native handle -- the GPU counterpart to
    `conditional_attention_forward`'s CPU path, and this project's answer
    to `docs/restart/hz0c_c8_model_level_integration_results.md`'s named
    next step (that report found the CPU FFI path 35-190x slower than MLX
    and named exposing the already-built GPU kernel through this same
    bridge as the real candidate for an actual speedup).

    Handle-based rather than one-shot: creating a Metal device/pipeline/
    queue is real, amortizable setup cost that a fair per-call latency
    comparison against MLX (which keeps its own compiled kernels resident)
    must not include in the timed region. Use as a context manager or call
    `close()` explicitly; the underlying Rust handle is destroyed exactly
    once either way (`close()` is idempotent).

    Raises `PMetalBridgeUnavailable` at construction if no Metal device is
    available or the shader fails to compile (surfaced by the Rust side
    returning a null handle) -- the same "real, expected, not silently
    worked around" convention as a missing frozen checkpoint elsewhere in
    this project."""

    def __init__(self) -> None:
        self._handle = _lib().hz0c_metal_conditional_attention_create()
        if not self._handle:
            raise PMetalBridgeUnavailable("hz0c_metal_conditional_attention_create returned null -- no Metal device available or shader compilation failed")

    def forward(self, x: np.ndarray, trigger: np.ndarray, *, qkv_w: np.ndarray, qkv_b: np.ndarray, out_w: np.ndarray, out_b: np.ndarray, heads: int) -> np.ndarray:
        if not self._handle:
            raise PMetalBridgeUnavailable("this MetalConditionalAttention handle has already been closed")
        _validate_shapes(dim=x.shape[-1], heads=heads, x_shape=x.shape, trigger_shape=trigger.shape, qkv_w_shape=qkv_w.shape, qkv_b_shape=qkv_b.shape, out_w_shape=out_w.shape, out_b_shape=out_b.shape)
        batch, seq, dim = x.shape
        x_f32 = _as_f32(x.reshape(-1))
        qkv_w_f32 = _as_f32(qkv_w.reshape(-1))
        qkv_b_f32 = _as_f32(qkv_b.reshape(-1))
        out_w_f32 = _as_f32(out_w.reshape(-1))
        out_b_f32 = _as_f32(out_b.reshape(-1))
        trigger_f32 = _as_f32(trigger.reshape(-1))
        output = np.zeros(batch * seq * dim, dtype=np.float32)
        status = _lib().hz0c_metal_conditional_attention_forward(
            ctypes.c_void_p(self._handle),
            ctypes.c_int64(batch), ctypes.c_int64(seq), ctypes.c_int64(dim), ctypes.c_int64(heads),
            _ptr(x_f32), _ptr(qkv_w_f32), _ptr(qkv_b_f32), _ptr(out_w_f32), _ptr(out_b_f32), _ptr(trigger_f32),
            _ptr(output),
        )
        if status != 0:
            raise RuntimeError(f"hz0c_metal_conditional_attention_forward failed with status {status} (likely a shape mismatch)")
        return output.reshape(batch, seq, dim)

    def close(self) -> None:
        if self._handle:
            _lib().hz0c_metal_conditional_attention_destroy(ctypes.c_void_p(self._handle))
            self._handle = None

    def __enter__(self) -> "MetalConditionalAttention":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
