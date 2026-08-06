//! C-ABI FFI boundary for the real Metal MoE forward kernel
//! (`hz0a-pmetal-gpu::MetalMoeSwiGlu::forward_logits`), loaded from
//! Python via `ctypes` (see `python/hz0e_moe_bridge.py`). No PyO3/
//! maturin dependency, matching this workspace's own zero-extra-
//! toolchain convention (`hz0b-pmetal-memory-bridge`,
//! `hz0d-pmetal-fastweights-bridge`).
//!
//! Unlike those two prior bridges (whose underlying state is a plain
//! data struct), the Metal kernel wrapper itself is expensive to
//! CREATE (compiles the shader, builds a compute pipeline) but cheap
//! to REUSE -- so this bridge exposes an explicit create-once/call-
//! many handle (`hz0e_moe_new` / `hz0e_moe_forward_logits` /
//! `hz0e_moe_free`), not a stateless per-call function. A real model
//! integration calling this once per MoE layer per token batch must
//! NOT recompile the shader on every call, matching how
//! `MetalMoeSwiGlu::new()` is meant to be used (compile once, dispatch
//! many) in its own Rust callers.
//!
//! All `unsafe` here is confined to marshalling raw pointers and
//! managing the opaque handle's lifetime; every actual MoE computation
//! still runs inside `hz0a-pmetal-gpu`/`hz0e-pmetal-moe`.

use hz0a_pmetal_gpu::{CachedMoeWeights, MetalMoeSwiGlu};

/// Opaque handle -- Python holds this as a raw pointer and passes it
/// back into `hz0e_moe_forward_logits`/`hz0e_moe_free`, never
/// dereferencing it itself.
pub struct Hz0eMoeHandle {
    kernel: MetalMoeSwiGlu,
}

#[no_mangle]
pub extern "C" fn hz0e_moe_new(out_error: *mut *mut std::os::raw::c_char) -> *mut Hz0eMoeHandle {
    match MetalMoeSwiGlu::new() {
        Ok(kernel) => Box::into_raw(Box::new(Hz0eMoeHandle { kernel })),
        Err(message) => {
            if !out_error.is_null() {
                let c_string = std::ffi::CString::new(message).unwrap_or_default();
                unsafe {
                    *out_error = c_string.into_raw();
                }
            }
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn hz0e_moe_free(handle: *mut Hz0eMoeHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

#[no_mangle]
pub unsafe extern "C" fn hz0e_moe_free_error(message: *mut std::os::raw::c_char) {
    if !message.is_null() {
        drop(std::ffi::CString::from_raw(message));
    }
}

/// Full real forward pass for one MoE layer: real top-1 routing from
/// `router_logits`, real Metal expert/fallback SwiGLU dispatch, real
/// gate scaling -- the SAME computation
/// `hz0a-pmetal-gpu::MetalMoeSwiGlu::forward_logits` performs, exposed
/// across the FFI boundary. Caller (Python) owns and pre-allocates
/// every buffer; shapes are fully determined by
/// `tokens`/`experts`/`dim`/`expert_d_ff`/`fallback_d_ff`, which the
/// caller already knows. Returns `0` on success, nonzero on error
/// (with `*out_error` set to a real, human-readable message the caller
/// must free via `hz0e_moe_free_error`).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn hz0e_moe_forward_logits(
    handle: *mut Hz0eMoeHandle,
    router_logits: *const f32,
    input: *const f32,
    expert_weights: *const f32,
    expert_biases: *const f32,
    fallback_weights: *const f32,
    fallback_biases: *const f32,
    tokens: usize,
    experts: usize,
    capacity_factor: f32,
    dim: usize,
    expert_d_ff: usize,
    fallback_d_ff: usize,
    out_output: *mut f32,
    out_error: *mut *mut std::os::raw::c_char,
) -> i32 {
    if handle.is_null() {
        set_error(out_error, "hz0e_moe_forward_logits: null handle");
        return 1;
    }
    let handle_ref = &*handle;
    let router_logits = std::slice::from_raw_parts(router_logits, tokens * experts);
    let input = std::slice::from_raw_parts(input, tokens * dim);
    let expert_weights = std::slice::from_raw_parts(expert_weights, experts * 3 * expert_d_ff * dim);
    let expert_biases = std::slice::from_raw_parts(expert_biases, experts * (2 * expert_d_ff + dim));
    let fallback_weights = std::slice::from_raw_parts(fallback_weights, 3 * fallback_d_ff * dim);
    let fallback_biases = std::slice::from_raw_parts(fallback_biases, 2 * fallback_d_ff + dim);

    match handle_ref.kernel.forward_logits(
        router_logits, input, expert_weights, expert_biases, fallback_weights, fallback_biases,
        tokens, experts, capacity_factor, dim, expert_d_ff, fallback_d_ff,
    ) {
        Ok((_plan, output)) => {
            std::slice::from_raw_parts_mut(out_output, tokens * dim).copy_from_slice(&output);
            0
        }
        Err(message) => {
            set_error(out_error, &message);
            1
        }
    }
}

/// Opaque handle wrapping device-resident weight buffers
/// (`hz0a_pmetal_gpu::CachedMoeWeights`), produced once by
/// `hz0e_moe_upload_weights` and reused across many
/// `hz0e_moe_forward_cached` calls -- the fix for the ~40x end-to-end
/// slowdown `hz0e_moe_forward_logits` incurs by re-uploading the same
/// expert/fallback weight buffers on every call.
pub struct Hz0eMoeWeightsHandle {
    weights: CachedMoeWeights,
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn hz0e_moe_upload_weights(
    handle: *mut Hz0eMoeHandle,
    expert_weights: *const f32,
    expert_biases: *const f32,
    fallback_weights: *const f32,
    fallback_biases: *const f32,
    experts: usize,
    dim: usize,
    expert_d_ff: usize,
    fallback_d_ff: usize,
    out_error: *mut *mut std::os::raw::c_char,
) -> *mut Hz0eMoeWeightsHandle {
    if handle.is_null() {
        set_error(out_error, "hz0e_moe_upload_weights: null handle");
        return std::ptr::null_mut();
    }
    let handle_ref = &*handle;
    let expert_weights = std::slice::from_raw_parts(expert_weights, experts * 3 * expert_d_ff * dim);
    let expert_biases = std::slice::from_raw_parts(expert_biases, experts * (2 * expert_d_ff + dim));
    let fallback_weights = std::slice::from_raw_parts(fallback_weights, 3 * fallback_d_ff * dim);
    let fallback_biases = std::slice::from_raw_parts(fallback_biases, 2 * fallback_d_ff + dim);
    match handle_ref.kernel.upload_weights(
        expert_weights, expert_biases, fallback_weights, fallback_biases,
        experts, dim, expert_d_ff, fallback_d_ff,
    ) {
        Ok(weights) => Box::into_raw(Box::new(Hz0eMoeWeightsHandle { weights })),
        Err(message) => {
            set_error(out_error, &message);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn hz0e_moe_free_weights(handle: *mut Hz0eMoeWeightsHandle) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// Real forward pass against already-resident weight buffers -- only
/// `router_logits`/`input`/`output` (all tokens-sized, small) cross the
/// FFI boundary per call; `expert_weights`/`fallback_weights` do not.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn hz0e_moe_forward_cached(
    handle: *mut Hz0eMoeHandle,
    weights_handle: *mut Hz0eMoeWeightsHandle,
    router_logits: *const f32,
    input: *const f32,
    tokens: usize,
    experts: usize,
    capacity_factor: f32,
    dim: usize,
    out_output: *mut f32,
    out_error: *mut *mut std::os::raw::c_char,
) -> i32 {
    if handle.is_null() || weights_handle.is_null() {
        set_error(out_error, "hz0e_moe_forward_cached: null handle");
        return 1;
    }
    let handle_ref = &*handle;
    let weights_ref = &(*weights_handle).weights;
    if weights_ref.experts() != experts || weights_ref.dim() != dim {
        set_error(out_error, "hz0e_moe_forward_cached: experts/dim mismatch with cached weights");
        return 1;
    }
    let router_logits = std::slice::from_raw_parts(router_logits, tokens * experts);
    let input = std::slice::from_raw_parts(input, tokens * dim);
    match handle_ref
        .kernel
        .forward_logits_cached(router_logits, input, tokens, capacity_factor, weights_ref)
    {
        Ok((_plan, output)) => {
            std::slice::from_raw_parts_mut(out_output, tokens * dim).copy_from_slice(&output);
            0
        }
        Err(message) => {
            set_error(out_error, &message);
            1
        }
    }
}

unsafe fn set_error(out_error: *mut *mut std::os::raw::c_char, message: &str) {
    if !out_error.is_null() {
        let c_string = std::ffi::CString::new(message).unwrap_or_default();
        *out_error = c_string.into_raw();
    }
}
