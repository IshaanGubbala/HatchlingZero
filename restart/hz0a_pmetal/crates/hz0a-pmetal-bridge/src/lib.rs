use hz0a_pmetal_gpu::MetalConditionalAnchorAttention;
use hz0a_pmetal_kernel::{
    conditional_anchor_attention_backward_f32, conditional_anchor_attention_f32,
    restart_kernel_scope, A1OperatorSpec, Gdn2CacheSpec,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartBridgeConfig {
    pub cache_spec: Gdn2CacheSpec,
    pub operator_spec: A1OperatorSpec,
    pub target_runtime: &'static str,
}

impl Default for RestartBridgeConfig {
    fn default() -> Self {
        Self {
            cache_spec: Gdn2CacheSpec::default(),
            operator_spec: A1OperatorSpec::default(),
            target_runtime: "pmetal",
        }
    }
}

/// C ABI: real Python<->Rust integration point for HZ-0C's conditional
/// anchor attention forward pass -- C8's previously-open "model-level
/// integration" item (no Python<->Rust FFI mechanism existed anywhere in
/// this repo before this function). Wraps
/// `hz0a_pmetal_kernel::conditional_anchor_attention_f32` (the same CPU
/// reference already verified against the real Python
/// `masked_anchor_attention` in `hz0a-pmetal-kernel/tests/parity_with_python_reference.rs`)
/// with a flat-pointer C ABI so a `ctypes`-based Python caller can dispatch
/// to it directly, with no intermediate serialization.
///
/// Caller (Python, via ctypes) owns every buffer -- this function only
/// reads the input pointers and writes into the caller-provided `output`
/// pointer; nothing is allocated on the Rust side that the caller would
/// need to free. Returns `0` on success, `-1` on a shape/argument mismatch
/// (the underlying function's own `Err` case; the specific message is not
/// passed across the FFI boundary -- the Python wrapper re-validates
/// shapes before calling and treats `-1` as a defensive fallback, not the
/// primary error-reporting path).
///
/// # Safety
/// `x` must point to `batch*seq*dim` valid `f32`s; `qkv_weight` to
/// `3*dim*dim`; `qkv_bias`/`out_bias` to `3*dim`/`dim`; `out_weight` to
/// `dim*dim`; `trigger` to `batch*seq`; `output` to `batch*seq*dim`
/// valid, WRITABLE `f32` slots that do not alias any input buffer. All
/// pointers must be aligned and non-null. The caller is responsible for
/// upholding this contract; this function cannot check it.
#[no_mangle]
pub unsafe extern "C" fn hz0c_conditional_attention_forward(
    batch: i64,
    seq: i64,
    dim: i64,
    heads: i64,
    x: *const f32,
    qkv_weight: *const f32,
    qkv_bias: *const f32,
    out_weight: *const f32,
    out_bias: *const f32,
    trigger: *const f32,
    output: *mut f32,
) -> i32 {
    if batch <= 0 || seq <= 0 || dim <= 0 || heads <= 0 {
        return -1;
    }
    let (batch, seq, dim, heads) = (batch as usize, seq as usize, dim as usize, heads as usize);
    let tokens = batch * seq;
    let x_slice = std::slice::from_raw_parts(x, tokens * dim);
    let qkv_w_slice = std::slice::from_raw_parts(qkv_weight, 3 * dim * dim);
    let qkv_b_slice = std::slice::from_raw_parts(qkv_bias, 3 * dim);
    let out_w_slice = std::slice::from_raw_parts(out_weight, dim * dim);
    let out_b_slice = std::slice::from_raw_parts(out_bias, dim);
    let trigger_slice = std::slice::from_raw_parts(trigger, tokens);
    match conditional_anchor_attention_f32(
        batch, seq, dim, heads, x_slice, qkv_w_slice, qkv_b_slice, out_w_slice, out_b_slice,
        trigger_slice,
    ) {
        Ok(result) => {
            let out_slice = std::slice::from_raw_parts_mut(output, tokens * dim);
            out_slice.copy_from_slice(&result);
            0
        }
        Err(_) => -1,
    }
}

/// C ABI backward counterpart to `hz0c_conditional_attention_forward`,
/// wrapping `conditional_anchor_attention_backward_f32`. Writes into five
/// separate caller-provided output buffers (`grad_x`, `grad_qkv_weight`,
/// `grad_qkv_bias`, `grad_out_weight`, `grad_out_bias`) rather than one,
/// matching the underlying function's own `ConditionalAttentionBackward`
/// struct fields.
///
/// # Safety
/// Same contract as `hz0c_conditional_attention_forward` for the shared
/// input buffers, plus `grad_output` (`batch*seq*dim` valid `f32`s) and
/// the five output pointers, each sized to match its corresponding
/// gradient's natural length (`grad_x`: `batch*seq*dim`;
/// `grad_qkv_weight`: `3*dim*dim`; `grad_qkv_bias`: `3*dim`;
/// `grad_out_weight`: `dim*dim`; `grad_out_bias`: `dim`), writable, and
/// non-aliasing.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn hz0c_conditional_attention_backward(
    batch: i64,
    seq: i64,
    dim: i64,
    heads: i64,
    x: *const f32,
    qkv_weight: *const f32,
    qkv_bias: *const f32,
    out_weight: *const f32,
    trigger: *const f32,
    grad_output: *const f32,
    grad_x: *mut f32,
    grad_qkv_weight: *mut f32,
    grad_qkv_bias: *mut f32,
    grad_out_weight: *mut f32,
    grad_out_bias: *mut f32,
) -> i32 {
    if batch <= 0 || seq <= 0 || dim <= 0 || heads <= 0 {
        return -1;
    }
    let (batch, seq, dim, heads) = (batch as usize, seq as usize, dim as usize, heads as usize);
    let tokens = batch * seq;
    let x_slice = std::slice::from_raw_parts(x, tokens * dim);
    let qkv_w_slice = std::slice::from_raw_parts(qkv_weight, 3 * dim * dim);
    let qkv_b_slice = std::slice::from_raw_parts(qkv_bias, 3 * dim);
    let out_w_slice = std::slice::from_raw_parts(out_weight, dim * dim);
    let trigger_slice = std::slice::from_raw_parts(trigger, tokens);
    let grad_output_slice = std::slice::from_raw_parts(grad_output, tokens * dim);
    match conditional_anchor_attention_backward_f32(
        batch, seq, dim, heads, x_slice, qkv_w_slice, qkv_b_slice, out_w_slice, trigger_slice,
        grad_output_slice,
    ) {
        Ok(result) => {
            std::slice::from_raw_parts_mut(grad_x, tokens * dim).copy_from_slice(&result.grad_x);
            std::slice::from_raw_parts_mut(grad_qkv_weight, 3 * dim * dim)
                .copy_from_slice(&result.grad_qkv_weight);
            std::slice::from_raw_parts_mut(grad_qkv_bias, 3 * dim)
                .copy_from_slice(&result.grad_qkv_bias);
            std::slice::from_raw_parts_mut(grad_out_weight, dim * dim)
                .copy_from_slice(&result.grad_out_weight);
            std::slice::from_raw_parts_mut(grad_out_bias, dim)
                .copy_from_slice(&result.grad_out_bias);
            0
        }
        Err(_) => -1,
    }
}

/// C ABI: creates a real Metal GPU dispatch handle for conditional anchor
/// attention (`hz0a_pmetal_gpu::MetalConditionalAnchorAttention`), for the
/// GPU counterpart to `hz0c_conditional_attention_forward`'s CPU dispatch
/// -- named as the real next step in
/// `docs/restart/hz0c_c8_model_level_integration_results.md` after that
/// CPU FFI path was honestly measured 35-190x slower than MLX. Handle-
/// based (not a fresh pipeline per call) because creating a Metal device/
/// pipeline/queue is real, amortizable setup cost -- exactly the kind of
/// per-call overhead a fair benchmark against MLX (which keeps its own
/// compiled kernels resident) must not smuggle into the timed region.
///
/// Returns a non-null opaque handle on success, `null` if no Metal device
/// is available (e.g. a CI/non-Apple-Silicon environment) or shader
/// compilation fails. The caller (Python) owns the handle and MUST call
/// `hz0c_metal_conditional_attention_destroy` exactly once when done with
/// it, and must not use it after destroying it.
#[no_mangle]
pub extern "C" fn hz0c_metal_conditional_attention_create() -> *mut MetalConditionalAnchorAttention {
    match MetalConditionalAnchorAttention::new() {
        Ok(instance) => Box::into_raw(Box::new(instance)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// C ABI: destroys a handle created by `hz0c_metal_conditional_attention_create`.
/// A null handle is accepted as a no-op (matches `free(NULL)`'s own
/// convention, so a Python caller does not need its own null check before
/// calling this in a cleanup/`__del__` path).
///
/// # Safety
/// `handle` must be either null or a value previously returned by
/// `hz0c_metal_conditional_attention_create` that has not already been
/// destroyed. Using the handle again after this call is undefined
/// behavior -- the caller (Python) is responsible for not doing so.
#[no_mangle]
pub unsafe extern "C" fn hz0c_metal_conditional_attention_destroy(handle: *mut MetalConditionalAnchorAttention) {
    if !handle.is_null() {
        drop(Box::from_raw(handle));
    }
}

/// C ABI: real Metal GPU dispatch for conditional anchor attention,
/// through a handle created by `hz0c_metal_conditional_attention_create`.
/// Same buffer-layout contract as `hz0c_conditional_attention_forward`
/// (this function's CPU counterpart) -- see that function's own doc
/// comment for the exact per-buffer length contract.
///
/// # Safety
/// `handle` must be a live handle from `hz0c_metal_conditional_attention_create`
/// (not null, not yet destroyed). All buffer-pointer requirements are
/// identical to `hz0c_conditional_attention_forward`'s own safety
/// contract.
#[no_mangle]
pub unsafe extern "C" fn hz0c_metal_conditional_attention_forward(
    handle: *const MetalConditionalAnchorAttention,
    batch: i64,
    seq: i64,
    dim: i64,
    heads: i64,
    x: *const f32,
    qkv_weight: *const f32,
    qkv_bias: *const f32,
    out_weight: *const f32,
    out_bias: *const f32,
    trigger: *const f32,
    output: *mut f32,
) -> i32 {
    if handle.is_null() || batch <= 0 || seq <= 0 || dim <= 0 || heads <= 0 {
        return -1;
    }
    let (batch, seq, dim, heads) = (batch as usize, seq as usize, dim as usize, heads as usize);
    let tokens = batch * seq;
    let instance = &*handle;
    let x_slice = std::slice::from_raw_parts(x, tokens * dim);
    let qkv_w_slice = std::slice::from_raw_parts(qkv_weight, 3 * dim * dim);
    let qkv_b_slice = std::slice::from_raw_parts(qkv_bias, 3 * dim);
    let out_w_slice = std::slice::from_raw_parts(out_weight, dim * dim);
    let out_b_slice = std::slice::from_raw_parts(out_bias, dim);
    let trigger_slice = std::slice::from_raw_parts(trigger, tokens);
    match instance.forward(
        batch, seq, dim, heads, x_slice, qkv_w_slice, qkv_b_slice, out_w_slice, out_b_slice,
        trigger_slice,
    ) {
        Ok(result) => {
            let out_slice = std::slice::from_raw_parts_mut(output, tokens * dim);
            out_slice.copy_from_slice(&result);
            0
        }
        Err(_) => -1,
    }
}

pub fn restart_bridge_summary() -> String {
    let cfg = RestartBridgeConfig::default();
    format!(
        "{} -> {} bridge scaffold (d_model={}, heads={}, d_k={}, d_v={})",
        restart_kernel_scope(),
        cfg.target_runtime,
        cfg.operator_spec.model_dim,
        cfg.operator_spec.num_heads,
        cfg.operator_spec.key_dim,
        cfg.operator_spec.value_dim
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridge_summary_mentions_pmetal_and_a1_shape() {
        let summary = restart_bridge_summary();
        assert!(summary.contains("pmetal"));
        assert!(summary.contains("d_model=768"));
        assert!(summary.contains("heads=12"));
    }

    #[test]
    fn ffi_forward_matches_safe_reference_directly() {
        let (batch, seq, dim, heads) = (2, 4, 4, 2);
        let mut state = 5u64;
        let lcg = |s: &mut u64, scale: f32| -> f32 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((*s >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * 2.0 * scale
        };
        let make = |n: usize, s: &mut u64| (0..n).map(|_| lcg(s, 0.3)).collect::<Vec<f32>>();
        let x = make(batch * seq * dim, &mut state);
        let qkv_w = make(3 * dim * dim, &mut state);
        let qkv_b = make(3 * dim, &mut state);
        let out_w = make(dim * dim, &mut state);
        let out_b = make(dim, &mut state);
        let trigger = vec![1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];

        let expected = conditional_anchor_attention_f32(batch, seq, dim, heads, &x, &qkv_w, &qkv_b, &out_w, &out_b, &trigger).unwrap();
        let mut ffi_out = vec![0.0f32; batch * seq * dim];
        let status = unsafe {
            hz0c_conditional_attention_forward(
                batch as i64, seq as i64, dim as i64, heads as i64,
                x.as_ptr(), qkv_w.as_ptr(), qkv_b.as_ptr(), out_w.as_ptr(), out_b.as_ptr(), trigger.as_ptr(),
                ffi_out.as_mut_ptr(),
            )
        };
        assert_eq!(status, 0);
        assert_eq!(ffi_out, expected);
    }

    #[test]
    fn gpu_ffi_handle_forward_matches_safe_reference_and_lifecycle_is_sound() {
        let (batch, seq, dim, heads) = (2, 4, 4, 2);
        let mut state = 5u64;
        let lcg = |s: &mut u64, scale: f32| -> f32 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((*s >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * 2.0 * scale
        };
        let make = |n: usize, s: &mut u64| (0..n).map(|_| lcg(s, 0.3)).collect::<Vec<f32>>();
        let x = make(batch * seq * dim, &mut state);
        let qkv_w = make(3 * dim * dim, &mut state);
        let qkv_b = make(3 * dim, &mut state);
        let out_w = make(dim * dim, &mut state);
        let out_b = make(dim, &mut state);
        let trigger = vec![1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];

        let expected = conditional_anchor_attention_f32(batch, seq, dim, heads, &x, &qkv_w, &qkv_b, &out_w, &out_b, &trigger).unwrap();
        let handle = hz0c_metal_conditional_attention_create();
        assert!(!handle.is_null(), "no Metal device available for this test");
        let mut ffi_out = vec![0.0f32; batch * seq * dim];
        // Call the SAME handle twice to lock in that it is genuinely
        // reusable (the whole point of a handle over a fresh pipeline per
        // call) and both calls stay correct, not just the first.
        for _ in 0..2 {
            let status = unsafe {
                hz0c_metal_conditional_attention_forward(
                    handle, batch as i64, seq as i64, dim as i64, heads as i64,
                    x.as_ptr(), qkv_w.as_ptr(), qkv_b.as_ptr(), out_w.as_ptr(), out_b.as_ptr(), trigger.as_ptr(),
                    ffi_out.as_mut_ptr(),
                )
            };
            assert_eq!(status, 0);
            let max_diff = ffi_out.iter().zip(expected.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            assert!(max_diff < 2e-3, "GPU FFI diff too large: {max_diff}");
        }
        unsafe { hz0c_metal_conditional_attention_destroy(handle) };
    }

    #[test]
    fn gpu_ffi_forward_rejects_null_handle_and_destroy_accepts_null() {
        let mut out = vec![0.0f32; 1];
        let buf = vec![0.0f32; 1];
        let status = unsafe {
            hz0c_metal_conditional_attention_forward(
                std::ptr::null(), 1, 1, 1, 1, buf.as_ptr(), buf.as_ptr(), buf.as_ptr(), buf.as_ptr(), buf.as_ptr(), buf.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        assert_eq!(status, -1);
        unsafe { hz0c_metal_conditional_attention_destroy(std::ptr::null_mut()) };
    }

    #[test]
    fn ffi_forward_rejects_invalid_shape() {
        let mut out = vec![0.0f32; 1];
        let buf = vec![0.0f32; 1];
        let status = unsafe {
            hz0c_conditional_attention_forward(
                0, 1, 1, 1, buf.as_ptr(), buf.as_ptr(), buf.as_ptr(), buf.as_ptr(), buf.as_ptr(), buf.as_ptr(),
                out.as_mut_ptr(),
            )
        };
        assert_eq!(status, -1);
    }

    #[test]
    fn ffi_backward_matches_safe_reference_directly() {
        let (batch, seq, dim, heads) = (1, 3, 4, 2);
        let x: Vec<f32> = (0..batch * seq * dim).map(|i| 0.03 + i as f32 * 0.02).collect();
        let qkv_w: Vec<f32> = (0..3 * dim * dim).map(|i| -0.08 + i as f32 * 0.007).collect();
        let qkv_b: Vec<f32> = (0..3 * dim).map(|i| -0.03 + i as f32 * 0.01).collect();
        let out_w: Vec<f32> = (0..dim * dim).map(|i| 0.04 - i as f32 * 0.005).collect();
        let trigger = vec![1.0, 0.0, 1.0];
        let grad_output: Vec<f32> = (0..batch * seq * dim).map(|i| 0.02 + i as f32 * 0.01).collect();

        let expected = conditional_anchor_attention_backward_f32(batch, seq, dim, heads, &x, &qkv_w, &qkv_b, &out_w, &trigger, &grad_output).unwrap();
        let mut grad_x = vec![0.0f32; batch * seq * dim];
        let mut grad_qkv_weight = vec![0.0f32; 3 * dim * dim];
        let mut grad_qkv_bias = vec![0.0f32; 3 * dim];
        let mut grad_out_weight = vec![0.0f32; dim * dim];
        let mut grad_out_bias = vec![0.0f32; dim];
        let status = unsafe {
            hz0c_conditional_attention_backward(
                batch as i64, seq as i64, dim as i64, heads as i64,
                x.as_ptr(), qkv_w.as_ptr(), qkv_b.as_ptr(), out_w.as_ptr(), trigger.as_ptr(), grad_output.as_ptr(),
                grad_x.as_mut_ptr(), grad_qkv_weight.as_mut_ptr(), grad_qkv_bias.as_mut_ptr(),
                grad_out_weight.as_mut_ptr(), grad_out_bias.as_mut_ptr(),
            )
        };
        assert_eq!(status, 0);
        assert_eq!(grad_x, expected.grad_x);
        assert_eq!(grad_qkv_weight, expected.grad_qkv_weight);
        assert_eq!(grad_qkv_bias, expected.grad_qkv_bias);
        assert_eq!(grad_out_weight, expected.grad_out_weight);
        assert_eq!(grad_out_bias, expected.grad_out_bias);
    }
}
