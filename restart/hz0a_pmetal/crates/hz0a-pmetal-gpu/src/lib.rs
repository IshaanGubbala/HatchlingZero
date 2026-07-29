//! Real Metal GPU dispatch from Rust for the GDN-2 forward recurrence --
//! the A8 "step 2: Metal forward + PMetal backward" milestone (plan's own
//! sequencing: don't attempt step 3, fully fused, until step 1 -- the CPU
//! `hz0a-pmetal-tensor` crate -- and this step are proven).
//!
//! The shader body is the same O(S) single-pass scan already fixed and
//! proven this session in `reference/hz0a_mlx_metal.py`'s `_SOURCE` (the
//! MLX-dispatched version) -- ported to a standalone Metal Shading
//! Language kernel with explicit buffer bindings instead of relying on
//! MLX's `mx.fast.metal_kernel` code generation, and dispatched directly
//! via the `metal` crate (the standard, widely-used Rust binding to
//! Apple's Metal API; the CPU tensor crate stays dependency-minimal, this
//! is a separate crate specifically for real device execution).

use hz0a_pmetal_kernel::{gdn2_backward_f32, gdn2_forward_f32, Gdn2ForwardShape};
use metal::{Device, MTLResourceOptions, MTLSize};

const FORWARD_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Gdn2Shape {
    uint B;
    uint S;
    uint H;
    uint K;
    uint V;
};

static inline float hz_sigmoid(float x) {
    return 1.0f / (1.0f + metal::exp(-x));
}

kernel void gdn2_forward(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* d [[buffer(3)]],
    device const float* e [[buffer(4)]],
    device const float* w [[buffer(5)]],
    device const float* initial [[buffer(6)]],
    device float* y [[buffer(7)]],
    device float* final_state [[buffer(8)]],
    constant Gdn2Shape& shape [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    uint B = shape.B;
    uint S = shape.S;
    uint H = shape.H;
    uint K = shape.K;
    uint V = shape.V;
    uint value = tid % V;
    uint head = (tid / V) % H;
    uint batch = tid / (V * H);
    if (batch >= B || K > 64) return;

    thread float state[64];
    uint state_base = ((batch * H + head) * V + value) * K;
    for (uint key = 0; key < K; ++key)
        state[key] = initial[state_base + key];

    for (uint t = 0; t < S; ++t) {
        uint key_row = ((batch * S + t) * H + head) * K;
        uint row = ((batch * S + t) * H + head) * V + value;
        float write = hz_sigmoid(w[row]);
        float value_t = v[row];
        float output = 0.0f;
        for (uint key = 0; key < K; ++key) {
            float decay = hz_sigmoid(d[key_row + key]);
            float erase = hz_sigmoid(e[key_row + key]);
            state[key] = decay * (1.0f - erase) * state[key] + write * value_t * k[key_row + key];
            output += state[key] * q[key_row + key];
        }
        y[row] = output;
    }
    for (uint key = 0; key < K; ++key)
        final_state[state_base + key] = state[key];
}
"#;

/// Same threadgroup-shared-memory value-axis reduction already proven this
/// session in `reference/hz0a_mlx_metal.py`'s `_BACKWARD_BODY_FUSED` (1.93x
/// faster than materializing (B,S,H,V,K) partial-gradient buffers, verified
/// there against the reference to 2e-5) -- ported to a standalone kernel.
/// Note: `d`/`e`/`w` inputs here are the sigmoid-*activated* gates (matching
/// the MLX version's contract), not raw logits; the caller (or a wrapping
/// gate-derivative step) is responsible for that if starting from logits.
const BACKWARD_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Gdn2Shape {
    uint B;
    uint S;
    uint H;
    uint K;
    uint V;
};

kernel void gdn2_backward(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* d [[buffer(3)]],
    device const float* e [[buffer(4)]],
    device const float* w [[buffer(5)]],
    device const float* initial [[buffer(6)]],
    device const float* grad_output [[buffer(7)]],
    device const float* grad_final [[buffer(8)]],
    device float* grad_q [[buffer(9)]],
    device float* grad_k [[buffer(10)]],
    device float* grad_v [[buffer(11)]],
    device float* grad_d [[buffer(12)]],
    device float* grad_e [[buffer(13)]],
    device float* grad_w [[buffer(14)]],
    device float* grad_initial [[buffer(15)]],
    constant Gdn2Shape& shape [[buffer(16)]],
    uint value [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]])
{
    uint B = shape.B;
    uint S = shape.S;
    uint H = shape.H;
    uint K = shape.K;
    uint V = shape.V;
    uint head = group_id % H;
    uint batch = group_id / H;
    if (batch >= B || S > 128 || K > 64 || V > 64) return;

    threadgroup float shared_buf[64][64];

    thread float states[129][64];
    for (uint key = 0; key < K; ++key)
        states[0][key] = initial[((batch * H + head) * V + value) * K + key];
    for (uint t = 0; t < S; ++t) {
        uint input_base = ((batch * S + t) * H + head) * K;
        uint value_base = ((batch * S + t) * H + head) * V;
        for (uint key = 0; key < K; ++key)
            states[t + 1][key] = d[input_base + key] * (1.0f - e[input_base + key]) * states[t][key]
                + w[value_base + value] * v[value_base + value] * k[input_base + key];
    }
    thread float grad_state[64];
    for (uint key = 0; key < K; ++key)
        grad_state[key] = grad_final[((batch * H + head) * V + value) * K + key];

    for (int reverse = int(S) - 1; reverse >= 0; --reverse) {
        uint t = uint(reverse);
        uint input_base = ((batch * S + t) * H + head) * K;
        uint value_base = ((batch * S + t) * H + head) * V;
        thread float local_grad_q[64];
        thread float local_grad_k[64];
        thread float local_grad_d[64];
        thread float local_grad_e[64];
        float value_gradient = 0.0f;
        float write_gradient = 0.0f;
        for (uint key = 0; key < K; ++key) {
            float total = grad_state[key] + grad_output[value_base + value] * q[input_base + key];
            local_grad_q[key] = grad_output[value_base + value] * states[t + 1][key];
            local_grad_k[key] = total * w[value_base + value] * v[value_base + value];
            local_grad_d[key] = total * (1.0f - e[input_base + key]) * states[t][key];
            local_grad_e[key] = -total * d[input_base + key] * states[t][key];
            value_gradient += total * w[value_base + value] * k[input_base + key];
            write_gradient += total * v[value_base + value] * k[input_base + key];
            grad_state[key] = total * d[input_base + key] * (1.0f - e[input_base + key]);
        }
        grad_v[value_base + value] = value_gradient;
        grad_w[value_base + value] = write_gradient;

        uint out_base = ((batch * S + t) * H + head) * K;

        for (uint key = 0; key < K; ++key) shared_buf[value][key] = local_grad_q[key];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (value < K) {
            float total_sum = 0.0f;
            for (uint vv = 0; vv < V; ++vv) total_sum += shared_buf[vv][value];
            grad_q[out_base + value] = total_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint key = 0; key < K; ++key) shared_buf[value][key] = local_grad_k[key];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (value < K) {
            float total_sum = 0.0f;
            for (uint vv = 0; vv < V; ++vv) total_sum += shared_buf[vv][value];
            grad_k[out_base + value] = total_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint key = 0; key < K; ++key) shared_buf[value][key] = local_grad_d[key];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (value < K) {
            float total_sum = 0.0f;
            for (uint vv = 0; vv < V; ++vv) total_sum += shared_buf[vv][value];
            grad_d[out_base + value] = total_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint key = 0; key < K; ++key) shared_buf[value][key] = local_grad_e[key];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (value < K) {
            float total_sum = 0.0f;
            for (uint vv = 0; vv < V; ++vv) total_sum += shared_buf[vv][value];
            grad_e[out_base + value] = total_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint key = 0; key < K; ++key)
        grad_initial[((batch * H + head) * V + value) * K + key] = grad_state[key];
}
"#;

#[repr(C)]
struct Gdn2ShapeUniform {
    b: u32,
    s: u32,
    h: u32,
    k: u32,
    v: u32,
}

pub struct MetalGdn2Forward {
    device: Device,
    pipeline: metal::ComputePipelineState,
    queue: metal::CommandQueue,
}

#[derive(Debug)]
pub struct Gdn2ForwardOutput {
    pub outputs: Vec<f32>,
    pub final_state: Vec<f32>,
}

impl MetalGdn2Forward {
    /// Fails (returns `Err`) if no Metal device is available (e.g. running
    /// in a CI/CPU-only environment) -- callers on Apple Silicon should not
    /// see this fail.
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device available")?;
        let library = device
            .new_library_with_source(FORWARD_SOURCE, &metal::CompileOptions::new())
            .map_err(|e| format!("Metal shader compilation failed: {e}"))?;
        let function = library
            .get_function("gdn2_forward", None)
            .map_err(|e| format!("could not find gdn2_forward function: {e}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("could not build compute pipeline: {e}"))?;
        let queue = device.new_command_queue();
        Ok(Self { device, pipeline, queue })
    }

    pub fn forward(
        &self,
        shape: &Gdn2ForwardShape,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        decay_logits: &[f32],
        erase_logits: &[f32],
        write_logits: &[f32],
        initial_state: &[f32],
    ) -> Gdn2ForwardOutput {
        let opts = MTLResourceOptions::StorageModeShared;
        let make_input = |data: &[f32]| {
            self.device.new_buffer_with_data(data.as_ptr() as *const std::ffi::c_void, (data.len() * std::mem::size_of::<f32>()) as u64, opts)
        };
        let buf_q = make_input(q);
        let buf_k = make_input(k);
        let buf_v = make_input(v);
        let buf_d = make_input(decay_logits);
        let buf_e = make_input(erase_logits);
        let buf_w = make_input(write_logits);
        let buf_initial = make_input(initial_state);

        let output_len = shape.batch * shape.seq * shape.heads * shape.value_dim;
        let state_len = shape.batch * shape.heads * shape.value_dim * shape.key_dim;
        let buf_y = self.device.new_buffer((output_len * std::mem::size_of::<f32>()) as u64, opts);
        let buf_final_state = self.device.new_buffer((state_len * std::mem::size_of::<f32>()) as u64, opts);

        let shape_uniform = Gdn2ShapeUniform {
            b: shape.batch as u32,
            s: shape.seq as u32,
            h: shape.heads as u32,
            k: shape.key_dim as u32,
            v: shape.value_dim as u32,
        };
        let buf_shape = self.device.new_buffer_with_data(
            &shape_uniform as *const Gdn2ShapeUniform as *const std::ffi::c_void,
            std::mem::size_of::<Gdn2ShapeUniform>() as u64,
            opts,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&buf_q), 0);
        encoder.set_buffer(1, Some(&buf_k), 0);
        encoder.set_buffer(2, Some(&buf_v), 0);
        encoder.set_buffer(3, Some(&buf_d), 0);
        encoder.set_buffer(4, Some(&buf_e), 0);
        encoder.set_buffer(5, Some(&buf_w), 0);
        encoder.set_buffer(6, Some(&buf_initial), 0);
        encoder.set_buffer(7, Some(&buf_y), 0);
        encoder.set_buffer(8, Some(&buf_final_state), 0);
        encoder.set_buffer(9, Some(&buf_shape), 0);

        let total_threads = (shape.batch * shape.heads * shape.value_dim) as u64;
        let threadgroup_size = total_threads.min(256);
        encoder.dispatch_threads(MTLSize::new(total_threads, 1, 1), MTLSize::new(threadgroup_size.max(1), 1, 1));
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let outputs = unsafe { std::slice::from_raw_parts(buf_y.contents() as *const f32, output_len) }.to_vec();
        let final_state = unsafe { std::slice::from_raw_parts(buf_final_state.contents() as *const f32, state_len) }.to_vec();
        Gdn2ForwardOutput { outputs, final_state }
    }
}

#[derive(Debug)]
pub struct Gdn2BackwardOutput {
    pub grad_q: Vec<f32>,
    pub grad_k: Vec<f32>,
    pub grad_v: Vec<f32>,
    /// Gradients w.r.t. the *activated* (post-sigmoid) gates -- matching
    /// the kernel body's own contract (same as the MLX-dispatched version
    /// this was ported from). Apply `* gate * (1 - gate)` to get gradients
    /// w.r.t. the raw pre-sigmoid logits, matching `gdn2_backward_f32`'s
    /// convention.
    pub grad_decay_activated: Vec<f32>,
    pub grad_erase_activated: Vec<f32>,
    pub grad_write_activated: Vec<f32>,
    pub grad_initial_state: Vec<f32>,
}

pub struct MetalGdn2Backward {
    device: Device,
    pipeline: metal::ComputePipelineState,
    queue: metal::CommandQueue,
}

impl MetalGdn2Backward {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device available")?;
        let library = device
            .new_library_with_source(BACKWARD_SOURCE, &metal::CompileOptions::new())
            .map_err(|e| format!("Metal backward shader compilation failed: {e}"))?;
        let function = library
            .get_function("gdn2_backward", None)
            .map_err(|e| format!("could not find gdn2_backward function: {e}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("could not build backward compute pipeline: {e}"))?;
        let queue = device.new_command_queue();
        Ok(Self { device, pipeline, queue })
    }

    /// `decay_activated`/`erase_activated`/`write_activated` must already be
    /// post-sigmoid (matching the kernel body's own contract).
    #[allow(clippy::too_many_arguments)]
    pub fn backward(
        &self,
        shape: &Gdn2ForwardShape,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        decay_activated: &[f32],
        erase_activated: &[f32],
        write_activated: &[f32],
        initial_state: &[f32],
        grad_output: &[f32],
        grad_final: &[f32],
    ) -> Gdn2BackwardOutput {
        let opts = MTLResourceOptions::StorageModeShared;
        let make_input = |data: &[f32]| {
            self.device.new_buffer_with_data(data.as_ptr() as *const std::ffi::c_void, (data.len() * std::mem::size_of::<f32>()) as u64, opts)
        };
        let buf_q = make_input(q);
        let buf_k = make_input(k);
        let buf_v = make_input(v);
        let buf_d = make_input(decay_activated);
        let buf_e = make_input(erase_activated);
        let buf_w = make_input(write_activated);
        let buf_initial = make_input(initial_state);
        let buf_grad_output = make_input(grad_output);
        let buf_grad_final = make_input(grad_final);

        let qk_len = shape.batch * shape.seq * shape.heads * shape.key_dim;
        let v_len = shape.batch * shape.seq * shape.heads * shape.value_dim;
        let state_len = shape.batch * shape.heads * shape.value_dim * shape.key_dim;
        let make_output = |n: usize| self.device.new_buffer((n * std::mem::size_of::<f32>()) as u64, opts);
        let buf_grad_q = make_output(qk_len);
        let buf_grad_k = make_output(qk_len);
        let buf_grad_v = make_output(v_len);
        let buf_grad_d = make_output(qk_len);
        let buf_grad_e = make_output(qk_len);
        let buf_grad_w = make_output(v_len);
        let buf_grad_initial = make_output(state_len);

        let shape_uniform = Gdn2ShapeUniform {
            b: shape.batch as u32,
            s: shape.seq as u32,
            h: shape.heads as u32,
            k: shape.key_dim as u32,
            v: shape.value_dim as u32,
        };
        let buf_shape = self.device.new_buffer_with_data(
            &shape_uniform as *const Gdn2ShapeUniform as *const std::ffi::c_void,
            std::mem::size_of::<Gdn2ShapeUniform>() as u64,
            opts,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        for (index, buffer) in [
            &buf_q, &buf_k, &buf_v, &buf_d, &buf_e, &buf_w, &buf_initial, &buf_grad_output, &buf_grad_final, &buf_grad_q, &buf_grad_k, &buf_grad_v, &buf_grad_d,
            &buf_grad_e, &buf_grad_w, &buf_grad_initial,
        ]
        .into_iter()
        .enumerate()
        {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_buffer(16, Some(&buf_shape), 0);

        let groups = (shape.batch * shape.heads) as u64;
        let threadgroup_size = shape.value_dim as u64;
        encoder.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(threadgroup_size, 1, 1));
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let read = |buffer: &metal::Buffer, n: usize| unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, n) }.to_vec();
        Gdn2BackwardOutput {
            grad_q: read(&buf_grad_q, qk_len),
            grad_k: read(&buf_grad_k, qk_len),
            grad_v: read(&buf_grad_v, v_len),
            grad_decay_activated: read(&buf_grad_d, qk_len),
            grad_erase_activated: read(&buf_grad_e, qk_len),
            grad_write_activated: read(&buf_grad_w, v_len),
            grad_initial_state: read(&buf_grad_initial, state_len),
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-30.0, 30.0)).exp())
}

/// CPU-vs-GPU backward parity: `decay_logits`/`erase_logits`/`write_logits`
/// are RAW (pre-sigmoid), matching `gdn2_backward_f32`'s contract. Applies
/// the same sigmoid-derivative chain rule the MLX Python wrapper applies
/// (`grad * gate * (1 - gate)`) to convert the GPU kernel's activated-gate
/// gradients into the same raw-logit-gradient convention before comparing.
#[allow(clippy::too_many_arguments)]
pub fn compare_cpu_and_gpu_backward(
    shape: &Gdn2ForwardShape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay_logits: &[f32],
    erase_logits: &[f32],
    write_logits: &[f32],
    initial_state: &[f32],
    grad_output: &[f32],
    grad_final: &[f32],
) -> Result<[f32; 7], String> {
    let cpu = gdn2_backward_f32(shape, q, k, v, decay_logits, erase_logits, write_logits, initial_state, grad_output, grad_final)?;

    let activated = |logits: &[f32]| logits.iter().map(|&x| sigmoid(x)).collect::<Vec<f32>>();
    let decay_activated = activated(decay_logits);
    let erase_activated = activated(erase_logits);
    let write_activated = activated(write_logits);

    let gpu = MetalGdn2Backward::new()?;
    let gpu_result = gpu.backward(shape, q, k, v, &decay_activated, &erase_activated, &write_activated, initial_state, grad_output, grad_final);

    let chain_rule = |gradient: &[f32], gate: &[f32]| -> Vec<f32> { gradient.iter().zip(gate.iter()).map(|(g, a)| g * a * (1.0 - a)).collect() };
    let gpu_grad_decay_logits = chain_rule(&gpu_result.grad_decay_activated, &decay_activated);
    let gpu_grad_erase_logits = chain_rule(&gpu_result.grad_erase_activated, &erase_activated);
    let gpu_grad_write_logits = chain_rule(&gpu_result.grad_write_activated, &write_activated);

    let max_diff = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max) };
    Ok([
        max_diff(&gpu_result.grad_q, &cpu.grad_q),
        max_diff(&gpu_result.grad_k, &cpu.grad_k),
        max_diff(&gpu_result.grad_v, &cpu.grad_v),
        max_diff(&gpu_grad_decay_logits, &cpu.grad_decay_logits),
        max_diff(&gpu_grad_erase_logits, &cpu.grad_erase_logits),
        max_diff(&gpu_grad_write_logits, &cpu.grad_write_logits),
        max_diff(&gpu_result.grad_initial_state, &cpu.grad_initial_state),
    ])
}

/// CPU-vs-GPU parity check, exposed for tests and callers: runs both the
/// existing `hz0a-pmetal-kernel` CPU reference and this Metal dispatch on
/// the same inputs and returns the max absolute difference for each output.
pub fn compare_cpu_and_gpu(
    shape: &Gdn2ForwardShape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay_logits: &[f32],
    erase_logits: &[f32],
    write_logits: &[f32],
    initial_state: &[f32],
) -> Result<(f32, f32), String> {
    let gpu = MetalGdn2Forward::new()?;
    let gpu_result = gpu.forward(shape, q, k, v, decay_logits, erase_logits, write_logits, initial_state);
    let cpu_result = gdn2_forward_f32(shape, q, k, v, decay_logits, erase_logits, write_logits, initial_state)?;
    let max_output_diff = gpu_result
        .outputs
        .iter()
        .zip(cpu_result.outputs.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let max_state_diff = gpu_result
        .final_state
        .iter()
        .zip(cpu_result.final_state.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    Ok((max_output_diff, max_state_diff))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg_f32(state: &mut u64, scale: f32) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = ((*state >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
        u * 2.0 * scale
    }

    #[test]
    fn gpu_forward_matches_cpu_reference_small_shape() {
        let shape = Gdn2ForwardShape { batch: 2, seq: 5, heads: 3, key_dim: 4, value_dim: 4 };
        let mut state = 7u64;
        let qk_len = shape.batch * shape.seq * shape.heads * shape.key_dim;
        let v_len = shape.batch * shape.seq * shape.heads * shape.value_dim;
        let state_len = shape.batch * shape.heads * shape.value_dim * shape.key_dim;
        let make = |n: usize, s: &mut u64| (0..n).map(|_| lcg_f32(s, 1.0)).collect::<Vec<f32>>();
        let q = make(qk_len, &mut state);
        let k = make(qk_len, &mut state);
        let v = make(v_len, &mut state);
        let d = make(v_len, &mut state);
        let e = make(v_len, &mut state);
        let w = make(v_len, &mut state);
        let initial = make(state_len, &mut state);

        match compare_cpu_and_gpu(&shape, &q, &k, &v, &d, &e, &w, &initial) {
            Ok((max_output_diff, max_state_diff)) => {
                assert!(max_output_diff < 1e-3, "output diff too large: {max_output_diff}");
                assert!(max_state_diff < 1e-3, "state diff too large: {max_state_diff}");
            }
            Err(message) => panic!("Metal GPU test failed: {message}"),
        }
    }

    #[test]
    fn gpu_forward_matches_cpu_reference_locked_a1_shape() {
        // The real locked A1 shape (dim=768, 12 heads, key/value_dim=64),
        // but small batch/sequence to keep the test fast.
        let shape = Gdn2ForwardShape { batch: 1, seq: 8, heads: 12, key_dim: 64, value_dim: 64 };
        let mut state = 23u64;
        let qk_len = shape.batch * shape.seq * shape.heads * shape.key_dim;
        let v_len = shape.batch * shape.seq * shape.heads * shape.value_dim;
        let state_len = shape.batch * shape.heads * shape.value_dim * shape.key_dim;
        let make = |n: usize, s: &mut u64| (0..n).map(|_| lcg_f32(s, 0.5)).collect::<Vec<f32>>();
        let q = make(qk_len, &mut state);
        let k = make(qk_len, &mut state);
        let v = make(v_len, &mut state);
        let d = make(v_len, &mut state);
        let e = make(v_len, &mut state);
        let w = make(v_len, &mut state);
        let initial = make(state_len, &mut state);

        match compare_cpu_and_gpu(&shape, &q, &k, &v, &d, &e, &w, &initial) {
            Ok((max_output_diff, max_state_diff)) => {
                assert!(max_output_diff < 1e-2, "output diff too large at locked shape: {max_output_diff}");
                assert!(max_state_diff < 1e-2, "state diff too large at locked shape: {max_state_diff}");
            }
            Err(message) => panic!("Metal GPU test failed: {message}"),
        }
    }

    #[test]
    fn gpu_backward_matches_cpu_reference_small_shape() {
        let shape = Gdn2ForwardShape { batch: 2, seq: 6, heads: 3, key_dim: 4, value_dim: 4 };
        let mut state = 41u64;
        let qk_len = shape.batch * shape.seq * shape.heads * shape.key_dim;
        let v_len = shape.batch * shape.seq * shape.heads * shape.value_dim;
        let state_len = shape.batch * shape.heads * shape.value_dim * shape.key_dim;
        let make = |n: usize, s: &mut u64| (0..n).map(|_| lcg_f32(s, 1.0)).collect::<Vec<f32>>();
        let q = make(qk_len, &mut state);
        let k = make(qk_len, &mut state);
        let v = make(v_len, &mut state);
        let d = make(qk_len, &mut state);
        let e = make(qk_len, &mut state);
        let w = make(v_len, &mut state);
        let initial = make(state_len, &mut state);
        let grad_output = make(v_len, &mut state);
        let grad_final = make(state_len, &mut state);

        match compare_cpu_and_gpu_backward(&shape, &q, &k, &v, &d, &e, &w, &initial, &grad_output, &grad_final) {
            Ok(diffs) => {
                let names = ["grad_q", "grad_k", "grad_v", "grad_decay_logits", "grad_erase_logits", "grad_write_logits", "grad_initial_state"];
                for (name, diff) in names.iter().zip(diffs.iter()) {
                    assert!(*diff < 1e-3, "{name} diff too large: {diff}");
                }
            }
            Err(message) => panic!("Metal GPU backward test failed: {message}"),
        }
    }

    #[test]
    fn gpu_backward_matches_cpu_reference_locked_a1_shape() {
        let shape = Gdn2ForwardShape { batch: 1, seq: 8, heads: 12, key_dim: 64, value_dim: 64 };
        let mut state = 59u64;
        let qk_len = shape.batch * shape.seq * shape.heads * shape.key_dim;
        let v_len = shape.batch * shape.seq * shape.heads * shape.value_dim;
        let state_len = shape.batch * shape.heads * shape.value_dim * shape.key_dim;
        let make = |n: usize, s: &mut u64| (0..n).map(|_| lcg_f32(s, 0.5)).collect::<Vec<f32>>();
        let q = make(qk_len, &mut state);
        let k = make(qk_len, &mut state);
        let v = make(v_len, &mut state);
        let d = make(qk_len, &mut state);
        let e = make(qk_len, &mut state);
        let w = make(v_len, &mut state);
        let initial = make(state_len, &mut state);
        let grad_output = make(v_len, &mut state);
        let grad_final = make(state_len, &mut state);

        match compare_cpu_and_gpu_backward(&shape, &q, &k, &v, &d, &e, &w, &initial, &grad_output, &grad_final) {
            Ok(diffs) => {
                let names = ["grad_q", "grad_k", "grad_v", "grad_decay_logits", "grad_erase_logits", "grad_write_logits", "grad_initial_state"];
                for (name, diff) in names.iter().zip(diffs.iter()) {
                    assert!(*diff < 1e-2, "{name} diff too large at locked shape: {diff}");
                }
            }
            Err(message) => panic!("Metal GPU backward test failed: {message}"),
        }
    }
}
