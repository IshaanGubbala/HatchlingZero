//! A6 -> A8 narrow milestone: one complete recurrent block -- RMSNorm, the
//! packed in-projection, the GDN-2 recurrence, the out-projection, the
//! residual add, and an AdamW parameter update -- executing entirely on
//! Metal from Rust, chained through GPU buffers so no intermediate
//! activation (normed input, packed projection, unpacked Q/K/V/gates,
//! mixed recurrence output) is ever read back to the host between steps.
//! Only the block's actual I/O boundary crosses to host memory: the input
//! tokens/hidden state going in, the final output coming out, and the
//! upstream loss gradient going back in for backward.
//!
//! Math mirrors, element-for-element, the existing CPU references this
//! session already proved correct: `hz0a_pmetal_tensor::{RmsNorm, Linear,
//! Gdn2Block}` for forward/packing layout, and
//! `hz0a_pmetal_kernel::gdn2_backward_f32` for the raw-logit-convention
//! GDN-2 backward math (the existing `BACKWARD_SOURCE` kernel in this
//! crate takes pre-activated gates and expects the caller to do the
//! sigmoid/chain-rule on the host, which would reintroduce exactly the
//! host round-trip this milestone exists to remove -- so this module adds
//! a self-contained raw-logit variant instead of reusing that kernel).
//!
//! Deliberately NOT in scope here (per the plan's own A6->A8 sequencing,
//! and to avoid building a general-purpose Rust autograd engine no one
//! asked for): embeddings, the LM head, cross-entropy loss, causal
//! attention, and the MLP. Those stay on the existing CPU path in
//! `hz0a-pmetal-tensor` until a similar narrow milestone is scoped for
//! them.

use metal::{Device, MTLResourceOptions, MTLSize};

const KERNELS_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

static inline float hz_sigmoid(float x) {
    return 1.0f / (1.0f + metal::exp(-x));
}

// ---- RMSNorm ----------------------------------------------------------

kernel void rmsnorm_forward(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* out [[buffer(2)]],
    device float* inv_rms [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint col [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    threadgroup float partial[256];
    float local_sum = 0.0f;
    for (uint c = col; c < dim; c += threads_per_group) {
        float v = x[row * dim + c];
        local_sum += v * v;
    }
    partial[col] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (col < stride) partial[col] += partial[col + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float rms = 1.0f / sqrt(partial[0] / float(dim) + 1e-6f);
    if (col == 0) inv_rms[row] = rms;
    for (uint c = col; c < dim; c += threads_per_group) {
        out[row * dim + c] = x[row * dim + c] * rms * weight[c];
    }
}

// grad_x[row,c] and a per-element partial (g*x*inv_rms) that a later
// reduce_rows_sum call turns into grad_weight -- avoids atomics entirely,
// matching this session's established workaround for the broken
// atomic_fetch_add_explicit in this MLX/Metal environment.
kernel void rmsnorm_backward(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* inv_rms [[buffer(2)]],
    device const float* grad_out [[buffer(3)]],
    device float* grad_x [[buffer(4)]],
    device float* grad_weight_partial [[buffer(5)]],
    constant uint& dim [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint col [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]])
{
    threadgroup float partial[256];
    float local_dot = 0.0f;
    for (uint c = col; c < dim; c += threads_per_group) {
        local_dot += grad_out[row * dim + c] * weight[c] * x[row * dim + c];
    }
    partial[col] = local_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (col < stride) partial[col] += partial[col + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dot = partial[0];
    float rms = inv_rms[row];
    for (uint c = col; c < dim; c += threads_per_group) {
        float g = grad_out[row * dim + c];
        float xv = x[row * dim + c];
        grad_x[row * dim + c] = weight[c] * rms * g - weight[c] * xv * rms * rms * rms * dot / float(dim);
        grad_weight_partial[row * dim + c] = g * xv * rms;
    }
}

// (rows, dim) -> (dim,), one thread per column, summed sequentially over
// rows -- reused for RmsNorm's grad_weight and Linear's grad_bias.
kernel void reduce_rows_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& dim [[buffer(3)]],
    uint col [[thread_position_in_grid]])
{
    if (col >= dim) return;
    float total = 0.0f;
    for (uint row = 0; row < rows; ++row) total += input[row * dim + col];
    output[col] = total;
}

// ---- Linear (naive GEMM, ordinary operations before fusion) ----------

kernel void linear_forward(
    device const float* x [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint& rows [[buffer(4)]],
    constant uint& in_features [[buffer(5)]],
    constant uint& out_features [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = rows * out_features;
    if (tid >= total) return;
    uint row = tid / out_features;
    uint o = tid % out_features;
    float total_sum = bias[o];
    for (uint i = 0; i < in_features; ++i) total_sum += x[row * in_features + i] * weight[o * in_features + i];
    out[tid] = total_sum;
}

kernel void linear_backward_dx(
    device const float* grad_out [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* grad_x [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = rows * in_features;
    if (tid >= total) return;
    uint row = tid / in_features;
    uint i = tid % in_features;
    float total_sum = 0.0f;
    for (uint o = 0; o < out_features; ++o) total_sum += grad_out[row * out_features + o] * weight[o * in_features + i];
    grad_x[tid] = total_sum;
}

kernel void linear_backward_dw(
    device const float* grad_out [[buffer(0)]],
    device const float* x [[buffer(1)]],
    device float* grad_weight [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& in_features [[buffer(4)]],
    constant uint& out_features [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    uint total = out_features * in_features;
    if (tid >= total) return;
    uint o = tid / in_features;
    uint i = tid % in_features;
    float total_sum = 0.0f;
    for (uint row = 0; row < rows; ++row) total_sum += grad_out[row * out_features + o] * x[row * in_features + i];
    grad_weight[tid] = total_sum;
}

// ---- pack/unpack the GDN-2 in-projection (matches Gdn2Block::forward) -

kernel void unpack_gdn2_inputs(
    device const float* projected [[buffer(0)]],
    device float* q [[buffer(1)]],
    device float* k [[buffer(2)]],
    device float* v [[buffer(3)]],
    device float* decay_logits [[buffer(4)]],
    device float* erase_logits [[buffer(5)]],
    device float* write_logits [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant uint& d_k [[buffer(8)]],
    constant uint& d_v [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    uint per_head = 4 * d_k + 2 * d_v;
    uint width = heads * per_head;
    uint step = tid / heads;
    uint head = tid % heads;
    uint base = step * width + head * per_head;
    uint qk_base = (step * heads + head) * d_k;
    uint v_base = (step * heads + head) * d_v;
    uint offset = 2 * d_k + d_v;
    for (uint i = 0; i < d_k; ++i) {
        q[qk_base + i] = projected[base + i];
        k[qk_base + i] = projected[base + d_k + i];
        decay_logits[qk_base + i] = projected[base + offset + i];
        erase_logits[qk_base + i] = projected[base + offset + d_k + i];
    }
    for (uint i = 0; i < d_v; ++i) {
        v[v_base + i] = projected[base + 2 * d_k + i];
        write_logits[v_base + i] = projected[base + offset + 2 * d_k + i];
    }
}

kernel void repack_grad_gdn2_inputs(
    device const float* grad_q [[buffer(0)]],
    device const float* grad_k [[buffer(1)]],
    device const float* grad_v [[buffer(2)]],
    device const float* grad_decay_logits [[buffer(3)]],
    device const float* grad_erase_logits [[buffer(4)]],
    device const float* grad_write_logits [[buffer(5)]],
    device float* grad_projected [[buffer(6)]],
    constant uint& heads [[buffer(7)]],
    constant uint& d_k [[buffer(8)]],
    constant uint& d_v [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    uint per_head = 4 * d_k + 2 * d_v;
    uint width = heads * per_head;
    uint step = tid / heads;
    uint head = tid % heads;
    uint base = step * width + head * per_head;
    uint qk_base = (step * heads + head) * d_k;
    uint v_base = (step * heads + head) * d_v;
    uint offset = 2 * d_k + d_v;
    for (uint i = 0; i < d_k; ++i) {
        grad_projected[base + i] = grad_q[qk_base + i];
        grad_projected[base + d_k + i] = grad_k[qk_base + i];
        grad_projected[base + offset + i] = grad_decay_logits[qk_base + i];
        grad_projected[base + offset + d_k + i] = grad_erase_logits[qk_base + i];
    }
    for (uint i = 0; i < d_v; ++i) {
        grad_projected[base + 2 * d_k + i] = grad_v[v_base + i];
        grad_projected[base + offset + 2 * d_k + i] = grad_write_logits[v_base + i];
    }
}

// ---- GDN-2 recurrence (forward: same O(S) scan already proven this
// session; backward: same math as hz0a_pmetal_kernel::gdn2_backward_f32,
// but self-contained on raw logits so no host sigmoid/chain-rule step is
// needed between forward and backward) -------------------------------

struct Gdn2Shape { uint B; uint S; uint H; uint K; uint V; };

kernel void gdn2_forward_block(
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
    uint B = shape.B; uint S = shape.S; uint H = shape.H; uint K = shape.K; uint V = shape.V;
    uint value = tid % V;
    uint head = (tid / V) % H;
    uint batch = tid / (V * H);
    if (batch >= B || K > 64) return;

    thread float state[64];
    uint state_base = ((batch * H + head) * V + value) * K;
    for (uint key = 0; key < K; ++key) state[key] = initial[state_base + key];

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
    for (uint key = 0; key < K; ++key) final_state[state_base + key] = state[key];
}

kernel void gdn2_backward_raw(
    device const float* q [[buffer(0)]],
    device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* decay_logits [[buffer(3)]],
    device const float* erase_logits [[buffer(4)]],
    device const float* write_logits [[buffer(5)]],
    device const float* initial [[buffer(6)]],
    device const float* grad_output [[buffer(7)]],
    device const float* grad_final [[buffer(8)]],
    device float* grad_q [[buffer(9)]],
    device float* grad_k [[buffer(10)]],
    device float* grad_v [[buffer(11)]],
    device float* grad_decay_logits [[buffer(12)]],
    device float* grad_erase_logits [[buffer(13)]],
    device float* grad_write_logits [[buffer(14)]],
    device float* grad_initial [[buffer(15)]],
    constant Gdn2Shape& shape [[buffer(16)]],
    uint value [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]])
{
    uint B = shape.B; uint S = shape.S; uint H = shape.H; uint K = shape.K; uint V = shape.V;
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
        float w = hz_sigmoid(write_logits[value_base + value]);
        for (uint key = 0; key < K; ++key) {
            float d = hz_sigmoid(decay_logits[input_base + key]);
            float e = hz_sigmoid(erase_logits[input_base + key]);
            states[t + 1][key] = d * (1.0f - e) * states[t][key] + w * v[value_base + value] * k[input_base + key];
        }
    }
    thread float grad_state[64];
    for (uint key = 0; key < K; ++key)
        grad_state[key] = grad_final[((batch * H + head) * V + value) * K + key];

    for (int reverse = int(S) - 1; reverse >= 0; --reverse) {
        uint t = uint(reverse);
        uint input_base = ((batch * S + t) * H + head) * K;
        uint value_base = ((batch * S + t) * H + head) * V;
        float w = hz_sigmoid(write_logits[value_base + value]);
        thread float local_grad_q[64];
        thread float local_grad_k[64];
        thread float local_grad_decay[64];
        thread float local_grad_erase[64];
        float value_gradient = 0.0f;
        float write_gradient = 0.0f;
        for (uint key = 0; key < K; ++key) {
            float d = hz_sigmoid(decay_logits[input_base + key]);
            float e = hz_sigmoid(erase_logits[input_base + key]);
            float a = d * (1.0f - e);
            float total = grad_state[key] + grad_output[value_base + value] * q[input_base + key];
            local_grad_q[key] = grad_output[value_base + value] * states[t + 1][key];
            local_grad_k[key] = total * w * v[value_base + value];
            local_grad_decay[key] = total * states[t][key] * (1.0f - e) * d * (1.0f - d);
            local_grad_erase[key] = total * states[t][key] * (-d) * e * (1.0f - e);
            value_gradient += total * w * k[input_base + key];
            write_gradient += total * v[value_base + value] * k[input_base + key];
            grad_state[key] = total * a;
        }
        grad_v[value_base + value] = value_gradient;
        grad_write_logits[value_base + value] = write_gradient * w * (1.0f - w);

        uint out_base = ((batch * S + t) * H + head) * K;

        for (uint key = 0; key < K; ++key) shared_buf[value][key] = local_grad_q[key];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (value < K) { float total_sum = 0.0f; for (uint vv = 0; vv < V; ++vv) total_sum += shared_buf[vv][value]; grad_q[out_base + value] = total_sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint key = 0; key < K; ++key) shared_buf[value][key] = local_grad_k[key];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (value < K) { float total_sum = 0.0f; for (uint vv = 0; vv < V; ++vv) total_sum += shared_buf[vv][value]; grad_k[out_base + value] = total_sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint key = 0; key < K; ++key) shared_buf[value][key] = local_grad_decay[key];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (value < K) { float total_sum = 0.0f; for (uint vv = 0; vv < V; ++vv) total_sum += shared_buf[vv][value]; grad_decay_logits[out_base + value] = total_sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint key = 0; key < K; ++key) shared_buf[value][key] = local_grad_erase[key];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (value < K) { float total_sum = 0.0f; for (uint vv = 0; vv < V; ++vv) total_sum += shared_buf[vv][value]; grad_erase_logits[out_base + value] = total_sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint key = 0; key < K; ++key)
        grad_initial[((batch * H + head) * V + value) * K + key] = grad_state[key];
}

// ---- residual + AdamW --------------------------------------------------

kernel void elementwise_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    out[tid] = a[tid] + b[tid];
}

kernel void adamw_update(
    device float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant float& lr [[buffer(4)]],
    constant float& beta1 [[buffer(5)]],
    constant float& beta2 [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    constant float& weight_decay [[buffer(8)]],
    constant float& bias_correction1 [[buffer(9)]],
    constant float& bias_correction2 [[buffer(10)]],
    uint tid [[thread_position_in_grid]])
{
    float g = grad[tid];
    float new_m = beta1 * m[tid] + (1.0f - beta1) * g;
    float new_v = beta2 * v[tid] + (1.0f - beta2) * g * g;
    m[tid] = new_m;
    v[tid] = new_v;
    float m_hat = new_m / bias_correction1;
    float v_hat = new_v / bias_correction2;
    param[tid] = param[tid] - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param[tid]);
}
"#;

fn make_pipeline(device: &Device, library: &metal::Library, name: &str) -> metal::ComputePipelineState {
    let function = library.get_function(name, None).unwrap_or_else(|e| panic!("missing kernel {name}: {e}"));
    device.new_compute_pipeline_state_with_function(&function).unwrap_or_else(|e| panic!("failed to build pipeline {name}: {e}"))
}

pub struct Gdn2FullBlockGpu {
    device: Device,
    queue: metal::CommandQueue,
    rmsnorm_forward: metal::ComputePipelineState,
    rmsnorm_backward: metal::ComputePipelineState,
    reduce_rows_sum: metal::ComputePipelineState,
    linear_forward: metal::ComputePipelineState,
    linear_backward_dx: metal::ComputePipelineState,
    linear_backward_dw: metal::ComputePipelineState,
    unpack_gdn2_inputs: metal::ComputePipelineState,
    repack_grad_gdn2_inputs: metal::ComputePipelineState,
    gdn2_forward_block: metal::ComputePipelineState,
    gdn2_backward_raw: metal::ComputePipelineState,
    elementwise_add: metal::ComputePipelineState,
    adamw_update: metal::ComputePipelineState,
}

pub struct BlockShape {
    pub steps: usize,
    pub dim: usize,
    pub heads: usize,
    pub d_k: usize,
    pub d_v: usize,
}

impl BlockShape {
    fn width(&self) -> usize {
        self.heads * (4 * self.d_k + 2 * self.d_v)
    }
    fn qk_len(&self) -> usize {
        self.steps * self.heads * self.d_k
    }
    fn v_len(&self) -> usize {
        self.steps * self.heads * self.d_v
    }
    fn state_len(&self) -> usize {
        self.heads * self.d_v * self.d_k
    }
}

/// Parameters (and, for AdamW, optimizer moment buffers) for one block, as
/// host-owned `Vec<f32>` -- these are small enough (one block's worth of
/// weights) that keeping the canonical copy on the host between training
/// steps is the params, not an "intermediate activation"; only what
/// happens *within* one forward/backward/update call stays GPU-resident.
pub struct BlockParameters {
    pub rmsnorm_weight: Vec<f32>,
    pub in_proj_weight: Vec<f32>,
    pub in_proj_bias: Vec<f32>,
    pub out_proj_weight: Vec<f32>,
    pub out_proj_bias: Vec<f32>,
}

pub struct AdamWMoments {
    pub m: BlockParameters,
    pub v: BlockParameters,
    pub step: u32,
}

impl AdamWMoments {
    pub fn zeros_like(params: &BlockParameters) -> Self {
        let zeros = |v: &Vec<f32>| vec![0.0f32; v.len()];
        Self {
            m: BlockParameters {
                rmsnorm_weight: zeros(&params.rmsnorm_weight),
                in_proj_weight: zeros(&params.in_proj_weight),
                in_proj_bias: zeros(&params.in_proj_bias),
                out_proj_weight: zeros(&params.out_proj_weight),
                out_proj_bias: zeros(&params.out_proj_bias),
            },
            v: BlockParameters {
                rmsnorm_weight: zeros(&params.rmsnorm_weight),
                in_proj_weight: zeros(&params.in_proj_weight),
                in_proj_bias: zeros(&params.in_proj_bias),
                out_proj_weight: zeros(&params.out_proj_weight),
                out_proj_bias: zeros(&params.out_proj_bias),
            },
            step: 0,
        }
    }
}

pub struct ForwardOutput {
    pub y: Vec<f32>,
    pub final_state: Vec<f32>,
}

impl Gdn2FullBlockGpu {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device available")?;
        let library = device.new_library_with_source(KERNELS_SOURCE, &metal::CompileOptions::new()).map_err(|e| format!("Metal shader compilation failed: {e}"))?;
        let queue = device.new_command_queue();
        Ok(Self {
            rmsnorm_forward: make_pipeline(&device, &library, "rmsnorm_forward"),
            rmsnorm_backward: make_pipeline(&device, &library, "rmsnorm_backward"),
            reduce_rows_sum: make_pipeline(&device, &library, "reduce_rows_sum"),
            linear_forward: make_pipeline(&device, &library, "linear_forward"),
            linear_backward_dx: make_pipeline(&device, &library, "linear_backward_dx"),
            linear_backward_dw: make_pipeline(&device, &library, "linear_backward_dw"),
            unpack_gdn2_inputs: make_pipeline(&device, &library, "unpack_gdn2_inputs"),
            repack_grad_gdn2_inputs: make_pipeline(&device, &library, "repack_grad_gdn2_inputs"),
            gdn2_forward_block: make_pipeline(&device, &library, "gdn2_forward_block"),
            gdn2_backward_raw: make_pipeline(&device, &library, "gdn2_backward_raw"),
            elementwise_add: make_pipeline(&device, &library, "elementwise_add"),
            adamw_update: make_pipeline(&device, &library, "adamw_update"),
            device,
            queue,
        })
    }

    fn buf_in(&self, data: &[f32]) -> metal::Buffer {
        self.device.new_buffer_with_data(data.as_ptr() as *const std::ffi::c_void, (data.len() * 4).max(4) as u64, MTLResourceOptions::StorageModeShared)
    }
    fn buf_out(&self, len: usize) -> metal::Buffer {
        self.device.new_buffer((len * 4).max(4) as u64, MTLResourceOptions::StorageModeShared)
    }
    fn read(buffer: &metal::Buffer, len: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, len) }.to_vec()
    }

    /// Runs RMSNorm -> in_proj -> unpack -> GDN-2 recurrence -> out_proj
    /// -> residual entirely as chained GPU buffers within one command
    /// buffer. `x` (the block's hidden-state input) and the returned `y`
    /// are the only activations that ever touch host memory; everything
    /// in between (normed, projected, q/k/v/gates, mixed) stays as
    /// `metal::Buffer` handles in the returned cache for backward to
    /// reuse without re-deriving or re-uploading them.
    pub fn forward(&self, shape: &BlockShape, x: &[f32], initial_state: &[f32], params: &BlockParameters) -> (ForwardOutput, ForwardCache) {
        assert_eq!(x.len(), shape.steps * shape.dim);
        assert_eq!(initial_state.len(), shape.state_len());
        let width = shape.width();

        let x_buf = self.buf_in(x);
        let rmsnorm_w_buf = self.buf_in(&params.rmsnorm_weight);
        let in_w_buf = self.buf_in(&params.in_proj_weight);
        let in_b_buf = self.buf_in(&params.in_proj_bias);
        let out_w_buf = self.buf_in(&params.out_proj_weight);
        let out_b_buf = self.buf_in(&params.out_proj_bias);
        let initial_buf = self.buf_in(initial_state);

        let normed_buf = self.buf_out(shape.steps * shape.dim);
        let inv_rms_buf = self.buf_out(shape.steps);
        let projected_buf = self.buf_out(shape.steps * width);
        let q_buf = self.buf_out(shape.qk_len());
        let k_buf = self.buf_out(shape.qk_len());
        let v_buf = self.buf_out(shape.v_len());
        let decay_buf = self.buf_out(shape.qk_len());
        let erase_buf = self.buf_out(shape.qk_len());
        let write_buf = self.buf_out(shape.v_len());
        let mixed_buf = self.buf_out(shape.v_len());
        let final_state_buf = self.buf_out(shape.state_len());
        let block_out_buf = self.buf_out(shape.steps * shape.dim);
        let y_buf = self.buf_out(shape.steps * shape.dim);

        let command_buffer = self.queue.new_command_buffer();

        // RMSNorm
        {
            let dim_buf = self.buf_in(&[shape.dim as f32]);
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.rmsnorm_forward);
            encoder.set_buffer(0, Some(&x_buf), 0);
            encoder.set_buffer(1, Some(&rmsnorm_w_buf), 0);
            encoder.set_buffer(2, Some(&normed_buf), 0);
            encoder.set_buffer(3, Some(&inv_rms_buf), 0);
            encoder.set_bytes(4, 4, [shape.dim as u32].as_ptr() as *const std::ffi::c_void);
            let threads = 256usize.min(shape.dim.next_power_of_two().max(1));
            encoder.dispatch_thread_groups(MTLSize::new(shape.steps as u64, 1, 1), MTLSize::new(threads as u64, 1, 1));
            encoder.end_encoding();
            drop(dim_buf);
        }
        // in_proj linear
        self.encode_linear_forward(command_buffer, &normed_buf, &in_w_buf, &in_b_buf, &projected_buf, shape.steps, shape.dim, width);
        // unpack
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.unpack_gdn2_inputs);
            encoder.set_buffer(0, Some(&projected_buf), 0);
            encoder.set_buffer(1, Some(&q_buf), 0);
            encoder.set_buffer(2, Some(&k_buf), 0);
            encoder.set_buffer(3, Some(&v_buf), 0);
            encoder.set_buffer(4, Some(&decay_buf), 0);
            encoder.set_buffer(5, Some(&erase_buf), 0);
            encoder.set_buffer(6, Some(&write_buf), 0);
            encoder.set_bytes(7, 4, [shape.heads as u32].as_ptr() as *const std::ffi::c_void);
            encoder.set_bytes(8, 4, [shape.d_k as u32].as_ptr() as *const std::ffi::c_void);
            encoder.set_bytes(9, 4, [shape.d_v as u32].as_ptr() as *const std::ffi::c_void);
            let total = (shape.steps * shape.heads) as u64;
            encoder.dispatch_threads(MTLSize::new(total, 1, 1), MTLSize::new(total.min(256).max(1), 1, 1));
            encoder.end_encoding();
        }
        // GDN-2 forward
        {
            #[repr(C)]
            struct Gdn2ShapeUniform { b: u32, s: u32, h: u32, k: u32, v: u32 }
            let shape_uniform = Gdn2ShapeUniform { b: 1, s: shape.steps as u32, h: shape.heads as u32, k: shape.d_k as u32, v: shape.d_v as u32 };
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.gdn2_forward_block);
            encoder.set_buffer(0, Some(&q_buf), 0);
            encoder.set_buffer(1, Some(&k_buf), 0);
            encoder.set_buffer(2, Some(&v_buf), 0);
            encoder.set_buffer(3, Some(&decay_buf), 0);
            encoder.set_buffer(4, Some(&erase_buf), 0);
            encoder.set_buffer(5, Some(&write_buf), 0);
            encoder.set_buffer(6, Some(&initial_buf), 0);
            encoder.set_buffer(7, Some(&mixed_buf), 0);
            encoder.set_buffer(8, Some(&final_state_buf), 0);
            encoder.set_bytes(9, std::mem::size_of::<Gdn2ShapeUniform>() as u64, &shape_uniform as *const _ as *const std::ffi::c_void);
            let total = (shape.heads * shape.d_v) as u64;
            encoder.dispatch_threads(MTLSize::new(total, 1, 1), MTLSize::new(total.min(256).max(1), 1, 1));
            encoder.end_encoding();
        }
        // out_proj linear
        self.encode_linear_forward(command_buffer, &mixed_buf, &out_w_buf, &out_b_buf, &block_out_buf, shape.steps, shape.heads * shape.d_v, shape.dim);
        // residual add
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.elementwise_add);
            encoder.set_buffer(0, Some(&x_buf), 0);
            encoder.set_buffer(1, Some(&block_out_buf), 0);
            encoder.set_buffer(2, Some(&y_buf), 0);
            let total = (shape.steps * shape.dim) as u64;
            encoder.dispatch_threads(MTLSize::new(total, 1, 1), MTLSize::new(total.min(256).max(1), 1, 1));
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let y = Self::read(&y_buf, shape.steps * shape.dim);
        let final_state = Self::read(&final_state_buf, shape.state_len());

        (
            ForwardOutput { y, final_state },
            ForwardCache { x_buf, rmsnorm_w_buf, normed_buf, inv_rms_buf, in_w_buf, q_buf, k_buf, v_buf, decay_buf, erase_buf, write_buf, initial_buf, mixed_buf, out_w_buf },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_linear_forward(&self, command_buffer: &metal::CommandBufferRef, x: &metal::Buffer, w: &metal::Buffer, b: &metal::Buffer, out: &metal::Buffer, rows: usize, in_features: usize, out_features: usize) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.linear_forward);
        encoder.set_buffer(0, Some(x), 0);
        encoder.set_buffer(1, Some(w), 0);
        encoder.set_buffer(2, Some(b), 0);
        encoder.set_buffer(3, Some(out), 0);
        encoder.set_bytes(4, 4, [rows as u32].as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, [in_features as u32].as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(6, 4, [out_features as u32].as_ptr() as *const std::ffi::c_void);
        let total = (rows * out_features) as u64;
        encoder.dispatch_threads(MTLSize::new(total, 1, 1), MTLSize::new(total.min(256).max(1), 1, 1));
        encoder.end_encoding();
    }

    /// Backward + in-place AdamW update, given the upstream gradient
    /// `grad_y` (dL/d block-output) and `grad_final_state`, both supplied
    /// by the caller (the block's I/O boundary -- not an intermediate).
    /// Mutates `params` and `moments` in place with the updated values.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_and_update(&self, shape: &BlockShape, cache: &ForwardCache, grad_y: &[f32], grad_final_state: &[f32], params: &mut BlockParameters, moments: &mut AdamWMoments, lr: f32) {
        assert_eq!(grad_y.len(), shape.steps * shape.dim);
        assert_eq!(grad_final_state.len(), shape.state_len());
        let width = shape.width();

        let grad_y_buf = self.buf_in(grad_y);
        let grad_final_buf = self.buf_in(grad_final_state);

        let grad_mixed_buf = self.buf_out(shape.v_len());
        let grad_out_w_buf = self.buf_out(shape.dim * shape.heads * shape.d_v);
        let grad_out_b_partial_buf = self.buf_out(shape.steps * shape.dim); // reduce_rows_sum input == grad_y itself, see below
        let grad_q_buf = self.buf_out(shape.qk_len());
        let grad_k_buf = self.buf_out(shape.qk_len());
        let grad_v_buf = self.buf_out(shape.v_len());
        let grad_decay_buf = self.buf_out(shape.qk_len());
        let grad_erase_buf = self.buf_out(shape.qk_len());
        let grad_write_buf = self.buf_out(shape.v_len());
        let grad_initial_buf = self.buf_out(shape.state_len());
        let grad_projected_buf = self.buf_out(shape.steps * width);
        let grad_normed_buf = self.buf_out(shape.steps * shape.dim);
        let grad_in_w_buf = self.buf_out(width * shape.dim);
        let grad_rmsnorm_w_partial_buf = self.buf_out(shape.steps * shape.dim);
        let grad_x_from_norm_buf = self.buf_out(shape.steps * shape.dim);
        let grad_x_total_buf = self.buf_out(shape.steps * shape.dim);

        let command_buffer = self.queue.new_command_buffer();

        // out_proj backward: dx (-> grad_mixed), dW
        self.encode_linear_backward_dx(command_buffer, &grad_y_buf, &cache.out_w_buf, &grad_mixed_buf, shape.steps, shape.heads * shape.d_v, shape.dim);
        self.encode_linear_backward_dw(command_buffer, &grad_y_buf, &cache.mixed_buf, &grad_out_w_buf, shape.steps, shape.heads * shape.d_v, shape.dim);
        let _ = &grad_out_b_partial_buf; // grad_out_bias reduction reads grad_y_buf directly, see below

        // GDN-2 backward (raw-logit, self-contained)
        {
            #[repr(C)]
            struct Gdn2ShapeUniform { b: u32, s: u32, h: u32, k: u32, v: u32 }
            let shape_uniform = Gdn2ShapeUniform { b: 1, s: shape.steps as u32, h: shape.heads as u32, k: shape.d_k as u32, v: shape.d_v as u32 };
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.gdn2_backward_raw);
            for (index, buffer) in [
                &cache.q_buf, &cache.k_buf, &cache.v_buf, &cache.decay_buf, &cache.erase_buf, &cache.write_buf, &cache.initial_buf, &grad_mixed_buf, &grad_final_buf,
                &grad_q_buf, &grad_k_buf, &grad_v_buf, &grad_decay_buf, &grad_erase_buf, &grad_write_buf, &grad_initial_buf,
            ]
            .into_iter()
            .enumerate()
            {
                encoder.set_buffer(index as u64, Some(buffer), 0);
            }
            encoder.set_bytes(16, std::mem::size_of::<Gdn2ShapeUniform>() as u64, &shape_uniform as *const _ as *const std::ffi::c_void);
            let groups = shape.heads as u64;
            encoder.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(shape.d_v as u64, 1, 1));
            encoder.end_encoding();
        }
        // repack grads back into (steps, width)
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.repack_grad_gdn2_inputs);
            encoder.set_buffer(0, Some(&grad_q_buf), 0);
            encoder.set_buffer(1, Some(&grad_k_buf), 0);
            encoder.set_buffer(2, Some(&grad_v_buf), 0);
            encoder.set_buffer(3, Some(&grad_decay_buf), 0);
            encoder.set_buffer(4, Some(&grad_erase_buf), 0);
            encoder.set_buffer(5, Some(&grad_write_buf), 0);
            encoder.set_buffer(6, Some(&grad_projected_buf), 0);
            encoder.set_bytes(7, 4, [shape.heads as u32].as_ptr() as *const std::ffi::c_void);
            encoder.set_bytes(8, 4, [shape.d_k as u32].as_ptr() as *const std::ffi::c_void);
            encoder.set_bytes(9, 4, [shape.d_v as u32].as_ptr() as *const std::ffi::c_void);
            let total = (shape.steps * shape.heads) as u64;
            encoder.dispatch_threads(MTLSize::new(total, 1, 1), MTLSize::new(total.min(256).max(1), 1, 1));
            encoder.end_encoding();
        }
        // in_proj backward: dx (-> grad_normed), dW
        self.encode_linear_backward_dx(command_buffer, &grad_projected_buf, &cache.in_w_buf, &grad_normed_buf, shape.steps, shape.dim, width);
        self.encode_linear_backward_dw(command_buffer, &grad_projected_buf, &cache.normed_buf, &grad_in_w_buf, shape.steps, shape.dim, width);

        // RMSNorm backward
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.rmsnorm_backward);
            encoder.set_buffer(0, Some(&cache.x_buf), 0);
            encoder.set_buffer(1, Some(&cache.rmsnorm_w_buf), 0);
            encoder.set_buffer(2, Some(&cache.inv_rms_buf), 0);
            encoder.set_buffer(3, Some(&grad_normed_buf), 0);
            encoder.set_buffer(4, Some(&grad_x_from_norm_buf), 0);
            encoder.set_buffer(5, Some(&grad_rmsnorm_w_partial_buf), 0);
            encoder.set_bytes(6, 4, [shape.dim as u32].as_ptr() as *const std::ffi::c_void);
            let threads = 256usize.min(shape.dim.next_power_of_two().max(1));
            encoder.dispatch_thread_groups(MTLSize::new(shape.steps as u64, 1, 1), MTLSize::new(threads as u64, 1, 1));
            encoder.end_encoding();
        }
        // residual: grad_x_total = grad_x_from_norm + grad_y (the skip connection)
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.elementwise_add);
            encoder.set_buffer(0, Some(&grad_x_from_norm_buf), 0);
            encoder.set_buffer(1, Some(&grad_y_buf), 0);
            encoder.set_buffer(2, Some(&grad_x_total_buf), 0);
            let total = (shape.steps * shape.dim) as u64;
            encoder.dispatch_threads(MTLSize::new(total, 1, 1), MTLSize::new(total.min(256).max(1), 1, 1));
            encoder.end_encoding();
        }

        // reductions: grad_rmsnorm_weight, grad_in_bias, grad_out_bias
        let grad_rmsnorm_w_buf = self.buf_out(shape.dim);
        self.encode_reduce_rows_sum(command_buffer, &grad_rmsnorm_w_partial_buf, &grad_rmsnorm_w_buf, shape.steps, shape.dim);
        let grad_in_b_buf = self.buf_out(width);
        self.encode_reduce_rows_sum(command_buffer, &grad_projected_buf, &grad_in_b_buf, shape.steps, width);
        let grad_out_b_buf = self.buf_out(shape.dim);
        self.encode_reduce_rows_sum(command_buffer, &grad_y_buf, &grad_out_b_buf, shape.steps, shape.dim);

        // AdamW updates, all five parameter tensors, entirely on-device
        moments.step += 1;
        let bc1 = 1.0f32 - 0.9f32.powi(moments.step as i32);
        let bc2 = 1.0f32 - 0.999f32.powi(moments.step as i32);
        let rmsnorm_w_param_buf = self.buf_in(&params.rmsnorm_weight);
        let rmsnorm_w_m_buf = self.buf_in(&moments.m.rmsnorm_weight);
        let rmsnorm_w_v_buf = self.buf_in(&moments.v.rmsnorm_weight);
        self.encode_adamw(command_buffer, &rmsnorm_w_param_buf, &grad_rmsnorm_w_buf, &rmsnorm_w_m_buf, &rmsnorm_w_v_buf, shape.dim, lr, bc1, bc2);

        let in_w_param_buf = self.buf_in(&params.in_proj_weight);
        let in_w_m_buf = self.buf_in(&moments.m.in_proj_weight);
        let in_w_v_buf = self.buf_in(&moments.v.in_proj_weight);
        self.encode_adamw(command_buffer, &in_w_param_buf, &grad_in_w_buf, &in_w_m_buf, &in_w_v_buf, width * shape.dim, lr, bc1, bc2);

        let in_b_param_buf = self.buf_in(&params.in_proj_bias);
        let in_b_m_buf = self.buf_in(&moments.m.in_proj_bias);
        let in_b_v_buf = self.buf_in(&moments.v.in_proj_bias);
        self.encode_adamw(command_buffer, &in_b_param_buf, &grad_in_b_buf, &in_b_m_buf, &in_b_v_buf, width, lr, bc1, bc2);

        let out_w_param_buf = self.buf_in(&params.out_proj_weight);
        let out_w_m_buf = self.buf_in(&moments.m.out_proj_weight);
        let out_w_v_buf = self.buf_in(&moments.v.out_proj_weight);
        self.encode_adamw(command_buffer, &out_w_param_buf, &grad_out_w_buf, &out_w_m_buf, &out_w_v_buf, shape.dim * shape.heads * shape.d_v, lr, bc1, bc2);

        let out_b_param_buf = self.buf_in(&params.out_proj_bias);
        let out_b_m_buf = self.buf_in(&moments.m.out_proj_bias);
        let out_b_v_buf = self.buf_in(&moments.v.out_proj_bias);
        self.encode_adamw(command_buffer, &out_b_param_buf, &grad_out_b_buf, &out_b_m_buf, &out_b_v_buf, shape.dim, lr, bc1, bc2);

        command_buffer.commit();
        command_buffer.wait_until_completed();

        params.rmsnorm_weight = Self::read(&rmsnorm_w_param_buf, shape.dim);
        params.in_proj_weight = Self::read(&in_w_param_buf, width * shape.dim);
        params.in_proj_bias = Self::read(&in_b_param_buf, width);
        params.out_proj_weight = Self::read(&out_w_param_buf, shape.dim * shape.heads * shape.d_v);
        params.out_proj_bias = Self::read(&out_b_param_buf, shape.dim);
        moments.m.rmsnorm_weight = Self::read(&rmsnorm_w_m_buf, shape.dim);
        moments.v.rmsnorm_weight = Self::read(&rmsnorm_w_v_buf, shape.dim);
        moments.m.in_proj_weight = Self::read(&in_w_m_buf, width * shape.dim);
        moments.v.in_proj_weight = Self::read(&in_w_v_buf, width * shape.dim);
        moments.m.in_proj_bias = Self::read(&in_b_m_buf, width);
        moments.v.in_proj_bias = Self::read(&in_b_v_buf, width);
        moments.m.out_proj_weight = Self::read(&out_w_m_buf, shape.dim * shape.heads * shape.d_v);
        moments.v.out_proj_weight = Self::read(&out_w_v_buf, shape.dim * shape.heads * shape.d_v);
        moments.m.out_proj_bias = Self::read(&out_b_m_buf, shape.dim);
        moments.v.out_proj_bias = Self::read(&out_b_v_buf, shape.dim);

        let _ = grad_x_total_buf; // grad wrt block input, available for a caller that chains further blocks; unused by this narrow single-block milestone
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_linear_backward_dx(&self, command_buffer: &metal::CommandBufferRef, grad_out: &metal::Buffer, weight: &metal::Buffer, grad_x: &metal::Buffer, rows: usize, in_features: usize, out_features: usize) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.linear_backward_dx);
        encoder.set_buffer(0, Some(grad_out), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(grad_x), 0);
        encoder.set_bytes(3, 4, [rows as u32].as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, [in_features as u32].as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, [out_features as u32].as_ptr() as *const std::ffi::c_void);
        let total = (rows * in_features) as u64;
        encoder.dispatch_threads(MTLSize::new(total, 1, 1), MTLSize::new(total.min(256).max(1), 1, 1));
        encoder.end_encoding();
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_linear_backward_dw(&self, command_buffer: &metal::CommandBufferRef, grad_out: &metal::Buffer, x: &metal::Buffer, grad_w: &metal::Buffer, rows: usize, in_features: usize, out_features: usize) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.linear_backward_dw);
        encoder.set_buffer(0, Some(grad_out), 0);
        encoder.set_buffer(1, Some(x), 0);
        encoder.set_buffer(2, Some(grad_w), 0);
        encoder.set_bytes(3, 4, [rows as u32].as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, [in_features as u32].as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, [out_features as u32].as_ptr() as *const std::ffi::c_void);
        let total = (out_features * in_features) as u64;
        encoder.dispatch_threads(MTLSize::new(total, 1, 1), MTLSize::new(total.min(256).max(1), 1, 1));
        encoder.end_encoding();
    }

    fn encode_reduce_rows_sum(&self, command_buffer: &metal::CommandBufferRef, input: &metal::Buffer, output: &metal::Buffer, rows: usize, dim: usize) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.reduce_rows_sum);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(2, 4, [rows as u32].as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(3, 4, [dim as u32].as_ptr() as *const std::ffi::c_void);
        encoder.dispatch_threads(MTLSize::new(dim as u64, 1, 1), MTLSize::new((dim as u64).min(256).max(1), 1, 1));
        encoder.end_encoding();
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_adamw(&self, command_buffer: &metal::CommandBufferRef, param: &metal::Buffer, grad: &metal::Buffer, m: &metal::Buffer, v: &metal::Buffer, n: usize, lr: f32, bc1: f32, bc2: f32) {
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.adamw_update);
        encoder.set_buffer(0, Some(param), 0);
        encoder.set_buffer(1, Some(grad), 0);
        encoder.set_buffer(2, Some(m), 0);
        encoder.set_buffer(3, Some(v), 0);
        encoder.set_bytes(4, 4, [lr].as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(5, 4, [0.9f32].as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(6, 4, [0.999f32].as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(7, 4, [1e-8f32].as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(8, 4, [0.01f32].as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(9, 4, [bc1].as_ptr() as *const std::ffi::c_void);
        encoder.set_bytes(10, 4, [bc2].as_ptr() as *const std::ffi::c_void);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new((n as u64).min(256).max(1), 1, 1));
        encoder.end_encoding();
    }
}

/// GPU-resident intermediate activations from `forward`, kept alive as
/// `metal::Buffer` handles for `backward_and_update` to reuse directly --
/// never read back to the host in between.
pub struct ForwardCache {
    x_buf: metal::Buffer,
    rmsnorm_w_buf: metal::Buffer,
    normed_buf: metal::Buffer,
    inv_rms_buf: metal::Buffer,
    in_w_buf: metal::Buffer,
    q_buf: metal::Buffer,
    k_buf: metal::Buffer,
    v_buf: metal::Buffer,
    decay_buf: metal::Buffer,
    erase_buf: metal::Buffer,
    write_buf: metal::Buffer,
    initial_buf: metal::Buffer,
    mixed_buf: metal::Buffer,
    out_w_buf: metal::Buffer,
}
