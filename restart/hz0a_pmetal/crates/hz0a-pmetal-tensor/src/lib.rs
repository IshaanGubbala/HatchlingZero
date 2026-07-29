#![forbid(unsafe_code)]
//! Real (non-Torch, non-Python) Rust tensor execution for the HZ-0A tiny
//! model: embeddings, RMSNorm, linear projections, causal attention, SwiGLU
//! MLP, tied LM head, cross-entropy, and a GDN-2 block wrapping the existing
//! `hz0a-pmetal-kernel` flat-buffer recurrence, all with manual (non-autodiff)
//! backward passes -- matching `restart/hz0a_pmetal/python/native_*.py`'s
//! math exactly so PMetal and the simple reference can be compared directly.
//!
//! This is deliberately the "ordinary PMetal forward and manual backward"
//! step the A8 plan calls for before any Metal-fused work, and closes part
//! of A6's real device-tensor-execution gap (the GDN-2 core already had a
//! working flat-buffer kernel; embeddings/RMSNorm/attention/MLP/optimizer
//! around it did not exist in Rust at all before this).

use hz0a_pmetal_kernel::{gdn2_backward_f32, gdn2_forward_f32, Gdn2ForwardShape};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x.clamp(-30.0, 30.0)).exp())
}

/// Round an f32 to the precision bf16 actually has: bf16 is exactly the
/// upper 16 bits of an IEEE754 f32 (8 exponent bits, 7 mantissa bits,
/// same exponent range as f32, unlike float16). This truncates-with-
/// round-to-nearest-even to that 7-bit mantissa and widens back to f32,
/// which is bit-for-bit what a real f32->bf16->f32 round trip produces --
/// no external bf16 crate needed to test precision loss faithfully.
pub fn round_to_bf16(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let bits = x.to_bits();
    let rounding_bias = 0x0000_7FFFu32 + ((bits >> 16) & 1);
    let rounded = bits.wrapping_add(rounding_bias) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

/// Round every parameter's data (not gradients) to bf16 precision in place.
pub fn round_parameters_to_bf16(parameters: &mut [&mut Parameter]) {
    for parameter in parameters.iter_mut() {
        for value in parameter.data.iter_mut() {
            *value = round_to_bf16(*value);
        }
    }
}

/// Row-major (rows, cols) matmul: (m x k) @ (k x n) -> (m x n).
fn matmul(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            if av == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += av * b[p * n + j];
            }
        }
    }
    out
}

/// a^T @ b where a is (m x k) and b is (m x n) -> (k x n).
fn matmul_at_b(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), m * n);
    let mut out = vec![0.0f32; k * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            if av == 0.0 {
                continue;
            }
            for j in 0..n {
                out[p * n + j] += av * b[i * n + j];
            }
        }
    }
    out
}

/// a @ b^T where a is (m x k) and b is (n x k) -> (m x n).
fn matmul_a_bt(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), n * k);
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[j * k + p];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct Parameter {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    pub grad: Vec<f32>,
}

impl Parameter {
    pub fn zeros(name: &str, shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self { name: name.to_string(), shape, data: vec![0.0; n], grad: vec![0.0; n] }
    }

    pub fn filled(name: &str, shape: Vec<usize>, value: f32) -> Self {
        let n: usize = shape.iter().product();
        Self { name: name.to_string(), shape, data: vec![value; n], grad: vec![0.0; n] }
    }

    /// Deterministic pseudo-random init (LCG) so the crate has zero external
    /// dependencies; not cryptographic, only used for weight init.
    pub fn random(name: &str, shape: Vec<usize>, std: f32, seed: u64) -> Self {
        let n: usize = shape.iter().product();
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut data = vec![0.0f32; n];
        for slot in data.iter_mut() {
            // Box-Muller from two LCG-drawn uniforms in (0, 1).
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = ((state >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0);
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;
            let radius = (-2.0 * u1.ln()).sqrt();
            let angle = 2.0 * std::f64::consts::PI * u2;
            *slot = (radius * angle.cos() * std as f64) as f32;
        }
        Self { name: name.to_string(), shape, data, grad: vec![0.0; n] }
    }

    pub fn zero_grad(&mut self) {
        self.grad.iter_mut().for_each(|g| *g = 0.0);
    }
}

pub struct Linear {
    pub weight: Parameter,
    pub bias: Option<Parameter>,
    in_features: usize,
    out_features: usize,
    cache_input: Vec<f32>,
    cache_rows: usize,
}

impl Linear {
    pub fn new(name: &str, in_features: usize, out_features: usize, bias: bool, seed: u64) -> Self {
        Self {
            weight: Parameter::random(&format!("{name}.weight"), vec![out_features, in_features], 0.02, seed),
            bias: if bias { Some(Parameter::zeros(&format!("{name}.bias"), vec![out_features])) } else { None },
            in_features,
            out_features,
            cache_input: Vec::new(),
            cache_rows: 0,
        }
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut out = vec![&mut self.weight];
        if let Some(bias) = &mut self.bias {
            out.push(bias);
        }
        out
    }

    /// `x` is (rows, in_features) row-major; returns (rows, out_features).
    pub fn forward(&mut self, x: &[f32], rows: usize) -> Vec<f32> {
        assert_eq!(x.len(), rows * self.in_features);
        self.cache_input = x.to_vec();
        self.cache_rows = rows;
        let mut out = matmul_a_bt(x, rows, self.in_features, &self.weight.data, self.out_features);
        if let Some(bias) = &self.bias {
            for row in 0..rows {
                for col in 0..self.out_features {
                    out[row * self.out_features + col] += bias.data[col];
                }
            }
        }
        out
    }

    /// `grad_output` is (rows, out_features); returns grad wrt input, (rows, in_features).
    pub fn backward(&mut self, grad_output: &[f32]) -> Vec<f32> {
        let rows = self.cache_rows;
        assert_eq!(grad_output.len(), rows * self.out_features);
        let grad_weight = matmul_at_b(grad_output, rows, self.out_features, &self.cache_input, self.in_features);
        for (g, dw) in self.weight.grad.iter_mut().zip(grad_weight.iter()) {
            *g += dw;
        }
        if let Some(bias) = &mut self.bias {
            for row in 0..rows {
                for col in 0..self.out_features {
                    bias.grad[col] += grad_output[row * self.out_features + col];
                }
            }
        }
        matmul(grad_output, rows, self.out_features, &self.weight.data, self.in_features)
    }
}

pub struct RmsNorm {
    pub weight: Parameter,
    dim: usize,
    eps: f32,
    cache_x: Vec<f32>,
    cache_inv_rms: Vec<f32>,
    cache_rows: usize,
}

impl RmsNorm {
    pub fn new(name: &str, dim: usize) -> Self {
        Self { weight: Parameter::filled(&format!("{name}.weight"), vec![dim], 1.0), dim, eps: 1e-6, cache_x: Vec::new(), cache_inv_rms: Vec::new(), cache_rows: 0 }
    }

    pub fn forward(&mut self, x: &[f32], rows: usize) -> Vec<f32> {
        assert_eq!(x.len(), rows * self.dim);
        self.cache_x = x.to_vec();
        self.cache_rows = rows;
        self.cache_inv_rms = vec![0.0; rows];
        let mut out = vec![0.0f32; rows * self.dim];
        for row in 0..rows {
            let slice = &x[row * self.dim..(row + 1) * self.dim];
            let mean_sq = slice.iter().map(|v| v * v).sum::<f32>() / self.dim as f32;
            let inv_rms = 1.0 / (mean_sq + self.eps).sqrt();
            self.cache_inv_rms[row] = inv_rms;
            for col in 0..self.dim {
                out[row * self.dim + col] = slice[col] * inv_rms * self.weight.data[col];
            }
        }
        out
    }

    pub fn backward(&mut self, grad_output: &[f32]) -> Vec<f32> {
        let rows = self.cache_rows;
        let dim = self.dim;
        let mut grad_input = vec![0.0f32; rows * dim];
        for row in 0..rows {
            let x = &self.cache_x[row * dim..(row + 1) * dim];
            let g = &grad_output[row * dim..(row + 1) * dim];
            let inv_rms = self.cache_inv_rms[row];
            let mut dot = 0.0f32;
            for col in 0..dim {
                self.weight.grad[col] += g[col] * x[col] * inv_rms;
                dot += g[col] * self.weight.data[col] * x[col];
            }
            for col in 0..dim {
                grad_input[row * dim + col] =
                    self.weight.data[col] * inv_rms * g[col] - self.weight.data[col] * x[col] * inv_rms.powi(3) * dot / dim as f32;
            }
        }
        grad_input
    }
}

pub fn silu_forward(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v * sigmoid(v)).collect()
}

pub fn silu_backward(x: &[f32], grad_output: &[f32]) -> Vec<f32> {
    x.iter()
        .zip(grad_output.iter())
        .map(|(&v, &g)| {
            let s = sigmoid(v);
            g * s * (1.0 + v * (1.0 - s))
        })
        .collect()
}

pub struct SwiGlu {
    pub gate: Linear,
    pub up: Linear,
    pub down: Linear,
    cache_gate: Vec<f32>,
    cache_up: Vec<f32>,
}

impl SwiGlu {
    pub fn new(name: &str, dim: usize, d_ff: usize, seed: u64) -> Self {
        Self {
            gate: Linear::new(&format!("{name}.gate"), dim, d_ff, true, seed.wrapping_add(1)),
            up: Linear::new(&format!("{name}.up"), dim, d_ff, true, seed.wrapping_add(2)),
            down: Linear::new(&format!("{name}.down"), d_ff, dim, true, seed.wrapping_add(3)),
            cache_gate: Vec::new(),
            cache_up: Vec::new(),
        }
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut out = self.gate.parameters_mut();
        out.extend(self.up.parameters_mut());
        out.extend(self.down.parameters_mut());
        out
    }

    pub fn forward(&mut self, x: &[f32], rows: usize) -> Vec<f32> {
        self.cache_gate = self.gate.forward(x, rows);
        self.cache_up = self.up.forward(x, rows);
        let activated = silu_forward(&self.cache_gate);
        let product: Vec<f32> = activated.iter().zip(self.cache_up.iter()).map(|(a, u)| a * u).collect();
        self.down.forward(&product, rows)
    }

    pub fn backward(&mut self, grad_output: &[f32]) -> Vec<f32> {
        let grad_product = self.down.backward(grad_output);
        let activated = silu_forward(&self.cache_gate);
        let grad_gate_act: Vec<f32> = grad_product.iter().zip(self.cache_up.iter()).map(|(g, u)| g * u).collect();
        let grad_gate = silu_backward(&self.cache_gate, &grad_gate_act);
        let grad_up: Vec<f32> = grad_product.iter().zip(activated.iter()).map(|(g, a)| g * a).collect();
        let grad_from_gate = self.gate.backward(&grad_gate);
        let grad_from_up = self.up.backward(&grad_up);
        grad_from_gate.iter().zip(grad_from_up.iter()).map(|(a, b)| a + b).collect()
    }
}

pub struct Embedding {
    pub weight: Parameter,
    dim: usize,
    cache_ids: Vec<usize>,
}

impl Embedding {
    pub fn new(name: &str, vocab_size: usize, dim: usize, seed: u64) -> Self {
        Self { weight: Parameter::random(&format!("{name}.weight"), vec![vocab_size, dim], 0.02, seed), dim, cache_ids: Vec::new() }
    }

    pub fn forward(&mut self, ids: &[usize]) -> Vec<f32> {
        self.cache_ids = ids.to_vec();
        let mut out = vec![0.0f32; ids.len() * self.dim];
        for (row, &id) in ids.iter().enumerate() {
            out[row * self.dim..(row + 1) * self.dim].copy_from_slice(&self.weight.data[id * self.dim..(id + 1) * self.dim]);
        }
        out
    }

    pub fn backward(&mut self, grad_output: &[f32]) {
        for (row, &id) in self.cache_ids.iter().enumerate() {
            for col in 0..self.dim {
                self.weight.grad[id * self.dim + col] += grad_output[row * self.dim + col];
            }
        }
    }

    /// Tied LM head forward: hidden (rows, dim) @ weight^T (dim, vocab) -> (rows, vocab).
    pub fn lm_head_forward(&self, hidden: &[f32], rows: usize) -> Vec<f32> {
        let vocab = self.weight.shape[0];
        matmul_a_bt(hidden, rows, self.dim, &self.weight.data, vocab)
    }

    /// Tied LM head backward: accumulates into `self.weight.grad` (shared with the
    /// embedding table, matching the Python NativeTiedLMHead) and returns grad wrt hidden.
    pub fn lm_head_backward(&mut self, grad_logits: &[f32], hidden: &[f32], rows: usize) -> Vec<f32> {
        let vocab = self.weight.shape[0];
        let grad_weight = matmul_at_b(grad_logits, rows, vocab, hidden, self.dim);
        for (g, dw) in self.weight.grad.iter_mut().zip(grad_weight.iter()) {
            *g += dw;
        }
        matmul(grad_logits, rows, vocab, &self.weight.data, self.dim)
    }
}

pub fn cross_entropy_forward(logits: &[f32], targets: &[usize], rows: usize, vocab: usize) -> (f32, Vec<f32>) {
    let mut probabilities = vec![0.0f32; rows * vocab];
    let mut loss = 0.0f64;
    for row in 0..rows {
        let slice = &logits[row * vocab..(row + 1) * vocab];
        let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for col in 0..vocab {
            let e = (slice[col] - max).exp();
            probabilities[row * vocab + col] = e;
            sum += e;
        }
        for col in 0..vocab {
            probabilities[row * vocab + col] /= sum;
        }
        let p_target = probabilities[row * vocab + targets[row]].max(1e-30);
        loss += -(p_target as f64).ln();
    }
    ((loss / rows as f64) as f32, probabilities)
}

pub fn cross_entropy_backward(probabilities: &[f32], targets: &[usize], rows: usize, vocab: usize) -> Vec<f32> {
    let mut grad = probabilities.to_vec();
    for row in 0..rows {
        grad[row * vocab + targets[row]] -= 1.0;
    }
    let scale = 1.0 / rows as f32;
    grad.iter_mut().for_each(|g| *g *= scale);
    grad
}

pub struct CausalAttention {
    pub qkv: Linear,
    pub out: Linear,
    dim: usize,
    heads: usize,
    head_dim: usize,
    cache_q: Vec<f32>,
    cache_k: Vec<f32>,
    cache_v: Vec<f32>,
    cache_weights: Vec<f32>,
    cache_steps: usize,
}

impl CausalAttention {
    pub fn new(name: &str, dim: usize, heads: usize, seed: u64) -> Self {
        assert_eq!(dim % heads, 0, "attention dim must divide evenly by heads");
        Self {
            qkv: Linear::new(&format!("{name}.qkv"), dim, 3 * dim, true, seed.wrapping_add(1)),
            out: Linear::new(&format!("{name}.out"), dim, dim, true, seed.wrapping_add(2)),
            dim,
            heads,
            head_dim: dim / heads,
            cache_q: Vec::new(),
            cache_k: Vec::new(),
            cache_v: Vec::new(),
            cache_weights: Vec::new(),
            cache_steps: 0,
        }
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut out = self.qkv.parameters_mut();
        out.extend(self.out.parameters_mut());
        out
    }

    // Layout: batch=1 assumed by caller composition (blocks operate per-sequence);
    // x is (steps, dim). Index helper: [step][head][channel].
    fn idx(&self, step: usize, head: usize, channel: usize) -> usize {
        (step * self.heads + head) * self.head_dim + channel
    }

    pub fn forward(&mut self, x: &[f32], steps: usize) -> Vec<f32> {
        self.cache_steps = steps;
        let packed = self.qkv.forward(x, steps); // (steps, 3*dim)
        let mut q = vec![0.0f32; steps * self.dim];
        let mut k = vec![0.0f32; steps * self.dim];
        let mut v = vec![0.0f32; steps * self.dim];
        // Python: packed.reshape(steps, heads, 3*head_dim) then split(axis=-1)
        // into q/k/v -- groups BY HEAD first, then q/k/v within each head's
        // 3*head_dim slice (not "all heads' q, then all heads' k, then v").
        for step in 0..steps {
            for head in 0..self.heads {
                let head_base = step * 3 * self.dim + head * 3 * self.head_dim;
                for c in 0..self.head_dim {
                    q[self.idx(step, head, c)] = packed[head_base + c];
                    k[self.idx(step, head, c)] = packed[head_base + self.head_dim + c];
                    v[self.idx(step, head, c)] = packed[head_base + 2 * self.head_dim + c];
                }
            }
        }
        let scale = (self.head_dim as f32).powf(-0.5);
        let mut weights = vec![0.0f32; self.heads * steps * steps];
        for head in 0..self.heads {
            for t in 0..steps {
                let mut scores = vec![f32::NEG_INFINITY; steps];
                for s in 0..=t {
                    let mut acc = 0.0f32;
                    for c in 0..self.head_dim {
                        acc += q[self.idx(t, head, c)] * k[self.idx(s, head, c)];
                    }
                    scores[s] = acc * scale;
                }
                let max = scores[..=t].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for s in 0..=t {
                    let e = (scores[s] - max).exp();
                    weights[(head * steps + t) * steps + s] = e;
                    sum += e;
                }
                for s in 0..=t {
                    weights[(head * steps + t) * steps + s] /= sum;
                }
            }
        }
        let mut mixed = vec![0.0f32; steps * self.dim];
        for head in 0..self.heads {
            for t in 0..steps {
                for c in 0..self.head_dim {
                    let mut acc = 0.0f32;
                    for s in 0..=t {
                        acc += weights[(head * steps + t) * steps + s] * v[self.idx(s, head, c)];
                    }
                    mixed[self.idx(t, head, c)] = acc;
                }
            }
        }
        self.cache_q = q;
        self.cache_k = k;
        self.cache_v = v;
        self.cache_weights = weights;
        self.out.forward(&mixed, steps)
    }

    pub fn backward(&mut self, grad_output: &[f32]) -> Vec<f32> {
        let steps = self.cache_steps;
        let grad_mixed = self.out.backward(grad_output);
        let scale = (self.head_dim as f32).powf(-0.5);
        let mut grad_q = vec![0.0f32; steps * self.dim];
        let mut grad_k = vec![0.0f32; steps * self.dim];
        let mut grad_v = vec![0.0f32; steps * self.dim];
        for head in 0..self.heads {
            for t in 0..steps {
                let mut grad_weights_row = vec![0.0f32; t + 1];
                for s in 0..=t {
                    let mut acc = 0.0f32;
                    for c in 0..self.head_dim {
                        acc += grad_mixed[self.idx(t, head, c)] * self.cache_v[self.idx(s, head, c)];
                    }
                    grad_weights_row[s] = acc;
                }
                let weighted_dot: f32 = (0..=t).map(|s| grad_weights_row[s] * self.cache_weights[(head * steps + t) * steps + s]).sum();
                let mut grad_scores_row = vec![0.0f32; t + 1];
                for s in 0..=t {
                    let w = self.cache_weights[(head * steps + t) * steps + s];
                    grad_scores_row[s] = w * (grad_weights_row[s] - weighted_dot);
                }
                for c in 0..self.head_dim {
                    let mut acc_q = 0.0f32;
                    for s in 0..=t {
                        acc_q += grad_scores_row[s] * self.cache_k[self.idx(s, head, c)];
                    }
                    grad_q[self.idx(t, head, c)] = acc_q * scale;
                }
                for s in 0..=t {
                    for c in 0..self.head_dim {
                        grad_k[self.idx(s, head, c)] += grad_scores_row[s] * self.cache_q[self.idx(t, head, c)] * scale;
                        grad_v[self.idx(s, head, c)] += self.cache_weights[(head * steps + t) * steps + s] * grad_mixed[self.idx(t, head, c)];
                    }
                }
            }
        }
        let mut grad_packed = vec![0.0f32; steps * 3 * self.dim];
        for step in 0..steps {
            for head in 0..self.heads {
                let head_base = step * 3 * self.dim + head * 3 * self.head_dim;
                for c in 0..self.head_dim {
                    grad_packed[head_base + c] = grad_q[self.idx(step, head, c)];
                    grad_packed[head_base + self.head_dim + c] = grad_k[self.idx(step, head, c)];
                    grad_packed[head_base + 2 * self.head_dim + c] = grad_v[self.idx(step, head, c)];
                }
            }
        }
        self.qkv.backward(&grad_packed)
    }
}

pub struct Gdn2Block {
    pub in_proj: Linear,
    pub out_proj: Linear,
    heads: usize,
    d_k: usize,
    d_v: usize,
    cache_projected: Vec<f32>,
    cache_shape: Option<Gdn2ForwardShape>,
    cache_forward: Option<hz0a_pmetal_kernel::Gdn2ForwardOutput>,
    cache_initial_state: Vec<f32>,
}

impl Gdn2Block {
    pub fn new(name: &str, dim: usize, heads: usize, d_k: usize, d_v: usize, seed: u64) -> Self {
        let width = heads * (4 * d_k + 2 * d_v);
        let mut in_proj = Linear::new(&format!("{name}.in_proj"), dim, width, true, seed.wrapping_add(1));
        let start = heads * (2 * d_k + d_v);
        for i in start..start + heads * d_k {
            in_proj.bias.as_mut().unwrap().data[i] = 4.59512;
        }
        for i in start + heads * d_k..start + 2 * heads * d_k {
            in_proj.bias.as_mut().unwrap().data[i] = -4.59512;
        }
        for i in start + 2 * heads * d_k..width {
            in_proj.bias.as_mut().unwrap().data[i] = -4.59512;
        }
        let out_proj = Linear::new(&format!("{name}.out_proj"), heads * d_v, dim, true, seed.wrapping_add(2));
        Self { in_proj, out_proj, heads, d_k, d_v, cache_projected: Vec::new(), cache_shape: None, cache_forward: None, cache_initial_state: Vec::new() }
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut out = self.in_proj.parameters_mut();
        out.extend(self.out_proj.parameters_mut());
        out
    }

    /// `x` is (steps, dim). `initial_state` is (heads, d_v, d_k) flat, batch=1.
    pub fn forward(&mut self, x: &[f32], steps: usize, initial_state: &[f32]) -> Vec<f32> {
        let width = self.heads * (4 * self.d_k + 2 * self.d_v);
        let projected = self.in_proj.forward(x, steps);
        assert_eq!(projected.len(), steps * width);
        let per_head = 4 * self.d_k + 2 * self.d_v;
        let mut q = vec![0.0f32; steps * self.heads * self.d_k];
        let mut k = vec![0.0f32; steps * self.heads * self.d_k];
        let mut v = vec![0.0f32; steps * self.heads * self.d_v];
        let mut decay = vec![0.0f32; steps * self.heads * self.d_k];
        let mut erase = vec![0.0f32; steps * self.heads * self.d_k];
        let mut write = vec![0.0f32; steps * self.heads * self.d_v];
        for step in 0..steps {
            for head in 0..self.heads {
                let base = step * width + head * per_head;
                let qk_base = (step * self.heads + head) * self.d_k;
                let v_base = (step * self.heads + head) * self.d_v;
                q[qk_base..qk_base + self.d_k].copy_from_slice(&projected[base..base + self.d_k]);
                k[qk_base..qk_base + self.d_k].copy_from_slice(&projected[base + self.d_k..base + 2 * self.d_k]);
                v[v_base..v_base + self.d_v].copy_from_slice(&projected[base + 2 * self.d_k..base + 2 * self.d_k + self.d_v]);
                let offset = 2 * self.d_k + self.d_v;
                decay[qk_base..qk_base + self.d_k].copy_from_slice(&projected[base + offset..base + offset + self.d_k]);
                erase[qk_base..qk_base + self.d_k].copy_from_slice(&projected[base + offset + self.d_k..base + offset + 2 * self.d_k]);
                write[v_base..v_base + self.d_v].copy_from_slice(&projected[base + offset + 2 * self.d_k..base + per_head]);
            }
        }
        let shape = Gdn2ForwardShape { batch: 1, seq: steps, heads: self.heads, key_dim: self.d_k, value_dim: self.d_v };
        let result = gdn2_forward_f32(&shape, &q, &k, &v, &decay, &erase, &write, initial_state).expect("gdn2 forward shape mismatch");
        let mixed = result.outputs.clone();
        self.cache_projected = projected;
        self.cache_shape = Some(shape);
        self.cache_initial_state = initial_state.to_vec();
        let output = self.out_proj.forward(&mixed, steps);
        self.cache_forward = Some(result);
        output
    }

    /// Returns (grad_input, grad_initial_state).
    pub fn backward(&mut self, grad_output: &[f32], grad_final_state: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let shape = self.cache_shape.clone().unwrap();
        let steps = shape.seq;
        let width = self.heads * (4 * self.d_k + 2 * self.d_v);
        let per_head = 4 * self.d_k + 2 * self.d_v;
        let grad_mixed = self.out_proj.backward(grad_output);
        let mut q = vec![0.0f32; steps * self.heads * self.d_k];
        let mut k = vec![0.0f32; steps * self.heads * self.d_k];
        let mut v = vec![0.0f32; steps * self.heads * self.d_v];
        let mut decay = vec![0.0f32; steps * self.heads * self.d_k];
        let mut erase = vec![0.0f32; steps * self.heads * self.d_k];
        let mut write = vec![0.0f32; steps * self.heads * self.d_v];
        for step in 0..steps {
            for head in 0..self.heads {
                let base = step * width + head * per_head;
                let qk_base = (step * self.heads + head) * self.d_k;
                let v_base = (step * self.heads + head) * self.d_v;
                q[qk_base..qk_base + self.d_k].copy_from_slice(&self.cache_projected[base..base + self.d_k]);
                k[qk_base..qk_base + self.d_k].copy_from_slice(&self.cache_projected[base + self.d_k..base + 2 * self.d_k]);
                v[v_base..v_base + self.d_v].copy_from_slice(&self.cache_projected[base + 2 * self.d_k..base + 2 * self.d_k + self.d_v]);
                let offset = 2 * self.d_k + self.d_v;
                decay[qk_base..qk_base + self.d_k].copy_from_slice(&self.cache_projected[base + offset..base + offset + self.d_k]);
                erase[qk_base..qk_base + self.d_k].copy_from_slice(&self.cache_projected[base + offset + self.d_k..base + offset + 2 * self.d_k]);
                write[v_base..v_base + self.d_v].copy_from_slice(&self.cache_projected[base + offset + 2 * self.d_k..base + per_head]);
            }
        }
        let grads = gdn2_backward_f32(&shape, &q, &k, &v, &decay, &erase, &write, &self.cache_initial_state, &grad_mixed, grad_final_state)
            .expect("gdn2 backward shape mismatch");
        let mut grad_projected = vec![0.0f32; steps * width];
        for step in 0..steps {
            for head in 0..self.heads {
                let base = step * width + head * per_head;
                let qk_base = (step * self.heads + head) * self.d_k;
                let v_base = (step * self.heads + head) * self.d_v;
                grad_projected[base..base + self.d_k].copy_from_slice(&grads.grad_q[qk_base..qk_base + self.d_k]);
                grad_projected[base + self.d_k..base + 2 * self.d_k].copy_from_slice(&grads.grad_k[qk_base..qk_base + self.d_k]);
                grad_projected[base + 2 * self.d_k..base + 2 * self.d_k + self.d_v].copy_from_slice(&grads.grad_v[v_base..v_base + self.d_v]);
                let offset = 2 * self.d_k + self.d_v;
                grad_projected[base + offset..base + offset + self.d_k].copy_from_slice(&grads.grad_decay_logits[qk_base..qk_base + self.d_k]);
                grad_projected[base + offset + self.d_k..base + offset + 2 * self.d_k].copy_from_slice(&grads.grad_erase_logits[qk_base..qk_base + self.d_k]);
                grad_projected[base + offset + 2 * self.d_k..base + per_head].copy_from_slice(&grads.grad_write_logits[v_base..v_base + self.d_v]);
            }
        }
        let grad_input = self.in_proj.backward(&grad_projected);
        (grad_input, grads.grad_initial_state)
    }
}

pub enum Mixer {
    Gdn2(Gdn2Block),
    Attention(CausalAttention),
}

pub struct Block {
    pub norm1: RmsNorm,
    pub mixer: Mixer,
    pub norm2: RmsNorm,
    pub mlp: SwiGlu,
    cache_x: Vec<f32>,
    cache_residual: Vec<f32>,
}

impl Block {
    pub fn new_gdn2(name: &str, dim: usize, heads: usize, d_k: usize, d_v: usize, d_ff: usize, seed: u64) -> Self {
        Self {
            norm1: RmsNorm::new(&format!("{name}.norm1"), dim),
            mixer: Mixer::Gdn2(Gdn2Block::new(&format!("{name}.gdn2"), dim, heads, d_k, d_v, seed.wrapping_add(10))),
            norm2: RmsNorm::new(&format!("{name}.norm2"), dim),
            mlp: SwiGlu::new(&format!("{name}.mlp"), dim, d_ff, seed.wrapping_add(20)),
            cache_x: Vec::new(),
            cache_residual: Vec::new(),
        }
    }

    pub fn new_attention(name: &str, dim: usize, heads: usize, d_ff: usize, seed: u64) -> Self {
        Self {
            norm1: RmsNorm::new(&format!("{name}.norm1"), dim),
            mixer: Mixer::Attention(CausalAttention::new(&format!("{name}.attention"), dim, heads, seed.wrapping_add(10))),
            norm2: RmsNorm::new(&format!("{name}.norm2"), dim),
            mlp: SwiGlu::new(&format!("{name}.mlp"), dim, d_ff, seed.wrapping_add(20)),
            cache_x: Vec::new(),
            cache_residual: Vec::new(),
        }
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut out = self.norm1.weight_mut_vec();
        match &mut self.mixer {
            Mixer::Gdn2(block) => out.extend(block.parameters_mut()),
            Mixer::Attention(attn) => out.extend(attn.parameters_mut()),
        }
        out.extend(self.norm2.weight_mut_vec());
        out.extend(self.mlp.parameters_mut());
        out
    }

    /// Returns (output, next_state). `state` is `Some` only for GDN-2 blocks.
    pub fn forward(&mut self, x: &[f32], steps: usize, dim: usize, state: Option<&[f32]>) -> (Vec<f32>, Option<Vec<f32>>) {
        self.cache_x = x.to_vec();
        let normalized = self.norm1.forward(x, steps);
        let (mixed, next_state) = match &mut self.mixer {
            Mixer::Attention(attn) => (attn.forward(&normalized, steps), None),
            Mixer::Gdn2(block) => {
                let default_state;
                let initial = match state {
                    Some(s) => s,
                    None => {
                        default_state = vec![0.0f32; block.heads * block.d_v * block.d_k];
                        &default_state
                    }
                };
                let out = block.forward(&normalized, steps, initial);
                let next = block.cache_forward.as_ref().unwrap().final_state.clone();
                (out, Some(next))
            }
        };
        let residual: Vec<f32> = x.iter().zip(mixed.iter()).map(|(a, b)| a + b).collect();
        self.cache_residual = residual.clone();
        let mlp_out = self.mlp.forward(&self.norm2.forward(&residual, steps), steps);
        let _ = dim;
        let output: Vec<f32> = residual.iter().zip(mlp_out.iter()).map(|(a, b)| a + b).collect();
        (output, next_state)
    }

    /// Returns (grad_input, grad_initial_state) -- the latter `None` for attention blocks.
    pub fn backward(&mut self, grad_output: &[f32], grad_final_state: Option<&[f32]>) -> (Vec<f32>, Option<Vec<f32>>) {
        let grad_mlp = self.mlp.backward(grad_output);
        let grad_norm2 = self.norm2.backward(&grad_mlp);
        let grad_residual: Vec<f32> = grad_output.iter().zip(grad_norm2.iter()).map(|(a, b)| a + b).collect();
        let (grad_normed, grad_initial) = match &mut self.mixer {
            Mixer::Attention(attn) => (attn.backward(&grad_residual), None),
            Mixer::Gdn2(block) => {
                let default_grad;
                let gfs = match grad_final_state {
                    Some(g) => g,
                    None => {
                        default_grad = vec![0.0f32; block.heads * block.d_v * block.d_k];
                        &default_grad
                    }
                };
                let (grad_in, grad_init) = block.backward(&grad_residual, gfs);
                (grad_in, Some(grad_init))
            }
        };
        let grad_norm1 = self.norm1.backward(&grad_normed);
        let grad_input: Vec<f32> = grad_residual.iter().zip(grad_norm1.iter()).map(|(a, b)| a + b).collect();
        (grad_input, grad_initial)
    }
}

impl RmsNorm {
    fn weight_mut_vec(&mut self) -> Vec<&mut Parameter> {
        vec![&mut self.weight]
    }
}

pub struct TinyModel {
    pub embedding: Embedding,
    pub blocks: Vec<Block>,
    pub final_norm: RmsNorm,
    dim: usize,
}

impl TinyModel {
    pub fn new(vocab_size: usize, dim: usize, heads: usize, d_k: usize, d_v: usize, d_ff: usize, attention_indices: &[usize], layers: usize, seed: u64) -> Self {
        let embedding = Embedding::new("embedding", vocab_size, dim, seed);
        let mut blocks = Vec::with_capacity(layers);
        for index in 0..layers {
            let name = format!("blocks.{index}");
            let block_seed = seed.wrapping_add(1000 + index as u64 * 100);
            if attention_indices.contains(&index) {
                blocks.push(Block::new_attention(&name, dim, heads, d_ff, block_seed));
            } else {
                blocks.push(Block::new_gdn2(&name, dim, heads, d_k, d_v, d_ff, block_seed));
            }
        }
        let final_norm = RmsNorm::new("final_norm", dim);
        Self { embedding, blocks, final_norm, dim }
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter> {
        let mut out = vec![&mut self.embedding.weight];
        for block in &mut self.blocks {
            out.extend(block.parameters_mut());
        }
        out.extend(self.final_norm.weight_mut_vec());
        out
    }

    pub fn forward(&mut self, token_ids: &[usize]) -> Vec<f32> {
        self.forward_with_states(token_ids).0
    }

    /// Same as `forward`, but also returns each block's final recurrent
    /// state (`None` for attention blocks) -- exposed for cross-language
    /// state-parity checks (A6's validation checklist explicitly lists
    /// "recurrent states" alongside block outputs/logits/loss/gradients).
    pub fn forward_with_states(&mut self, token_ids: &[usize]) -> (Vec<f32>, Vec<Option<Vec<f32>>>) {
        let steps = token_ids.len();
        let mut x = self.embedding.forward(token_ids);
        let mut states = Vec::with_capacity(self.blocks.len());
        for block in &mut self.blocks {
            let (out, next_state) = block.forward(&x, steps, self.dim, None);
            x = out;
            states.push(next_state);
        }
        let hidden = self.final_norm.forward(&x, steps);
        (self.embedding.lm_head_forward(&hidden, steps), states)
    }

    /// One forward + full backward pass; returns the scalar loss. Zeroes no
    /// gradients itself -- call `zero_grad` first if accumulation isn't wanted.
    pub fn loss_and_backward(&mut self, token_ids: &[usize], targets: &[usize]) -> f32 {
        let steps = token_ids.len();
        let vocab = self.embedding.weight.shape[0];
        let mut x = self.embedding.forward(token_ids);
        for block in &mut self.blocks {
            let (out, _next_state) = block.forward(&x, steps, self.dim, None);
            x = out;
        }
        let hidden = self.final_norm.forward(&x, steps);
        let logits = self.embedding.lm_head_forward(&hidden, steps);
        let (loss, probabilities) = cross_entropy_forward(&logits, targets, steps, vocab);
        let grad_logits = cross_entropy_backward(&probabilities, targets, steps, vocab);
        let grad_hidden_from_head = self.embedding.lm_head_backward(&grad_logits, &hidden, steps);
        let mut grad_hidden = self.final_norm.backward(&grad_hidden_from_head);
        for block in self.blocks.iter_mut().rev() {
            let (grad_in, _grad_state) = block.backward(&grad_hidden, None);
            grad_hidden = grad_in;
        }
        self.embedding.backward(&grad_hidden);
        loss
    }

    pub fn zero_grad(&mut self) {
        for parameter in self.parameters_mut() {
            parameter.zero_grad();
        }
    }

    /// Load parameter values from a flat slice, in `parameters_mut()` order.
    /// Used for cross-language parity tests -- loading the exact weights the
    /// Python reference used, rather than relying on matching RNG output
    /// across languages (which two different RNG algorithms never give,
    /// even with "the same seed").
    pub fn load_flat_parameters(&mut self, values: &[f64]) {
        let mut offset = 0;
        for parameter in self.parameters_mut() {
            let len = parameter.data.len();
            for (slot, value) in parameter.data.iter_mut().zip(&values[offset..offset + len]) {
                *slot = *value as f32;
            }
            offset += len;
        }
        assert_eq!(offset, values.len(), "flat parameter count mismatch");
    }

    pub fn flat_gradients(&mut self) -> Vec<f64> {
        self.parameters_mut().iter().flat_map(|p| p.grad.iter().map(|&g| g as f64)).collect()
    }
}

pub struct AdamW {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    step: u64,
    moments: std::collections::HashMap<String, (Vec<f32>, Vec<f32>)>,
}

impl AdamW {
    pub fn new(lr: f32) -> Self {
        Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.01, step: 0, moments: std::collections::HashMap::new() }
    }

    pub fn new_with_weight_decay(lr: f32, weight_decay: f32) -> Self {
        Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay, step: 0, moments: std::collections::HashMap::new() }
    }

    pub fn update(&mut self, parameters: &mut [&mut Parameter]) {
        self.step += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);
        for parameter in parameters.iter_mut() {
            let entry = self.moments.entry(parameter.name.clone()).or_insert_with(|| (vec![0.0; parameter.data.len()], vec![0.0; parameter.data.len()]));
            let (m, v) = entry;
            for i in 0..parameter.data.len() {
                let g = parameter.grad[i];
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * g;
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * g * g;
                let m_hat = m[i] / bias_correction1;
                let v_hat = v[i] / bias_correction2;
                parameter.data[i] -= self.lr * (m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * parameter.data[i]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finite_difference_check<F: Fn(&mut TinyModel) -> f32>(model: &mut TinyModel, loss_fn: F, tolerance: f32) {
        let eps = 1e-3f32;
        model.zero_grad();
        let _ = model.loss_and_backward(&[0, 1, 2], &[1, 2, 0]);
        let mut checked = 0;
        let mut params_snapshot: Vec<(String, usize, f32, f32)> = Vec::new();
        for parameter in model.parameters_mut() {
            // Sample a handful of entries per tensor to keep the test fast.
            let stride = (parameter.data.len() / 3).max(1);
            for i in (0..parameter.data.len()).step_by(stride) {
                params_snapshot.push((parameter.name.clone(), i, parameter.data[i], parameter.grad[i]));
            }
        }
        for (name, index, _original, analytic_grad) in params_snapshot {
            let plus = {
                for parameter in model.parameters_mut() {
                    if parameter.name == name {
                        parameter.data[index] += eps;
                    }
                }
                loss_fn(model)
            };
            let minus = {
                for parameter in model.parameters_mut() {
                    if parameter.name == name {
                        parameter.data[index] -= 2.0 * eps;
                    }
                }
                loss_fn(model)
            };
            for parameter in model.parameters_mut() {
                if parameter.name == name {
                    parameter.data[index] += eps; // restore
                }
            }
            let numeric = (plus - minus) / (2.0 * eps);
            assert!(
                (numeric - analytic_grad).abs() < tolerance,
                "{name}[{index}]: analytic={analytic_grad} numeric={numeric}"
            );
            checked += 1;
        }
        assert!(checked > 0, "no parameters were checked");
    }

    #[test]
    fn linear_matches_finite_difference() {
        let mut layer = Linear::new("test", 3, 2, true, 1);
        let x = vec![0.5, -0.2, 0.1, 0.3, 0.7, -0.4];
        let out = layer.forward(&x, 2);
        let grad_output = vec![1.0, -1.0, 0.5, 0.5];
        let grad_input = layer.backward(&grad_output);
        assert_eq!(out.len(), 4);
        assert_eq!(grad_input.len(), 6);
        let eps = 1e-3f32;
        let mut plus_x = x.clone();
        plus_x[0] += eps;
        let mut minus_x = x.clone();
        minus_x[0] -= eps;
        let mut probe = Linear::new("test", 3, 2, true, 1);
        let out_plus = probe.forward(&plus_x, 2);
        let out_minus = probe.forward(&minus_x, 2);
        let numeric: f32 = out_plus.iter().zip(out_minus.iter()).zip(grad_output.iter()).map(|((p, m), g)| (p - m) / (2.0 * eps) * g).sum();
        assert!((numeric - grad_input[0]).abs() < 5e-3, "numeric={numeric} analytic={}", grad_input[0]);
    }

    #[test]
    fn rmsnorm_matches_finite_difference() {
        let mut norm = RmsNorm::new("test", 4);
        let x = vec![0.5, -0.2, 0.1, 0.3];
        let out = norm.forward(&x, 1);
        assert_eq!(out.len(), 4);
        let grad_output = vec![1.0, 0.5, -0.5, 0.25];
        let grad_input = norm.backward(&grad_output);
        let eps = 1e-3f32;
        let mut plus = x.clone();
        plus[1] += eps;
        let mut minus = x.clone();
        minus[1] -= eps;
        let mut probe = RmsNorm::new("test", 4);
        let out_plus = probe.forward(&plus, 1);
        let mut probe2 = RmsNorm::new("test", 4);
        let out_minus = probe2.forward(&minus, 1);
        let numeric: f32 = out_plus.iter().zip(out_minus.iter()).zip(grad_output.iter()).map(|((p, m), g)| (p - m) / (2.0 * eps) * g).sum();
        assert!((numeric - grad_input[1]).abs() < 5e-3, "numeric={numeric} analytic={}", grad_input[1]);
    }

    #[test]
    fn tiny_model_forward_produces_finite_logits() {
        let mut model = TinyModel::new(6, 4, 2, 2, 2, 8, &[1], 2, 42);
        let logits = model.forward(&[0, 1, 2]);
        assert_eq!(logits.len(), 3 * 6);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn tiny_model_gradients_match_finite_difference() {
        let mut model = TinyModel::new(5, 4, 2, 2, 2, 6, &[1], 2, 7);
        finite_difference_check(&mut model, |m| {
            let (loss, probs) = {
                let logits = m.forward(&[0, 1, 2]);
                cross_entropy_forward(&logits, &[1, 2, 0], 3, 5)
            };
            let _ = probs;
            loss
        }, 3e-2);
    }

    #[test]
    fn adamw_step_changes_parameters_and_stays_finite() {
        let mut model = TinyModel::new(6, 4, 2, 2, 2, 8, &[1], 2, 3);
        let mut optimizer = AdamW::new(1e-2);
        model.zero_grad();
        let loss_before = model.loss_and_backward(&[0, 1, 2], &[1, 2, 0]);
        let before: Vec<f32> = model.parameters_mut().iter().flat_map(|p| p.data.clone()).collect();
        optimizer.update(&mut model.parameters_mut());
        let after: Vec<f32> = model.parameters_mut().iter().flat_map(|p| p.data.clone()).collect();
        assert!(after.iter().all(|v| v.is_finite()));
        assert!(before.iter().zip(after.iter()).any(|(a, b)| (a - b).abs() > 1e-9));
        model.zero_grad();
        let loss_after = model.loss_and_backward(&[0, 1, 2], &[1, 2, 0]);
        assert!(loss_before.is_finite() && loss_after.is_finite());
    }

    #[test]
    fn all_gdn2_model_trains_multiple_steps() {
        let mut model = TinyModel::new(8, 4, 2, 2, 2, 8, &[], 3, 11);
        let mut optimizer = AdamW::new(5e-3);
        let mut last_loss = f32::INFINITY;
        for _ in 0..10 {
            model.zero_grad();
            let loss = model.loss_and_backward(&[0, 1, 2, 3], &[1, 2, 3, 0]);
            assert!(loss.is_finite());
            optimizer.update(&mut model.parameters_mut());
            last_loss = loss;
        }
        assert!(last_loss.is_finite());
    }

    #[test]
    fn round_to_bf16_matches_known_values() {
        // 1/3 in f32 is 0x3EAAAAAB; bf16 truncates/rounds to 0x3EAB (~0.333984375).
        let value = 1.0f32 / 3.0f32;
        let rounded = round_to_bf16(value);
        assert!((rounded - 0.333984375).abs() < 1e-9, "rounded={rounded}");
        assert_eq!(round_to_bf16(1.0), 1.0);
        assert_eq!(round_to_bf16(0.0), 0.0);
        assert_eq!(round_to_bf16(-2.5), -2.5); // exactly representable, no rounding needed
    }

    /// A6's plan text: "Use float32 for reference checks, then BF16." The
    /// float32 cross-language parity check lives in
    /// tests/parity_with_python_reference.rs; this validates the same
    /// full model (embedding + GDN-2 + attention + MLP + tied head +
    /// cross-entropy + AdamW) stays finite and trains under BF16-rounded
    /// weight precision, mirroring the same stability question already
    /// answered for the MLX/Metal path this session (float16 rejected for
    /// NaN, BF16 verified stable there too).
    #[test]
    fn bf16_precision_model_trains_and_stays_finite() {
        let mut model = TinyModel::new(9, 8, 2, 4, 4, 12, &[1], 3, 23);
        round_parameters_to_bf16(&mut model.parameters_mut());
        let mut optimizer = AdamW::new(1e-3);
        let mut last_loss = f32::INFINITY;
        for _ in 0..8 {
            model.zero_grad();
            let loss = model.loss_and_backward(&[0, 1, 2, 3, 4], &[1, 2, 3, 4, 0]);
            assert!(loss.is_finite(), "loss went non-finite at BF16 weight precision");
            for parameter in model.parameters_mut() {
                assert!(parameter.grad.iter().all(|g| g.is_finite()), "{}: non-finite gradient at BF16 precision", parameter.name);
            }
            optimizer.update(&mut model.parameters_mut());
            for parameter in model.parameters_mut() {
                assert!(parameter.data.iter().all(|v| v.is_finite()), "{}: non-finite parameter after AdamW at BF16 precision", parameter.name);
            }
            last_loss = loss;
        }
        assert!(last_loss.is_finite());
    }

    #[test]
    fn bf16_precision_forward_close_to_float32_forward() {
        // Not bit-identical (that's the point of testing precision loss),
        // but a BF16-rounded model should still be numerically close to its
        // float32 twin on the same input -- catches a BF16 path that's
        // finite but silently wrong (e.g. a rounding bug that discards a
        // whole tensor) without demanding exact agreement.
        let mut model_f32 = TinyModel::new(9, 8, 2, 4, 4, 12, &[1], 3, 23);
        let logits_f32 = model_f32.forward(&[0, 1, 2, 3, 4]);

        let mut model_bf16 = TinyModel::new(9, 8, 2, 4, 4, 12, &[1], 3, 23);
        round_parameters_to_bf16(&mut model_bf16.parameters_mut());
        let logits_bf16 = model_bf16.forward(&[0, 1, 2, 3, 4]);

        assert_eq!(logits_f32.len(), logits_bf16.len());
        let max_abs_logit = logits_f32.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        let max_diff = logits_f32.iter().zip(logits_bf16.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(
            max_diff < 0.15 * max_abs_logit.max(1.0),
            "BF16 forward diverged too far from float32: max_diff={max_diff}, max_abs_logit={max_abs_logit}"
        );
    }
}
