#![forbid(unsafe_code)]

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gdn2ForwardShape {
    pub batch: usize,
    pub seq: usize,
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Gdn2CacheSpec {
    pub saves_decay: bool,
    pub saves_erase: bool,
    pub saves_write: bool,
    pub saves_hidden_checkpoints: bool,
}

impl Default for Gdn2CacheSpec {
    fn default() -> Self {
        Self {
            saves_decay: true,
            saves_erase: true,
            saves_write: true,
            saves_hidden_checkpoints: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct A1OperatorSpec {
    pub model_dim: usize,
    pub num_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

impl Default for A1OperatorSpec {
    fn default() -> Self {
        Self {
            model_dim: 768,
            num_heads: 12,
            key_dim: 64,
            value_dim: 64,
        }
    }
}

impl A1OperatorSpec {
    pub fn state_shape(&self, batch: usize) -> (usize, usize, usize, usize) {
        (batch, self.num_heads, self.value_dim, self.key_dim)
    }

    pub fn a1_recurrent_input_width(&self) -> usize {
        self.num_heads * (4 * self.key_dim + 2 * self.value_dim)
    }
}

pub fn restart_kernel_scope() -> &'static str {
    "hz0a-pmetal-kernel"
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gdn2ForwardOutput {
    pub outputs: Vec<f32>,
    pub final_state: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gdn2BackwardOutput {
    pub grad_q: Vec<f32>,
    pub grad_k: Vec<f32>,
    pub grad_v: Vec<f32>,
    pub grad_decay_logits: Vec<f32>,
    pub grad_erase_logits: Vec<f32>,
    pub grad_write_logits: Vec<f32>,
    pub grad_initial_state: Vec<f32>,
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value.clamp(-30.0, 30.0)).exp())
}

fn expected_len(shape: &Gdn2ForwardShape, width: usize) -> usize {
    shape.batch * shape.seq * shape.heads * width
}

/// Dependency-free CPU reference for the PMetal GDN-2 forward contract.
/// Tensor layout is [batch, sequence, heads, channel] and state is
/// [batch, heads, value, key].
pub fn gdn2_forward_f32(
    shape: &Gdn2ForwardShape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay_logits: &[f32],
    erase_logits: &[f32],
    write_logits: &[f32],
    initial_state: &[f32],
) -> Result<Gdn2ForwardOutput, String> {
    let q_len = expected_len(shape, shape.key_dim);
    let v_len = expected_len(shape, shape.value_dim);
    let state_len = shape.batch * shape.heads * shape.value_dim * shape.key_dim;
    for (name, actual, expected) in [
        ("q", q.len(), q_len),
        ("k", k.len(), q_len),
        ("v", v.len(), v_len),
        ("decay_logits", decay_logits.len(), q_len),
        ("erase_logits", erase_logits.len(), q_len),
        ("write_logits", write_logits.len(), v_len),
        ("initial_state", initial_state.len(), state_len),
    ] {
        if actual != expected {
            return Err(format!("{name} length {actual} does not match expected {expected}"));
        }
    }
    let mut state = initial_state.to_vec();
    let mut outputs = vec![0.0; v_len];
    for batch in 0..shape.batch {
        for step in 0..shape.seq {
            for head in 0..shape.heads {
                let state_base = (batch * shape.heads + head) * shape.value_dim * shape.key_dim;
                let token_base = (batch * shape.seq + step) * shape.heads + head;
                let key_base = token_base * shape.key_dim;
                let value_base = token_base * shape.value_dim;
                for value in 0..shape.value_dim {
                    let write = sigmoid(write_logits[value_base + value]);
                    for key in 0..shape.key_dim {
                        let decay = sigmoid(decay_logits[key_base + key]);
                        let erase = sigmoid(erase_logits[key_base + key]);
                        let old = state[state_base + value * shape.key_dim + key];
                        let update = v[value_base + value] * k[key_base + key];
                        state[state_base + value * shape.key_dim + key] = decay * (1.0 - erase) * old + write * update;
                    }
                }
                for value in 0..shape.value_dim {
                    let mut readout = 0.0;
                    for key in 0..shape.key_dim {
                        readout += state[state_base + value * shape.key_dim + key] * q[key_base + key];
                    }
                    outputs[value_base + value] = readout;
                }
            }
        }
    }
    Ok(Gdn2ForwardOutput { outputs, final_state: state })
}

/// Dependency-free reverse scan for the same flat-buffer contract. The
/// forward states are recomputed internally so callers need only retain inputs.
pub fn gdn2_backward_f32(
    shape: &Gdn2ForwardShape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay_logits: &[f32],
    erase_logits: &[f32],
    write_logits: &[f32],
    initial_state: &[f32],
    grad_outputs: &[f32],
    grad_final_state: &[f32],
) -> Result<Gdn2BackwardOutput, String> {
    let _forward = gdn2_forward_f32(shape, q, k, v, decay_logits, erase_logits, write_logits, initial_state)?;
    let q_len = expected_len(shape, shape.key_dim);
    let v_len = expected_len(shape, shape.value_dim);
    let state_len = shape.batch * shape.heads * shape.value_dim * shape.key_dim;
    if grad_outputs.len() != v_len || grad_final_state.len() != state_len {
        return Err("gradient cotangent shape does not match forward output".to_string());
    }
    let mut snapshots = vec![0.0; (shape.seq + 1) * state_len];
    snapshots[..state_len].copy_from_slice(initial_state);
    for step in 0..shape.seq {
        let prior = &snapshots[step * state_len..(step + 1) * state_len];
        let mut next = prior.to_vec();
        for batch in 0..shape.batch {
            for head in 0..shape.heads {
                let state_base = (batch * shape.heads + head) * shape.value_dim * shape.key_dim;
                let token_base = (batch * shape.seq + step) * shape.heads + head;
                let kb = token_base * shape.key_dim;
                let vb = token_base * shape.value_dim;
                for value in 0..shape.value_dim {
                    for key in 0..shape.key_dim {
                        let d = sigmoid(decay_logits[kb + key]);
                        let e = sigmoid(erase_logits[kb + key]);
                        let w = sigmoid(write_logits[vb + value]);
                        next[state_base + value * shape.key_dim + key] = d * (1.0 - e) * prior[state_base + value * shape.key_dim + key]
                            + w * v[vb + value] * k[kb + key];
                    }
                }
            }
        }
        snapshots[(step + 1) * state_len..(step + 2) * state_len].copy_from_slice(&next);
    }
    let mut result = Gdn2BackwardOutput {
        grad_q: vec![0.0; q_len],
        grad_k: vec![0.0; q_len],
        grad_v: vec![0.0; v_len],
        grad_decay_logits: vec![0.0; q_len],
        grad_erase_logits: vec![0.0; q_len],
        grad_write_logits: vec![0.0; v_len],
        grad_initial_state: vec![0.0; state_len],
    };
    let mut grad_state = grad_final_state.to_vec();
    for step in (0..shape.seq).rev() {
        let current = &snapshots[(step + 1) * state_len..(step + 2) * state_len];
        let prior = &snapshots[step * state_len..(step + 1) * state_len];
        let mut grad_prev = vec![0.0; state_len];
        for batch in 0..shape.batch {
            for head in 0..shape.heads {
                let sb = (batch * shape.heads + head) * shape.value_dim * shape.key_dim;
                let tb = (batch * shape.seq + step) * shape.heads + head;
                let kb = tb * shape.key_dim;
                let vb = tb * shape.value_dim;
                let token_vb = (batch * shape.seq + step) * shape.heads * shape.value_dim + head * shape.value_dim;
                for key in 0..shape.key_dim {
                    let d = sigmoid(decay_logits[kb + key]);
                    let e = sigmoid(erase_logits[kb + key]);
                    let a = d * (1.0 - e);
                    let mut grad_a = 0.0;
                    for value in 0..shape.value_dim {
                        let cell = sb + value * shape.key_dim + key;
                        let total = grad_state[cell] + grad_outputs[token_vb + value] * q[kb + key];
                        grad_prev[cell] = total * a;
                        grad_a += total * prior[cell];
                        let w = sigmoid(write_logits[vb + value]);
                        result.grad_v[token_vb + value] += total * w * k[kb + key];
                        result.grad_k[kb + key] += total * w * v[token_vb + value];
                    }
                    result.grad_decay_logits[kb + key] += grad_a * (1.0 - e) * d * (1.0 - d);
                    result.grad_erase_logits[kb + key] += grad_a * (-d) * e * (1.0 - e);
                }
                for value in 0..shape.value_dim {
                    let mut grad_write = 0.0;
                    for key in 0..shape.key_dim {
                        let cell = sb + value * shape.key_dim + key;
                        let total = grad_state[cell] + grad_outputs[token_vb + value] * q[kb + key];
                        grad_write += total * v[token_vb + value] * k[kb + key];
                    }
                    let w = sigmoid(write_logits[vb + value]);
                    result.grad_write_logits[vb + value] += grad_write * w * (1.0 - w);
                }
                for value in 0..shape.value_dim {
                    for key in 0..shape.key_dim {
                        let cell = sb + value * shape.key_dim + key;
                        result.grad_q[kb + key] += grad_outputs[token_vb + value] * current[cell];
                    }
                }
            }
        }
        grad_state = grad_prev;
    }
    result.grad_initial_state = grad_state;
    Ok(result)
}

/// Correctness-first conditional causal attention reference for HZ-0C.
/// Queries and keys are restricted to triggered positions; keys are also
/// causal. The layout is row-major: qkv_weight is [3*dim, dim], out_weight is
/// [dim, dim], and all sequence buffers are [batch, seq, dim].
pub fn conditional_anchor_attention_f32(
    batch: usize,
    seq: usize,
    dim: usize,
    heads: usize,
    x: &[f32],
    qkv_weight: &[f32],
    qkv_bias: &[f32],
    out_weight: &[f32],
    out_bias: &[f32],
    trigger: &[f32],
) -> Result<Vec<f32>, String> {
    if heads == 0 || dim == 0 || dim % heads != 0 {
        return Err("dim must be positive and divisible by heads".into());
    }
    let tokens = batch * seq;
    if x.len() != tokens * dim || trigger.len() != tokens
        || qkv_weight.len() != 3 * dim * dim || qkv_bias.len() != 3 * dim
        || out_weight.len() != dim * dim || out_bias.len() != dim
    {
        return Err("conditional attention buffer shape mismatch".into());
    }
    let mut qkv = vec![0.0; tokens * 3 * dim];
    for token in 0..tokens {
        for row in 0..3 * dim {
            let mut value = qkv_bias[row];
            for col in 0..dim {
                value += x[token * dim + col] * qkv_weight[row * dim + col];
            }
            qkv[token * 3 * dim + row] = value;
        }
    }
    let mut attention = vec![0.0; tokens * dim];
    let head_dim = dim / heads;
    let scale = (head_dim as f32).sqrt().recip();
    for b in 0..batch {
        for t in 0..seq {
            let token = b * seq + t;
            if trigger[token] <= 0.0 {
                continue;
            }
            for h in 0..heads {
                let offset = h * head_dim;
                let mut scores = vec![f32::NEG_INFINITY; t + 1];
                for s in 0..=t {
                    let source = b * seq + s;
                    if trigger[source] <= 0.0 {
                        continue;
                    }
                    let mut score = 0.0;
                    for k in 0..head_dim {
                        score += qkv[token * 3 * dim + offset + k]
                            * qkv[source * 3 * dim + dim + offset + k];
                    }
                    scores[s] = score * scale;
                }
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                if !max_score.is_finite() {
                    continue;
                }
                let mut denominator = 0.0;
                for score in &mut scores {
                    if score.is_finite() {
                        *score = (*score - max_score).exp();
                        denominator += *score;
                    } else {
                        *score = 0.0;
                    }
                }
                for s in 0..=t {
                    let weight = scores[s] / denominator;
                    let source = b * seq + s;
                    for k in 0..head_dim {
                        attention[token * dim + offset + k] +=
                            weight * qkv[source * 3 * dim + 2 * dim + offset + k];
                    }
                }
            }
        }
    }
    let mut output = vec![0.0; tokens * dim];
    for token in 0..tokens {
        for row in 0..dim {
            let mut value = out_bias[row];
            for col in 0..dim {
                value += attention[token * dim + col] * out_weight[row * dim + col];
            }
            output[token * dim + row] = value;
        }
    }
    Ok(output)
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConditionalAttentionBackward {
    pub grad_x: Vec<f32>,
    pub grad_qkv_weight: Vec<f32>,
    pub grad_qkv_bias: Vec<f32>,
    pub grad_out_weight: Vec<f32>,
    pub grad_out_bias: Vec<f32>,
}

/// Explicit backward reference for `conditional_anchor_attention_f32`.
/// Trigger values are routing decisions and are intentionally treated as
/// constants; all trainable projections receive gradients.
#[allow(clippy::too_many_arguments)]
pub fn conditional_anchor_attention_backward_f32(
    batch: usize,
    seq: usize,
    dim: usize,
    heads: usize,
    x: &[f32],
    qkv_weight: &[f32],
    qkv_bias: &[f32],
    out_weight: &[f32],
    trigger: &[f32],
    grad_output: &[f32],
) -> Result<ConditionalAttentionBackward, String> {
    if heads == 0 || dim == 0 || dim % heads != 0 {
        return Err("dim must be positive and divisible by heads".into());
    }
    let tokens = batch * seq;
    if x.len() != tokens * dim || trigger.len() != tokens || grad_output.len() != tokens * dim
        || qkv_weight.len() != 3 * dim * dim || qkv_bias.len() != 3 * dim
        || out_weight.len() != dim * dim
    {
        return Err("conditional attention backward buffer shape mismatch".into());
    }
    let mut qkv = vec![0.0; tokens * 3 * dim];
    for token in 0..tokens {
        for row in 0..3 * dim {
            let mut value = qkv_bias[row];
            for col in 0..dim { value += x[token * dim + col] * qkv_weight[row * dim + col]; }
            qkv[token * 3 * dim + row] = value;
        }
    }
    let mut grad_x = vec![0.0; tokens * dim];
    let mut grad_qkv_weight = vec![0.0; 3 * dim * dim];
    let mut grad_qkv_bias = vec![0.0; 3 * dim];
    let mut grad_out_weight = vec![0.0; dim * dim];
    let mut grad_out_bias = vec![0.0; dim];
    let head_dim = dim / heads;
    let scale = (head_dim as f32).sqrt().recip();

    for token in 0..tokens {
        for row in 0..dim {
            grad_out_bias[row] += grad_output[token * dim + row];
            for col in 0..dim {
                grad_out_weight[row * dim + col] += grad_output[token * dim + row]
                    * if trigger[token] > 0.0 { // attention is zero otherwise
                        0.0
                    } else { 0.0 };
            }
        }
    }
    for b in 0..batch {
        for t in 0..seq {
            let token = b * seq + t;
            if trigger[token] <= 0.0 { continue; }
            for h in 0..heads {
                let offset = h * head_dim;
                let mut scores = vec![0.0; t + 1];
                let mut valid = vec![false; t + 1];
                for s in 0..=t {
                    let source = b * seq + s;
                    if trigger[source] <= 0.0 { continue; }
                    let mut score = 0.0;
                    for k in 0..head_dim {
                        score += qkv[token * 3 * dim + offset + k]
                            * qkv[source * 3 * dim + dim + offset + k];
                    }
                    scores[s] = score * scale;
                    valid[s] = true;
                }
                let max_score = scores.iter().enumerate().filter(|(s, _)| valid[*s])
                    .map(|(_, score)| *score).fold(f32::NEG_INFINITY, f32::max);
                if !max_score.is_finite() { continue; }
                let mut denom = 0.0;
                for s in 0..=t { if valid[s] { scores[s] = (scores[s] - max_score).exp(); denom += scores[s]; } }
                for s in 0..=t { if valid[s] { scores[s] /= denom; } }

                let mut grad_prob = vec![0.0; t + 1];
                for row in 0..dim {
                    let go = grad_output[token * dim + row];
                    for col in 0..dim {
                        grad_out_weight[row * dim + col] += go * (if col / head_dim == h {
                            let component = col % head_dim;
                            let mut v = 0.0;
                            for s in 0..=t { if valid[s] { v += scores[s] * qkv[(b * seq + s) * 3 * dim + 2 * dim + h * head_dim + component]; } }
                            v
                        } else { 0.0 });
                    }
                }
                for s in 0..=t { if valid[s] {
                    let source = b * seq + s;
                    for component in 0..head_dim {
                        let col = offset + component;
                        let mut upstream = 0.0;
                        for row in 0..dim { upstream += grad_output[token * dim + row] * out_weight[row * dim + col]; }
                        grad_prob[s] += upstream * qkv[source * 3 * dim + 2 * dim + col];
                        let gv = scores[s] * upstream;
                        grad_qkv_bias[2 * dim + col] += gv;
                        for c in 0..dim { grad_qkv_weight[(2 * dim + col) * dim + c] += gv * x[source * dim + c]; grad_x[source * dim + c] += gv * qkv_weight[(2 * dim + col) * dim + c]; }
                    }
                }}
                let mut dot = 0.0;
                for s in 0..=t { if valid[s] { dot += grad_prob[s] * scores[s]; } }
                for s in 0..=t { if valid[s] {
                    let source = b * seq + s;
                    let grad_score = scores[s] * (grad_prob[s] - dot) * scale;
                    for k in 0..head_dim {
                        let qrow = offset + k; let krow = dim + offset + k;
                        let qv = qkv[token * 3 * dim + qrow]; let kv = qkv[source * 3 * dim + krow];
                        grad_qkv_bias[qrow] += grad_score * kv; grad_qkv_bias[krow] += grad_score * qv;
                        for c in 0..dim {
                            grad_qkv_weight[qrow * dim + c] += grad_score * kv * x[token * dim + c];
                            grad_qkv_weight[krow * dim + c] += grad_score * qv * x[source * dim + c];
                            grad_x[token * dim + c] += grad_score * kv * qkv_weight[qrow * dim + c];
                            grad_x[source * dim + c] += grad_score * qv * qkv_weight[krow * dim + c];
                        }
                    }
                }}
            }
        }
    }
    Ok(ConditionalAttentionBackward { grad_x, grad_qkv_weight, grad_qkv_bias, grad_out_weight, grad_out_bias })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a1_default_matches_locked_spec() {
        let spec = A1OperatorSpec::default();
        assert_eq!(spec.model_dim, 768);
        assert_eq!(spec.num_heads, 12);
        assert_eq!(spec.key_dim, 64);
        assert_eq!(spec.value_dim, 64);
        assert_eq!(spec.state_shape(2), (2, 12, 64, 64));
        assert_eq!(spec.a1_recurrent_input_width(), 4608);
    }

    #[test]
    fn cache_spec_defaults_to_saved_gates_and_checkpoints() {
        let cache = Gdn2CacheSpec::default();
        assert!(cache.saves_decay);
        assert!(cache.saves_erase);
        assert!(cache.saves_write);
        assert!(cache.saves_hidden_checkpoints);
    }

    #[test]
    fn cpu_forward_carries_state_and_rejects_bad_shapes() {
        let shape = Gdn2ForwardShape { batch: 1, seq: 2, heads: 1, key_dim: 2, value_dim: 2 };
        let q = vec![1.0, 0.0, 0.0, 1.0];
        let k = vec![1.0, 0.0, 0.0, 1.0];
        let v = vec![2.0, 3.0, 4.0, 5.0];
        let gates = vec![0.0; 4];
        let write = vec![0.0; 4];
        let initial = vec![0.0; 4];
        let result = gdn2_forward_f32(&shape, &q, &k, &v, &gates, &gates, &write, &initial).unwrap();
        assert_eq!(result.outputs.len(), 4);
        assert_eq!(result.final_state.len(), 4);
        assert!(gdn2_forward_f32(&shape, &q[..2], &k, &v, &gates, &gates, &write, &initial).is_err());
    }

    #[test]
    fn cpu_forward_chunking_matches_full_scan() {
        let shape = Gdn2ForwardShape { batch: 1, seq: 4, heads: 1, key_dim: 1, value_dim: 1 };
        let q = vec![1.0, 2.0, 3.0, 4.0];
        let k = vec![0.5, 1.0, 1.5, 2.0];
        let v = vec![2.0, 2.0, 2.0, 2.0];
        let gates = vec![0.0; 4];
        let initial = vec![0.0];
        let full = gdn2_forward_f32(&shape, &q, &k, &v, &gates, &gates, &gates, &initial).unwrap();
        let first_shape = Gdn2ForwardShape { seq: 2, ..shape.clone() };
        let first = gdn2_forward_f32(&first_shape, &q[..2], &k[..2], &v[..2], &gates[..2], &gates[..2], &gates[..2], &initial).unwrap();
        let second = gdn2_forward_f32(&first_shape, &q[2..], &k[2..], &v[2..], &gates[2..], &gates[2..], &gates[2..], &first.final_state).unwrap();
        assert_eq!(&full.outputs[..2], &first.outputs[..]);
        assert_eq!(&full.outputs[2..], &second.outputs[..]);
        assert_eq!(full.final_state, second.final_state);
    }

    #[test]
    fn cpu_backward_q_matches_finite_difference() {
        let shape = Gdn2ForwardShape { batch: 1, seq: 2, heads: 1, key_dim: 1, value_dim: 1 };
        let q = vec![0.7, -0.2];
        let k = vec![0.4, 0.8];
        let v = vec![1.2, -0.5];
        let gates = vec![0.3, -0.4];
        let initial = vec![0.1];
        let grad_outputs = vec![1.5, -0.7];
        let grad_final = vec![0.9];
        let analytic = gdn2_backward_f32(&shape, &q, &k, &v, &gates, &gates, &gates, &initial, &grad_outputs, &grad_final).unwrap();
        let eps = 1e-3;
        let mut positive = q.clone();
        let mut negative = q.clone();
        positive[0] += eps;
        negative[0] -= eps;
        let objective = |query: &[f32]| {
            let output = gdn2_forward_f32(&shape, query, &k, &v, &gates, &gates, &gates, &initial).unwrap();
            output.outputs.iter().zip(&grad_outputs).map(|(a, b)| a * b).sum::<f32>() + output.final_state[0] * grad_final[0]
        };
        let numeric = (objective(&positive) - objective(&negative)) / (2.0 * eps);
        assert!((analytic.grad_q[0] - numeric).abs() < 2e-3, "analytic={} numeric={}", analytic.grad_q[0], numeric);
    }

    #[test]
    fn conditional_attention_zeroes_nontriggered_queries_and_future_keys() {
        let dim = 2;
        let x = vec![1.0, 0.0, 0.0, 1.0, 2.0, 1.0];
        let qkv_weight = {
            let mut w = vec![0.0; 3 * dim * dim];
            for i in 0..dim {
                w[i * dim + i] = 1.0;
                w[(dim + i) * dim + i] = 1.0;
                w[(2 * dim + i) * dim + i] = 1.0;
            }
            w
        };
        let bias = vec![0.0; 3 * dim];
        let out_weight = vec![1.0, 0.0, 0.0, 1.0];
        let out_bias = vec![0.0; dim];
        let only_first = conditional_anchor_attention_f32(
            1, 3, dim, 1, &x, &qkv_weight, &bias, &out_weight, &out_bias,
            &[1.0, 0.0, 0.0],
        ).unwrap();
        assert_eq!(&only_first[2..], &[0.0, 0.0, 0.0, 0.0]);
        let both = conditional_anchor_attention_f32(
            1, 3, dim, 1, &x, &qkv_weight, &bias, &out_weight, &out_bias,
            &[1.0, 0.0, 1.0],
        ).unwrap();
        assert!(both[4].is_finite() && both[5].is_finite());
        assert_ne!(&both[4..], &[0.0, 0.0]);
    }

    #[test]
    fn conditional_attention_matches_scalar_causal_reference_and_isolates_batches() {
        // One-dimensional attention makes the expected softmax closed-form.
        let x = vec![1.0, 2.0, 3.0, 10.0];
        let qkv_weight = vec![1.0, 1.0, 1.0];
        let qkv_bias = vec![0.0; 3];
        let out_weight = vec![1.0];
        let out_bias = vec![0.25];
        let trigger = vec![1.0, 0.0, 1.0, 0.0];
        let output = conditional_anchor_attention_f32(
            2, 2, 1, 1, &x, &qkv_weight, &qkv_bias, &out_weight, &out_bias, &trigger,
        ).unwrap();

        // Batch 0 token 0 attends only to value 1; token 1 is not triggered.
        assert!((output[0] - 1.25).abs() < 1e-6);
        assert_eq!(output[1], 0.25);
        // Batch 1 has its own causal prefix and cannot see batch 0.
        assert!((output[2] - 3.25).abs() < 1e-6);
        assert!((output[3] - 0.25).abs() < 1e-6);

        assert!(conditional_anchor_attention_f32(
            1, 1, 3, 2, &[1.0, 2.0, 3.0], &[0.0; 27], &[0.0; 9],
            &[0.0; 9], &[0.0; 3], &[1.0],
        ).is_err());
    }

    #[test]
    fn conditional_attention_backward_matches_finite_difference() {
        let (batch, seq, dim, heads) = (1, 3, 4, 2);
        let mut x: Vec<f32> = (0..batch * seq * dim).map(|i| 0.03 + i as f32 * 0.02).collect();
        let mut qkv_w: Vec<f32> = (0..3 * dim * dim).map(|i| -0.08 + i as f32 * 0.007).collect();
        let mut qkv_b: Vec<f32> = (0..3 * dim).map(|i| -0.03 + i as f32 * 0.01).collect();
        let mut out_w: Vec<f32> = (0..dim * dim).map(|i| 0.04 - i as f32 * 0.005).collect();
        let mut out_b: Vec<f32> = (0..dim).map(|i| 0.01 * i as f32).collect();
        let trigger = vec![1.0, 0.0, 1.0];
        let grad_output: Vec<f32> = (0..batch * seq * dim).map(|i| 0.02 + i as f32 * 0.01).collect();
        let loss = |x: &[f32], qw: &[f32], qb: &[f32], ow: &[f32], ob: &[f32]| {
            let y = conditional_anchor_attention_f32(batch, seq, dim, heads, x, qw, qb, ow, ob, &trigger).unwrap();
            y.iter().zip(&grad_output).map(|(a, b)| a * b).sum::<f32>()
        };
        let analytic = conditional_anchor_attention_backward_f32(batch, seq, dim, heads, &x, &qkv_w, &qkv_b, &out_w, &trigger, &grad_output).unwrap();
        let eps = 1e-3;
        macro_rules! check_family {
            ($values:expr, $expected:expr, $which:expr, $plus:expr, $minus:expr) => {{
                for i in 0..$values.len() {
                    let original = $values[i];
                    $values[i] = original + eps;
                    let plus = $plus;
                    $values[i] = original - eps;
                    let minus = $minus;
                    $values[i] = original;
                    let numerical = (plus - minus) / (2.0 * eps);
                    assert!((numerical - $expected[i]).abs() < 3e-3, "family {} index {i}: numerical {numerical} analytic {}", $which, $expected[i]);
                }
            }};
        }
        check_family!(x, analytic.grad_x, 0, loss(&x, &qkv_w, &qkv_b, &out_w, &out_b), loss(&x, &qkv_w, &qkv_b, &out_w, &out_b));
        check_family!(qkv_w, analytic.grad_qkv_weight, 1, loss(&x, &qkv_w, &qkv_b, &out_w, &out_b), loss(&x, &qkv_w, &qkv_b, &out_w, &out_b));
        check_family!(qkv_b, analytic.grad_qkv_bias, 2, loss(&x, &qkv_w, &qkv_b, &out_w, &out_b), loss(&x, &qkv_w, &qkv_b, &out_w, &out_b));
        check_family!(out_w, analytic.grad_out_weight, 3, loss(&x, &qkv_w, &qkv_b, &out_w, &out_b), loss(&x, &qkv_w, &qkv_b, &out_w, &out_b));
        check_family!(out_b, analytic.grad_out_bias, 4, loss(&x, &qkv_w, &qkv_b, &out_w, &out_b), loss(&x, &qkv_w, &qkv_b, &out_w, &out_b));
    }
}
