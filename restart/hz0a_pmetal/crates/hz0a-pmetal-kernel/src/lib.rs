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
}
