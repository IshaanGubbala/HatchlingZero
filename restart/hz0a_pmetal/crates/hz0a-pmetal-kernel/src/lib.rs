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
}
