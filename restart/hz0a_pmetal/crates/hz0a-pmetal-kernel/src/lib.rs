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
}
