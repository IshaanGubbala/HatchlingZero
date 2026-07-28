#![forbid(unsafe_code)]

/// Minimal restart-facing description of the fused GDN-2 interface.
///
/// This crate is intentionally tiny at initialization time. It exists to pin
/// down the API shape before any substantial PMetal operator work begins.
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

pub fn restart_kernel_scope() -> &'static str {
    "hz0a-pmetal-kernel"
}

