use hz0a_pmetal_kernel::{restart_kernel_scope, A1OperatorSpec, Gdn2CacheSpec};

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
}
