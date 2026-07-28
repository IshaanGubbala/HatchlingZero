use hz0a_pmetal_kernel::{restart_kernel_scope, Gdn2CacheSpec};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartBridgeConfig {
    pub cache_spec: Gdn2CacheSpec,
    pub target_runtime: &'static str,
}

impl Default for RestartBridgeConfig {
    fn default() -> Self {
        Self {
            cache_spec: Gdn2CacheSpec::default(),
            target_runtime: "pmetal",
        }
    }
}

pub fn restart_bridge_summary() -> String {
    format!("{} -> pmetal bridge scaffold", restart_kernel_scope())
}

