//! Factory for creating load balancing policies

use std::sync::Arc;

use super::{
    BucketConfig, BucketPolicy, CacheAwareConfig, CacheAwarePolicy, CacheAwarePowerOfTwoPolicy,
    CacheAwareRankConfig, CacheAwareRankPolicy, ChunkLMetricPolicy, ConsistentHashingPolicy,
    DualMapConfig, DualMapPolicy, LMetricPolicy, LoadBalancingPolicy, ManualConfig, ManualPolicy,
    PowerOfTwoPolicy, PrebleE2PrefillPolicy, PrefixHashConfig, PrefixHashPolicy,
    PrefixOnlyLpmPolicy, RandomPolicy, RankConsistentHashPolicy, RankLeastLoadedPolicy,
    RankPowerOfTwoPolicy, RankTotalTokensConfig, RankTotalTokensPolicy, RoundRobinPolicy,
    SMetricPolicy,
};
use crate::config::PolicyConfig;

/// Factory for creating policy instances
pub struct PolicyFactory;

impl PolicyFactory {
    /// Create a policy from configuration
    pub fn create_from_config(config: &PolicyConfig) -> Arc<dyn LoadBalancingPolicy> {
        match config {
            PolicyConfig::Random => Arc::new(RandomPolicy::new()),
            PolicyConfig::RoundRobin => Arc::new(RoundRobinPolicy::new()),
            PolicyConfig::PowerOfTwo { .. } => Arc::new(PowerOfTwoPolicy::new()),
            PolicyConfig::RankPowerOfTwo => Arc::new(RankPowerOfTwoPolicy::new()),
            PolicyConfig::RankLeastLoaded => Arc::new(RankLeastLoadedPolicy::new()),
            PolicyConfig::RankTotalTokens {
                max_staleness_ms,
                request_timeout_ms,
            } => Arc::new(RankTotalTokensPolicy::new(RankTotalTokensConfig {
                max_staleness_ms: *max_staleness_ms,
                request_timeout_ms: *request_timeout_ms,
            })),
            PolicyConfig::RankConsistentHash => Arc::new(RankConsistentHashPolicy::new()),
            PolicyConfig::PrefixOnlyLpm => Arc::new(PrefixOnlyLpmPolicy::new()),
            PolicyConfig::LMetric => Arc::new(LMetricPolicy::new()),
            PolicyConfig::PrebleE2Prefill {
                history_window_secs,
            } => Arc::new(PrebleE2PrefillPolicy::new(*history_window_secs)),
            PolicyConfig::DualMap {
                slo_token_threshold,
                prefix_window_size,
                prefix_min_samples,
                prefix_block_tokens,
            } => Arc::new(DualMapPolicy::new(DualMapConfig {
                slo_token_threshold: *slo_token_threshold,
                prefix_window_size: *prefix_window_size,
                prefix_min_samples: *prefix_min_samples,
                prefix_block_tokens: *prefix_block_tokens,
            })),
            PolicyConfig::SMetric { min_match_tokens } => {
                Arc::new(SMetricPolicy::new(*min_match_tokens))
            }
            PolicyConfig::ChunkLMetric { chunk_size } => {
                Arc::new(ChunkLMetricPolicy::new(*chunk_size))
            }
            PolicyConfig::CacheAwarePowerOfTwo => Arc::new(CacheAwarePowerOfTwoPolicy::new()),
            PolicyConfig::CacheAwareRank {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
            } => Arc::new(CacheAwareRankPolicy::new(CacheAwareRankConfig {
                cache_threshold: *cache_threshold,
                balance_abs_threshold: *balance_abs_threshold,
                balance_rel_threshold: *balance_rel_threshold,
            })),
            PolicyConfig::CacheAware {
                cache_threshold,
                balance_abs_threshold,
                balance_rel_threshold,
                eviction_interval_secs,
                max_tree_size,
            } => {
                let config = CacheAwareConfig {
                    cache_threshold: *cache_threshold,
                    balance_abs_threshold: *balance_abs_threshold,
                    balance_rel_threshold: *balance_rel_threshold,
                    eviction_interval_secs: *eviction_interval_secs,
                    max_tree_size: *max_tree_size,
                };
                Arc::new(CacheAwarePolicy::with_config(config))
            }
            PolicyConfig::Bucket {
                balance_abs_threshold,
                balance_rel_threshold,
                bucket_adjust_interval_secs,
            } => {
                let config = BucketConfig {
                    balance_abs_threshold: *balance_abs_threshold,
                    balance_rel_threshold: *balance_rel_threshold,
                    bucket_adjust_interval_secs: *bucket_adjust_interval_secs,
                };
                Arc::new(BucketPolicy::with_config(config))
            }
            PolicyConfig::Manual {
                eviction_interval_secs,
                max_idle_secs,
                assignment_mode,
            } => {
                let config = ManualConfig {
                    eviction_interval_secs: *eviction_interval_secs,
                    max_idle_secs: *max_idle_secs,
                    assignment_mode: *assignment_mode,
                };
                Arc::new(ManualPolicy::with_config(config))
            }
            PolicyConfig::ConsistentHashing => Arc::new(ConsistentHashingPolicy::new()),
            PolicyConfig::PrefixHash {
                prefix_token_count,
                load_factor,
            } => {
                let config = PrefixHashConfig {
                    prefix_token_count: *prefix_token_count,
                    load_factor: *load_factor,
                };
                Arc::new(PrefixHashPolicy::new(config))
            }
        }
    }

    /// Create a policy by name (for dynamic loading)
    pub fn create_by_name(name: &str) -> Option<Arc<dyn LoadBalancingPolicy>> {
        match name.to_lowercase().as_str() {
            "random" => Some(Arc::new(RandomPolicy::new())),
            "round_robin" | "roundrobin" => Some(Arc::new(RoundRobinPolicy::new())),
            "power_of_two" | "poweroftwo" => Some(Arc::new(PowerOfTwoPolicy::new())),
            "rank_power_of_two" | "rankpoweroftwo" => Some(Arc::new(RankPowerOfTwoPolicy::new())),
            "rank_least_loaded" | "rankleastloaded" => Some(Arc::new(RankLeastLoadedPolicy::new())),
            "rank_total_tokens" | "ranktotaltokens" => {
                Some(Arc::new(RankTotalTokensPolicy::default()))
            }
            "rank_consistent_hash" | "rankconsistenthash" => {
                Some(Arc::new(RankConsistentHashPolicy::new()))
            }
            "prefix_only_lpm" | "prefixonlylpm" => Some(Arc::new(PrefixOnlyLpmPolicy::new())),
            "lmetric" => Some(Arc::new(LMetricPolicy::new())),
            "preble_e2_prefill" | "preblee2prefill" => {
                Some(Arc::new(PrebleE2PrefillPolicy::default()))
            }
            "dualmap" => Some(Arc::new(DualMapPolicy::default())),
            "smetric" => Some(Arc::new(SMetricPolicy::new(512))),
            "chunk_lmetric" | "chunklmetric" => Some(Arc::new(ChunkLMetricPolicy::new(4096))),
            "cache_aware_p2c" | "cacheawarep2c" => {
                Some(Arc::new(CacheAwarePowerOfTwoPolicy::new()))
            }
            "cache_aware_rank" | "cacheawarerank" => {
                Some(Arc::new(CacheAwareRankPolicy::default()))
            }
            "cache_aware" | "cacheaware" => Some(Arc::new(CacheAwarePolicy::new())),
            "bucket" => Some(Arc::new(BucketPolicy::new())),
            "manual" => Some(Arc::new(ManualPolicy::new())),
            "consistent_hashing" | "consistenthashing" => {
                Some(Arc::new(ConsistentHashingPolicy::new()))
            }
            "prefix_hash" | "prefixhash" => Some(Arc::new(PrefixHashPolicy::with_defaults())),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_from_config() {
        let policy = PolicyFactory::create_from_config(&PolicyConfig::Random);
        assert_eq!(policy.name(), "random");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::RoundRobin);
        assert_eq!(policy.name(), "round_robin");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::PowerOfTwo {
            load_check_interval_secs: 60,
        });
        assert_eq!(policy.name(), "power_of_two");

        assert_eq!(
            PolicyFactory::create_from_config(&PolicyConfig::RankPowerOfTwo).name(),
            "rank_power_of_two"
        );
        assert_eq!(
            PolicyFactory::create_from_config(&PolicyConfig::RankLeastLoaded).name(),
            "rank_least_loaded"
        );
        assert_eq!(
            PolicyFactory::create_from_config(&PolicyConfig::RankTotalTokens {
                max_staleness_ms: 250,
                request_timeout_ms: 200,
            })
            .name(),
            "rank_total_tokens"
        );
        assert_eq!(
            PolicyFactory::create_from_config(&PolicyConfig::RankConsistentHash).name(),
            "rank_consistent_hash"
        );
        assert_eq!(
            PolicyFactory::create_from_config(&PolicyConfig::PrefixOnlyLpm).name(),
            "prefix_only_lpm"
        );
        assert_eq!(
            PolicyFactory::create_from_config(&PolicyConfig::LMetric).name(),
            "lmetric"
        );
        assert_eq!(
            PolicyFactory::create_from_config(&PolicyConfig::PrebleE2Prefill {
                history_window_secs: 180,
            })
            .name(),
            "preble_e2_prefill"
        );
        assert_eq!(
            PolicyFactory::create_from_config(&PolicyConfig::DualMap {
                slo_token_threshold: 16_384,
                prefix_window_size: 200,
                prefix_min_samples: 20,
                prefix_block_tokens: 512,
            })
            .name(),
            "dualmap"
        );
        assert_eq!(
            PolicyFactory::create_from_config(&PolicyConfig::SMetric {
                min_match_tokens: 512,
            })
            .name(),
            "smetric"
        );
        assert_eq!(
            PolicyFactory::create_from_config(&PolicyConfig::ChunkLMetric { chunk_size: 4096 })
                .name(),
            "chunk_lmetric"
        );
        assert_eq!(
            PolicyFactory::create_from_config(&PolicyConfig::CacheAwarePowerOfTwo).name(),
            "cache_aware_p2c"
        );
        assert_eq!(
            PolicyFactory::create_from_config(&PolicyConfig::CacheAwareRank {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
            })
            .name(),
            "cache_aware_rank"
        );

        let policy = PolicyFactory::create_from_config(&PolicyConfig::CacheAware {
            cache_threshold: 0.7,
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            eviction_interval_secs: 30,
            max_tree_size: 1000,
        });
        assert_eq!(policy.name(), "cache_aware");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::Bucket {
            balance_abs_threshold: 10,
            balance_rel_threshold: 1.5,
            bucket_adjust_interval_secs: 5,
        });
        assert_eq!(policy.name(), "bucket");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::Manual {
            eviction_interval_secs: 60,
            max_idle_secs: 4 * 3600,
            assignment_mode: Default::default(),
        });
        assert_eq!(policy.name(), "manual");

        let policy = PolicyFactory::create_from_config(&PolicyConfig::ConsistentHashing);
        assert_eq!(policy.name(), "consistent_hashing");
    }

    #[tokio::test]
    async fn test_create_by_name() {
        assert!(PolicyFactory::create_by_name("random").is_some());
        assert!(PolicyFactory::create_by_name("RANDOM").is_some());
        assert!(PolicyFactory::create_by_name("round_robin").is_some());
        assert!(PolicyFactory::create_by_name("RoundRobin").is_some());
        assert!(PolicyFactory::create_by_name("power_of_two").is_some());
        assert!(PolicyFactory::create_by_name("PowerOfTwo").is_some());
        assert!(PolicyFactory::create_by_name("rank_least_loaded").is_some());
        assert!(PolicyFactory::create_by_name("rank_total_tokens").is_some());
        assert!(PolicyFactory::create_by_name("rank_consistent_hash").is_some());
        assert!(PolicyFactory::create_by_name("preble_e2_prefill").is_some());
        assert!(PolicyFactory::create_by_name("dualmap").is_some());
        assert!(PolicyFactory::create_by_name("smetric").is_some());
        assert!(PolicyFactory::create_by_name("chunk_lmetric").is_some());
        assert!(PolicyFactory::create_by_name("cache_aware_p2c").is_some());
        assert!(PolicyFactory::create_by_name("cache_aware").is_some());
        assert!(PolicyFactory::create_by_name("CacheAware").is_some());
        assert!(PolicyFactory::create_by_name("bucket").is_some());
        assert!(PolicyFactory::create_by_name("Bucket").is_some());
        assert!(PolicyFactory::create_by_name("manual").is_some());
        assert!(PolicyFactory::create_by_name("Manual").is_some());
        assert!(PolicyFactory::create_by_name("consistent_hashing").is_some());
        assert!(PolicyFactory::create_by_name("ConsistentHashing").is_some());
        assert!(PolicyFactory::create_by_name("unknown").is_none());
    }
}
