// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! DP-attention-rank policies for the Rust model gateway.
//!
//! In DP-aware mode SGLang discovery exposes every attention-DP rank as a
//! separate [`DPAwareWorker`](crate::core::DPAwareWorker).  These policies
//! therefore implement the normal [`LoadBalancingPolicy`] interface: the
//! router selects a `(base_url, dp_rank)` worker and the transport commits
//! that decision through `routed_dp_rank`.
//!
//! Cache-backed policies consume SGLang's authoritative KV events.  They do
//! not build an approximate request-history tree and do not silently treat a
//! missing publisher, tokenizer, or rank stream as a cold cache.

use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex as StdMutex,
    },
    time::{Duration, Instant},
};

use async_trait::async_trait;
use rand::Rng;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use tokio::sync::{Mutex, OnceCell};
use tracing::debug;
use xxhash_rust::xxh3::xxh3_64_with_seed;

use super::{
    get_healthy_worker_indices,
    kv_events::{compute_block_hashes, compute_block_hashes_bigram, KvEventIndex, KvWorkerId},
    LoadBalancingPolicy, SelectWorkerInfo,
};
use crate::core::Worker;

#[derive(Debug, Clone, Copy)]
pub struct CacheAwareRankConfig {
    pub cache_threshold: f32,
    pub balance_abs_threshold: usize,
    pub balance_rel_threshold: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct RankTotalTokensConfig {
    /// Maximum age of one successfully fetched `/v1/loads` snapshot before
    /// the next request refreshes it.
    pub max_staleness_ms: u64,
    /// Per-engine timeout for `/v1/loads`.
    pub request_timeout_ms: u64,
}

impl Default for RankTotalTokensConfig {
    fn default() -> Self {
        Self {
            max_staleness_ms: 250,
            request_timeout_ms: 200,
        }
    }
}

/// Routing-only DualMap configuration.
///
/// `slo_token_threshold` is the calibrated amount of virtual prefill work
/// that fits inside the target TTFT SLO.  Expressing the threshold in tokens
/// matches the upstream implementation and keeps model-specific throughput
/// calibration outside the routing hot path.
#[derive(Debug, Clone, Copy)]
pub struct DualMapConfig {
    pub slo_token_threshold: usize,
    pub prefix_window_size: usize,
    pub prefix_min_samples: usize,
    pub prefix_block_tokens: usize,
}

impl Default for DualMapConfig {
    fn default() -> Self {
        Self {
            slo_token_threshold: 16_384,
            prefix_window_size: 200,
            prefix_min_samples: 20,
            prefix_block_tokens: 512,
        }
    }
}

impl Default for CacheAwareRankConfig {
    fn default() -> Self {
        Self {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
        }
    }
}

/// Lazily-created event index shared by all requests handled by one policy
/// instance. Lazy creation keeps config parsing and synchronous policy unit
/// tests independent of a Tokio runtime.
struct ExactKvRankState {
    index: OnceCell<Arc<KvEventIndex>>,
    attach_lock: Mutex<()>,
}

impl std::fmt::Debug for ExactKvRankState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExactKvRankState")
            .field("initialized", &self.index.initialized())
            .finish_non_exhaustive()
    }
}

impl ExactKvRankState {
    fn new() -> Self {
        Self {
            index: OnceCell::new(),
            attach_lock: Mutex::new(()),
        }
    }

    async fn index(&self) -> Arc<KvEventIndex> {
        self.index
            .get_or_init(|| async { KvEventIndex::new() })
            .await
            .clone()
    }

    /// Discover and subscribe each unique engine base URL once. The lock is
    /// deliberately held across discovery so concurrent first requests do
    /// not create duplicate ZMQ subscribers.
    async fn prepare(&self, workers: &[Arc<dyn Worker>], healthy: &[usize]) -> Arc<KvEventIndex> {
        let index = self.index().await;
        let _attach = self.attach_lock.lock().await;
        let urls: BTreeSet<&str> = healthy
            .iter()
            .map(|&idx| workers[idx].base_url().trim_end_matches('/'))
            .collect();
        for url in urls {
            if !index.has_worker(url) {
                index.add_worker(url, None).await;
            }
        }
        index
    }
}

#[derive(Debug, Clone)]
struct RankScore {
    worker_index: usize,
    base_url: String,
    dp_rank: u32,
    matched_blocks: usize,
    matched_tokens: usize,
    load: usize,
}

fn stable_rank_key(score: &RankScore) -> (&str, u32, usize) {
    (&score.base_url, score.dp_rank, score.worker_index)
}

fn pick_rr(indices: &mut Vec<&RankScore>, rr: &AtomicUsize) -> Option<usize> {
    indices.sort_by_key(|score| stable_rank_key(score));
    (!indices.is_empty()).then(|| {
        let offset = rr.fetch_add(1, Ordering::Relaxed) % indices.len();
        indices[offset].worker_index
    })
}

fn request_id(info: &SelectWorkerInfo<'_>) -> String {
    info.headers
        .and_then(|headers| headers.get("x-request-id"))
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_string()
}

const SELECTION_ID_HEADER: &str = "x-async-moe-selection-id";

fn selection_id(info: &SelectWorkerInfo<'_>) -> Option<u64> {
    info.headers?
        .get(SELECTION_ID_HEADER)?
        .to_str()
        .ok()?
        .parse()
        .ok()
}

fn audit_decision(
    policy: &'static str,
    info: &SelectWorkerInfo<'_>,
    scores: &[RankScore],
    selected: usize,
    reason: &'static str,
) {
    if !tracing::enabled!(target: "async_moe::dp_rank_decision", tracing::Level::DEBUG) {
        return;
    }
    let Some(chosen) = scores.iter().find(|score| score.worker_index == selected) else {
        return;
    };
    let prompt_tokens = info.tokens.map(<[u32]>::len);
    let candidates: Vec<_> = scores
        .iter()
        .map(|score| {
            serde_json::json!({
                "base_url": score.base_url,
                "dp_rank": score.dp_rank,
                "matched_blocks": score.matched_blocks,
                "matched_tokens": score.matched_tokens,
                "uncached_tokens": prompt_tokens
                    .map(|tokens| tokens.saturating_sub(score.matched_tokens)),
                "load": score.load,
            })
        })
        .collect();
    debug!(
        target: "async_moe::dp_rank_decision",
        policy,
        request_id = request_id(info),
        selected_base_url = chosen.base_url,
        selected_dp_rank = chosen.dp_rank,
        selected_matched_blocks = chosen.matched_blocks,
        selected_matched_tokens = chosen.matched_tokens,
        selected_load = chosen.load,
        reason,
        candidates = %serde_json::Value::Array(candidates),
        "DP-rank routing decision"
    );
}

fn lmetric_cost(prompt_tokens: usize, score: &RankScore) -> usize {
    prompt_tokens
        .saturating_sub(score.matched_tokens)
        .saturating_mul(score.load.saturating_add(1))
}

fn chunk_lmetric_cost(prompt_tokens: usize, chunk_size: usize, score: &RankScore) -> usize {
    let uncached_tokens = prompt_tokens.saturating_sub(score.matched_tokens);
    let chunks = uncached_tokens.div_ceil(chunk_size);
    chunks.saturating_mul(score.load.saturating_add(1))
}

type RankHistoryKey = (String, u32);

#[derive(Debug, Clone, Deserialize)]
struct EngineRankLoad {
    #[serde(default)]
    timestamp: f64,
    dp_rank: u32,
    #[serde(default)]
    num_running_reqs: usize,
    #[serde(default)]
    num_waiting_reqs: usize,
    #[serde(default)]
    num_total_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct EngineRankLoadsResponse {
    #[serde(default)]
    loads: Vec<EngineRankLoad>,
}

#[derive(Debug, Clone)]
struct RankTokenBudget {
    source_timestamp: f64,
    source_total_tokens: usize,
    source_total_requests: usize,
    overlay_tokens: usize,
    overlay_requests: usize,
}

impl RankTokenBudget {
    fn from_load(load: &EngineRankLoad) -> Self {
        Self {
            source_timestamp: load.timestamp,
            source_total_tokens: load.num_total_tokens,
            source_total_requests: load.num_running_reqs.saturating_add(load.num_waiting_reqs),
            overlay_tokens: 0,
            overlay_requests: 0,
        }
    }

    fn effective_total_tokens(&self) -> usize {
        self.source_total_tokens.saturating_add(self.overlay_tokens)
    }

    fn effective_total_requests(&self) -> usize {
        self.source_total_requests
            .saturating_add(self.overlay_requests)
    }
}

#[derive(Debug, Default)]
struct RankTokenLoadState {
    budgets: BTreeMap<RankHistoryKey, RankTokenBudget>,
    refreshed_at: Option<Instant>,
}

impl RankTokenLoadState {
    fn refresh_due(&self, now: Instant, max_staleness: Duration) -> bool {
        self.refreshed_at
            .is_none_or(|at| now.saturating_duration_since(at) >= max_staleness)
    }

    fn topology_matches(&self, expected: &BTreeSet<RankHistoryKey>) -> bool {
        self.budgets.len() == expected.len()
            && self.budgets.keys().all(|key| expected.contains(key))
    }

    /// Apply an all-ranks snapshot atomically. A missing, duplicate, invalid,
    /// or older rank makes the complete refresh unusable; callers fail closed
    /// rather than mixing generations. Equal source timestamps preserve the
    /// selection-time overlay. A newer source timestamp acknowledges that
    /// generation and starts a fresh overlay, matching SGLang DPC's heuristic.
    fn apply_snapshot(
        &mut self,
        expected: &BTreeSet<RankHistoryKey>,
        loads: BTreeMap<RankHistoryKey, EngineRankLoad>,
        now: Instant,
    ) -> bool {
        if loads.len() != expected.len() || loads.keys().any(|key| !expected.contains(key)) {
            return false;
        }
        if loads
            .values()
            .any(|load| !load.timestamp.is_finite() || load.timestamp < 0.0)
        {
            return false;
        }

        // Timestamp comparisons need the full `(base_url, rank)` key; keep
        // this separate from the wire-value validation above.
        if loads.iter().any(|(key, load)| {
            self.budgets
                .get(key)
                .is_some_and(|budget| load.timestamp < budget.source_timestamp)
        }) {
            return false;
        }

        self.budgets.retain(|key, _| expected.contains(key));
        for (key, load) in loads {
            match self.budgets.get_mut(&key) {
                Some(budget) if load.timestamp == budget.source_timestamp => {}
                Some(budget) => *budget = RankTokenBudget::from_load(&load),
                None => {
                    self.budgets.insert(key, RankTokenBudget::from_load(&load));
                }
            }
        }
        self.refreshed_at = Some(now);
        true
    }

    fn reserve(&mut self, key: &RankHistoryKey, prompt_tokens: usize) -> bool {
        let Some(budget) = self.budgets.get_mut(key) else {
            return false;
        };
        budget.overlay_tokens = budget.overlay_tokens.saturating_add(prompt_tokens);
        budget.overlay_requests = budget.overlay_requests.saturating_add(1);
        true
    }

    fn clear(&mut self) {
        self.budgets.clear();
        self.refreshed_at = None;
    }
}

#[derive(Debug, Clone)]
struct RankTokenScore {
    worker_index: usize,
    base_url: String,
    dp_rank: u32,
    source_total_tokens: usize,
    source_total_requests: usize,
    overlay_tokens: usize,
    overlay_requests: usize,
    effective_total_tokens: usize,
    effective_total_requests: usize,
}

fn rank_token_scores(
    ranks: &[RankScore],
    budgets: &BTreeMap<RankHistoryKey, RankTokenBudget>,
) -> Option<Vec<RankTokenScore>> {
    ranks
        .iter()
        .map(|rank| {
            let budget = budgets.get(&(rank.base_url.clone(), rank.dp_rank))?;
            Some(RankTokenScore {
                worker_index: rank.worker_index,
                base_url: rank.base_url.clone(),
                dp_rank: rank.dp_rank,
                source_total_tokens: budget.source_total_tokens,
                source_total_requests: budget.source_total_requests,
                overlay_tokens: budget.overlay_tokens,
                overlay_requests: budget.overlay_requests,
                effective_total_tokens: budget.effective_total_tokens(),
                effective_total_requests: budget.effective_total_requests(),
            })
        })
        .collect()
}

fn select_rank_total_tokens(scores: &[RankTokenScore]) -> Option<usize> {
    scores
        .iter()
        .min_by_key(|score| {
            (
                score.effective_total_tokens,
                score.effective_total_requests,
                score.base_url.as_str(),
                score.dp_rank,
                score.worker_index,
            )
        })
        .map(|score| score.worker_index)
}

const RANK_HASH_VIRTUAL_NODES: u32 = 160;

fn rank_routing_key(info: &SelectWorkerInfo<'_>) -> Option<(&'static str, Vec<u8>)> {
    for (header, source) in [
        ("x-smg-routing-key", "x-smg-routing-key"),
        ("x-session-id", "x-session-id"),
        ("x-user-id", "x-user-id"),
        ("x-tenant-id", "x-tenant-id"),
    ] {
        if let Some(value) = info.headers.and_then(|headers| headers.get(header)) {
            if !value.as_bytes().is_empty() {
                return Some((source, value.as_bytes().to_vec()));
            }
        }
    }

    if let Some(tokens) = info.tokens.filter(|tokens| !tokens.is_empty()) {
        let mut bytes = b"async-moe-token-ids-v1\0".to_vec();
        for token in tokens {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        return Some(("token_ids", bytes));
    }
    info.request_text
        .filter(|text| !text.is_empty())
        .map(|text| ("request_text", text.as_bytes().to_vec()))
}

fn sha256_digest(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn hash_position(digest: &[u8; 32]) -> u64 {
    u64::from_be_bytes(digest[..8].try_into().expect("SHA-256 prefix is 8 bytes"))
}

fn rank_ring_position(identity: &str, virtual_node: u32) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(b"async-moe-rank-ring-v1\0");
    hasher.update(identity.as_bytes());
    hasher.update([0]);
    hasher.update(virtual_node.to_le_bytes());
    hash_position(&hasher.finalize().into())
}

fn select_rank_consistent_hash(scores: &[RankScore], key_digest: &[u8; 32]) -> Option<usize> {
    let mut ring = Vec::with_capacity(
        scores
            .len()
            .saturating_mul(RANK_HASH_VIRTUAL_NODES as usize),
    );
    let mut identities = BTreeSet::new();
    for score in scores {
        let identity = format!("{}@{}", score.base_url, score.dp_rank);
        if !identities.insert(identity.clone()) {
            return None;
        }
        for virtual_node in 0..RANK_HASH_VIRTUAL_NODES {
            ring.push((
                rank_ring_position(&identity, virtual_node),
                identity.clone(),
                virtual_node,
                score.worker_index,
            ));
        }
    }
    ring.sort_unstable_by(|a, b| (a.0, &a.1, a.2).cmp(&(b.0, &b.1, b.2)));
    let key_position = hash_position(key_digest);
    let index = ring
        .partition_point(|entry| entry.0 < key_position)
        .checked_rem(ring.len())?;
    Some(ring[index].3)
}

fn digest_hex(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in digest {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

#[derive(Debug)]
struct PrefillHistoryEntry {
    assigned_at: Instant,
    rank: RankHistoryKey,
    uncached_tokens: usize,
}

/// Request-assignment history used by the prefill-only E2 reproduction.
///
/// Preble defines recent load as the prefill/decode computation incurred by
/// requests in a history window H. In prefill-only mode, uncached prompt
/// tokens are the paper's scheduler-side proxy for prefill computation. The
/// history is updated at the same atomic selection point as the route, so
/// concurrent arrivals cannot all observe a rank before any of them charge it.
#[derive(Debug)]
struct RecentPrefillHistory {
    window: Duration,
    entries: VecDeque<PrefillHistoryEntry>,
    totals: BTreeMap<RankHistoryKey, usize>,
}

impl RecentPrefillHistory {
    fn new(window: Duration) -> Self {
        Self {
            window,
            entries: VecDeque::new(),
            totals: BTreeMap::new(),
        }
    }

    fn prune(&mut self, now: Instant) {
        while self
            .entries
            .front()
            .is_some_and(|entry| now.saturating_duration_since(entry.assigned_at) >= self.window)
        {
            let expired = self.entries.pop_front().expect("front was present");
            if let Some(total) = self.totals.get_mut(&expired.rank) {
                *total = total.saturating_sub(expired.uncached_tokens);
                if *total == 0 {
                    self.totals.remove(&expired.rank);
                }
            }
        }
    }

    fn loads(&mut self, now: Instant) -> BTreeMap<RankHistoryKey, usize> {
        self.prune(now);
        self.totals.clone()
    }

    fn record(&mut self, now: Instant, score: &RankScore, uncached_tokens: usize) {
        if uncached_tokens == 0 {
            return;
        }
        let rank = (score.base_url.clone(), score.dp_rank);
        self.totals
            .entry(rank.clone())
            .and_modify(|total| *total = total.saturating_add(uncached_tokens))
            .or_insert(uncached_tokens);
        self.entries.push_back(PrefillHistoryEntry {
            assigned_at: now,
            rank,
            uncached_tokens,
        });
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.totals.clear();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PreblePrefillCost {
    recent_prefill_work: usize,
    new_prefill_work: usize,
    total: usize,
}

fn preble_prefill_cost(
    prompt_tokens: usize,
    score: &RankScore,
    recent_loads: &BTreeMap<RankHistoryKey, usize>,
) -> PreblePrefillCost {
    let recent_prefill_work = recent_loads
        .get(&(score.base_url.clone(), score.dp_rank))
        .copied()
        .unwrap_or(0);
    let new_prefill_work = prompt_tokens.saturating_sub(score.matched_tokens);
    PreblePrefillCost {
        recent_prefill_work,
        new_prefill_work,
        total: recent_prefill_work.saturating_add(new_prefill_work),
    }
}

fn select_preble_prefill_candidate(
    scores: &[RankScore],
    prompt_tokens: usize,
    recent_loads: &BTreeMap<RankHistoryKey, usize>,
    tie_rr: &AtomicUsize,
) -> Option<(usize, &'static str)> {
    let cached_len = scores.iter().map(|score| score.matched_tokens).max()?;
    let missed_len = prompt_tokens.saturating_sub(cached_len);
    let exploit = missed_len < cached_len;
    let candidates: Vec<&RankScore> = if exploit {
        scores
            .iter()
            .filter(|score| score.matched_tokens == cached_len)
            .collect()
    } else {
        scores.iter().collect()
    };
    let best = candidates
        .iter()
        .map(|score| preble_prefill_cost(prompt_tokens, score, recent_loads).total)
        .min()?;
    let mut tied = candidates
        .into_iter()
        .filter(|score| preble_prefill_cost(prompt_tokens, score, recent_loads).total == best)
        .collect();
    let selected = pick_rr(&mut tied, tie_rr)?;
    Some((
        selected,
        if exploit {
            "preble_exploit"
        } else {
            "preble_explore"
        },
    ))
}

fn audit_preble_prefill_decision(
    info: &SelectWorkerInfo<'_>,
    scores: &[RankScore],
    recent_loads: &BTreeMap<RankHistoryKey, usize>,
    selected: usize,
    reason: &'static str,
) {
    if !tracing::enabled!(target: "async_moe::dp_rank_decision", tracing::Level::DEBUG) {
        return;
    }
    let Some(prompt_tokens) = info.tokens.map(<[u32]>::len) else {
        return;
    };
    let Some(chosen) = scores.iter().find(|score| score.worker_index == selected) else {
        return;
    };
    let candidates: Vec<_> = scores
        .iter()
        .map(|score| {
            let cost = preble_prefill_cost(prompt_tokens, score, recent_loads);
            serde_json::json!({
                "base_url": score.base_url,
                "dp_rank": score.dp_rank,
                "matched_blocks": score.matched_blocks,
                "matched_tokens": score.matched_tokens,
                "uncached_tokens": cost.new_prefill_work,
                "recent_prefill_work": cost.recent_prefill_work,
                "load_cost": cost.total,
            })
        })
        .collect();
    debug!(
        target: "async_moe::dp_rank_decision",
        policy = "preble_e2_prefill",
        request_id = request_id(info),
        selected_base_url = chosen.base_url,
        selected_dp_rank = chosen.dp_rank,
        selected_matched_blocks = chosen.matched_blocks,
        selected_matched_tokens = chosen.matched_tokens,
        reason,
        candidates = %serde_json::Value::Array(candidates),
        "DP-rank routing decision"
    );
}

/// Sliding-window adaptive prefix table from DualMap.  The table starts with
/// one 512-token routing block and expands one level when a prefix accounts
/// for more than `2 / num_ranks` of the recent requests.  It contracts when
/// the parent falls below `1 / num_ranks`.
#[derive(Debug)]
struct AdaptivePrefixTracker {
    window_size: usize,
    min_samples: usize,
    window: VecDeque<Vec<u64>>,
    expanded: BTreeSet<Vec<u64>>,
}

impl AdaptivePrefixTracker {
    fn new(window_size: usize, min_samples: usize) -> Self {
        Self {
            window_size,
            min_samples,
            window: VecDeque::new(),
            expanded: BTreeSet::new(),
        }
    }

    fn ratio(&self, prefix: &[u64]) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        let matches = self
            .window
            .iter()
            .filter(|observed| observed.starts_with(prefix))
            .count();
        matches as f64 / self.window.len() as f64
    }

    fn select_and_observe(&mut self, blocks: &[u64], num_ranks: usize) -> Vec<u64> {
        debug_assert!(!blocks.is_empty());
        debug_assert!(num_ranks > 0);

        let mut depth = 1usize;
        while depth < blocks.len() && self.expanded.contains(&blocks[..depth]) {
            depth += 1;
        }
        let selected = blocks[..depth].to_vec();

        self.window.push_back(blocks.to_vec());
        while self.window.len() > self.window_size {
            self.window.pop_front();
        }

        if self.window.len() >= self.min_samples {
            let hot_threshold = 2.0 / num_ranks as f64;
            let cold_threshold = 1.0 / num_ranks as f64;
            if depth < blocks.len() && self.ratio(&selected) > hot_threshold {
                self.expanded.insert(selected.clone());
            } else if depth > 1 {
                let parent = &blocks[..depth - 1];
                if self.ratio(parent) < cold_threshold {
                    self.expanded.remove(parent);
                }
            }
        }

        selected
    }

    fn clear(&mut self) {
        self.window.clear();
        self.expanded.clear();
    }
}

#[derive(Debug, Clone)]
struct DualMapReservation {
    rank: RankHistoryKey,
    uncached_tokens: usize,
}

#[derive(Debug)]
struct DualMapRoutingState {
    prefixes: AdaptivePrefixTracker,
    pending_prefill_tokens: BTreeMap<RankHistoryKey, usize>,
    reservations: BTreeMap<u64, DualMapReservation>,
}

impl DualMapRoutingState {
    fn new(config: DualMapConfig) -> Self {
        Self {
            prefixes: AdaptivePrefixTracker::new(
                config.prefix_window_size,
                config.prefix_min_samples,
            ),
            pending_prefill_tokens: BTreeMap::new(),
            reservations: BTreeMap::new(),
        }
    }

    #[cfg(test)]
    fn pending(&self, score: &RankScore) -> usize {
        self.pending_prefill_tokens
            .get(&(score.base_url.clone(), score.dp_rank))
            .copied()
            .unwrap_or(0)
    }

    fn reserve(&mut self, selection_id: u64, score: &RankScore, uncached_tokens: usize) {
        // The router supplies monotonically increasing IDs.  Handling a
        // duplicate defensively prevents a stale reservation from leaking.
        self.release(selection_id);
        let rank = (score.base_url.clone(), score.dp_rank);
        self.pending_prefill_tokens
            .entry(rank.clone())
            .and_modify(|pending| *pending = pending.saturating_add(uncached_tokens))
            .or_insert(uncached_tokens);
        self.reservations.insert(
            selection_id,
            DualMapReservation {
                rank,
                uncached_tokens,
            },
        );
    }

    fn release(&mut self, selection_id: u64) {
        let Some(reservation) = self.reservations.remove(&selection_id) else {
            return;
        };
        if let Some(pending) = self.pending_prefill_tokens.get_mut(&reservation.rank) {
            *pending = pending.saturating_sub(reservation.uncached_tokens);
            if *pending == 0 {
                self.pending_prefill_tokens.remove(&reservation.rank);
            }
        }
    }

    fn clear(&mut self) {
        self.prefixes.clear();
        self.pending_prefill_tokens.clear();
        self.reservations.clear();
    }
}

fn hash_token_blocks(tokens: &[u32], block_tokens: usize) -> Vec<u64> {
    tokens
        .chunks(block_tokens)
        .map(|chunk| {
            let mut bytes = Vec::with_capacity(chunk.len().saturating_mul(4));
            for token in chunk {
                bytes.extend_from_slice(&token.to_le_bytes());
            }
            xxh3_64_with_seed(&bytes, 0x4455_414c_4d41_5000)
        })
        .collect()
}

fn hash_prefix_blocks(prefix: &[u64], seed: u64) -> u64 {
    let mut bytes = Vec::with_capacity(prefix.len().saturating_mul(8));
    for block in prefix {
        bytes.extend_from_slice(&block.to_le_bytes());
    }
    xxh3_64_with_seed(&bytes, seed)
}

fn dualmap_ring_candidate(scores: &[RankScore], prefix: &[u64], seed: u64) -> Option<usize> {
    let key_position = hash_prefix_blocks(prefix, seed);
    scores
        .iter()
        .min_by_key(|score| {
            let identity = format!("{}@{}", score.base_url, score.dp_rank);
            let anchor = xxh3_64_with_seed(identity.as_bytes(), seed);
            anchor.wrapping_sub(key_position)
        })
        .map(|score| score.worker_index)
}

fn dualmap_candidates(scores: &[RankScore], prefix: &[u64]) -> Option<Vec<usize>> {
    let first = dualmap_ring_candidate(scores, prefix, 0x4455_414c_4d41_5001)?;
    if scores.len() == 1 {
        return Some(vec![first]);
    }
    let second = dualmap_ring_candidate(scores, prefix, 0x4455_414c_4d41_5002)?;
    if first != second {
        return Some(vec![first, second]);
    }

    // DualMap requires two distinct choices.  Its reference implementation
    // uses `(first + 1) % n`; stable rank order is the topology-independent
    // equivalent when worker indices are not dense rank IDs.
    let mut stable: Vec<&RankScore> = scores.iter().collect();
    stable.sort_by_key(|score| stable_rank_key(score));
    let first_pos = stable
        .iter()
        .position(|score| score.worker_index == first)?;
    Some(vec![
        first,
        stable[(first_pos + 1) % stable.len()].worker_index,
    ])
}

fn virtual_prefill_tokens(
    prompt_tokens: usize,
    score: &RankScore,
    pending: &BTreeMap<RankHistoryKey, usize>,
) -> usize {
    pending
        .get(&(score.base_url.clone(), score.dp_rank))
        .copied()
        .unwrap_or(0)
        .saturating_add(prompt_tokens.saturating_sub(score.matched_tokens))
}

fn select_dualmap_candidate(
    scores: &[RankScore],
    candidates: &[usize],
    prompt_tokens: usize,
    pending: &BTreeMap<RankHistoryKey, usize>,
    slo_token_threshold: usize,
    tie_rr: &AtomicUsize,
) -> Option<(usize, &'static str)> {
    let candidate_scores: Vec<&RankScore> = candidates
        .iter()
        .filter_map(|worker_index| {
            scores
                .iter()
                .find(|score| score.worker_index == *worker_index)
        })
        .collect();
    let max_cache = candidate_scores
        .iter()
        .map(|score| score.matched_tokens)
        .max()?;
    let min_virtual_for_cache = candidate_scores
        .iter()
        .filter(|score| score.matched_tokens == max_cache)
        .map(|score| virtual_prefill_tokens(prompt_tokens, score, pending))
        .min()?;
    let mut cache_ties: Vec<&RankScore> = candidate_scores
        .iter()
        .copied()
        .filter(|score| {
            score.matched_tokens == max_cache
                && virtual_prefill_tokens(prompt_tokens, score, pending) == min_virtual_for_cache
        })
        .collect();
    let cache_choice = pick_rr(&mut cache_ties, tie_rr)?;

    if min_virtual_for_cache <= slo_token_threshold {
        return Some((
            cache_choice,
            if candidate_scores
                .iter()
                .all(|score| score.matched_tokens == max_cache)
            {
                "dualmap_equal_cache_min_ttft"
            } else {
                "dualmap_cache_affinity"
            },
        ));
    }

    let min_virtual = candidate_scores
        .iter()
        .map(|score| virtual_prefill_tokens(prompt_tokens, score, pending))
        .min()?;
    let mut ttft_ties: Vec<&RankScore> = candidate_scores
        .into_iter()
        .filter(|score| virtual_prefill_tokens(prompt_tokens, score, pending) == min_virtual)
        .collect();
    Some((pick_rr(&mut ttft_ties, tie_rr)?, "dualmap_slo_min_ttft"))
}

fn audit_dualmap_decision(
    info: &SelectWorkerInfo<'_>,
    scores: &[RankScore],
    candidate_indices: &[usize],
    pending: &BTreeMap<RankHistoryKey, usize>,
    selected: usize,
    reason: &'static str,
    routing_block_count: usize,
    adaptive_prefix_depth: usize,
    slo_token_threshold: usize,
) {
    if !tracing::enabled!(target: "async_moe::dp_rank_decision", tracing::Level::DEBUG) {
        return;
    }
    let Some(prompt_tokens) = info.tokens.map(<[u32]>::len) else {
        return;
    };
    let Some(chosen) = scores.iter().find(|score| score.worker_index == selected) else {
        return;
    };
    let candidates: Vec<_> = candidate_indices
        .iter()
        .filter_map(|worker_index| {
            let score = scores
                .iter()
                .find(|score| score.worker_index == *worker_index)?;
            let pending_tokens = pending
                .get(&(score.base_url.clone(), score.dp_rank))
                .copied()
                .unwrap_or(0);
            Some(serde_json::json!({
                "base_url": score.base_url,
                "dp_rank": score.dp_rank,
                "matched_tokens": score.matched_tokens,
                "uncached_tokens": prompt_tokens.saturating_sub(score.matched_tokens),
                "pending_prefill_tokens": pending_tokens,
                "virtual_prefill_tokens": virtual_prefill_tokens(prompt_tokens, score, pending),
            }))
        })
        .collect();
    debug!(
        target: "async_moe::dp_rank_decision",
        policy = "dualmap",
        request_id = request_id(info),
        selected_base_url = chosen.base_url,
        selected_dp_rank = chosen.dp_rank,
        selected_matched_tokens = chosen.matched_tokens,
        routing_block_count,
        adaptive_prefix_depth,
        slo_token_threshold,
        reason,
        candidates = %serde_json::Value::Array(candidates),
        "DP-rank routing decision"
    );
}

fn load_scores(workers: &[Arc<dyn Worker>]) -> Option<Vec<RankScore>> {
    let healthy = get_healthy_worker_indices(workers);
    if healthy.is_empty() || healthy.iter().any(|&idx| workers[idx].dp_rank().is_none()) {
        return None;
    }
    healthy
        .into_iter()
        .map(|worker_index| {
            let worker = &workers[worker_index];
            Some(RankScore {
                worker_index,
                base_url: worker.base_url().trim_end_matches('/').to_string(),
                dp_rank: u32::try_from(worker.dp_rank()?).ok()?,
                matched_blocks: 0,
                matched_tokens: 0,
                load: worker.load(),
            })
        })
        .collect()
}

async fn exact_scores(
    state: &ExactKvRankState,
    workers: &[Arc<dyn Worker>],
    info: &SelectWorkerInfo<'_>,
) -> Option<Vec<RankScore>> {
    let healthy = get_healthy_worker_indices(workers);
    if healthy.is_empty() {
        return None;
    }
    let tokens = info.tokens.filter(|tokens| !tokens.is_empty())?;
    let index = state.prepare(workers, &healthy).await;
    let block_size = index.block_size_oracle().get()? as usize;
    let block_hashes = if index.block_size_oracle().is_bigram() {
        compute_block_hashes_bigram(tokens, block_size)
    } else {
        compute_block_hashes(tokens, block_size)
    };

    let mut identities = Vec::with_capacity(healthy.len());
    for worker_index in healthy {
        let worker = &workers[worker_index];
        let dp_rank = u32::try_from(worker.dp_rank()?).ok()?;
        let base_url = worker.base_url().trim_end_matches('/').to_string();
        let kv_worker = KvWorkerId::new(base_url.clone(), dp_rank);
        identities.push((worker_index, base_url, dp_rank, kv_worker));
    }

    // Every subscriber starts concurrently. Join their bounded readiness
    // waits so a completely unavailable DP=8 publisher costs one timeout,
    // not eight serial timeouts.
    let readiness = futures::future::join_all(
        identities
            .iter()
            .map(|(_, _, _, kv_worker)| index.wait_ready(kv_worker)),
    )
    .await;
    if readiness.iter().any(|ready| !ready) {
        // Tear down the failed attachment so a later request can rediscover
        // and retry after a transient publisher or network failure.
        let failed_urls: BTreeSet<String> = identities
            .iter()
            .zip(readiness.iter())
            .filter(|(_, ready)| !**ready)
            .map(|((_, base_url, _, _), _)| base_url.clone())
            .collect();
        for url in failed_urls {
            index.remove_worker(&url).await;
        }
        return None;
    }

    let tree = index.tree();
    let mut scores = Vec::with_capacity(identities.len());
    for (worker_index, base_url, dp_rank, kv_worker) in identities {
        let worker = &workers[worker_index];
        let matched_blocks = tree.match_prefix_for_worker(&kv_worker, &block_hashes);
        scores.push(RankScore {
            worker_index,
            base_url,
            dp_rank,
            matched_blocks,
            matched_tokens: matched_blocks.saturating_mul(block_size).min(tokens.len()),
            load: worker.load(),
        });
    }
    Some(scores)
}

fn p2c_second_index(first: usize, offset: usize, candidate_count: usize) -> usize {
    debug_assert!(candidate_count >= 2);
    debug_assert!(first < candidate_count);
    debug_assert!(offset < candidate_count - 1);
    (first + 1 + offset) % candidate_count
}

fn sample_two_distinct(candidate_count: usize) -> (usize, usize) {
    debug_assert!(candidate_count >= 2);
    let mut rng = rand::rng();
    let first = rng.random_range(0..candidate_count);
    let offset = rng.random_range(0..candidate_count - 1);
    (first, p2c_second_index(first, offset, candidate_count))
}

/// Classic power-of-two choices over DP-rank candidates, using gateway-local
/// in-flight requests as load. This intentionally ignores engine-wide load
/// snapshots keyed only by base URL because those cannot distinguish ranks.
#[derive(Debug, Default)]
pub struct RankPowerOfTwoPolicy;

impl RankPowerOfTwoPolicy {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl LoadBalancingPolicy for RankPowerOfTwoPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let healthy = get_healthy_worker_indices(workers);
        if healthy.is_empty() {
            return None;
        }
        if healthy.iter().any(|&idx| workers[idx].dp_rank().is_none()) {
            return None;
        }
        if healthy.len() == 1 {
            return Some(healthy[0]);
        }

        let (first, second) = sample_two_distinct(healthy.len());
        let a = healthy[first];
        let b = healthy[second];
        let selected = if workers[a].load() <= workers[b].load() {
            a
        } else {
            b
        };
        let scores = [a, b]
            .into_iter()
            .map(|worker_index| RankScore {
                worker_index,
                base_url: workers[worker_index].base_url().to_string(),
                dp_rank: workers[worker_index].dp_rank().unwrap_or_default() as u32,
                matched_blocks: 0,
                matched_tokens: 0,
                load: workers[worker_index].load(),
            })
            .collect::<Vec<_>>();
        audit_decision(self.name(), info, &scores, selected, "power_of_two");
        workers[selected].increment_processed();
        Some(selected)
    }

    fn name(&self) -> &'static str {
        "rank_power_of_two"
    }

    fn tracks_inflight_load(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Join-the-shortest-queue over all healthy DP ranks. Exact load ties rotate
/// in stable rank order so an all-idle system does not pin rank zero.
#[derive(Debug)]
pub struct RankLeastLoadedPolicy {
    tie_rr: AtomicUsize,
}

impl RankLeastLoadedPolicy {
    pub fn new() -> Self {
        Self {
            tie_rr: AtomicUsize::new(0),
        }
    }
}

impl Default for RankLeastLoadedPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LoadBalancingPolicy for RankLeastLoadedPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let scores = load_scores(workers)?;
        let min_load = scores.iter().map(|score| score.load).min()?;
        let mut tied = scores
            .iter()
            .filter(|score| score.load == min_load)
            .collect();
        let selected = pick_rr(&mut tied, &self.tie_rr)?;
        audit_decision(self.name(), info, &scores, selected, "least_inflight");
        workers[selected].increment_processed();
        Some(selected)
    }

    fn name(&self) -> &'static str {
        "rank_least_loaded"
    }

    fn tracks_inflight_load(&self) -> bool {
        true
    }

    fn reset(&self) {
        self.tie_rr.store(0, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// SGLang DPC `total_tokens` adapted to gateway-visible DP ranks.
///
/// The engine's per-rank `num_total_tokens` is authoritative. Between newer
/// source timestamps, this policy adds prompt-token and request-count
/// reservations at the atomic selection point so a burst cannot repeatedly
/// choose the same rank from one snapshot. The overlay is intentionally the
/// same generation heuristic as SGLang DPC: a newer source timestamp replaces
/// it. It is not response-lifetime accounting or an admission acknowledgement.
#[derive(Debug)]
pub struct RankTotalTokensPolicy {
    config: RankTotalTokensConfig,
    client: reqwest::Client,
    state: Mutex<RankTokenLoadState>,
}

impl RankTotalTokensPolicy {
    pub fn new(config: RankTotalTokensConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
            state: Mutex::new(RankTokenLoadState::default()),
        }
    }

    async fn fetch_snapshot(
        &self,
        workers: &[Arc<dyn Worker>],
        ranks: &[RankScore],
        expected: &BTreeSet<RankHistoryKey>,
    ) -> Option<BTreeMap<RankHistoryKey, EngineRankLoad>> {
        let mut engines = BTreeMap::<String, Option<String>>::new();
        for rank in ranks {
            let api_key = workers[rank.worker_index].api_key().clone();
            match engines.get(&rank.base_url) {
                Some(existing) if existing != &api_key => return None,
                Some(_) => {}
                None => {
                    engines.insert(rank.base_url.clone(), api_key);
                }
            }
        }

        let timeout = Duration::from_millis(self.config.request_timeout_ms);
        let responses =
            futures::future::join_all(engines.into_iter().map(|(base_url, api_key)| {
                let client = self.client.clone();
                async move {
                    let url = format!("{base_url}/v1/loads?include=core");
                    let mut request = client.get(url).timeout(timeout);
                    if let Some(key) = api_key {
                        request = request.bearer_auth(key);
                    }
                    let response = request.send().await.ok()?.error_for_status().ok()?;
                    let loads = response.json::<EngineRankLoadsResponse>().await.ok()?;
                    Some((base_url, loads.loads))
                }
            }))
            .await;

        let mut snapshot = BTreeMap::new();
        for response in responses {
            let (base_url, loads) = response?;
            for load in loads {
                let key = (base_url.clone(), load.dp_rank);
                if expected.contains(&key) && snapshot.insert(key, load).is_some() {
                    return None;
                }
            }
        }
        Some(snapshot)
    }

    fn audit(&self, info: &SelectWorkerInfo<'_>, scores: &[RankTokenScore], selected: usize) {
        if !tracing::enabled!(target: "async_moe::dp_rank_decision", tracing::Level::DEBUG) {
            return;
        }
        let Some(chosen) = scores.iter().find(|score| score.worker_index == selected) else {
            return;
        };
        let candidates: Vec<_> = scores
            .iter()
            .map(|score| {
                serde_json::json!({
                    "base_url": score.base_url,
                    "dp_rank": score.dp_rank,
                    "source_total_tokens": score.source_total_tokens,
                    "source_total_requests": score.source_total_requests,
                    "overlay_tokens": score.overlay_tokens,
                    "overlay_requests": score.overlay_requests,
                    "effective_total_tokens": score.effective_total_tokens,
                    "effective_total_requests": score.effective_total_requests,
                })
            })
            .collect();
        debug!(
            target: "async_moe::dp_rank_decision",
            policy = "rank_total_tokens",
            request_id = request_id(info),
            prompt_tokens = info.tokens.map(<[u32]>::len),
            selected_base_url = chosen.base_url,
            selected_dp_rank = chosen.dp_rank,
            selected_effective_total_tokens = chosen.effective_total_tokens,
            selected_effective_total_requests = chosen.effective_total_requests,
            reason = "least_total_tokens",
            candidates = %serde_json::Value::Array(candidates),
            "DP-rank routing decision"
        );
    }
}

impl Default for RankTotalTokensPolicy {
    fn default() -> Self {
        Self::new(RankTotalTokensConfig::default())
    }
}

#[async_trait]
impl LoadBalancingPolicy for RankTotalTokensPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let prompt_tokens = info.tokens?.len();
        let ranks = load_scores(workers)?;
        let expected: BTreeSet<_> = ranks
            .iter()
            .map(|rank| (rank.base_url.clone(), rank.dp_rank))
            .collect();
        let now = Instant::now();
        let mut state = self.state.lock().await;
        if !state.topology_matches(&expected)
            || state.refresh_due(now, Duration::from_millis(self.config.max_staleness_ms))
        {
            let snapshot = self.fetch_snapshot(workers, &ranks, &expected).await?;
            if !state.apply_snapshot(&expected, snapshot, now) {
                return None;
            }
        }

        let scores = rank_token_scores(&ranks, &state.budgets)?;
        let selected = select_rank_total_tokens(&scores)?;
        let chosen = scores.iter().find(|score| score.worker_index == selected)?;
        let key = (chosen.base_url.clone(), chosen.dp_rank);
        self.audit(info, &scores, selected);
        if !state.reserve(&key, prompt_tokens) {
            return None;
        }
        drop(state);
        workers[selected].increment_processed();
        Some(selected)
    }

    fn name(&self) -> &'static str {
        "rank_total_tokens"
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn reset(&self) {
        if let Ok(mut state) = self.state.try_lock() {
            state.clear();
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Stable DP-rank session affinity with a policy-local SHA-256 hash ring.
/// Header keys take explicit priority; token IDs provide deterministic
/// affinity when callers do not supply a key. Only the key digest is audited.
#[derive(Debug, Default)]
pub struct RankConsistentHashPolicy;

impl RankConsistentHashPolicy {
    pub fn new() -> Self {
        Self
    }

    fn audit(
        &self,
        info: &SelectWorkerInfo<'_>,
        scores: &[RankScore],
        selected: usize,
        key_source: &'static str,
        key_digest: &[u8; 32],
    ) {
        if !tracing::enabled!(target: "async_moe::dp_rank_decision", tracing::Level::DEBUG) {
            return;
        }
        let Some(chosen) = scores.iter().find(|score| score.worker_index == selected) else {
            return;
        };
        let candidates: Vec<_> = scores
            .iter()
            .map(|score| {
                serde_json::json!({
                    "base_url": score.base_url,
                    "dp_rank": score.dp_rank,
                })
            })
            .collect();
        debug!(
            target: "async_moe::dp_rank_decision",
            policy = "rank_consistent_hash",
            request_id = request_id(info),
            selected_base_url = chosen.base_url,
            selected_dp_rank = chosen.dp_rank,
            key_source,
            routing_key_sha256 = digest_hex(key_digest),
            virtual_nodes = RANK_HASH_VIRTUAL_NODES,
            reason = "consistent_hash",
            candidates = %serde_json::Value::Array(candidates),
            "DP-rank routing decision"
        );
    }
}

#[async_trait]
impl LoadBalancingPolicy for RankConsistentHashPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let scores = load_scores(workers)?;
        let (key_source, key) = rank_routing_key(info)?;
        let key_digest = sha256_digest(&key);
        let selected = select_rank_consistent_hash(&scores, &key_digest)?;
        self.audit(info, &scores, selected, key_source, &key_digest);
        workers[selected].increment_processed();
        Some(selected)
    }

    fn name(&self) -> &'static str {
        "rank_consistent_hash"
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Cache-only longest-prefix-match. Load is intentionally excluded; ties,
/// including an all-cold cache, rotate in stable `(base_url, rank)` order.
#[derive(Debug)]
pub struct PrefixOnlyLpmPolicy {
    state: ExactKvRankState,
    tie_rr: AtomicUsize,
}

impl PrefixOnlyLpmPolicy {
    pub fn new() -> Self {
        Self {
            state: ExactKvRankState::new(),
            tie_rr: AtomicUsize::new(0),
        }
    }
}

impl Default for PrefixOnlyLpmPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LoadBalancingPolicy for PrefixOnlyLpmPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let scores = exact_scores(&self.state, workers, info).await?;
        let best = scores.iter().map(|score| score.matched_blocks).max()?;
        let mut tied = scores
            .iter()
            .filter(|score| score.matched_blocks == best)
            .collect();
        let selected = pick_rr(&mut tied, &self.tie_rr)?;
        audit_decision(self.name(), info, &scores, selected, "longest_prefix");
        workers[selected].increment_processed();
        Some(selected)
    }

    fn name(&self) -> &'static str {
        "prefix_only_lpm"
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn reset(&self) {
        self.tie_rr.store(0, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// LMetric's rank-adapted cost: `(prompt_tokens - matched_tokens) *
/// (inflight_requests + 1)`. Lowest cost wins; exact ties rotate instead of
/// pinning the lowest rank.
#[derive(Debug)]
pub struct LMetricPolicy {
    state: ExactKvRankState,
    tie_rr: AtomicUsize,
}

impl LMetricPolicy {
    pub fn new() -> Self {
        Self {
            state: ExactKvRankState::new(),
            tie_rr: AtomicUsize::new(0),
        }
    }
}

impl Default for LMetricPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LoadBalancingPolicy for LMetricPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let scores = exact_scores(&self.state, workers, info).await?;
        let prompt_tokens = info.tokens?.len();
        let cost = |score: &RankScore| lmetric_cost(prompt_tokens, score);
        let best = scores.iter().map(&cost).min()?;
        let mut tied = scores.iter().filter(|score| cost(score) == best).collect();
        let selected = pick_rr(&mut tied, &self.tie_rr)?;
        audit_decision(self.name(), info, &scores, selected, "lmetric_min_cost");
        workers[selected].increment_processed();
        Some(selected)
    }

    fn name(&self) -> &'static str {
        "lmetric"
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn tracks_inflight_load(&self) -> bool {
        true
    }

    fn reset(&self) {
        self.tie_rr.store(0, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Routing-only, prefill-only reproduction of Preble's E2 global scheduler.
///
/// The paper's exploit/explore gate and `recent load + new request` prefill
/// cost are preserved. Recent load is the uncached-token work assigned in the
/// configured history window (Preble's default H is three minutes). Decode
/// cost, eviction simulation, the local priority scheduler, prefix
/// replication, and autoscaling are intentionally outside this point policy.
#[derive(Debug)]
pub struct PrebleE2PrefillPolicy {
    state: ExactKvRankState,
    history: StdMutex<RecentPrefillHistory>,
    tie_rr: AtomicUsize,
}

impl PrebleE2PrefillPolicy {
    pub fn new(history_window_secs: u64) -> Self {
        Self {
            state: ExactKvRankState::new(),
            history: StdMutex::new(RecentPrefillHistory::new(Duration::from_secs(
                history_window_secs,
            ))),
            tie_rr: AtomicUsize::new(0),
        }
    }
}

impl Default for PrebleE2PrefillPolicy {
    fn default() -> Self {
        Self::new(180)
    }
}

#[async_trait]
impl LoadBalancingPolicy for PrebleE2PrefillPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let scores = exact_scores(&self.state, workers, info).await?;
        let prompt_tokens = info.tokens?.len();
        let now = Instant::now();
        let mut history = self.history.lock().ok()?;
        let recent_loads = history.loads(now);
        let (selected, reason) =
            select_preble_prefill_candidate(&scores, prompt_tokens, &recent_loads, &self.tie_rr)?;
        let chosen = scores.iter().find(|score| score.worker_index == selected)?;
        let selected_cost = preble_prefill_cost(prompt_tokens, chosen, &recent_loads);
        audit_preble_prefill_decision(info, &scores, &recent_loads, selected, reason);
        history.record(now, chosen, selected_cost.new_prefill_work);
        workers[selected].increment_processed();
        Some(selected)
    }

    fn name(&self) -> &'static str {
        "preble_e2_prefill"
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn reset(&self) {
        self.tie_rr.store(0, Ordering::Relaxed);
        if let Ok(mut history) = self.history.lock() {
            history.clear();
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Routing-only implementation of DualMap's global entrance scheduler.
///
/// This preserves the paper's adaptive partial-prefix hashing, two independent
/// consistent-hash candidates, exact KV-prefix scoring, and SLO-aware switch
/// from cache affinity to minimum predicted TTFT.  `pending_prefill_tokens`
/// are exact router-side reservations for active prefill-only requests and
/// are released by the HTTP response-lifetime guard.  Queue migration and
/// scale-out ring maintenance require DualMap's global waiting queue and are
/// intentionally not claimed by this point policy.
#[derive(Debug)]
pub struct DualMapPolicy {
    config: DualMapConfig,
    kv_state: ExactKvRankState,
    routing_state: StdMutex<DualMapRoutingState>,
    tie_rr: AtomicUsize,
}

impl DualMapPolicy {
    pub fn new(config: DualMapConfig) -> Self {
        Self {
            config,
            kv_state: ExactKvRankState::new(),
            routing_state: StdMutex::new(DualMapRoutingState::new(config)),
            tie_rr: AtomicUsize::new(0),
        }
    }
}

impl Default for DualMapPolicy {
    fn default() -> Self {
        Self::new(DualMapConfig::default())
    }
}

#[async_trait]
impl LoadBalancingPolicy for DualMapPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let selection_id = selection_id(info)?;
        let tokens = info.tokens.filter(|tokens| !tokens.is_empty())?;
        let scores = exact_scores(&self.kv_state, workers, info).await?;
        let routing_blocks = hash_token_blocks(tokens, self.config.prefix_block_tokens);
        let mut state = self.routing_state.lock().ok()?;
        let adaptive_prefix = state
            .prefixes
            .select_and_observe(&routing_blocks, scores.len());
        let candidates = dualmap_candidates(&scores, &adaptive_prefix)?;
        let (selected, reason) = select_dualmap_candidate(
            &scores,
            &candidates,
            tokens.len(),
            &state.pending_prefill_tokens,
            self.config.slo_token_threshold,
            &self.tie_rr,
        )?;
        let chosen = scores.iter().find(|score| score.worker_index == selected)?;
        let uncached_tokens = tokens.len().saturating_sub(chosen.matched_tokens);
        audit_dualmap_decision(
            info,
            &scores,
            &candidates,
            &state.pending_prefill_tokens,
            selected,
            reason,
            routing_blocks.len(),
            adaptive_prefix.len(),
            self.config.slo_token_threshold,
        );
        state.reserve(selection_id, chosen, uncached_tokens);
        workers[selected].increment_processed();
        Some(selected)
    }

    fn on_request_finished(&self, selection_id: u64) {
        if let Ok(mut state) = self.routing_state.lock() {
            state.release(selection_id);
        }
    }

    fn name(&self) -> &'static str {
        "dualmap"
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn tracks_inflight_load(&self) -> bool {
        true
    }

    fn reset(&self) {
        self.tie_rr.store(0, Ordering::Relaxed);
        if let Ok(mut state) = self.routing_state.lock() {
            state.clear();
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Session-aware cache routing. Requests with no substantial exact prefix are
/// treated as first turns and sent to the least-loaded rank. Requests with at
/// least `min_match_tokens` remain on a rank with the longest exact prefix.
#[derive(Debug)]
pub struct SMetricPolicy {
    min_match_tokens: usize,
    state: ExactKvRankState,
    tie_rr: AtomicUsize,
}

impl SMetricPolicy {
    pub fn new(min_match_tokens: usize) -> Self {
        Self {
            min_match_tokens,
            state: ExactKvRankState::new(),
            tie_rr: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl LoadBalancingPolicy for SMetricPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let scores = exact_scores(&self.state, workers, info).await?;
        let best_match = scores.iter().map(|score| score.matched_tokens).max()?;
        let (mut tied, reason): (Vec<&RankScore>, &'static str) = if best_match
            >= self.min_match_tokens
        {
            let min_load = scores
                .iter()
                .filter(|score| score.matched_tokens == best_match)
                .map(|score| score.load)
                .min()?;
            (
                scores
                    .iter()
                    .filter(|score| score.matched_tokens == best_match && score.load == min_load)
                    .collect(),
                "session_prefix",
            )
        } else {
            let min_load = scores.iter().map(|score| score.load).min()?;
            (
                scores
                    .iter()
                    .filter(|score| score.load == min_load)
                    .collect(),
                "first_turn_least_inflight",
            )
        };
        let selected = pick_rr(&mut tied, &self.tie_rr)?;
        audit_decision(self.name(), info, &scores, selected, reason);
        workers[selected].increment_processed();
        Some(selected)
    }

    fn name(&self) -> &'static str {
        "smetric"
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn tracks_inflight_load(&self) -> bool {
        true
    }

    fn reset(&self) {
        self.tie_rr.store(0, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Chunk-granular LMetric. This preserves LMetric's cache/load interaction but
/// scores the scheduler-visible number of uncached prefill chunks, making the
/// configured effective per-rank chunk size an explicit part of the policy.
#[derive(Debug)]
pub struct ChunkLMetricPolicy {
    chunk_size: usize,
    state: ExactKvRankState,
    tie_rr: AtomicUsize,
}

impl ChunkLMetricPolicy {
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            state: ExactKvRankState::new(),
            tie_rr: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl LoadBalancingPolicy for ChunkLMetricPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let scores = exact_scores(&self.state, workers, info).await?;
        let prompt_tokens = info.tokens?.len();
        let cost = |score: &RankScore| chunk_lmetric_cost(prompt_tokens, self.chunk_size, score);
        let best = scores.iter().map(&cost).min()?;
        let mut tied = scores.iter().filter(|score| cost(score) == best).collect();
        let selected = pick_rr(&mut tied, &self.tie_rr)?;
        audit_decision(
            self.name(),
            info,
            &scores,
            selected,
            "chunk_lmetric_min_cost",
        );
        workers[selected].increment_processed();
        Some(selected)
    }

    fn name(&self) -> &'static str {
        "chunk_lmetric"
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn tracks_inflight_load(&self) -> bool {
        true
    }

    fn reset(&self) {
        self.tie_rr.store(0, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Cache-aware power of two: sample two distinct healthy ranks and compare
/// their exact LMetric costs. Exact scores are still materialized for every
/// rank to preserve KV-readiness checks and decision audit; this policy tests
/// the effect of restricting the decision candidate set, not router scaling.
#[derive(Debug)]
pub struct CacheAwarePowerOfTwoPolicy {
    state: ExactKvRankState,
    tie_rr: AtomicUsize,
}

impl CacheAwarePowerOfTwoPolicy {
    pub fn new() -> Self {
        Self {
            state: ExactKvRankState::new(),
            tie_rr: AtomicUsize::new(0),
        }
    }
}

impl Default for CacheAwarePowerOfTwoPolicy {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LoadBalancingPolicy for CacheAwarePowerOfTwoPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let scores = exact_scores(&self.state, workers, info).await?;
        if scores.len() == 1 {
            let selected = scores[0].worker_index;
            audit_decision(self.name(), info, &scores, selected, "single_candidate");
            workers[selected].increment_processed();
            return Some(selected);
        }

        let prompt_tokens = info.tokens?.len();
        let (first, second) = sample_two_distinct(scores.len());
        let candidates = vec![scores[first].clone(), scores[second].clone()];
        let best = candidates
            .iter()
            .map(|score| lmetric_cost(prompt_tokens, score))
            .min()?;
        let mut tied = candidates
            .iter()
            .filter(|score| lmetric_cost(prompt_tokens, score) == best)
            .collect();
        let selected = pick_rr(&mut tied, &self.tie_rr)?;
        audit_decision(self.name(), info, &candidates, selected, "cache_aware_p2c");
        workers[selected].increment_processed();
        Some(selected)
    }

    fn name(&self) -> &'static str {
        "cache_aware_p2c"
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn tracks_inflight_load(&self) -> bool {
        true
    }

    fn reset(&self) {
        self.tie_rr.store(0, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// SGLang-style cache-aware policy over exact per-rank KV residency. Severe
/// load imbalance takes precedence; otherwise a sufficiently long best prefix
/// is selected, with local load used within the best-match set.
#[derive(Debug)]
pub struct CacheAwareRankPolicy {
    config: CacheAwareRankConfig,
    state: ExactKvRankState,
    tie_rr: AtomicUsize,
}

impl CacheAwareRankPolicy {
    pub fn new(config: CacheAwareRankConfig) -> Self {
        Self {
            config,
            state: ExactKvRankState::new(),
            tie_rr: AtomicUsize::new(0),
        }
    }
}

impl Default for CacheAwareRankPolicy {
    fn default() -> Self {
        Self::new(CacheAwareRankConfig::default())
    }
}

#[async_trait]
impl LoadBalancingPolicy for CacheAwareRankPolicy {
    async fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        info: &SelectWorkerInfo<'_>,
    ) -> Option<usize> {
        let scores = exact_scores(&self.state, workers, info).await?;
        let min_load = scores.iter().map(|score| score.load).min()?;
        let max_load = scores.iter().map(|score| score.load).max()?;
        let imbalanced = max_load.saturating_sub(min_load) > self.config.balance_abs_threshold
            && (max_load as f32) > (min_load as f32 * self.config.balance_rel_threshold);

        let (mut tied, reason): (Vec<&RankScore>, &'static str) = if imbalanced {
            (
                scores
                    .iter()
                    .filter(|score| score.load == min_load)
                    .collect(),
                "load_imbalance",
            )
        } else {
            let best_blocks = scores.iter().map(|score| score.matched_blocks).max()?;
            let prompt_tokens = info.tokens?.len();
            let best_tokens = scores
                .iter()
                .filter(|score| score.matched_blocks == best_blocks)
                .map(|score| score.matched_tokens)
                .max()
                .unwrap_or(0);
            let match_rate = if prompt_tokens == 0 {
                0.0
            } else {
                best_tokens as f32 / prompt_tokens as f32
            };
            if best_blocks > 0 && match_rate > self.config.cache_threshold {
                let best_match_load = scores
                    .iter()
                    .filter(|score| score.matched_blocks == best_blocks)
                    .map(|score| score.load)
                    .min()?;
                (
                    scores
                        .iter()
                        .filter(|score| {
                            score.matched_blocks == best_blocks && score.load == best_match_load
                        })
                        .collect(),
                    "cache_hit",
                )
            } else {
                (
                    scores
                        .iter()
                        .filter(|score| score.load == min_load)
                        .collect(),
                    "cache_below_threshold",
                )
            }
        };

        let selected = pick_rr(&mut tied, &self.tie_rr)?;
        audit_decision(self.name(), info, &scores, selected, reason);
        workers[selected].increment_processed();
        Some(selected)
    }

    fn name(&self) -> &'static str {
        "cache_aware_rank"
    }

    fn needs_request_tokens(&self) -> bool {
        true
    }

    fn tracks_inflight_load(&self) -> bool {
        true
    }

    fn reset(&self) {
        self.tie_rr.store(0, Ordering::Relaxed);
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score(rank: u32, blocks: usize, tokens: usize, load: usize) -> RankScore {
        RankScore {
            worker_index: rank as usize,
            base_url: "http://engine:30000".into(),
            dp_rank: rank,
            matched_blocks: blocks,
            matched_tokens: tokens,
            load,
        }
    }

    #[test]
    fn stable_rr_rotates_exact_ties() {
        let scores = [score(2, 0, 0, 0), score(0, 0, 0, 0), score(1, 0, 0, 0)];
        let rr = AtomicUsize::new(0);
        let picks: Vec<_> = (0..5)
            .map(|_| {
                let mut tied = scores.iter().collect();
                pick_rr(&mut tied, &rr).unwrap()
            })
            .collect();
        assert_eq!(picks, vec![0, 1, 2, 0, 1]);
    }

    #[test]
    fn p2c_dp8_second_choice_is_distinct_and_uniform_over_other_ranks() {
        const DP_SIZE: usize = 8;
        for first in 0..DP_SIZE {
            let seconds: BTreeSet<_> = (0..DP_SIZE - 1)
                .map(|offset| p2c_second_index(first, offset, DP_SIZE))
                .collect();
            assert_eq!(seconds.len(), DP_SIZE - 1);
            assert!(!seconds.contains(&first));
            assert_eq!(
                seconds,
                (0..DP_SIZE).filter(|rank| *rank != first).collect()
            );
        }
    }

    #[test]
    fn lmetric_formula_can_prefer_less_cached_idle_rank() {
        let cached_hot = score(0, 7, 700, 5);
        let cold_idle = score(1, 0, 0, 0);
        let prompt_tokens = 1000usize;
        let cost = |s: &RankScore| {
            prompt_tokens
                .saturating_sub(s.matched_tokens)
                .saturating_mul(s.load + 1)
        };
        assert_eq!(cost(&cached_hot), 1800);
        assert_eq!(cost(&cold_idle), 1000);
        assert!(cost(&cold_idle) < cost(&cached_hot));
    }

    #[test]
    fn preble_prefill_exploits_when_cached_work_exceeds_miss() {
        let warm_busy = score(0, 7, 700, 0);
        let cold_idle = score(1, 0, 0, 0);
        let scores = [warm_busy, cold_idle];
        let recent_loads = BTreeMap::from([(("http://engine:30000".into(), 0), 2000)]);
        let rr = AtomicUsize::new(0);

        let (selected, reason) =
            select_preble_prefill_candidate(&scores, 1000, &recent_loads, &rr).unwrap();

        // E2 exploitation restricts candidates to the longest-prefix rank,
        // even though exploring the cold rank has a lower immediate cost.
        assert_eq!(selected, 0);
        assert_eq!(reason, "preble_exploit");
    }

    #[test]
    fn preble_prefill_explores_all_ranks_when_miss_dominates() {
        let warm_busy = score(0, 4, 400, 0);
        let cold_idle = score(1, 0, 0, 0);
        let scores = [warm_busy, cold_idle];
        let recent_loads = BTreeMap::from([(("http://engine:30000".into(), 0), 1000)]);
        let rr = AtomicUsize::new(0);

        let (selected, reason) =
            select_preble_prefill_candidate(&scores, 1000, &recent_loads, &rr).unwrap();

        assert_eq!(selected, 1);
        assert_eq!(reason, "preble_explore");
    }

    #[test]
    fn preble_prefill_history_accumulates_and_expires_work() {
        let rank = score(0, 0, 0, 0);
        let start = Instant::now();
        let mut history = RecentPrefillHistory::new(Duration::from_secs(5));
        history.record(start, &rank, 100);
        history.record(start + Duration::from_secs(1), &rank, 50);

        let key = (rank.base_url.clone(), rank.dp_rank);
        assert_eq!(history.loads(start + Duration::from_secs(2))[&key], 150);
        assert!(history.loads(start + Duration::from_secs(6)).is_empty());
    }

    #[test]
    fn chunk_lmetric_uses_scheduler_visible_chunks() {
        let cached_busy = score(0, 16, 4096, 2);
        let cold_idle = score(1, 0, 0, 0);
        let prompt_tokens = 8192usize;
        assert_eq!(chunk_lmetric_cost(prompt_tokens, 4096, &cached_busy), 3);
        assert_eq!(chunk_lmetric_cost(prompt_tokens, 4096, &cold_idle), 2);
    }

    #[test]
    fn smetric_threshold_distinguishes_first_and_later_turns() {
        let cold = score(0, 0, 0, 0);
        let warm = score(1, 2, 512, 9);
        assert!(cold.matched_tokens < 512);
        assert!(warm.matched_tokens >= 512);
    }

    #[test]
    fn prefix_order_is_independent_of_load() {
        let scores = [score(0, 9, 900, 100), score(1, 8, 800, 0)];
        let best = scores.iter().max_by_key(|s| s.matched_blocks).unwrap();
        assert_eq!(best.dp_rank, 0);
    }

    #[test]
    fn dualmap_adaptive_prefix_expands_hot_paths_one_level_at_a_time() {
        let mut tracker = AdaptivePrefixTracker::new(8, 4);
        let blocks = [11, 22, 33];
        for _ in 0..4 {
            assert_eq!(tracker.select_and_observe(&blocks, 8).len(), 1);
        }
        assert_eq!(tracker.select_and_observe(&blocks, 8).len(), 2);
        assert_eq!(tracker.select_and_observe(&blocks, 8).len(), 3);
    }

    #[test]
    fn dualmap_two_rings_always_return_distinct_candidates() {
        let scores = [score(0, 0, 0, 0), score(1, 0, 0, 0), score(2, 0, 0, 0)];
        let candidates = dualmap_candidates(&scores, &[11, 22]).unwrap();
        assert_eq!(candidates.len(), 2);
        assert_ne!(candidates[0], candidates[1]);
        assert_eq!(candidates, dualmap_candidates(&scores, &[11, 22]).unwrap());
    }

    #[test]
    fn dualmap_prefers_cache_until_virtual_work_exceeds_slo() {
        let warm = score(0, 8, 800, 0);
        let cold = score(1, 0, 0, 0);
        let scores = [warm, cold];
        let candidates = [0, 1];
        let rr = AtomicUsize::new(0);

        let light_pending = BTreeMap::from([(("http://engine:30000".into(), 0), 100)]);
        let (selected, reason) =
            select_dualmap_candidate(&scores, &candidates, 1000, &light_pending, 1000, &rr)
                .unwrap();
        assert_eq!(selected, 0);
        assert_eq!(reason, "dualmap_cache_affinity");

        let overloaded = BTreeMap::from([(("http://engine:30000".into(), 0), 1000)]);
        let (selected, reason) =
            select_dualmap_candidate(&scores, &candidates, 1000, &overloaded, 1000, &rr).unwrap();
        assert_eq!(selected, 1);
        assert_eq!(reason, "dualmap_slo_min_ttft");
    }

    #[test]
    fn dualmap_pending_reservation_is_released_by_selection_id() {
        let config = DualMapConfig::default();
        let mut state = DualMapRoutingState::new(config);
        let rank = score(3, 0, 0, 0);
        state.reserve(17, &rank, 4096);
        assert_eq!(state.pending(&rank), 4096);
        state.release(17);
        assert_eq!(state.pending(&rank), 0);
        assert!(state.reservations.is_empty());
    }

    fn engine_load(rank: u32, timestamp: f64, tokens: usize, requests: usize) -> EngineRankLoad {
        EngineRankLoad {
            timestamp,
            dp_rank: rank,
            num_running_reqs: requests,
            num_waiting_reqs: 0,
            num_total_tokens: tokens,
        }
    }

    #[test]
    fn total_token_overlay_survives_equal_snapshot_and_resets_on_newer_source() {
        let key = ("http://engine:30000".to_string(), 0);
        let expected = BTreeSet::from([key.clone()]);
        let start = Instant::now();
        let mut state = RankTokenLoadState::default();
        assert!(state.apply_snapshot(
            &expected,
            BTreeMap::from([(key.clone(), engine_load(0, 10.0, 100, 2))]),
            start,
        ));
        assert!(state.reserve(&key, 40));
        assert_eq!(state.budgets[&key].effective_total_tokens(), 140);
        assert_eq!(state.budgets[&key].effective_total_requests(), 3);

        // The source has not advanced, so this is the same generation and
        // must not erase reservations made after the prior fetch.
        assert!(state.apply_snapshot(
            &expected,
            BTreeMap::from([(key.clone(), engine_load(0, 10.0, 100, 2))]),
            start + Duration::from_millis(1),
        ));
        assert_eq!(state.budgets[&key].overlay_tokens, 40);
        assert_eq!(state.budgets[&key].overlay_requests, 1);

        assert!(state.apply_snapshot(
            &expected,
            BTreeMap::from([(key.clone(), engine_load(0, 11.0, 125, 3))]),
            start + Duration::from_millis(2),
        ));
        assert_eq!(state.budgets[&key].source_total_tokens, 125);
        assert_eq!(state.budgets[&key].overlay_tokens, 0);
        assert_eq!(state.budgets[&key].overlay_requests, 0);
    }

    #[test]
    fn total_token_snapshot_fails_closed_on_missing_or_older_rank() {
        let keys = BTreeSet::from([
            ("http://engine:30000".to_string(), 0),
            ("http://engine:30000".to_string(), 1),
        ]);
        let now = Instant::now();
        let complete = BTreeMap::from([
            (keys.first().unwrap().clone(), engine_load(0, 20.0, 10, 1)),
            (keys.last().unwrap().clone(), engine_load(1, 20.0, 20, 1)),
        ]);
        let mut state = RankTokenLoadState::default();
        assert!(state.apply_snapshot(&keys, complete, now));
        assert!(state.topology_matches(&keys));
        assert!(!state.topology_matches(&BTreeSet::from([
            ("http://engine:30000".to_string(), 0),
            ("http://engine:30000".to_string(), 2),
        ])));
        assert!(!state.apply_snapshot(
            &keys,
            BTreeMap::from([(keys.first().unwrap().clone(), engine_load(0, 21.0, 5, 1))]),
            now + Duration::from_millis(1),
        ));
        assert!(!state.apply_snapshot(
            &keys,
            BTreeMap::from([
                (keys.first().unwrap().clone(), engine_load(0, 19.0, 5, 1)),
                (keys.last().unwrap().clone(), engine_load(1, 21.0, 5, 1)),
            ]),
            now + Duration::from_millis(2),
        ));
        assert_eq!(state.budgets[keys.first().unwrap()].source_timestamp, 20.0);
    }

    #[test]
    fn total_tokens_uses_tokens_then_requests_then_stable_rank() {
        let scores = vec![
            RankTokenScore {
                worker_index: 2,
                base_url: "http://engine:30000".into(),
                dp_rank: 2,
                source_total_tokens: 90,
                source_total_requests: 7,
                overlay_tokens: 10,
                overlay_requests: 0,
                effective_total_tokens: 100,
                effective_total_requests: 7,
            },
            RankTokenScore {
                worker_index: 1,
                base_url: "http://engine:30000".into(),
                dp_rank: 1,
                source_total_tokens: 100,
                source_total_requests: 3,
                overlay_tokens: 0,
                overlay_requests: 1,
                effective_total_tokens: 100,
                effective_total_requests: 4,
            },
            RankTokenScore {
                worker_index: 0,
                base_url: "http://engine:30000".into(),
                dp_rank: 0,
                source_total_tokens: 80,
                source_total_requests: 3,
                overlay_tokens: 20,
                overlay_requests: 1,
                effective_total_tokens: 100,
                effective_total_requests: 4,
            },
        ];
        assert_eq!(select_rank_total_tokens(&scores), Some(0));
    }

    #[test]
    fn total_tokens_wire_format_accepts_omitted_zero_counters() {
        let response: EngineRankLoadsResponse = serde_json::from_str(
            r#"{"loads":[{"timestamp":7.5,"dp_rank":3,"num_total_tokens":42}]}"#,
        )
        .unwrap();
        assert_eq!(response.loads.len(), 1);
        assert_eq!(response.loads[0].num_running_reqs, 0);
        assert_eq!(response.loads[0].num_waiting_reqs, 0);
    }

    fn rank_topology(size: u32) -> Vec<RankScore> {
        (0..size).map(|rank| score(rank, 0, 0, 0)).collect()
    }

    #[test]
    fn rank_consistent_hash_is_repeatable_and_topology_order_independent() {
        let scores = rank_topology(8);
        let digest = sha256_digest(b"session-42");
        let expected = select_rank_consistent_hash(&scores, &digest);
        assert_eq!(expected, select_rank_consistent_hash(&scores, &digest));

        let mut reversed = scores.clone();
        reversed.reverse();
        assert_eq!(expected, select_rank_consistent_hash(&reversed, &digest));
    }

    #[test]
    fn rank_consistent_hash_distributes_keys_and_only_moves_to_added_rank() {
        let original = rank_topology(4);
        let expanded = rank_topology(5);
        let mut original_counts = [0usize; 4];
        let mut unchanged = 0usize;
        let mut moved = 0usize;
        for key in 0u32..10_000 {
            let digest = sha256_digest(&key.to_le_bytes());
            let before = select_rank_consistent_hash(&original, &digest).unwrap();
            let after = select_rank_consistent_hash(&expanded, &digest).unwrap();
            original_counts[before] += 1;
            if before == after {
                unchanged += 1;
            } else {
                moved += 1;
                assert_eq!(after, 4);
            }
        }
        assert!(original_counts.into_iter().all(|count| count > 1_000));
        assert!(unchanged > 7_000);
        assert!(moved > 500);
    }

    #[test]
    fn rank_consistent_hash_uses_documented_header_priority() {
        let mut headers = http::HeaderMap::new();
        headers.insert("x-tenant-id", "tenant".parse().unwrap());
        headers.insert("x-user-id", "user".parse().unwrap());
        headers.insert("x-session-id", "session".parse().unwrap());
        headers.insert("x-smg-routing-key", "explicit".parse().unwrap());
        let tokens = [1, 2, 3];
        let info = SelectWorkerInfo {
            request_text: Some("text"),
            tokens: Some(&tokens),
            headers: Some(&headers),
            hash_ring: None,
        };
        let (source, key) = rank_routing_key(&info).unwrap();
        assert_eq!(source, "x-smg-routing-key");
        assert_eq!(key, b"explicit");
        let encoded = digest_hex(&sha256_digest(&key));
        assert_eq!(encoded.len(), 64);
        assert!(!encoded.contains("explicit"));
    }
}
