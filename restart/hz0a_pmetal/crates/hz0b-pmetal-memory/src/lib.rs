#![forbid(unsafe_code)]
//! HZ-0B Phase B10: Rust CPU-tensor port of `reference/hz0b_memory_simulator.py`
//! (B2's pure memory simulator, as fixed 2026-07-30: match threshold
//! 0.999, confidence-weighted reads -- see
//! `docs/restart/hz0b_b8_stage5_results.md`). Flat f32/i32 buffers, no
//! external tensor library, matching `hz0a-pmetal-tensor`'s own
//! established convention in this workspace.
//!
//! This is the CPU reference tier of B10's plan -- "Once reference
//! semantics are stable" is satisfied by B2/B3 being complete and tested
//! (20 + 11 Python tests, plus today's 2 real bug fixes). A real Metal
//! GPU kernel crate (`hz0b-pmetal-memory-gpu`, mirroring
//! `hz0a-pmetal-gpu`'s structure) is the next phase, not built here --
//! B2's operations are cheap enough (num_slots ~8-16, key/value_dim
//! ~16-32) that a fused Metal kernel is a real speed win at HZ-0A's real
//! 301M-scale integration, but not yet load-bearing at the scale B6-B9's
//! probes have run at so far, matching how HZ-0A's own PMetal work
//! prioritized the GDN-2 recurrence (the actual per-token bottleneck)
//! over smaller surrounding ops.
//!
//! Batch dimension supported throughout via flat indexing
//! (`batch * num_slots * dim + slot * dim + d`), matching this state's
//! Python shape convention `[batch, num_slots, dim]`.

pub const PROTECTION_BLOCK_THRESHOLD: f32 = 0.5;
pub const MATCH_THRESHOLD: f32 = 0.999; // raised from 0.95 -- see docs/restart/hz0b_b8_stage5_results.md finding 1
pub const EMPTY_THRESHOLD: f32 = 1e-6;

#[derive(Debug, Clone, PartialEq)]
pub struct MemoryState {
    pub batch: usize,
    pub num_slots: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub keys: Vec<f32>,            // [batch * num_slots * key_dim]
    pub values: Vec<f32>,          // [batch * num_slots * value_dim]
    pub confidence: Vec<f32>,      // [batch * num_slots]
    pub age: Vec<i32>,             // [batch * num_slots]
    pub protection: Vec<f32>,      // [batch * num_slots]
    pub write_count: Vec<i32>,     // [batch * num_slots]
    pub last_write_step: Vec<i32>, // [batch * num_slots]
    pub write_source: Vec<i32>,    // [batch * num_slots]
}

/// Contract op: reset(batch_size). Full zero -- matches
/// `reference/hz0b_memory_simulator.py::reset` exactly.
pub fn reset(batch: usize, num_slots: usize, key_dim: usize, value_dim: usize) -> MemoryState {
    MemoryState {
        batch,
        num_slots,
        key_dim,
        value_dim,
        keys: vec![0.0; batch * num_slots * key_dim],
        values: vec![0.0; batch * num_slots * value_dim],
        confidence: vec![0.0; batch * num_slots],
        age: vec![0; batch * num_slots],
        protection: vec![0.0; batch * num_slots],
        write_count: vec![0; batch * num_slots],
        last_write_step: vec![0; batch * num_slots],
        write_source: vec![0; batch * num_slots],
    }
}

/// query: [batch * dim], keys: [batch * num_slots * dim] -> [batch * num_slots].
/// Epsilon inside the sqrt (not a post-hoc clamp), matching the Python
/// reference's own differentiability-at-zero reasoning even though this
/// Rust port has no autodiff -- keeping the exact same numerics as the
/// reference it's meant to match is the point.
fn cosine_similarity(query: &[f32], keys: &[f32], batch: usize, num_slots: usize, dim: usize) -> Vec<f32> {
    let mut scores = vec![0.0f32; batch * num_slots];
    for b in 0..batch {
        let q = &query[b * dim..(b + 1) * dim];
        let q_norm = (q.iter().map(|x| x * x).sum::<f32>() + 1e-8).sqrt();
        for s in 0..num_slots {
            let k = &keys[(b * num_slots + s) * dim..(b * num_slots + s + 1) * dim];
            let k_norm = (k.iter().map(|x| x * x).sum::<f32>() + 1e-8).sqrt();
            let dot: f32 = q.iter().zip(k.iter()).map(|(a, c)| (a / q_norm) * (c / k_norm)).sum();
            scores[b * num_slots + s] = dot;
        }
    }
    scores
}

fn softmax(scores: &[f32], batch: usize, num_slots: usize) -> Vec<f32> {
    let mut weights = vec![0.0f32; batch * num_slots];
    for b in 0..batch {
        let row = &scores[b * num_slots..(b + 1) * num_slots];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for s in 0..num_slots {
            weights[b * num_slots + s] = exps[s] / sum;
        }
    }
    weights
}

fn argmax_row(row: &[f32]) -> usize {
    row.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0
}

/// Contract op: read(query) -> (readout, read_weights).
/// `confidence_weighted` (default true in the Python reference, matching
/// the 2026-07-30 fix): adds `ln(confidence + eps)` to each slot's score
/// before ranking, so a low-confidence slot competes on worse footing --
/// see `docs/restart/hz0b_b8_stage5_results.md` finding 2.
pub fn read(state: &MemoryState, query: &[f32], slot_idx: Option<&[i32]>, hard: bool, confidence_weighted: bool) -> (Vec<f32>, Vec<f32>) {
    let (batch, num_slots, key_dim, value_dim) = (state.batch, state.num_slots, state.key_dim, state.value_dim);
    let weights = if let Some(idx) = slot_idx {
        let mut w = vec![0.0f32; batch * num_slots];
        for b in 0..batch {
            w[b * num_slots + idx[b] as usize] = 1.0;
        }
        w
    } else {
        let mut scores = cosine_similarity(query, &state.keys, batch, num_slots, key_dim);
        if confidence_weighted {
            for b in 0..batch {
                for s in 0..num_slots {
                    scores[b * num_slots + s] += (state.confidence[b * num_slots + s] + 1e-6).ln();
                }
            }
        }
        if hard {
            let mut w = vec![0.0f32; batch * num_slots];
            for b in 0..batch {
                let best = argmax_row(&scores[b * num_slots..(b + 1) * num_slots]);
                w[b * num_slots + best] = 1.0;
            }
            w
        } else {
            softmax(&scores, batch, num_slots)
        }
    };
    let mut readout = vec![0.0f32; batch * value_dim];
    for b in 0..batch {
        for s in 0..num_slots {
            let w = weights[b * num_slots + s];
            if w == 0.0 {
                continue;
            }
            for d in 0..value_dim {
                readout[b * value_dim + d] += state.values[(b * num_slots + s) * value_dim + d] * w;
            }
        }
    }
    (readout, weights)
}

/// Mirrors `_choose_write_slot`: returns (slot_idx per batch row,
/// forced_reject per batch row).
fn choose_write_slot(state: &MemoryState, key: &[f32]) -> (Vec<i32>, Vec<bool>) {
    let (batch, num_slots, key_dim) = (state.batch, state.num_slots, state.key_dim);
    let similarity = cosine_similarity(key, &state.keys, batch, num_slots, key_dim);
    let mut slot_idx = vec![0i32; batch];
    let mut forced_reject = vec![false; batch];
    for b in 0..batch {
        let row = &similarity[b * num_slots..(b + 1) * num_slots];
        let is_match: Vec<bool> = row.iter().map(|&s| s > MATCH_THRESHOLD).collect();
        let has_match = is_match.iter().any(|&m| m);
        if has_match {
            // argmax of similarity among matching slots only
            let (matched_slot, _) = row.iter().enumerate().filter(|(i, _)| is_match[*i]).fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) });
            let is_protected = state.protection[b * num_slots + matched_slot] >= PROTECTION_BLOCK_THRESHOLD;
            slot_idx[b] = matched_slot as i32;
            forced_reject[b] = is_protected;
        } else {
            let mut best_slot = 0usize;
            let mut best_score = f32::INFINITY;
            let mut all_protected = true;
            for s in 0..num_slots {
                let idx = b * num_slots + s;
                let is_protected = state.protection[idx] >= PROTECTION_BLOCK_THRESHOLD;
                if !is_protected {
                    all_protected = false;
                }
                let is_empty = state.confidence[idx] < EMPTY_THRESHOLD;
                let eviction_score = if is_protected {
                    f32::INFINITY
                } else {
                    state.confidence[idx] * (1.0 - state.protection[idx]) - (state.age[idx] as f32) * 1e-6
                };
                let empty_score = if is_empty && !is_protected { -1.0 } else { f32::INFINITY };
                let score = empty_score.min(eviction_score);
                if score < best_score {
                    best_score = score;
                    best_slot = s;
                }
            }
            slot_idx[b] = best_slot as i32;
            forced_reject[b] = all_protected;
        }
    }
    (slot_idx, forced_reject)
}

/// Contract op: write(key, value, strength) -> (new_state, chosen_slot_idx, rejected).
pub fn write(state: &MemoryState, key: &[f32], value: &[f32], strength: &[f32], step: i32, source: i32, slot_idx: Option<&[i32]>) -> (MemoryState, Vec<i32>, Vec<bool>) {
    let (batch, num_slots, key_dim, value_dim) = (state.batch, state.num_slots, state.key_dim, state.value_dim);
    let (chosen_slot, rejected) = match slot_idx {
        Some(idx) => {
            let mut rej = vec![false; batch];
            for b in 0..batch {
                rej[b] = state.protection[b * num_slots + idx[b] as usize] >= PROTECTION_BLOCK_THRESHOLD;
            }
            (idx.to_vec(), rej)
        }
        None => choose_write_slot(state, key),
    };
    let mut new_state = state.clone();
    for b in 0..batch {
        if rejected[b] {
            continue;
        }
        let s = chosen_slot[b] as usize;
        let idx = b * num_slots + s;
        for d in 0..key_dim {
            new_state.keys[idx * key_dim + d] = key[b * key_dim + d];
        }
        for d in 0..value_dim {
            new_state.values[idx * value_dim + d] = value[b * value_dim + d];
        }
        new_state.confidence[idx] = strength[b];
        new_state.age[idx] = 0;
        new_state.write_count[idx] += 1;
        new_state.last_write_step[idx] = step;
        new_state.write_source[idx] = source;
    }
    (new_state, chosen_slot, rejected)
}

/// Contract op: reinforce(slot). Confidence up (+0.2, clamped to 1.0),
/// age reset, value UNCHANGED.
pub fn reinforce(state: &MemoryState, slot_idx: &[i32]) -> MemoryState {
    let mut new_state = state.clone();
    for b in 0..state.batch {
        let idx = b * state.num_slots + slot_idx[b] as usize;
        new_state.confidence[idx] = (state.confidence[idx] + 0.2).min(1.0);
        new_state.age[idx] = 0;
    }
    new_state
}

/// Contract op: update(slot, new_value). Value replaced; key and
/// protection_strength UNCHANGED.
pub fn update(state: &MemoryState, slot_idx: &[i32], new_value: &[f32]) -> MemoryState {
    let mut new_state = state.clone();
    for b in 0..state.batch {
        let s = slot_idx[b] as usize;
        let idx = b * state.num_slots + s;
        for d in 0..state.value_dim {
            new_state.values[idx * state.value_dim + d] = new_value[b * state.value_dim + d];
        }
    }
    new_state
}

/// Contract op: protect(slot, strength). Sets protection_strength explicitly.
pub fn protect(state: &MemoryState, slot_idx: &[i32], strength: &[f32]) -> MemoryState {
    let mut new_state = state.clone();
    for b in 0..state.batch {
        let idx = b * state.num_slots + slot_idx[b] as usize;
        new_state.protection[idx] = strength[b];
    }
    new_state
}

/// Contract op: forget_or_decay(step). Global per-step decay,
/// protection-aware: protected slots decay much slower.
pub fn forget_or_decay(state: &MemoryState, decay_rate: f32) -> MemoryState {
    let mut new_state = state.clone();
    for i in 0..state.confidence.len() {
        let effective_decay = decay_rate * (1.0 - state.protection[i]) + state.protection[i];
        new_state.confidence[i] = state.confidence[i] * effective_decay;
        new_state.age[i] = state.age[i] + 1;
    }
    new_state
}

/// Contract op: delete(slot). Explicit hard clear -- distinct from decay,
/// immediate and total for that slot.
pub fn delete(state: &MemoryState, slot_idx: &[i32]) -> MemoryState {
    let mut new_state = state.clone();
    for b in 0..state.batch {
        let s = slot_idx[b] as usize;
        let idx = b * state.num_slots + s;
        for d in 0..state.key_dim {
            new_state.keys[idx * state.key_dim + d] = 0.0;
        }
        for d in 0..state.value_dim {
            new_state.values[idx * state.value_dim + d] = 0.0;
        }
        new_state.confidence[idx] = 0.0;
        new_state.age[idx] = 0;
        new_state.protection[idx] = 0.0;
        new_state.write_count[idx] = 0;
        new_state.write_source[idx] = 0;
        // last_write_step intentionally left untouched, matching the
        // Python reference (delete() never resets it either).
    }
    new_state
}

/// Contract op: serialize(). Plain, flat, framework-agnostic fields --
/// this Rust struct already IS that; provided for API completeness
/// against the B1 contract's 10-operation list, and as the natural place
/// a future JSON/binary format would hang off.
pub fn serialize(state: &MemoryState) -> MemoryState {
    state.clone()
}

/// Contract op: restore(). Exact inverse of serialize() -- a clone, for
/// the same reason.
pub fn restore(blob: &MemoryState) -> MemoryState {
    blob.clone()
}
