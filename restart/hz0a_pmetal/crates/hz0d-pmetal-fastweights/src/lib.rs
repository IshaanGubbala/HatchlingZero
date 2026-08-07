#![forbid(unsafe_code)]
//! HZ-0D Phase D9: Rust CPU-tensor port of `reference/hz0d_fast_weights.py`
//! (D1's fast-weight state/lifecycle contract, as it stands after D3's
//! v4 adaptive-ridge selection and D6/D7/D8's real-model integration).
//! Flat f32/i32 buffers, no external tensor library, matching
//! `hz0b-pmetal-memory`'s own established convention in this workspace
//! (itself matching `hz0a-pmetal-tensor`'s).
//!
//! **New relative to the Python reference: an explicit `session` (batch)
//! dimension.** `reference/hz0d_fast_weights.py::FastWeightState` has no
//! batch axis -- it is one session's state. D9's plan text names
//! "batched deterministic sessions" as a real requirement (many users'
//! independent, isolated sessions processed together, sharing the SAME
//! frozen backbone weights but never sharing or leaking fast-weight
//! state between each other). This crate adds that dimension for real:
//! every operation takes a `session` index (or, for `apply`, processes
//! every session in the batch in one call) and is verified (`tests/`)
//! to keep each session's state fully independent and every result
//! deterministic given the same inputs.
//!
//! **RNG is intentionally NOT reimplemented here.** `init_fast_weights`'s
//! asymmetric init (`a_fast` small random, `b_fast` exactly zero) draws
//! `a_fast` from `mx.random.normal`; replicating MLX's PRNG algorithm
//! bit-for-bit in Rust would be real work for no real benefit (nothing
//! downstream needs Rust to independently generate that first draw). As
//! with every other manual-matmul/state-machine port in this project,
//! randomness is generated once by the Python/MLX reference and handed
//! in as data -- `reset` here takes a caller-supplied `a_fast_init`
//! buffer (already drawn by `reference/hz0d_fast_weights.py::init_fast_weights`
//! or any other real MLX call) and constructs `b_fast` as the exact
//! zeros the asymmetric-init contract requires; that IS the real,
//! deterministic part of reset, and it is what's ported.

pub const DEFAULT_MAX_DELTA_NORM: f32 = 1.0;

#[derive(Debug, Clone, PartialEq)]
pub struct FastWeightState {
    pub sessions: usize,
    pub num_layers: usize,
    pub dim: usize,
    pub rank: usize,
    pub a_fast: Vec<f32>,       // [sessions * num_layers * dim * rank]
    pub b_fast: Vec<f32>,       // [sessions * num_layers * rank * dim]
    pub update_count: Vec<i32>, // [sessions]
}

fn layer_slices(
    sessions: usize,
    num_layers: usize,
    dim: usize,
    rank: usize,
    session: usize,
    layer: usize,
) -> (usize, usize) {
    let a_off = (session * num_layers + layer) * dim * rank;
    let b_off = (session * num_layers + layer) * rank * dim;
    debug_assert!(session < sessions && layer < num_layers);
    (a_off, b_off)
}

/// Contract op: reset(sessions, num_layers, dim, rank, a_fast_init).
/// `a_fast_init`: `[sessions * num_layers * dim * rank]`, caller-supplied
/// (see module docs). `b_fast` is exactly zero -- the asymmetric-init
/// invariant (`docs/restart/hz0d_d1_contract.md` section 2 addendum):
/// the realized delta `A @ B` is exactly zero at every layer for every
/// session immediately after reset, matching the "inactive fast weights
/// reproduce HZ-0C behavior" requirement D6 verified end to end.
pub fn reset(
    sessions: usize,
    num_layers: usize,
    dim: usize,
    rank: usize,
    a_fast_init: &[f32],
) -> FastWeightState {
    assert_eq!(
        a_fast_init.len(),
        sessions * num_layers * dim * rank,
        "a_fast_init length mismatch"
    );
    FastWeightState {
        sessions,
        num_layers,
        dim,
        rank,
        a_fast: a_fast_init.to_vec(),
        b_fast: vec![0.0; sessions * num_layers * rank * dim],
        update_count: vec![0; sessions],
    }
}

/// The realized `[dim, dim]` weight delta for one session's one layer --
/// `reference/hz0d_fast_weights.py::effective_delta`.
pub fn effective_delta(state: &FastWeightState, session: usize, layer: usize) -> Vec<f32> {
    let (dim, rank) = (state.dim, state.rank);
    let (a_off, b_off) = layer_slices(state.sessions, state.num_layers, dim, rank, session, layer);
    let a = &state.a_fast[a_off..a_off + dim * rank];
    let b = &state.b_fast[b_off..b_off + rank * dim];
    let mut delta = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for r in 0..rank {
            let av = a[i * rank + r];
            if av == 0.0 {
                continue;
            }
            for j in 0..dim {
                delta[i * dim + j] += av * b[r * dim + j];
            }
        }
    }
    delta
}

/// Contract op: `apply_fast_linear` for EVERY session in the batch in one
/// call -- `x`: `[sessions * dim]` (one activation vector per session;
/// callers with a real `[sessions, seq, dim]` tensor call this once per
/// sequence position, matching how `reference/hz0d_d6_integration.py`'s
/// real-model wiring processes one token position at a time). Each
/// session's OWN fast-weight delta is applied to its OWN `x` row --
/// `base_weight`/`base_bias` are the SAME frozen backbone weights shared
/// by every session (the one real production topology: many isolated
/// user sessions, one frozen pretrained model). Returns `[sessions * dim]`.
///
/// Computes the low-rank contribution as `(x @ B.T) @ A.T` (`2 * dim *
/// rank` multiply-adds) rather than materializing the dense `[dim, dim]`
/// delta first (`dim * dim * rank` multiply-adds just to build it, on
/// top of applying it) -- mathematically identical
/// (`x @ (A @ B).T == (x @ B.T) @ A.T`), but at the D1 contract's real
/// scale (`dim=768`, `rank=16`) the dense path costs ~9.4M multiply-adds
/// for the fast-weight term alone versus ~25K here, a real ~380x
/// reduction in the part of `apply` that scales with `rank`. Found and
/// fixed the same day this crate was built, via direct benchmarking
/// (see `docs/restart/hz0d_d9_pmetal_results.md`), matching C8's own
/// standing discipline of not leaving a known-fixable redundancy in a
/// kernel once it's been measured.
pub fn apply(
    state: &FastWeightState,
    x: &[f32],
    base_weight: &[f32],
    base_bias: &[f32],
    layer: usize,
) -> Vec<f32> {
    let dim = state.dim;
    let rank = state.rank;
    assert_eq!(x.len(), state.sessions * dim);
    assert_eq!(base_weight.len(), dim * dim);
    assert_eq!(base_bias.len(), dim);
    let mut y = vec![0.0f32; state.sessions * dim];
    for s in 0..state.sessions {
        let (a_off, b_off) = layer_slices(state.sessions, state.num_layers, dim, rank, s, layer);
        let a_layer = &state.a_fast[a_off..a_off + dim * rank];
        let b_layer = &state.b_fast[b_off..b_off + rank * dim];
        let x_s = &x[s * dim..(s + 1) * dim];

        // code[r] = sum_j x_s[j] * b_layer[r, j]  -- [rank]
        let mut code = vec![0.0f32; rank];
        for r in 0..rank {
            let row = r * dim;
            let mut acc = 0.0f32;
            for j in 0..dim {
                acc += x_s[j] * b_layer[row + j];
            }
            code[r] = acc;
        }
        for out_i in 0..dim {
            let mut acc = base_bias[out_i];
            let row = out_i * dim;
            for in_j in 0..dim {
                acc += x_s[in_j] * base_weight[row + in_j];
            }
            // fast delta contribution: sum_r a_layer[out_i, r] * code[r]
            let a_row = out_i * rank;
            for r in 0..rank {
                acc += a_layer[a_row + r] * code[r];
            }
            y[s * dim + out_i] = acc;
        }
    }
    y
}

/// `reference/hz0d_fast_weights.py::clip_layer_factors`, in place on one
/// session's one layer -- bounds the REALIZED delta's Frobenius norm,
/// splitting the scale evenly across both factors via its square root
/// (so the low-rank representation is never destroyed by materializing
/// and reclipping a dense delta).
fn clip_layer(
    a_layer: &mut [f32],
    b_layer: &mut [f32],
    dim: usize,
    rank: usize,
    max_delta_norm: f32,
) {
    // delta[i,j] = sum_r a[i,r]*b[r,j] -- materialized densely (matching
    // the Python reference exactly) since the norm needs the SUM over r
    // squared, not a sum of per-r squares.
    let mut delta = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for r in 0..rank {
            let av = a_layer[i * rank + r];
            if av == 0.0 {
                continue;
            }
            for j in 0..dim {
                delta[i * dim + j] += av * b_layer[r * dim + j];
            }
        }
    }
    let delta_norm = (delta.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()).sqrt() as f32;
    let scale = (max_delta_norm / (delta_norm + 1e-8)).min(1.0);
    let factor_scale = scale.sqrt();
    for v in a_layer.iter_mut() {
        *v *= factor_scale;
    }
    for v in b_layer.iter_mut() {
        *v *= factor_scale;
    }
}

/// Contract op: `update_fast_weights` for one session's one layer, using
/// REAL, externally-computed gradients (`grad_a`/`grad_b`) -- never
/// approximated here, matching `docs/restart/hz0d_history_audit.md`'s
/// core lesson (the archived prior implementation's mistake) applied to
/// this port too. Clips via `clip_layer`.
pub fn update(
    state: &FastWeightState,
    session: usize,
    layer: usize,
    grad_a: &[f32],
    grad_b: &[f32],
    lr: f32,
    max_delta_norm: f32,
) -> FastWeightState {
    let (dim, rank) = (state.dim, state.rank);
    let (a_off, b_off) = layer_slices(state.sessions, state.num_layers, dim, rank, session, layer);
    assert_eq!(grad_a.len(), dim * rank);
    assert_eq!(grad_b.len(), rank * dim);

    let mut new_state = state.clone();
    let mut a_layer: Vec<f32> = state.a_fast[a_off..a_off + dim * rank]
        .iter()
        .zip(grad_a)
        .map(|(v, g)| v - lr * g)
        .collect();
    let mut b_layer: Vec<f32> = state.b_fast[b_off..b_off + rank * dim]
        .iter()
        .zip(grad_b)
        .map(|(v, g)| v - lr * g)
        .collect();
    clip_layer(&mut a_layer, &mut b_layer, dim, rank, max_delta_norm);
    new_state.a_fast[a_off..a_off + dim * rank].copy_from_slice(&a_layer);
    new_state.b_fast[b_off..b_off + rank * dim].copy_from_slice(&b_layer);
    new_state.update_count[session] += 1;
    new_state
}

/// Contract op: `decay_fast_weights` -- multiplicative decay of BOTH
/// factors, applied to EVERY session (the same policy `decay_rate`
/// across the whole batch, matching how a real deployment would apply
/// one session-boundary policy uniformly). `decay_rate=1.0` is an exact
/// no-op.
pub fn decay(state: &FastWeightState, decay_rate: f32) -> FastWeightState {
    let mut new_state = state.clone();
    for v in new_state.a_fast.iter_mut() {
        *v *= decay_rate;
    }
    for v in new_state.b_fast.iter_mut() {
        *v *= decay_rate;
    }
    new_state
}

/// Contract op: `snapshot`. A plain clone, matching
/// `hz0b-pmetal-memory::serialize`'s own reasoning: this Rust struct
/// already IS the flat, framework-agnostic representation.
pub fn snapshot(state: &FastWeightState) -> FastWeightState {
    state.clone()
}

/// Contract op: `rollback`. Exact inverse of `snapshot` -- a clone.
pub fn rollback(checkpoint: &FastWeightState) -> FastWeightState {
    checkpoint.clone()
}

/// Bounds one session's real fast-state memory, matching
/// `reference/hz0d_fast_weights.py::fast_state_memory_bytes` -- computed
/// from real shapes, not a separately hand-maintained estimate.
pub fn fast_state_memory_bytes(num_layers: usize, dim: usize, rank: usize) -> usize {
    2 * num_layers * dim * rank * 4
}
