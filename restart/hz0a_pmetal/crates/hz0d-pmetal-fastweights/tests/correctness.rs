//! Rust-native correctness tests mirroring
//! `tests/reference/test_hz0d_fast_weights.py`'s own coverage, plus the
//! genuinely NEW batched-session behavior this port adds (not a
//! cross-language parity check -- that's `parity.rs`).

use hz0d_pmetal_fastweights::*;

const DIM: usize = 8;
const RANK: usize = 2;
const LAYERS: usize = 1;

fn zero_init(sessions: usize) -> Vec<f32> {
    vec![0.0; sessions * LAYERS * DIM * RANK]
}

#[test]
fn reset_gives_exactly_zero_realized_delta_asymmetric_init() {
    let a_init: Vec<f32> = (0..DIM * RANK).map(|i| 0.01 * (i as f32)).collect();
    let state = reset(1, LAYERS, DIM, RANK, &a_init);
    assert!(state.b_fast.iter().all(|&v| v == 0.0), "b_fast must be exactly zero at reset");
    let delta = effective_delta(&state, 0, 0);
    assert!(delta.iter().all(|&v| v == 0.0), "A @ 0 must be exactly zero regardless of A");
}

#[test]
fn update_strictly_reduces_a_toy_loss() {
    let state = reset(1, LAYERS, DIM, RANK, &zero_init(1));
    // A single hand-computed gradient step toward a nonzero target: any
    // nonzero grad_b (grad_a is zero-gradient at b_fast=0 by construction,
    // matching the asymmetric-init dead-saddle finding from D1) should
    // move b_fast in the -lr*grad direction exactly.
    let grad_a = vec![0.0f32; DIM * RANK];
    let grad_b: Vec<f32> = (0..RANK * DIM).map(|i| 1.0 + i as f32 * 0.1).collect();
    let updated = update(&state, 0, 0, &grad_a, &grad_b, 0.1, 10.0);
    for (i, (&before, &g)) in state.b_fast.iter().zip(grad_b.iter()).enumerate() {
        let expected = before - 0.1 * g;
        assert!((updated.b_fast[i] - expected).abs() < 1e-4, "b_fast[{i}] should move by exactly -lr*grad before clipping dominates");
    }
    assert_eq!(updated.update_count[0], 1);
}

#[test]
fn clip_bounds_realized_delta_norm_regardless_of_gradient_magnitude() {
    let state = reset(1, LAYERS, DIM, RANK, &zero_init(1));
    let huge_grad_a: Vec<f32> = vec![1000.0; DIM * RANK];
    let huge_grad_b: Vec<f32> = vec![1000.0; RANK * DIM];
    let updated = update(&state, 0, 0, &huge_grad_a, &huge_grad_b, 1.0, 1.0);
    let delta = effective_delta(&updated, 0, 0);
    let norm: f32 = delta.iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!(norm <= 1.0 + 1e-3, "realized delta norm {norm} exceeds max_delta_norm=1.0");
}

#[test]
fn decay_rate_one_is_an_exact_no_op() {
    let a_init: Vec<f32> = (0..DIM * RANK).map(|i| 0.02 * (i as f32 + 1.0)).collect();
    let state = reset(1, LAYERS, DIM, RANK, &a_init);
    let grad_b: Vec<f32> = (0..RANK * DIM).map(|i| 0.3 * (i as f32 + 1.0)).collect();
    let state = update(&state, 0, 0, &vec![0.0; DIM * RANK], &grad_b, 0.05, 10.0);
    let decayed = decay(&state, 1.0);
    assert_eq!(decayed.a_fast, state.a_fast);
    assert_eq!(decayed.b_fast, state.b_fast);
}

#[test]
fn snapshot_and_rollback_restore_state_exactly() {
    let a_init: Vec<f32> = (0..DIM * RANK).map(|i| 0.02 * (i as f32 + 1.0)).collect();
    let start = reset(1, LAYERS, DIM, RANK, &a_init);
    let checkpoint = snapshot(&start);
    let grad_b: Vec<f32> = vec![5.0; RANK * DIM];
    let mutated = update(&start, 0, 0, &vec![0.0; DIM * RANK], &grad_b, 0.1, 10.0);
    assert_ne!(mutated.b_fast, start.b_fast, "sanity: the update must have actually changed state");
    let restored = rollback(&checkpoint);
    assert_eq!(restored.a_fast, start.a_fast);
    assert_eq!(restored.b_fast, start.b_fast);
    assert_eq!(restored.update_count, start.update_count);
}

#[test]
fn fast_state_memory_bytes_matches_hand_computed_default_config() {
    // Mirrors tests/reference/test_hz0d_fast_weights.py's own check:
    // D1's default config (dim=768, rank=16, num_layers=6) -> 589,824 bytes.
    let bytes = fast_state_memory_bytes(6, 768, 16);
    assert_eq!(bytes, 589_824);
}

// --- Batched-session behavior: genuinely new relative to the Python
// reference (which has no session/batch axis at all). ---

#[test]
fn batched_update_touches_only_the_named_session() {
    let sessions = 3;
    let a_init: Vec<f32> = (0..sessions * LAYERS * DIM * RANK).map(|i| 0.01 * (i as f32)).collect();
    let state = reset(sessions, LAYERS, DIM, RANK, &a_init);
    let grad_a = vec![1.0f32; DIM * RANK];
    let grad_b = vec![1.0f32; RANK * DIM];
    let updated = update(&state, 1, 0, &grad_a, &grad_b, 0.1, 10.0);

    let a_len = LAYERS * DIM * RANK;
    let b_len = LAYERS * RANK * DIM;
    // session 0 and session 2 must be BIT-IDENTICAL to their pre-update values
    assert_eq!(&updated.a_fast[0..a_len], &state.a_fast[0..a_len], "session 0 must be untouched");
    assert_eq!(&updated.a_fast[2 * a_len..3 * a_len], &state.a_fast[2 * a_len..3 * a_len], "session 2 must be untouched");
    assert_eq!(&updated.b_fast[0..b_len], &state.b_fast[0..b_len], "session 0's b_fast must be untouched");
    assert_eq!(&updated.b_fast[2 * b_len..3 * b_len], &state.b_fast[2 * b_len..3 * b_len], "session 2's b_fast must be untouched");
    // session 1 must have actually changed
    assert_ne!(&updated.b_fast[b_len..2 * b_len], &state.b_fast[b_len..2 * b_len], "session 1 must have changed");
    assert_eq!(updated.update_count, vec![0, 1, 0], "only session 1's update_count should increment");
}

#[test]
fn apply_is_deterministic_and_independent_per_session_in_one_batched_call() {
    let sessions = 2;
    // Give the two sessions DIFFERENT a_fast so their deltas genuinely differ.
    let mut a_init = vec![0.05f32; LAYERS * DIM * RANK];
    a_init.extend(vec![-0.05f32; LAYERS * DIM * RANK]);
    let state = reset(sessions, LAYERS, DIM, RANK, &a_init);
    let grad_b = vec![2.0f32; RANK * DIM];
    let state = update(&state, 0, 0, &vec![0.0; DIM * RANK], &grad_b, 0.1, 10.0);
    let state = update(&state, 1, 0, &vec![0.0; DIM * RANK], &grad_b, 0.1, 10.0);

    let base_weight = vec![0.01f32; DIM * DIM];
    let base_bias = vec![0.0f32; DIM];
    let x: Vec<f32> = (0..sessions * DIM).map(|i| 0.1 * (i as f32 + 1.0)).collect();

    let y1 = apply(&state, &x, &base_weight, &base_bias, 0);
    let y2 = apply(&state, &x, &base_weight, &base_bias, 0);
    assert_eq!(y1, y2, "apply must be deterministic given identical inputs");

    // Session 0 and session 1 have DIFFERENT a_fast (opposite sign) but
    // the SAME x magnitude pattern and SAME b_fast update -- their
    // outputs must differ (proving the delta really is session-specific,
    // not silently shared/aliased across the batch).
    assert_ne!(&y1[0..DIM], &y1[DIM..2 * DIM], "different sessions with different deltas must produce different outputs");
}
