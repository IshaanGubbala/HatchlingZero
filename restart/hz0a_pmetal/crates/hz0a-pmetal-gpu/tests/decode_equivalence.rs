//! A12 ("fused Metal inference"): validates the existing Metal-dispatched
//! GDN-2 forward kernel (already a single Metal dispatch -- "PMetal-native
//! fused GDN-2 forward" per the plan's own phrasing, not a new kernel) as a
//! real decode path: full-sequence prefill versus token-by-token decode
//! must produce numerically equivalent outputs and final state, state must
//! reset and serialize/round-trip correctly, and prefill/decode speed must
//! be measured separately from each other and from the CPU-only baseline
//! (matching the plan's explicit checklist: full-sequence vs token-by-token
//! equivalence, chunk boundaries, reset/serialization, prefill vs decode
//! timing, kernel speed vs end-to-end speed).

use hz0a_pmetal_gpu::MetalGdn2Forward;
use hz0a_pmetal_kernel::{gdn2_forward_f32, Gdn2ForwardShape};
use std::time::Instant;

fn lcg_f32(state: &mut u64, scale: f32) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let u = ((*state >> 40) as f32 / (1u64 << 24) as f32) - 0.5;
    u * 2.0 * scale
}

struct Fixture {
    shape: Gdn2ForwardShape,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    d: Vec<f32>,
    e: Vec<f32>,
    w: Vec<f32>,
    initial_state: Vec<f32>,
}

fn build_fixture(batch: usize, seq: usize, heads: usize, key_dim: usize, value_dim: usize, seed: u64) -> Fixture {
    let shape = Gdn2ForwardShape { batch, seq, heads, key_dim, value_dim };
    let mut state = seed;
    let qk_len = batch * seq * heads * key_dim;
    let v_len = batch * seq * heads * value_dim;
    let state_len = batch * heads * value_dim * key_dim;
    let make = |n: usize, s: &mut u64| (0..n).map(|_| lcg_f32(s, 0.5)).collect::<Vec<f32>>();
    Fixture {
        q: make(qk_len, &mut state),
        k: make(qk_len, &mut state),
        v: make(v_len, &mut state),
        d: make(qk_len, &mut state),
        e: make(qk_len, &mut state),
        w: make(v_len, &mut state),
        initial_state: make(state_len, &mut state),
        shape,
    }
}

/// Slices out timestep `t`'s inputs from a fixture spanning the full
/// sequence, matching the layout `gdn2_forward_f32`/`MetalGdn2Forward`
/// expect: (batch, seq, heads, dim) row-major, batch=1 assumed here.
fn slice_timestep(fixture: &Fixture, t: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let h = fixture.shape.heads;
    let k = fixture.shape.key_dim;
    let v = fixture.shape.value_dim;
    let qk_row = h * k;
    let v_row = h * v;
    let qk_start = t * qk_row;
    let v_start = t * v_row;
    (
        fixture.q[qk_start..qk_start + qk_row].to_vec(),
        fixture.k[qk_start..qk_start + qk_row].to_vec(),
        fixture.v[v_start..v_start + v_row].to_vec(),
        fixture.d[qk_start..qk_start + qk_row].to_vec(),
        fixture.e[qk_start..qk_start + qk_row].to_vec(),
        fixture.w[v_start..v_start + v_row].to_vec(),
    )
}

#[test]
fn full_sequence_prefill_matches_token_by_token_decode() {
    let fixture = build_fixture(1, 16, 4, 8, 8, 7);
    let gpu = MetalGdn2Forward::new().expect("Metal device required");

    // Prefill: one call over the whole sequence.
    let prefill = gpu.forward(&fixture.shape, &fixture.q, &fixture.k, &fixture.v, &fixture.d, &fixture.e, &fixture.w, &fixture.initial_state);

    // Decode: one call per timestep, state carried token-to-token.
    let single_step_shape = Gdn2ForwardShape { batch: 1, seq: 1, heads: fixture.shape.heads, key_dim: fixture.shape.key_dim, value_dim: fixture.shape.value_dim };
    let mut state = fixture.initial_state.clone();
    let mut decode_outputs = Vec::new();
    for t in 0..fixture.shape.seq {
        let (q, k, v, d, e, w) = slice_timestep(&fixture, t);
        let result = gpu.forward(&single_step_shape, &q, &k, &v, &d, &e, &w, &state);
        decode_outputs.extend(result.outputs);
        state = result.final_state;
    }

    let max_output_diff = prefill.outputs.iter().zip(decode_outputs.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let max_state_diff = prefill.final_state.iter().zip(state.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_output_diff < 1e-3, "prefill vs decode output diff too large: {max_output_diff}");
    assert!(max_state_diff < 1e-3, "prefill vs decode final-state diff too large: {max_state_diff}");
}

#[test]
fn chunk_boundary_decode_matches_full_sequence() {
    // Decode in irregular chunk sizes (not all size-1, not all the same
    // size) crossing several "boundaries" -- proves state carry is correct
    // at arbitrary chunk splits, not just the single-token decode case.
    let fixture = build_fixture(1, 24, 4, 8, 8, 11);
    let gpu = MetalGdn2Forward::new().expect("Metal device required");
    let prefill = gpu.forward(&fixture.shape, &fixture.q, &fixture.k, &fixture.v, &fixture.d, &fixture.e, &fixture.w, &fixture.initial_state);

    let chunk_sizes = [5usize, 3, 7, 9]; // sums to 24
    assert_eq!(chunk_sizes.iter().sum::<usize>(), fixture.shape.seq);
    let mut state = fixture.initial_state.clone();
    let mut chunked_outputs = Vec::new();
    let mut start = 0;
    for &size in &chunk_sizes {
        let chunk_shape = Gdn2ForwardShape { batch: 1, seq: size, heads: fixture.shape.heads, key_dim: fixture.shape.key_dim, value_dim: fixture.shape.value_dim };
        let h = fixture.shape.heads;
        let k = fixture.shape.key_dim;
        let v = fixture.shape.value_dim;
        let qk_start = start * h * k;
        let qk_end = (start + size) * h * k;
        let v_start = start * h * v;
        let v_end = (start + size) * h * v;
        let result = gpu.forward(
            &chunk_shape,
            &fixture.q[qk_start..qk_end],
            &fixture.k[qk_start..qk_end],
            &fixture.v[v_start..v_end],
            &fixture.d[qk_start..qk_end],
            &fixture.e[qk_start..qk_end],
            &fixture.w[v_start..v_end],
            &state,
        );
        chunked_outputs.extend(result.outputs);
        state = result.final_state;
        start += size;
    }

    let max_output_diff = prefill.outputs.iter().zip(chunked_outputs.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let max_state_diff = prefill.final_state.iter().zip(state.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_output_diff < 1e-3, "chunked vs full-sequence output diff too large: {max_output_diff}");
    assert!(max_state_diff < 1e-3, "chunked vs full-sequence final-state diff too large: {max_state_diff}");
}

#[test]
fn state_reset_produces_fresh_generation() {
    let fixture = build_fixture(1, 8, 4, 8, 8, 5);
    let gpu = MetalGdn2Forward::new().expect("Metal device required");
    let state_len = fixture.shape.heads * fixture.shape.value_dim * fixture.shape.key_dim;

    let zero_state = vec![0.0f32; state_len];
    let (q, k, v, d, e, w) = slice_timestep(&fixture, 0);
    let single_step_shape = Gdn2ForwardShape { batch: 1, seq: 1, heads: fixture.shape.heads, key_dim: fixture.shape.key_dim, value_dim: fixture.shape.value_dim };

    let from_zero_a = gpu.forward(&single_step_shape, &q, &k, &v, &d, &e, &w, &zero_state);
    let from_zero_b = gpu.forward(&single_step_shape, &q, &k, &v, &d, &e, &w, &zero_state);
    // Same inputs from the same reset state must give bit-identical output --
    // no hidden carry from a previous call leaking through.
    assert_eq!(from_zero_a.outputs, from_zero_b.outputs);
    assert_eq!(from_zero_a.final_state, from_zero_b.final_state);

    // A non-zero prior state must produce a DIFFERENT result than a reset --
    // proves the state argument is actually being used, not silently ignored.
    let nonzero_state = fixture.initial_state[..state_len].to_vec();
    let from_nonzero = gpu.forward(&single_step_shape, &q, &k, &v, &d, &e, &w, &nonzero_state);
    let diff = from_zero_a.outputs.iter().zip(from_nonzero.outputs.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(diff > 1e-6, "reset vs non-reset state produced suspiciously identical output: diff={diff}");
}

#[test]
fn state_serialization_round_trip_preserves_decode() {
    // "Serialization" for this state representation is just Vec<f32> ->
    // bytes -> Vec<f32> (the real checkpoint format already does this via
    // .npy files); the property under test is that decode continues
    // identically after a save/reload, not the byte format itself.
    let fixture = build_fixture(1, 12, 4, 8, 8, 13);
    let gpu = MetalGdn2Forward::new().expect("Metal device required");
    let single_step_shape = Gdn2ForwardShape { batch: 1, seq: 1, heads: fixture.shape.heads, key_dim: fixture.shape.key_dim, value_dim: fixture.shape.value_dim };

    let mut state = fixture.initial_state.clone();
    for t in 0..6 {
        let (q, k, v, d, e, w) = slice_timestep(&fixture, t);
        let result = gpu.forward(&single_step_shape, &q, &k, &v, &d, &e, &w, &state);
        state = result.final_state;
    }

    // "Serialize": pack to bytes and back.
    let bytes: Vec<u8> = state.iter().flat_map(|value| value.to_le_bytes()).collect();
    let reloaded_state: Vec<f32> = bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();
    assert_eq!(state, reloaded_state, "serialize/deserialize round trip must be exact");

    // Continue decode from both the in-memory and the reloaded state; must match.
    let (q, k, v, d, e, w) = slice_timestep(&fixture, 6);
    let continued_from_memory = gpu.forward(&single_step_shape, &q, &k, &v, &d, &e, &w, &state);
    let continued_from_reload = gpu.forward(&single_step_shape, &q, &k, &v, &d, &e, &w, &reloaded_state);
    assert_eq!(continued_from_memory.outputs, continued_from_reload.outputs);
}

#[test]
fn prefill_and_decode_speed_measured_separately_and_vs_cpu() {
    // The locked A1 shape (dim=768, 12 heads, key/value_dim=64), a
    // realistic prefill length, batch=1 (the natural inference shape,
    // unlike training's batched shape).
    let shape = Gdn2ForwardShape { batch: 1, seq: 256, heads: 12, key_dim: 64, value_dim: 64 };
    let fixture = build_fixture(shape.batch, shape.seq, shape.heads, shape.key_dim, shape.value_dim, 23);
    let gpu = MetalGdn2Forward::new().expect("Metal device required");

    // Warm up (first dispatch pays one-time pipeline/library compile cost).
    let _ = gpu.forward(&fixture.shape, &fixture.q, &fixture.k, &fixture.v, &fixture.d, &fixture.e, &fixture.w, &fixture.initial_state);

    let prefill_start = Instant::now();
    let prefill_runs = 10;
    for _ in 0..prefill_runs {
        let _ = gpu.forward(&fixture.shape, &fixture.q, &fixture.k, &fixture.v, &fixture.d, &fixture.e, &fixture.w, &fixture.initial_state);
    }
    let prefill_gpu_ms = prefill_start.elapsed().as_secs_f64() * 1000.0 / prefill_runs as f64;

    let single_step_shape = Gdn2ForwardShape { batch: 1, seq: 1, heads: shape.heads, key_dim: shape.key_dim, value_dim: shape.value_dim };
    let decode_tokens = 64; // separate, shorter measurement -- decode is dispatch-overhead-bound, not compute-bound, so this isolates that cost cleanly
    let mut state = fixture.initial_state.clone();
    let decode_start = Instant::now();
    for t in 0..decode_tokens {
        let (q, k, v, d, e, w) = slice_timestep(&fixture, t % fixture.shape.seq);
        let result = gpu.forward(&single_step_shape, &q, &k, &v, &d, &e, &w, &state);
        state = result.final_state;
    }
    let decode_gpu_ms_per_token = decode_start.elapsed().as_secs_f64() * 1000.0 / decode_tokens as f64;

    // CPU-only baseline for the same prefill shape, for the "improves
    // end-to-end inference" half of A12's exit gate -- the historical
    // ~9.5x figure is a target to reproduce, not a retained result, so
    // this reports the real ratio on this hardware rather than assuming it.
    let cpu_start = Instant::now();
    let cpu_runs = 3; // CPU path is much slower; fewer runs keeps this test fast
    for _ in 0..cpu_runs {
        let _ = gdn2_forward_f32(&fixture.shape, &fixture.q, &fixture.k, &fixture.v, &fixture.d, &fixture.e, &fixture.w, &fixture.initial_state).unwrap();
    }
    let prefill_cpu_ms = cpu_start.elapsed().as_secs_f64() * 1000.0 / cpu_runs as f64;

    println!("prefill (seq={}): GPU {:.3}ms, CPU {:.3}ms, speedup {:.2}x", shape.seq, prefill_gpu_ms, prefill_cpu_ms, prefill_cpu_ms / prefill_gpu_ms);
    println!("decode: {:.4}ms/token GPU (dispatch-overhead-bound, single-token calls)", decode_gpu_ms_per_token);

    assert!(prefill_gpu_ms > 0.0 && prefill_cpu_ms > 0.0 && decode_gpu_ms_per_token > 0.0);
    assert!(prefill_cpu_ms > prefill_gpu_ms, "GPU prefill should be faster than CPU prefill at this shape, was not: GPU={prefill_gpu_ms}ms CPU={prefill_cpu_ms}ms");
}
