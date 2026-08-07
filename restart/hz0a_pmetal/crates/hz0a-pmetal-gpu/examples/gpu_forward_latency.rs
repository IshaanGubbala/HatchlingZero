//! Isolates how much of the GPU-FFI-vs-MLX gap
//! (`docs/restart/hz0c_c8_model_level_integration_results.md`) is real
//! Metal kernel dispatch cost versus Python/NumPy/`ctypes` marshaling
//! overhead -- named as a real, uninvestigated candidate cause in that
//! doc rather than left unattributed. This times
//! `MetalConditionalAnchorAttention::forward` directly from Rust, with NO
//! Python process, NO `ctypes`, NO NumPy array construction involved --
//! the purest possible measurement of the kernel's own dispatch cost at
//! the same shapes `scripts/hz0c_c8_ffi_latency_benchmark.py` already
//! measured through the full Python bridge.
//!
//! Run: `cargo run --release --example gpu_forward_latency -p hz0a-pmetal-gpu --manifest-path restart/hz0a_pmetal/Cargo.toml`

use std::time::Instant;

use hz0a_pmetal_gpu::MetalConditionalAnchorAttention;

fn lcg_f32(state: &mut u64, scale: f32) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*state >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * 2.0 * scale
}

fn make(n: usize, state: &mut u64, scale: f32) -> Vec<f32> {
    (0..n).map(|_| lcg_f32(state, scale)).collect()
}

fn trigger_at_rate(seq: usize, rate: f32) -> Vec<f32> {
    let count = ((rate * seq as f32).round() as usize).max(1);
    let mut row = vec![0.0f32; seq];
    for slot in row.iter_mut().take(count) {
        *slot = 1.0;
    }
    row
}

fn main() {
    let gpu = MetalConditionalAnchorAttention::new().expect("no Metal device available");
    let (batch, dim, heads) = (1usize, 768usize, 12usize);
    let repeats = 8;
    let warmup = 2;

    println!("shape,seq,rate,mean_ms,std_ms");
    for &seq in &[40usize, 128usize] {
        let mut state = 555u64;
        let x = make(batch * seq * dim, &mut state, 0.3);
        let qkv_w = make(3 * dim * dim, &mut state, 0.02);
        let qkv_b = make(3 * dim, &mut state, 0.02);
        let out_w = make(dim * dim, &mut state, 0.02);
        let out_b = make(dim, &mut state, 0.02);

        for &rate in &[0.0f32, 0.15, 1.0] {
            let trigger = trigger_at_rate(seq, rate);
            for _ in 0..warmup {
                gpu.forward(
                    batch, seq, dim, heads, &x, &qkv_w, &qkv_b, &out_w, &out_b, &trigger,
                )
                .unwrap();
            }
            let mut timings = Vec::with_capacity(repeats);
            for _ in 0..repeats {
                let started = Instant::now();
                gpu.forward(
                    batch, seq, dim, heads, &x, &qkv_w, &qkv_b, &out_w, &out_b, &trigger,
                )
                .unwrap();
                timings.push(started.elapsed().as_secs_f64() * 1000.0);
            }
            let mean = timings.iter().sum::<f64>() / timings.len() as f64;
            let variance =
                timings.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / timings.len() as f64;
            println!("seq={seq},rate={rate},{mean:.4},{:.4}", variance.sqrt());
        }
    }
}
