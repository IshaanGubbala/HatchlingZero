//! Rust-native correctness tests mirroring
//! `tests/reference/test_hz0b_memory_simulator.py`'s own coverage --
//! not a cross-language parity check (that's `parity.rs`), just
//! confirming this Rust port's own logic is internally correct against
//! the same behavioral contract the Python reference is tested against.

use hz0b_pmetal_memory::*;

fn onehot(dim: usize, index: usize) -> Vec<f32> {
    (0..dim).map(|i| if i == index { 1.0 } else { 0.0 }).collect()
}

#[test]
fn store_and_retrieve_exact() {
    let state = reset(1, 8, 4, 4);
    let key = onehot(4, 0);
    let value: Vec<f32> = onehot(4, 0).iter().map(|v| v * 3.0).collect();
    let (state, slot, rejected) = write(&state, &key, &value, &[1.0], 0, 1, None);
    assert!(!rejected[0]);
    let (readout, _) = read(&state, &key, Some(&slot), false, true);
    assert!((0..4).all(|i| (readout[i] - value[i]).abs() < 1e-5));
}

#[test]
fn overwrite_existing_fact_same_key_same_slot() {
    let state = reset(1, 8, 4, 4);
    let key = onehot(4, 0);
    let value1: Vec<f32> = onehot(4, 0).iter().map(|v| v * 1.0).collect();
    let value2: Vec<f32> = onehot(4, 0).iter().map(|v| v * 99.0).collect();
    let (state, slot1, _) = write(&state, &key, &value1, &[1.0], 0, 1, None);
    let (state, slot2, rejected) = write(&state, &key, &value2, &[1.0], 1, 1, None);
    assert!(!rejected[0]);
    assert_eq!(slot1[0], slot2[0], "same key must route to the same slot (update, not a new fact)");
    let (readout, _) = read(&state, &key, Some(&slot2), false, true);
    assert!((0..4).all(|i| (readout[i] - value2[i]).abs() < 1e-5));
}

#[test]
fn near_identical_keys_stay_distinct_after_the_0_999_threshold_fix() {
    // cosine ~0.995 -- below the 0.999 threshold, must NOT be treated as
    // a match. Mirrors the exact scenario from
    // reference/hz0b_b8_stage5_adversarial.py::scenario_near_identical_keys.
    let state = reset(1, 8, 16, 16);
    let key_a = onehot(16, 0);
    let mut key_b_raw = onehot(16, 0);
    key_b_raw.iter_mut().for_each(|v| *v *= 0.995);
    key_b_raw[1] = (1.0f32 - 0.995f32 * 0.995f32).sqrt();
    let norm = (key_b_raw.iter().map(|v| v * v).sum::<f32>()).sqrt();
    let key_b: Vec<f32> = key_b_raw.iter().map(|v| v / norm).collect();

    let value_a: Vec<f32> = onehot(16, 0).iter().map(|v| v * 5.0).collect();
    let value_b: Vec<f32> = onehot(16, 1).iter().map(|v| v * 5.0).collect();
    let (state, slot_a, _) = write(&state, &key_a, &value_a, &[1.0], 0, 1, None);
    let (state, slot_b, _) = write(&state, &key_b, &value_b, &[1.0], 1, 1, None);
    assert_ne!(slot_a[0], slot_b[0], "near-identical-but-different keys must land in distinct slots");
    let (readout_a, _) = read(&state, &key_a, Some(&slot_a), false, true);
    let (readout_b, _) = read(&state, &key_b, Some(&slot_b), false, true);
    assert!((0..16).all(|i| (readout_a[i] - value_a[i]).abs() < 1e-4));
    assert!((0..16).all(|i| (readout_b[i] - value_b[i]).abs() < 1e-4));
}

#[test]
fn protection_blocks_a_direct_overwrite() {
    let state = reset(1, 8, 4, 4);
    let key = onehot(4, 0);
    let legit: Vec<f32> = onehot(4, 0).iter().map(|v| v * 5.0).collect();
    let (state, slot, _) = write(&state, &key, &legit, &[1.0], 0, 1, None);
    let state = protect(&state, &slot, &[1.0]);
    let attacker: Vec<f32> = onehot(4, 1).iter().map(|v| v * 99.0).collect();
    let (state_after, _, rejected) = write(&state, &key, &attacker, &[1.0], 1, 1, Some(&slot));
    assert!(rejected[0]);
    let (readout, _) = read(&state_after, &key, Some(&slot), false, true);
    assert!((0..4).all(|i| (readout[i] - legit[i]).abs() < 1e-5));
}

#[test]
fn reset_wipes_everything() {
    let state = reset(1, 8, 4, 4);
    let key = onehot(4, 0);
    let value: Vec<f32> = onehot(4, 0).iter().map(|v| v * 5.0).collect();
    let (state, slot, _) = write(&state, &key, &value, &[1.0], 0, 1, None);
    let state = protect(&state, &slot, &[1.0]);
    assert!(state.confidence.iter().sum::<f32>() > 0.0);

    let fresh = reset(1, 8, 4, 4);
    assert!(fresh.confidence.iter().all(|&c| c == 0.0));
    assert!(fresh.protection.iter().all(|&p| p == 0.0));
    assert!(fresh.keys.iter().all(|&k| k == 0.0));
}

#[test]
fn forget_or_decay_reduces_confidence_and_protection_slows_it() {
    let state = reset(1, 8, 4, 4);
    let key0 = onehot(4, 0);
    let key1 = onehot(4, 1);
    let value: Vec<f32> = onehot(4, 0).iter().map(|v| v * 5.0).collect();
    let (state, slot0, _) = write(&state, &key0, &value, &[1.0], 0, 1, None);
    let (state, slot1, _) = write(&state, &key1, &value, &[1.0], 1, 1, None);
    let state = protect(&state, &slot1, &[1.0]);

    let mut decayed = state.clone();
    for _ in 0..20 {
        decayed = forget_or_decay(&decayed, 0.9);
    }
    let unprotected_conf = decayed.confidence[slot0[0] as usize];
    let protected_conf = decayed.confidence[slot1[0] as usize];
    assert!(unprotected_conf < 0.2, "unprotected confidence should decay substantially over 20 steps");
    assert!(protected_conf > 0.99, "protected confidence should barely decay");
}

#[test]
fn confidence_weighted_read_prefers_fresh_over_stale() {
    let state = reset(1, 8, 4, 4);
    let key = onehot(4, 0);
    let stale_value: Vec<f32> = onehot(4, 0).iter().map(|v| v * 5.0).collect();
    let fresh_value: Vec<f32> = onehot(4, 1).iter().map(|v| v * 5.0).collect();
    let (state, slot0, _) = write(&state, &key, &stale_value, &[1.0], 0, 1, Some(&[0]));
    let mut decayed = state.clone();
    for _ in 0..50 {
        decayed = forget_or_decay(&decayed, 0.9);
    }
    let (decayed, slot1, _) = write(&decayed, &key, &fresh_value, &[1.0], 50, 1, Some(&[1]));
    let _ = slot0;

    let (readout_weighted, _) = read(&decayed, &key, None, true, true);
    let (readout_unweighted, _) = read(&decayed, &key, None, true, false);
    assert!((0..4).all(|i| (readout_weighted[i] - fresh_value[i]).abs() < 1e-4), "confidence-weighted hard read must prefer the fresh memory");
    assert!((0..4).all(|i| (readout_unweighted[i] - stale_value[i]).abs() < 1e-4), "unweighted read ties go to the lower-index (stale) slot");
    let _ = slot1;
}

#[test]
fn capacity_pressure_protected_memory_survives() {
    let mut state = reset(1, 8, 8, 8);
    let protected_key = onehot(8, 0);
    let protected_value: Vec<f32> = onehot(8, 0).iter().map(|v| v * 5.0).collect();
    let (s, protected_slot, _) = write(&state, &protected_key, &protected_value, &[1.0], 0, 1, None);
    state = protect(&s, &protected_slot, &[1.0]);
    for i in 1..12 {
        let key = onehot(8, i % 8);
        let value: Vec<f32> = onehot(8, i % 8).iter().map(|v| v * 2.0).collect();
        let (s, _, _) = write(&state, &key, &value, &[1.0], i as i32, 1, None);
        state = s;
    }
    let (readout, _) = read(&state, &protected_key, Some(&protected_slot), false, true);
    assert!((0..8).all(|i| (readout[i] - protected_value[i]).abs() < 1e-4));
}
