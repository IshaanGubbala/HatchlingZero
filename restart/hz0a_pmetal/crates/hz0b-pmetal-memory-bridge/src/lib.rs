//! C-ABI FFI boundary for `hz0b-pmetal-memory`, loaded from Python via
//! `ctypes` (see `python/hz0b_memory_bridge.py`). No PyO3/maturin
//! dependency -- a plain `cdylib` with `extern "C"` functions, matching
//! this workspace's own zero-extra-toolchain convention.
//!
//! All `unsafe` in this crate is confined to marshalling raw pointers at
//! the boundary (`from_c`/`write_out`); every actual memory-op
//! computation still runs inside `hz0b-pmetal-memory`, which stays
//! `#![forbid(unsafe_code)]`. Caller (Python) owns and pre-allocates every
//! buffer -- shapes are fully determined by `batch`/`num_slots`/
//! `key_dim`/`value_dim`, which the caller already knows, so no
//! allocation crosses the FFI boundary in either direction and there is
//! no `free`-style function to pair.
//!
//! `serialize`/`restore` are intentionally NOT exposed here: in the Rust
//! reference they are pure clones (see `hz0b-pmetal-memory/src/lib.rs`),
//! and a Python caller that already holds every field it passed in gains
//! nothing by round-tripping them through the FFI boundary -- exposing
//! them would be a copy-function for its own sake, not a real capability.

use hz0b_pmetal_memory as mem;
use hz0b_pmetal_memory::MemoryState;

#[repr(C)]
pub struct CMemoryStateIn {
    pub batch: usize,
    pub num_slots: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub keys: *const f32,
    pub values: *const f32,
    pub confidence: *const f32,
    pub age: *const i32,
    pub protection: *const f32,
    pub write_count: *const i32,
    pub last_write_step: *const i32,
    pub write_source: *const i32,
}

#[repr(C)]
pub struct CMemoryStateOut {
    pub keys: *mut f32,
    pub values: *mut f32,
    pub confidence: *mut f32,
    pub age: *mut i32,
    pub protection: *mut f32,
    pub write_count: *mut i32,
    pub last_write_step: *mut i32,
    pub write_source: *mut i32,
}

unsafe fn from_c(s: &CMemoryStateIn) -> MemoryState {
    let n = s.batch * s.num_slots;
    MemoryState {
        batch: s.batch,
        num_slots: s.num_slots,
        key_dim: s.key_dim,
        value_dim: s.value_dim,
        keys: std::slice::from_raw_parts(s.keys, n * s.key_dim).to_vec(),
        values: std::slice::from_raw_parts(s.values, n * s.value_dim).to_vec(),
        confidence: std::slice::from_raw_parts(s.confidence, n).to_vec(),
        age: std::slice::from_raw_parts(s.age, n).to_vec(),
        protection: std::slice::from_raw_parts(s.protection, n).to_vec(),
        write_count: std::slice::from_raw_parts(s.write_count, n).to_vec(),
        last_write_step: std::slice::from_raw_parts(s.last_write_step, n).to_vec(),
        write_source: std::slice::from_raw_parts(s.write_source, n).to_vec(),
    }
}

unsafe fn write_out(state: &MemoryState, out: &CMemoryStateOut) {
    std::slice::from_raw_parts_mut(out.keys, state.keys.len()).copy_from_slice(&state.keys);
    std::slice::from_raw_parts_mut(out.values, state.values.len()).copy_from_slice(&state.values);
    std::slice::from_raw_parts_mut(out.confidence, state.confidence.len()).copy_from_slice(&state.confidence);
    std::slice::from_raw_parts_mut(out.age, state.age.len()).copy_from_slice(&state.age);
    std::slice::from_raw_parts_mut(out.protection, state.protection.len()).copy_from_slice(&state.protection);
    std::slice::from_raw_parts_mut(out.write_count, state.write_count.len()).copy_from_slice(&state.write_count);
    std::slice::from_raw_parts_mut(out.last_write_step, state.last_write_step.len()).copy_from_slice(&state.last_write_step);
    std::slice::from_raw_parts_mut(out.write_source, state.write_source.len()).copy_from_slice(&state.write_source);
}

#[no_mangle]
pub unsafe extern "C" fn hz0b_reset(batch: usize, num_slots: usize, key_dim: usize, value_dim: usize, out: *mut CMemoryStateOut) {
    let state = mem::reset(batch, num_slots, key_dim, value_dim);
    write_out(&state, &*out);
}

#[no_mangle]
pub unsafe extern "C" fn hz0b_read(
    state_in: *const CMemoryStateIn,
    query: *const f32,
    slot_idx: *const i32,
    hard: u8,
    confidence_weighted: u8,
    out_readout: *mut f32,
    out_weights: *mut f32,
) {
    let state = from_c(&*state_in);
    let query = std::slice::from_raw_parts(query, state.batch * state.key_dim);
    let idx_opt = if slot_idx.is_null() { None } else { Some(std::slice::from_raw_parts(slot_idx, state.batch)) };
    let (readout, weights) = mem::read(&state, query, idx_opt, hard != 0, confidence_weighted != 0);
    std::slice::from_raw_parts_mut(out_readout, readout.len()).copy_from_slice(&readout);
    std::slice::from_raw_parts_mut(out_weights, weights.len()).copy_from_slice(&weights);
}

#[no_mangle]
pub unsafe extern "C" fn hz0b_write(
    state_in: *const CMemoryStateIn,
    key: *const f32,
    value: *const f32,
    strength: *const f32,
    step: i32,
    source: i32,
    slot_idx: *const i32,
    out_state: *mut CMemoryStateOut,
    out_slot: *mut i32,
    out_rejected: *mut u8,
) {
    let state = from_c(&*state_in);
    let batch = state.batch;
    let key = std::slice::from_raw_parts(key, batch * state.key_dim);
    let value = std::slice::from_raw_parts(value, batch * state.value_dim);
    let strength = std::slice::from_raw_parts(strength, batch);
    let idx_opt = if slot_idx.is_null() { None } else { Some(std::slice::from_raw_parts(slot_idx, batch)) };
    let (new_state, slot, rejected) = mem::write(&state, key, value, strength, step, source, idx_opt);
    write_out(&new_state, &*out_state);
    std::slice::from_raw_parts_mut(out_slot, batch).copy_from_slice(&slot);
    let rejected_u8: Vec<u8> = rejected.iter().map(|&b| b as u8).collect();
    std::slice::from_raw_parts_mut(out_rejected, batch).copy_from_slice(&rejected_u8);
}

#[no_mangle]
pub unsafe extern "C" fn hz0b_reinforce(state_in: *const CMemoryStateIn, slot_idx: *const i32, out_state: *mut CMemoryStateOut) {
    let state = from_c(&*state_in);
    let idx = std::slice::from_raw_parts(slot_idx, state.batch);
    let new_state = mem::reinforce(&state, idx);
    write_out(&new_state, &*out_state);
}

#[no_mangle]
pub unsafe extern "C" fn hz0b_update(state_in: *const CMemoryStateIn, slot_idx: *const i32, new_value: *const f32, out_state: *mut CMemoryStateOut) {
    let state = from_c(&*state_in);
    let idx = std::slice::from_raw_parts(slot_idx, state.batch);
    let new_value = std::slice::from_raw_parts(new_value, state.batch * state.value_dim);
    let new_state = mem::update(&state, idx, new_value);
    write_out(&new_state, &*out_state);
}

#[no_mangle]
pub unsafe extern "C" fn hz0b_protect(state_in: *const CMemoryStateIn, slot_idx: *const i32, strength: *const f32, out_state: *mut CMemoryStateOut) {
    let state = from_c(&*state_in);
    let idx = std::slice::from_raw_parts(slot_idx, state.batch);
    let strength = std::slice::from_raw_parts(strength, state.batch);
    let new_state = mem::protect(&state, idx, strength);
    write_out(&new_state, &*out_state);
}

#[no_mangle]
pub unsafe extern "C" fn hz0b_forget_or_decay(state_in: *const CMemoryStateIn, decay_rate: f32, out_state: *mut CMemoryStateOut) {
    let state = from_c(&*state_in);
    let new_state = mem::forget_or_decay(&state, decay_rate);
    write_out(&new_state, &*out_state);
}

#[no_mangle]
pub unsafe extern "C" fn hz0b_delete(state_in: *const CMemoryStateIn, slot_idx: *const i32, out_state: *mut CMemoryStateOut) {
    let state = from_c(&*state_in);
    let idx = std::slice::from_raw_parts(slot_idx, state.batch);
    let new_state = mem::delete(&state, idx);
    write_out(&new_state, &*out_state);
}
