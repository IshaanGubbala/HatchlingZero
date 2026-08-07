//! C-ABI FFI boundary for `hz0d-pmetal-fastweights`, loaded from Python
//! via `ctypes` (see `python/hz0d_fastweights_bridge.py`). No PyO3/
//! maturin dependency, matching this workspace's own zero-extra-
//! toolchain convention (`hz0b-pmetal-memory-bridge`'s own precedent).
//!
//! All `unsafe` in this crate is confined to marshalling raw pointers at
//! the boundary; every actual fast-weight computation still runs inside
//! `hz0d-pmetal-fastweights`, which stays `#![forbid(unsafe_code)]`.
//! Caller (Python) owns and pre-allocates every buffer -- shapes are
//! fully determined by `sessions`/`num_layers`/`dim`/`rank`, which the
//! caller already knows.

use hz0d_pmetal_fastweights as fw;
use hz0d_pmetal_fastweights::FastWeightState;

#[repr(C)]
pub struct CFastWeightStateIn {
    pub sessions: usize,
    pub num_layers: usize,
    pub dim: usize,
    pub rank: usize,
    pub a_fast: *const f32,
    pub b_fast: *const f32,
    pub update_count: *const i32,
}

#[repr(C)]
pub struct CFastWeightStateOut {
    pub a_fast: *mut f32,
    pub b_fast: *mut f32,
    pub update_count: *mut i32,
}

unsafe fn from_c(s: &CFastWeightStateIn) -> FastWeightState {
    FastWeightState {
        sessions: s.sessions,
        num_layers: s.num_layers,
        dim: s.dim,
        rank: s.rank,
        a_fast: std::slice::from_raw_parts(s.a_fast, s.sessions * s.num_layers * s.dim * s.rank)
            .to_vec(),
        b_fast: std::slice::from_raw_parts(s.b_fast, s.sessions * s.num_layers * s.rank * s.dim)
            .to_vec(),
        update_count: std::slice::from_raw_parts(s.update_count, s.sessions).to_vec(),
    }
}

unsafe fn write_out(state: &FastWeightState, out: &CFastWeightStateOut) {
    std::slice::from_raw_parts_mut(out.a_fast, state.a_fast.len()).copy_from_slice(&state.a_fast);
    std::slice::from_raw_parts_mut(out.b_fast, state.b_fast.len()).copy_from_slice(&state.b_fast);
    std::slice::from_raw_parts_mut(out.update_count, state.update_count.len())
        .copy_from_slice(&state.update_count);
}

#[no_mangle]
pub unsafe extern "C" fn hz0d_reset(
    sessions: usize,
    num_layers: usize,
    dim: usize,
    rank: usize,
    a_fast_init: *const f32,
    out: *mut CFastWeightStateOut,
) {
    let a_init = std::slice::from_raw_parts(a_fast_init, sessions * num_layers * dim * rank);
    let state = fw::reset(sessions, num_layers, dim, rank, a_init);
    write_out(&state, &*out);
}

#[no_mangle]
pub unsafe extern "C" fn hz0d_apply(
    state_in: *const CFastWeightStateIn,
    x: *const f32,
    base_weight: *const f32,
    base_bias: *const f32,
    layer: usize,
    out_y: *mut f32,
) {
    let state = from_c(&*state_in);
    let x = std::slice::from_raw_parts(x, state.sessions * state.dim);
    let base_weight = std::slice::from_raw_parts(base_weight, state.dim * state.dim);
    let base_bias = std::slice::from_raw_parts(base_bias, state.dim);
    let y = fw::apply(&state, x, base_weight, base_bias, layer);
    std::slice::from_raw_parts_mut(out_y, y.len()).copy_from_slice(&y);
}

#[no_mangle]
pub unsafe extern "C" fn hz0d_update(
    state_in: *const CFastWeightStateIn,
    session: usize,
    layer: usize,
    grad_a: *const f32,
    grad_b: *const f32,
    lr: f32,
    max_delta_norm: f32,
    out: *mut CFastWeightStateOut,
) {
    let state = from_c(&*state_in);
    let grad_a = std::slice::from_raw_parts(grad_a, state.dim * state.rank);
    let grad_b = std::slice::from_raw_parts(grad_b, state.rank * state.dim);
    let new_state = fw::update(&state, session, layer, grad_a, grad_b, lr, max_delta_norm);
    write_out(&new_state, &*out);
}

#[no_mangle]
pub unsafe extern "C" fn hz0d_decay(
    state_in: *const CFastWeightStateIn,
    decay_rate: f32,
    out: *mut CFastWeightStateOut,
) {
    let state = from_c(&*state_in);
    let new_state = fw::decay(&state, decay_rate);
    write_out(&new_state, &*out);
}

#[no_mangle]
pub unsafe extern "C" fn hz0d_effective_delta(
    state_in: *const CFastWeightStateIn,
    session: usize,
    layer: usize,
    out_delta: *mut f32,
) {
    let state = from_c(&*state_in);
    let delta = fw::effective_delta(&state, session, layer);
    std::slice::from_raw_parts_mut(out_delta, delta.len()).copy_from_slice(&delta);
}
