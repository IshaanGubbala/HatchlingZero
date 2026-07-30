"""Controlled overwrite benchmark: current HZ-0A recurrence vs. candidate
GDN-3 variants (`reference/hz0a_gdn3_candidate_recurrence.py`), on a tiny,
synthetic, fully controlled task -- per the plan: don't retrain HZ-0A at
301M scale to test this, first check whether the delta-rule projection
term solves a real, measurable weakness at small scale where ground truth
is exactly known.

Task, exactly as specified:
  write key A -> value X
  write unrelated keys (B -> Z1, C -> Z2)
  overwrite key A -> value Y
  query key A          (want: Y, not X, not a blend)
  query B, C           (want: unchanged -- the A-overwrite should not
                         have disturbed unrelated memories)

Two parts:
  Part A (deterministic, no training): hand-set orthogonal keys and
  distinct values, run each variant's exact recurrence step-by-step, no
  gradients involved -- isolates the pure MECHANISM's overwrite/
  interference behavior, independent of whether a real system could learn
  good gate values.

  Part B (gradient + rough throughput sanity): confirms each variant is
  differentiable with finite gradients (a real training run needs this),
  and reports a rough Python-loop wall-clock per step as a coarse
  systems-gate proxy -- explicitly NOT a substitute for a real fused
  Metal kernel benchmark, which would be needed before trusting any
  throughput conclusion.
"""
from __future__ import annotations

import time

import mlx.core as mx

from reference.hz0a_gdn3_candidate_recurrence import (
    step_current,
    step_current_strong_erase,
    step_delta_projection,
    step_delta_projection_plus_decay,
)

DK = DV = 16


def cosine(a: mx.array, b: mx.array) -> float:
    return float(mx.sum(a * b) / (mx.sqrt(mx.sum(a * a)) * mx.sqrt(mx.sum(b * b)) + 1e-8))


def make_orthogonal_keys_and_values(seed: int = 0):
    key_a, key_b, key_c = mx.eye(DK)[0], mx.eye(DK)[1], mx.eye(DK)[2]  # perfectly orthogonal by construction
    rng = mx.random.key(seed)
    k1, k2, k3, k4 = mx.random.split(rng, 4)
    value_x = mx.random.normal((DV,), key=k1)
    value_y = mx.random.normal((DV,), key=k2)
    value_z1 = mx.random.normal((DV,), key=k3)
    value_z2 = mx.random.normal((DV,), key=k4)
    return key_a, key_b, key_c, value_x, value_y, value_z1, value_z2


def run_current(erase_value: float) -> dict:
    key_a, key_b, key_c, value_x, value_y, value_z1, value_z2 = make_orthogonal_keys_and_values()
    state = mx.zeros((DV, DK))
    decay, erase, write = mx.ones((DK,)), mx.full((DK,), erase_value), mx.ones((DV,))
    _, state = step_current(state, mx.zeros((DK,)), key_a, value_x, decay, erase, write)
    _, state = step_current(state, mx.zeros((DK,)), key_b, value_z1, decay, erase, write)
    _, state = step_current(state, mx.zeros((DK,)), key_c, value_z2, decay, erase, write)
    _, state = step_current(state, mx.zeros((DK,)), key_a, value_y, decay, erase, write)  # the overwrite
    return {
        "query_a": state @ key_a, "query_b": state @ key_b, "query_c": state @ key_c,
        "value_x": value_x, "value_y": value_y, "value_z1": value_z1, "value_z2": value_z2,
        "state_norm": float(mx.sqrt(mx.sum(state * state))),
    }


def run_current_strong_erase(boost: float = 4.0) -> dict:
    key_a, key_b, key_c, value_x, value_y, value_z1, value_z2 = make_orthogonal_keys_and_values()
    state = mx.zeros((DV, DK))
    decay, erase_logit, write = mx.ones((DK,)), mx.zeros((DK,)), mx.ones((DV,))
    _, state = step_current_strong_erase(state, mx.zeros((DK,)), key_a, value_x, decay, erase_logit, write, boost=boost)
    _, state = step_current_strong_erase(state, mx.zeros((DK,)), key_b, value_z1, decay, erase_logit, write, boost=boost)
    _, state = step_current_strong_erase(state, mx.zeros((DK,)), key_c, value_z2, decay, erase_logit, write, boost=boost)
    _, state = step_current_strong_erase(state, mx.zeros((DK,)), key_a, value_y, decay, erase_logit, write, boost=boost)
    return {
        "query_a": state @ key_a, "query_b": state @ key_b, "query_c": state @ key_c,
        "value_x": value_x, "value_y": value_y, "value_z1": value_z1, "value_z2": value_z2,
        "state_norm": float(mx.sqrt(mx.sum(state * state))),
    }


def run_delta_projection() -> dict:
    key_a, key_b, key_c, value_x, value_y, value_z1, value_z2 = make_orthogonal_keys_and_values()
    state = mx.zeros((DV, DK))
    beta = mx.array(1.0)
    _, state = step_delta_projection(state, mx.zeros((DK,)), key_a, value_x, beta)
    _, state = step_delta_projection(state, mx.zeros((DK,)), key_b, value_z1, beta)
    _, state = step_delta_projection(state, mx.zeros((DK,)), key_c, value_z2, beta)
    _, state = step_delta_projection(state, mx.zeros((DK,)), key_a, value_y, beta)  # the overwrite
    return {
        "query_a": state @ key_a, "query_b": state @ key_b, "query_c": state @ key_c,
        "value_x": value_x, "value_y": value_y, "value_z1": value_z1, "value_z2": value_z2,
        "state_norm": float(mx.sqrt(mx.sum(state * state))),
    }


def run_delta_projection_plus_decay(decay_value: float = 0.95) -> dict:
    key_a, key_b, key_c, value_x, value_y, value_z1, value_z2 = make_orthogonal_keys_and_values()
    state = mx.zeros((DV, DK))
    decay, beta = mx.full((DK,), decay_value), mx.array(1.0)
    _, state = step_delta_projection_plus_decay(state, mx.zeros((DK,)), key_a, value_x, decay, beta)
    _, state = step_delta_projection_plus_decay(state, mx.zeros((DK,)), key_b, value_z1, decay, beta)
    _, state = step_delta_projection_plus_decay(state, mx.zeros((DK,)), key_c, value_z2, decay, beta)
    _, state = step_delta_projection_plus_decay(state, mx.zeros((DK,)), key_a, value_y, decay, beta)
    return {
        "query_a": state @ key_a, "query_b": state @ key_b, "query_c": state @ key_c,
        "value_x": value_x, "value_y": value_y, "value_z1": value_z1, "value_z2": value_z2,
        "state_norm": float(mx.sqrt(mx.sum(state * state))),
    }


def report(name: str, result: dict) -> None:
    overwrite_toward_y = cosine(result["query_a"], result["value_y"])
    overwrite_toward_x = cosine(result["query_a"], result["value_x"])  # baseline: random cosine(X,Y) if overwrite is perfect, NOT necessarily 0
    interference_b_direction = 1.0 - cosine(result["query_b"], result["value_z1"])  # want ~0
    interference_c_direction = 1.0 - cosine(result["query_c"], result["value_z2"])  # want ~0
    # Magnitude retained, relative to what was originally written -- cosine
    # similarity is blind to uniform shrinkage (a channel-wise decay that
    # scales a column down without changing its direction still reads as
    # cosine=1.0), so this is the metric that actually catches "unrelated
    # memories quietly eroded by every subsequent write," which cosine
    # alone in the orthogonal-key case below will not reveal.
    magnitude_b = float(mx.sqrt(mx.sum(result["query_b"] ** 2))) / float(mx.sqrt(mx.sum(result["value_z1"] ** 2)) + 1e-8)
    magnitude_c = float(mx.sqrt(mx.sum(result["query_c"] ** 2))) / float(mx.sqrt(mx.sum(result["value_z2"] ** 2)) + 1e-8)
    print(f"{name:32s}  overwrite->Y: {overwrite_toward_y:+.4f}  cos(X,Y)baseline: {overwrite_toward_x:+.4f}  interfere_B(dir): {interference_b_direction:.4f}  interfere_C(dir): {interference_c_direction:.4f}  magnitude_B_retained: {magnitude_b:.3f}  magnitude_C_retained: {magnitude_c:.3f}  state_norm: {result['state_norm']:.3f}")


def make_near_duplicate_keys_and_values(seed: int = 0, similarity: float = 0.85):
    """key_b is NOT orthogonal to key_a (cosine ~= `similarity`) -- the
    realistic, harder case: HZ-0A's real learned keys will never be
    perfectly orthogonal like Part A's basis-vector test above. This is
    where a per-channel, key-blind erase gate and a key-TARGETED
    projection should actually behave differently."""
    key_a = mx.eye(DK)[0]
    orthogonal_component = mx.eye(DK)[1]
    key_b_raw = similarity * key_a + (1 - similarity**2) ** 0.5 * orthogonal_component
    key_b = key_b_raw / mx.sqrt(mx.sum(key_b_raw * key_b_raw))
    key_c = mx.eye(DK)[2]  # kept orthogonal, a control -- should be unaffected by either mechanism
    rng = mx.random.key(seed)
    k1, k2, k3, k4 = mx.random.split(rng, 4)
    value_x = mx.random.normal((DV,), key=k1)
    value_y = mx.random.normal((DV,), key=k2)
    value_z1 = mx.random.normal((DV,), key=k3)
    value_z2 = mx.random.normal((DV,), key=k4)
    return key_a, key_b, key_c, value_x, value_y, value_z1, value_z2


def run_near_duplicate(variant: str, **kwargs) -> dict:
    key_a, key_b, key_c, value_x, value_y, value_z1, value_z2 = make_near_duplicate_keys_and_values()
    state = mx.zeros((DV, DK))
    if variant == "current":
        decay, erase, write = mx.ones((DK,)), mx.full((DK,), kwargs["erase_value"]), mx.ones((DV,))
        for key, value in ((key_a, value_x), (key_b, value_z1), (key_c, value_z2), (key_a, value_y)):
            _, state = step_current(state, mx.zeros((DK,)), key, value, decay, erase, write)
    else:
        beta = mx.array(1.0)
        for key, value in ((key_a, value_x), (key_b, value_z1), (key_c, value_z2), (key_a, value_y)):
            _, state = step_delta_projection(state, mx.zeros((DK,)), key, value, beta)
    return {
        "query_a": state @ key_a, "query_b": state @ key_b, "query_c": state @ key_c,
        "value_x": value_x, "value_y": value_y, "value_z1": value_z1, "value_z2": value_z2,
        "state_norm": float(mx.sqrt(mx.sum(state * state))),
    }


def part_a():
    print("=== Part A: deterministic overwrite/interference mechanism test ===")
    print("(cosine similarity + magnitude retained; overwrite->Y should be HIGH, interference should be LOW, magnitude near 1.0)\n")
    print("-- current HZ-0A recurrence, erase-gate sweep, ORTHOGONAL keys (the easy case) --")
    for erase_value in (0.0, 0.3, 0.5, 0.7, 0.9, 0.99):
        report(f"current (erase={erase_value})", run_current(erase_value))
    print("\n-- current + strong erase (boosted logit, ablation) --")
    report("current_strong_erase(boost=4)", run_current_strong_erase(4.0))
    print("\n-- delta-rule projection candidates, ORTHOGONAL keys --")
    report("delta_projection (no decay)", run_delta_projection())
    report("delta_projection_plus_decay(0.95)", run_delta_projection_plus_decay(0.95))

    print("\n-- NEAR-DUPLICATE keys (key_B at cosine~0.85 to key_A) -- the realistic, harder case --")
    for erase_value in (0.5, 0.9, 0.99):
        report(f"current, near-dup (erase={erase_value})", run_near_duplicate("current", erase_value=erase_value))
    report("delta_projection, near-dup", run_near_duplicate("delta"))


def part_b():
    print("\n=== Part B: gradient finiteness + rough step-cost sanity ===")
    print("(NOT a real kernel benchmark -- Python-loop wall-clock only, orders of magnitude off from a fused Metal kernel; systems-gate conclusions need real kernel work, not this number)\n")

    key_a, key_b, key_c, value_x, value_y, value_z1, value_z2 = make_orthogonal_keys_and_values()

    def loss_current(erase_logit_scalar):
        state = mx.zeros((DV, DK))
        decay, write = mx.ones((DK,)), mx.ones((DV,))
        erase = mx.sigmoid(mx.full((DK,), erase_logit_scalar))
        _, state = step_current(state, mx.zeros((DK,)), key_a, value_x, decay, erase, write)
        _, state = step_current(state, mx.zeros((DK,)), key_b, value_z1, decay, erase, write)
        _, state = step_current(state, mx.zeros((DK,)), key_a, value_y, decay, erase, write)
        readout = state @ key_a
        return mx.sum((readout - value_y) ** 2)

    def loss_delta(beta_scalar):
        state = mx.zeros((DV, DK))
        beta = beta_scalar
        _, state = step_delta_projection(state, mx.zeros((DK,)), key_a, value_x, beta)
        _, state = step_delta_projection(state, mx.zeros((DK,)), key_b, value_z1, beta)
        _, state = step_delta_projection(state, mx.zeros((DK,)), key_a, value_y, beta)
        readout = state @ key_a
        return mx.sum((readout - value_y) ** 2)

    grad_current = mx.grad(loss_current)(mx.array(0.0))
    grad_delta = mx.grad(loss_delta)(mx.array(1.0))
    print(f"current recurrence: grad finite = {bool(mx.isfinite(grad_current))}, value = {float(grad_current):.6f}")
    print(f"delta projection:   grad finite = {bool(mx.isfinite(grad_delta))}, value = {float(grad_delta):.6f}")

    n_steps = 2000
    state = mx.zeros((DV, DK))
    decay, erase, write = mx.ones((DK,)), mx.full((DK,), 0.5), mx.ones((DV,))
    mx.eval(state)
    start = time.perf_counter()
    for _ in range(n_steps):
        _, state = step_current(state, key_a, key_a, value_x, decay, erase, write)
        mx.eval(state)
    current_time = time.perf_counter() - start

    state = mx.zeros((DV, DK))
    beta = mx.array(1.0)
    mx.eval(state)
    start = time.perf_counter()
    for _ in range(n_steps):
        _, state = step_delta_projection(state, key_a, key_a, value_x, beta)
        mx.eval(state)
    delta_time = time.perf_counter() - start

    print(f"\n{n_steps} steps, Python-loop wall-clock (rough, unfused, NOT representative of a real kernel):")
    print(f"  current recurrence:  {current_time*1000:.1f} ms  ({current_time/n_steps*1e6:.1f} us/step)")
    print(f"  delta projection:    {delta_time*1000:.1f} ms  ({delta_time/n_steps*1e6:.1f} us/step)")
    print(f"  ratio (delta/current): {delta_time/current_time:.2f}x")


if __name__ == "__main__":
    part_a()
    part_b()
