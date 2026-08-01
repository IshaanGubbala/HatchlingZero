"""HZ-0B reopening criterion 5: "Memory state does not degrade
monotonically as write count grows" (plans/HZ-0B_Progress_Tracker.md).

Pure B2 simulator test, no LM/gradient descent needed (same reasoning
as B8 Stage 5 -- capacity/eviction behavior is a property of the
mechanism itself, not something that needs a real model to exercise).

The trivial version of this test (protect some anchor facts, confirm
they survive) would pass by construction -- protection exists
specifically to guarantee that, already verified in B8 Stage 5's
capacity-pressure scenario. The real, non-trivial question this
criterion is actually asking: over a LONG sustained sequence of
UNPROTECTED writes (far more writes than slots), does the memory's
ability to hold onto recently-written facts degrade over time (a real
"wears out" failure mode), or does it reach a stable steady-state?

Test: write NUM_WRITES facts sequentially into an unprotected,
fixed-capacity memory. After each write, check whether the
JUST-WRITTEN fact is exactly retrievable. Track this "immediate
retrievability" rate over a sliding window as write count grows --
if it degrades monotonically toward 0 (memory "wearing out"), that's a
real failure. If it stays flat/stable (each new write correctly claims
a slot regardless of how many writes came before), that's a real pass.
"""
from __future__ import annotations

import mlx.core as mx

from reference.hz0b_memory_simulator import read, reset, write

NUM_SLOTS, KEY_DIM, VALUE_DIM = 8, 16, 16
NUM_WRITES = 200
WINDOW = 20


def _onehot(dim: int, index: int) -> mx.array:
    row = [1.0 if i == index else 0.0 for i in range(dim)]
    return mx.array([row])


def main():
    state = reset(1, NUM_SLOTS, KEY_DIM, VALUE_DIM)
    immediate_hits = []
    for w in range(NUM_WRITES):
        key, value = _onehot(KEY_DIM, w % KEY_DIM), _onehot(VALUE_DIM, w % VALUE_DIM) * 2.0
        state, slot, rejected = write(state, key, value, mx.array([1.0]), step=w)
        readout, _ = read(state, key, hard=True)
        hit = (not bool(rejected[0])) and bool(mx.all(mx.abs(readout - value) < 1e-3))
        immediate_hits.append(hit)

    windowed_rates = []
    for start in range(0, NUM_WRITES, WINDOW):
        chunk = immediate_hits[start:start + WINDOW]
        windowed_rates.append(sum(chunk) / len(chunk))

    print(f"Immediate-retrievability rate (fraction of writes where the JUST-written fact is exactly\n"
          f"retrievable right after writing it), in windows of {WINDOW} writes, over {NUM_WRITES} total unprotected writes into {NUM_SLOTS} slots:\n")
    for i, rate in enumerate(windowed_rates):
        start_write = i * WINDOW
        print(f"  writes {start_write:4d}-{start_write + WINDOW - 1:4d}: {rate:.3f}")

    first_half = windowed_rates[:len(windowed_rates) // 2]
    second_half = windowed_rates[len(windowed_rates) // 2:]
    first_half_mean = sum(first_half) / len(first_half)
    second_half_mean = sum(second_half) / len(second_half)
    print(f"\nFirst-half mean: {first_half_mean:.3f}   Second-half mean: {second_half_mean:.3f}")

    degraded = second_half_mean < first_half_mean - 0.05
    if not degraded:
        print("\nRESULT: PASS. Immediate-retrievability rate does NOT meaningfully degrade over "
              f"{NUM_WRITES} sustained unprotected writes ({NUM_WRITES // NUM_SLOTS}x memory capacity) -- "
              "the mechanism reaches stable steady-state eviction behavior, not a 'wears out over time' "
              "failure mode. Reopening criterion 5 is MET.")
    else:
        print(f"\nRESULT: FAIL. Retrievability dropped from {first_half_mean:.3f} to {second_half_mean:.3f} "
              "over the course of the run -- a real degradation-over-time problem. Criterion 5 NOT met.")


if __name__ == "__main__":
    main()
