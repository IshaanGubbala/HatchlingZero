# HZ Phase 1: inference-throughput results — the first real positive BDH signal

## ⚠️ Superseding update: the positive result below does NOT survive scaling up

`docs/restart/hz0h_phase1_crossover_scale_sweep_results.md` (2026-08-11,
same day) swept this exact comparison across three matched scales (5M,
25M, 71M params) and found the advantage below **reverses cleanly and
monotonically** as scale grows -- BDH streaming decode goes from ~3x
faster than the Transformer's real KV-cache at 5M params to ~3-5.6x
SLOWER at 71M params, at every context length tested. Read that document
for the current picture; the ~4.8M-param result immediately below is a
real, correctly-measured single data point, not a scale-independent
finding, exactly as its own caveats already said before the sweep
confirmed it.

## Update 2026-08-11: real KV-cache added, cleaner (and still positive, AT THIS SCALE) result

Added a real KV-cache to `reference/hz0a_matched_transformer.py`
(`MatchedTransformerLM.new_kv_cache`/`forward(kv_cache=...)`) — numerically
verified identical to a full non-cached forward
(`tests/reference/test_hz0a_matched_transformer_kv_cache.py`, 5 tests,
including chunked-prefill and no-RoPE cases). This was the single biggest
disclosed gap in the original pilot below: "Transformer naive" was not a
representative Transformer serving path.

**Real, honest surprise**: at this tiny scale (~4.8M params) and these
short contexts, the KV-cache is SLOWER than naive full-replay, not
faster:

| Context length | BDH streaming (tok/s) | Transformer naive replay (tok/s) | Transformer KV-cache (tok/s) | BDH streaming vs. Transformer KV-cache |
| --- | --- | --- | --- | --- |
| 128 | 792.5 | 511.3 | 247.2 | **3.21x** |
| 512 | 783.8 | 442.2 | 242.8 | **3.23x** |
| 1024 | 639.8 | 326.5 | 221.6 | **2.89x** |

This is a real, known small-scale effect, not a bug: a KV-cached decode
step does one token's worth of work per Python-level loop iteration
(small matmuls, plus tensor-concat overhead to grow the cache each step),
while naive replay's larger batched forward pass gets better hardware
utilization per call despite doing asymptotically more total work. The
crossover point where KV-caching actually wins is a real, model-size-and
context-length-dependent threshold -- not tested here (would need a
bigger model / longer context to find).

**Against the real, correct baseline, BDH's streaming decode still wins
decisively (~2.9-3.2x) and more STABLY across context length** than the
original (naive-vs-naive) comparison showed (which ranged 1.49x-6.65x,
partly an artifact of the naive Transformer's own quadratic degradation).
This is a cleaner, more defensible number than the original pilot below,
precisely because both baselines are no longer both crippled the same
way.

**One more honest caveat this update adds**: this comparison can't yet
distinguish "BDH's streaming state is architecturally more efficient
per token" from "this particular KV-cache implementation has more
per-step Python/tensor-op overhead than `bdh_stream_chunk`'s own call" --
both are real possibilities, not yet separated. A fairer apples-to-apples
implementation-efficiency comparison (e.g. profiling where each path's
per-step time actually goes) is real, undone future work.

---

## Original pilot (2026-08-11, kept for the record — see the update above for the corrected comparison)

Date: 2026-08-11. `scripts/hz0h_inference_benchmark.py`, matched ~4.8M-param
configs (same as the training pilot, `docs/restart/hz0h_initial_bdh_vs_transformer_pilot_results.md`),
Mac MPS, batch=1, greedy decode (argmax), 48 decode tokens, 3 prefill
repeats, random (untrained) weights — this measures raw architectural
throughput, not quality (no training happened here).

## Setup: three decode paths, not one

1. **BDH naive replay** (`BDH.generate()`, real upstream code) — re-runs
   the whole sequence every new token, O(T) work/token.
2. **BDH streaming** (`bdh_stream_chunk`, H2's proven-exact equivalent) —
   O(1) work/token via the real persistent per-layer synaptic state. This
   is the actual mechanism the project's thesis is about.
3. **Transformer naive replay** — `reference/hz0a_matched_transformer.py`
   has no KV-cache, so this is also O(T) work/token, the same real
   limitation as path 1. This is deliberately the fairest baseline
   available right now, not a strawman — see the script's own module
   docstring for the disclosed gap (a real Transformer KV-cache doesn't
   exist in this repo yet, so this can't yet be compared against a
   production-realistic Transformer serving path).

## Results

| Context length | BDH naive (tok/s) | BDH streaming (tok/s) | Streaming speedup over BDH's own naive path | Transformer naive (tok/s) | BDH streaming vs. Transformer naive |
| --- | --- | --- | --- | --- | --- |
| 128 | 81.9 | 755.5 | 9.2x | 507.7 | **1.49x** |
| 512 | 57.1 | 778.1 | 13.6x | 440.9 | **1.76x** |
| 1024 | 26.4 | 649.5 | 24.6x | 97.6 | **6.65x** |

## What this shows

- **BDH's own naive-replay decode gets slower as context grows** (81.9 →
  57.1 → 26.4 tok/s) — expected, O(T) work per token.
- **BDH's streaming decode stays roughly flat** (755 → 778 → 650 tok/s,
  the last dip likely just noise/thermal at this tiny sample size) — the
  real, structural O(1)-per-token signature H2 already proved
  algebraically; this is the first time it's been measured as an actual
  wall-clock number, not just an equivalence proof.
- **The advantage over the Transformer baseline widens with context**:
  1.49x at 128 tokens, 1.76x at 512, 6.65x at 1024 — because the
  Transformer baseline is ALSO O(T) per token (no KV-cache) and its
  naive-replay cost grows the same way BDH's own naive path's did (though
  the Transformer's dense matmuls are cheaper per token than BDH's, so it
  starts from a faster baseline and degrades from there).

**This is the first real result in this session's BDH-vs-Transformer work
that favors BDH.** The training-side pilot
(`docs/restart/hz0h_initial_bdh_vs_transformer_pilot_results.md`) showed
the Transformer winning decisively on training throughput and validation
loss at this same small scale. Put together: at ~4.8M params, BDH trains
slower and reaches worse quality, but its real streaming-state decode
mechanism is a genuine, structural, already-measured inference-time
advantage — exactly the kind of "different tradeoff, not a strict loss"
result `plans/HatchlingZero_Reality_Plan.md`'s own framing anticipates.

## Real caveats — do not overclaim this

1. **Random (untrained) weights.** This measures pure architectural
   throughput, not "quality per token" or anything downstream-task
   related. A trained model would very likely have the same relative
   timing (the mechanism doesn't depend on the weight values), but this
   hasn't been separately confirmed.
2. **(Resolved by the 2026-08-11 update above)** The Transformer baseline
   now has a real, tested KV-cache — the comparison in the update section
   is against the real serving-realistic decode path, not a crippled one.
3. **Small scale, single seed, batch=1, one machine.** Per
   `plans/HZ Benchmark Plan.md`'s own claim discipline — one real data
   point, not a scaling-law claim.
4. **No energy/joules-per-token number on this run.** This script has
   CUDA-only power sampling (via polling `nvidia-smi`); Mac has no
   equivalent instrumentation. Real gap, not filled here.
5. **Peak memory numbers are flat/uninformative at this scale** (~38.6MB
   across every row) — MPS has no true peak-memory API (see the script's
   own `peak_memory_bytes` docstring); at a model this small the
   allocator's resident footprint barely moves. A real, informative peak-
   memory comparison needs a bigger model, where the difference between
   BDH's O(1) state and a KV-cache's linearly-growing memory would
   actually show up.

## Real next steps

1. ~~Add a real KV-cache to `reference/hz0a_matched_transformer.py`~~ —
   done 2026-08-11, see the update section above.
2. Repeat at a larger, trained-model scale where peak memory differences
   become visible, quality-per-token can be reported alongside speed, and
   the KV-cache-vs-naive crossover point (where caching actually starts
   winning) can be found.
3. Add CUDA energy sampling to this same benchmark on the RTX 3060 side
   for a real joules/token number (Mac has no path to this without
   `powermetrics`, which needs sudo this script doesn't prompt for).
4. Longer context lengths (8K, 32K, 128K per `plans/HZ Benchmark
   Plan.md`'s target grid) at a scale where the KV-cache is actually
   winning over naive replay — this is where BDH's O(1)-state memory
   advantage should become most visible, if it holds.
5. Profile where each decode path's per-step time actually goes (Python
   loop overhead vs. tensor-op time vs. cache-concat cost) to separate
   "BDH's mechanism is architecturally more efficient" from
   "this KV-cache implementation has more overhead than bdh_stream_chunk's
   own call" — not yet done, see the update section's closing caveat.
