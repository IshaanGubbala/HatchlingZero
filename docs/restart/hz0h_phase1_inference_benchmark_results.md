# HZ Phase 1: inference-throughput results — the first real positive BDH signal

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
2. **The Transformer baseline has no KV-cache.** A real, comparable,
   production Transformer implementation would NOT replay the full
   sequence every decode step — it would cache K/V and do O(1)
   (well, O(context) attention lookup, but not a full forward) work per
   token too, closing much or all of this gap. This result says "BDH's
   real inference mechanism beats a Transformer WITHOUT a KV-cache," not
   "BDH beats Transformers at inference" — the latter claim needs a KV-
   cached Transformer baseline, real future work, not done here.
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

1. Add a real KV-cache to `reference/hz0a_matched_transformer.py` so
   "Transformer naive" becomes "Transformer, real serving-realistic
   decode" — the single biggest fairness gap left in this comparison.
2. Repeat at a larger, trained-model scale where peak memory differences
   become visible and quality-per-token can be reported alongside speed.
3. Add CUDA energy sampling to this same benchmark on the RTX 3060 side
   for a real joules/token number (Mac has no path to this yet).
4. Longer context lengths (8K, 32K, 128K per `plans/HZ Benchmark
   Plan.md`'s target grid) once a trained checkpoint and a KV-cached
   Transformer baseline both exist — this is where BDH's O(1)-state
   memory advantage should become most visible, if it holds.
