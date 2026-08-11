# HZ Phase 2 (exact streaming BDH): the equivalence proof is done — the "Major Risk" is real

Date: 2026-08-11. `plans/HatchlingZero_Reality_Plan.md`'s Phase 2 has two
parts: (1) prove the streaming/chunked form is exactly equivalent to the
parallel form, and (2) check whether the resulting state's absolute
memory footprint is actually reasonable — explicitly warning "Phase 2 is
not complete merely because streaming works."

## Part 1: equivalence — already done, re-confirmed

`reference/hz0h_bdh_torch.py`'s `bdh_stream_chunk`/`bdh_stream_sequence`/
`init_bdh_states` (H2, built earlier this project) already proves
`full-sequence == token-by-token == arbitrary chunked` within float32
precision — `tests/reference/test_hz0h_bdh_h2_streaming.py`, 9 tests
covering lengths 1/16/128/1024, token-by-token streaming, arbitrary
irregular chunk boundaries, 5 different partitions of one sequence all
agreeing, reset, real `torch.save`/`load` resume, and deepcopy-resume.
Structurally independent of the RoPE bug fix (both the parallel and
streaming forms call the same `Attention.rope`/`phases_cos_sin`, so the
equivalence property holds regardless of which RoPE formula is
correct) — still passing in every full-suite run since that fix. This
part of Phase 2 does not need new work.

## Part 2: state size — the real risk, now measured directly

State shape per layer: `(batch, n_head, N, n_embd)` where
`N = n_embd * mlp_internal_dim_multiplier / n_head` — so elements per
layer = `n_head * N * n_embd = mlp_internal_dim_multiplier * n_embd²`,
independent of `n_head`. Total state size scales with
`n_layer * m * D²` — quadratic in model width `D`, exactly the same
`D²` term identified as the cost driver in
`docs/restart/hz0h_phase1_kv_cache_bdh_results.md`'s decode-speed
analysis. This is the same mechanism showing up as a MEMORY problem now,
not just a speed one.

Measured directly (`init_bdh_states`, real tensor byte counts, not
estimated) at this session's three matched pilot scales:

| Scale | D | m | layers | State bytes (fp32, batch=1) | Model weight bytes | State/weights ratio |
| --- | --- | --- | --- | --- | --- | --- |
| ~5M | 256 | 24 | 6 | 37.7 MB | 19.4 MB | **1.95x** |
| ~25M | 512 | 32 | 8 | 268.4 MB | 101.7 MB | **2.64x** |
| ~71M | 768 | 40 | 10 | 943.7 MB | 284.7 MB | **3.31x** |

**The state is already bigger than the model's own weights at every
scale tested, and the ratio gets WORSE as scale grows.** This is the
opposite of what a "RAM-efficient alternative to a growing KV-cache"
should look like at small-to-moderate context.

## At realistic serving batch sizes, this is severe

| Scale | batch=1 | batch=8 | batch=32 |
| --- | --- | --- | --- |
| ~5M | 37.7 MB | 302.0 MB | 1.21 GB |
| ~25M | 268.4 MB | 2.15 GB | 8.59 GB |
| ~71M | 943.7 MB | 7.55 GB | **30.20 GB** |

State size scales linearly with batch size (each batch item needs its
own independent state) — at batch=32, the 71M-param model's state ALONE
needs 30GB, dwarfing anything a same-size Transformer's KV-cache would
need at any realistic context length.

## When does BDH's fixed state actually win on memory?

Crossover context length (state_bytes == KV-cache_bytes at matched D):
solving `n_layer * m * D² = n_layer * 2 * context * D` gives
`context = m * D / 2`:

| Scale | Crossover context length |
| --- | --- |
| ~5M | 3,072 tokens |
| ~25M | 8,192 tokens |
| ~71M | 15,360 tokens |

**Below these context lengths, BDH's "O(1) state" is actually using MORE
memory than a real KV-cache would, not less.** This crossover point
itself grows with model width (`m*D/2`, roughly linear in D at fixed
`m`) — meaning the memory advantage requires proportionally LONGER
context as the model scales up, the same unfavorable direction found for
decode speed in the prior document. Every context length tested in this
session's inference benchmarks (128–2048) is well below every one of
these crossover points — meaning BDH's state was a real memory cost, not
a memory saving, in every measurement taken so far.

## This is `plans/HatchlingZero_Reality_Plan.md`'s own "Risk 1," materializing early

> Risk 1 — BDH State Is Too Large. This could kill the RAM advantage.

The plan's own prescribed backup order: block-sparse state → state
quantization → low-rank consolidation → smaller latent multiplier →
state sharing across repeated depth → hybrid BDH/GDN-style state if
necessary (Phase 3 in the plan, "Synaptic State Compression"). This
result is direct, real, quantified evidence that Phase 3's work is not
optional polish — the raw state size found here would need something
like an 8-16x reduction just to stop being LARGER than the model's own
weights at the scales tested, before it can even begin competing with a
real KV-cache at practical context lengths.

## Honest framing

This does not mean BDH's state mechanism is useless — the O(1)-per-token
decode speed advantage at small scale
(`docs/restart/hz0h_phase1_kv_cache_bdh_results.md`) is real and
measured. It means the "fixed-size state instead of a growing KV-cache"
framing needs a caveat this session hadn't quantified until now: fixed
does not mean small, and at the model widths and context lengths tested,
BDH's state costs MORE memory than the thing it's meant to replace. Per
`plans/HatchlingZero_Reality_Plan.md`'s own Risk-1 framing: "If none
preserves quality, the >30% RAM target may be false for pure BDH. Report
that honestly" — reporting it honestly now, before Phase 3 compression
work starts, so that work has a real, quantified baseline to improve
against rather than an assumed-fine starting point.

## Real next steps (Phase 3, not started)

1. Sparse-row / block-sparse state (plan §6.1/6.2): only allocate/update
   state rows for active neuronal regions — needs BDH's real activation
   sparsity numbers (`docs/restart/hz0h_phase1_...` activation-sparsity
   work, already measured at ~13-53% depending on layer/path) as the
   starting budget for how much reduction is even plausible.
2. State quantization (plan §6.4): INT8/FP8 state would roughly
   quarter/halve these numbers directly — cheapest first experiment.
3. Recompute the crossover-context table after any compression method,
   to see whether it moves into a practically relevant range (hundreds
   to low thousands of tokens, not tens of thousands).
