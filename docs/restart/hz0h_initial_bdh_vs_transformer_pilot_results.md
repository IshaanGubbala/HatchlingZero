# HZ-0H initial BDH-vs-Transformer pilot: real results

## Update 2026-08-11: Transformer rerun with RoPE + matched weight_decay

Per this doc's own "real next steps" #1-2 below: added an opt-in `use_rope`
flag to `reference/hz0a_matched_transformer.py` (standard Llama/GPT-NeoX/
Qwen-style rotate-half RoPE, `_rope_cos_sin`/`_apply_rope`, applied to q/k
before `scaled_dot_product_attention`) -- **off by default**, since
`MatchedTransformerConfig` backs many established HZ-0A/HZ-0H results
(T1/T2 ternary comparisons, HZ-0A hybrid-vs-transformer controls) that
were run and reported without any positional encoding at all; adding it
unconditionally would have silently changed what those historical results
measured. Wired a `--rope` and `--weight-decay` CLI flag into
`scripts/hz0a_torch_stage2_runner.py` (weight_decay was previously
hardcoded to 0.01, not overridable). Full suite re-run after the change:
559 passed, 100 skipped, 1 pre-existing unrelated failure
(`test_hz0a_data_pipeline.py::test_token_packing_is_reproducible_for_a_seed`,
a stale manifest path pointing at a plan doc moved during an unrelated
concurrent reorg of `plans/` -- confirmed via `git stash` to predate and
be unrelated to this change).

Reran the Transformer side locally (Mac MPS, float32, same matched config,
same 25M-token budget, `--rope --weight-decay 0.1` to also match BDH's
real weight_decay, which the first run had NOT matched):

| | Transformer, no RoPE (original) | Transformer, +RoPE +wd=0.1 (this rerun) | BDH (RTX 3060, bf16, wd=0.1) |
| --- | --- | --- | --- |
| Best validation loss | 2.269 | **1.288** | 1.623 |
| tokens/sec (Mac MPS) | 107,744 | 95,213 | n/a (different machine) |
| Training wall-clock | 232.0s | 262.6s | 940.1s |

Adding real position information changed the result: the Transformer now
**beats** BDH on validation loss at matched params/tokens/optimizer
(1.288 vs 1.623), reversing the original (confounded) 2.269-vs-1.623
result reported below. RoPE cost ~12% throughput (95,213 vs 107,744
tok/s) on this Mac -- a real, modest compute cost, not the dominant
factor in the loss change.

**Still not a fully clean comparison** -- caveats that still apply: BDH's
own run was not rerun (still the original bf16/RTX3060/940s numbers,
weight_decay does now match at 0.1 but the run predates that match on the
BDH side incidentally already using 0.1 as its own default -- see
setup below); dtype+hardware still differ across the two architectures'
numbers (Transformer float32/MPS, BDH bfloat16/CUDA), so tok/s still is
not cross-machine comparable; BDH's own RoPE is a different, real,
upstream-verified formula (quantized frequency bins, cycles-to-radians
wrap) than the standard rotate-half RoPE added here for the Transformer
-- both now HAVE real position information, but via different real
formulas, which is itself worth being explicit about rather than treating
"both have RoPE" as "both have the same RoPE."

---

## Original pilot writeup (2026-08-10/11, kept for the record)

Date: 2026-08-10/11. First real run under `plans/HZ Benchmark Plan.md`'s
methodology (iso-parameter, matched everything possible at this scale).
Explicitly a small pilot to validate the pipeline (matched configs, real
recipes, multi-machine dispatch), NOT a claim about architecture
superiority at any meaningful scale -- see that plan's "Claim discipline"
and "Hardware reality" sections.

## Setup

Byte-level (vocab=256) sequences packed from `data/external_corpus/code.jsonl`
via `scripts/hz0h_pack_byte_corpus.py` (91MB source, 85.6M train bytes /
1.75M val bytes at sequence_length=256). Matched configs (both ~4.8M
params, within 0.9%):

- **Transformer** (`reference/hz0a_matched_transformer.py`): dim=256,
  layers=6, heads=4, d_ff=683 -- 4,804,868 params.
- **BDH** (`reference/hz0h_bdh_torch.py`): n_embd=256, n_layer=6, n_head=4,
  mlp_internal_dim_multiplier=24 -- 4,849,664 params.

Both: AdamW, max_lr=1e-3, weight_decay=0.1 (BDH) / 0.01 (Transformer's own
runner default, not overridden -- real, disclosed asymmetry, not matched),
cosine schedule with 100 warmup steps, batch_size=32, target_tokens=25,000,000,
seed=7.

**Dispatch**: benchmarked both on the Mac (MPS, float32) at this config
first -- Transformer 100,928 tok/s, BDH 16,251 tok/s (~6.2x gap). Ran
Transformer locally (fast) and dispatched BDH to the Windows RTX 3060 via
the LAN relay, in parallel, for best combined wall-clock.

## Results

| | Transformer (Mac, MPS, float32) | BDH (RTX 3060, CUDA, bfloat16) |
| --- | --- | --- |
| Parameters | 4,804,868 | 4,849,664 |
| tokens/sec | 107,744 | 26,596 |
| Training wall-clock | 232.0s (~3.9 min) | 940.1s (~15.7 min) |
| Best validation loss | 2.269 | 1.623 |
| Milestones hit | 5/5 | 5/5 |

## Real, honest caveats -- read before drawing any conclusion from this table

1. **tokens/sec is NOT cross-machine comparable.** Transformer ran float32
   on Mac MPS; BDH ran bfloat16 on an RTX 3060. Different hardware AND
   different precision, confounded together. This table's tokens/sec
   column says "Transformer is faster than BDH on the specific
   hardware/dtype each happened to run under," not "Transformer is
   faster than BDH" as an architecture claim. A same-machine, same-dtype
   comparison (both float32 or both bfloat16, both on the same GPU) is
   needed before that claim is even attemptable -- not done here.
2. **The Transformer baseline has no positional encoding at all** (see
   `plans/HZ Benchmark Plan.md`'s "real, current gap" section) -- no RoPE,
   no learned position embedding, nothing beyond the causal mask. BDH has
   real, verified RoPE. The validation-loss gap (2.269 vs 1.623) is
   confounded by this: BDH may simply be the only one of the two that can
   use token position at all on this task. **This makes the loss
   comparison in this table not a fair architecture comparison as-is** --
   fixing the Transformer's missing RoPE is a real prerequisite (already
   flagged as the top action item in `plans/HZ Benchmark Plan.md`) before
   any loss number here is used to argue anything about BDH vs
   Transformer quality.
3. **weight_decay differs** (0.1 BDH vs 0.01 Transformer, the latter's own
   runner default, not matched) -- a real, unintentional mismatch, not
   large enough to explain the loss gap alone but not clean either.
4. **Batch size 32 was not re-swept for either machine.** The RTX 3060 run
   used it "as literally specified" per the dispatch request, without
   checking whether a larger batch would improve its own throughput; the
   Mac run likewise used the same fixed value from the earlier smoke
   benchmark. Neither number should be read as each machine's real ceiling
   throughput.
5. **A real bug was found and fixed during this run** (RTX 3060 side, then
   ported back into `scripts/hz0h_stage2_runner_bdh.py`): casting the
   whole BDH model to bfloat16 via `.to(device=..., dtype=torch_dtype)`
   also casts `attn.freqs` (the RoPE frequency buffer), which crashes
   `Attention.forward`'s own `assert self.freqs.dtype == torch.float32`.
   Fixed by restoring `model.attn.freqs` to float32 immediately after the
   cast. Only affects float16/bfloat16 training (the Mac's float32 run
   never hit it). Confirms the RoPE-precision sensitivity from
   `docs/restart/hz0h_rope_bug_critical_correction.md` extends to dtype
   handling, not just the formula itself -- another real, disclosed
   consequence of that same subsystem needing care.
6. **Peak VRAM was not captured** for the CUDA run (this runner doesn't
   log `peak_memory_bytes` the way `scripts/hz0a_torch_stage2_runner.py`
   does) -- a real, disclosed gap versus the fuller metric set
   `plans/HZ Benchmark Plan.md` calls for at target scale.

## What this pilot actually establishes

- The dispatch pipeline works end to end: matched configs computed, real
  recipes used (no `targets=idx` bug, no missing RoPE conversion), data
  packed once and shared identically across machines, both runs completed
  their full 25M-token budget, results reported back over the LAN relay.
- A real, reproducible bug (bfloat16-cast RoPE buffer) was found and fixed
  because this pilot actually ran on real hardware with real dtype
  settings, not caught by any existing test (none of this project's
  existing BDH tests train under float16/bfloat16).
- The specific numbers in the results table are NOT yet a fair
  architecture comparison -- caveats 1-2 above are real prerequisites, not
  nitpicks, before this pilot's loss/throughput numbers support any
  BDH-vs-Transformer claim.

## Real next steps (not done here)

1. Add RoPE to `reference/hz0a_matched_transformer.py` (top action item,
   `plans/HZ Benchmark Plan.md`).
2. Match weight_decay between the two runs.
3. Re-run both architectures on the SAME machine and SAME dtype (at least
   one same-hardware pair) before citing a throughput number as an
   architecture property rather than a hardware/dtype artifact.
4. Capture peak VRAM on both sides.
5. Only after 1-4: treat this pilot's methodology as validated for the
   real target-scale study in `plans/HZ Benchmark Plan.md`.
