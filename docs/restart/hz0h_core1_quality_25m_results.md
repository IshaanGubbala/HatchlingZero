# HZ-Core-1 quality at 25M params: real, significant negative result -- VB does NOT preserve quality at real-data scale

Date: 2026-08-12. Real validation-loss results for the three trained
arms of HZ-Core-1's 4-way ablation (`plans/HZ Integrated Candidate Plan.md`
Step 6), all at the matched 25M-param config on the real byte-level
corpus (`data/packed/hz0h_bytes_25m_train.jsonl`, 25M-token budget,
`n_embd=512, n_layer=8, n_head=8, mlp_internal_dim_multiplier=32`).
Exact BDH and BDH+VB trained on the Windows RTX 3060 (CUDA, bfloat16,
batch=12 -- see note below); matched Transformer trained locally on the
Mac (MPS, float32, batch=32).

## Real results

| Arm | Params | best_validation_loss | tok/s | Device/dtype |
| --- | --- | --- | --- | --- |
| Exact BDH | 25,427,968 | **1.6484** | 6,151.6 | CUDA/bf16 |
| BDH + Value Bottleneck | 25,559,040 | **1.7988** | 6,201.9 | CUDA/bf16 |
| Matched Transformer | 25,343,824 | **1.2829** | 21,098.6 | MPS/fp32 |

Relative cross-entropy increase:

```text
VB vs exact BDH:           +9.12%  (real, meaningful, NOT noise -- consistent
                                     from best-checkpoint to final-checkpoint
                                     on both runs)
exact BDH vs Transformer:  +28.49%
VB vs Transformer:         +40.21%
```

## Real, honest headline: this is a negative result for the value bottleneck, not a positive one, at this scale

**The value bottleneck was the ONE mechanism that survived every stress
test so far this session** -- 0% degradation up to 16x combined
reduction on H5's passkey and reassignment synthetic tasks
(`docs/restart/hz0h_phase2r_combined_vb_int8_results.md`). This is the
first time it has been tested against REAL language-modeling data at
real scale rather than tiny synthetic in-context tasks, and the result
is a real, meaningful quality regression: **+9.12% relative validation
CE, far outside the plan's own "<=3-5% degradation" promotion target.**

This is not attributed to noise or an implementation bug: both runs hit
all 5 milestones, showed no NaN/Inf, decreased monotonically-with-noise
throughout, matched their expected parameter counts and `d_state=128`
exactly, and the gap is consistent from the best checkpoint to the
final checkpoint on both runs (not a single unlucky evaluation point).
Real, reproducible, and matches the standing lesson of this whole
session (headline results on tiny synthetic tasks don't reliably
predict real-scale behavior) -- except this time the check that caught
it was real text data, not merely a harder synthetic task.

## Real, honest caveats -- read before drawing further conclusions

1. **Dtype/device confound, disclosed by the training side itself**:
   the two BDH arms ran bfloat16 on CUDA; the Transformer ran float32
   on MPS. The 28-40% gap between BDH-family and Transformer is large
   enough that dtype alone is an unlikely full explanation, but it is
   not a fully controlled comparison -- a real open question, not
   resolved here. The exact-BDH-vs-VB comparison (9.12%) is NOT
   affected by this confound (both ran identically: same machine, same
   dtype, same batch size, same recipe) -- that specific number is the
   most trustworthy one in this table.
2. **BDH's own training recipe/LR schedule may simply be less mature**
   than the Transformer's at this scale -- this project's BDH work has
   focused on architecture-level questions (state compression,
   sparsity, depth) far more than on tuning the plain training recipe
   itself at real scale. The Transformer-vs-BDH gap (28.5%) should not
   be read as a settled architecture-superiority claim without ruling
   this out first.
3. Only ONE seed per arm -- no seed-variance check yet, unlike several
   other results this session (BlockBDH, soft-grouped state) that
   turned out to hinge heavily on seed. Given how much this result
   changes the picture, a repeat at a second seed is a real, warranted
   next step before treating +9.12% as fully settled.
4. Passkey/reassignment/interference quality checks on these REAL
   trained checkpoints have not been run yet (checkpoints are currently
   on the RTX 3060 side; the harness to load and evaluate a real
   checkpoint at this scale doesn't exist yet either -- both are the
   next real step). The synthetic-task numbers this session already has
   are all from small, freshly-initialized models trained directly for
   those tasks, not from this real 25M-scale checkpoint.
5. Real batch-size finding from the training side, worth keeping: at
   this larger `n_embd=512` scale, `--batch-size 32` caused a real WDDM
   shared-memory-paging stall on the RTX 3060 (VRAM pinned near the
   ~12GB ceiling, near-zero real throughput despite 100% reported GPU
   utilization) -- fixed by dropping to `--batch-size 12`. Confirmed as
   a real hardware/driver interaction, not a training bug, matching a
   previously documented pattern for this card
   (`docs/rtx3060_windows_setup.md`). Both CUDA arms used batch=12 for
   a fair within-machine comparison; the Mac's Transformer run used
   batch=32 (no such issue on MPS at this scale) -- another real,
   disclosed asymmetry in the comparison, on top of the dtype one.
6. joules/token measured (nvidia-smi power sampling, not torch-level
   instrumentation): 0.02638 J/token (exact BDH) vs 0.02634 J/token
   (BDH+VB) -- essentially identical, no measurable efficiency
   difference at this resolution. VB is not winning on power either at
   this scale, at least not by a margin this measurement can resolve.

## What this means for HZ-Core-1's promotion gate

Per `plans/HZ Integrated Candidate Plan.md`'s revised gate ("not all
five required, but more than one real, simultaneous advantage is"):

```text
State RAM:        MET (15.98x reduction, see hz0h_core1_efficiency_25m_results.md)
Total inference RAM: not yet measured at this scale
Decode:            partially favorable (VB fp32 1.37-1.44x faster than exact
                   BDH; INT8 arm has a real long-context regression, see
                   hz0h_core1_efficiency_25m_results.md) -- not yet compared
                   directly against the matched Transformer in one harness
Quality:           NOT MET -- real +9.12% relative CE degradation vs exact
                   BDH, far outside the <=3-5% target
Stateful tasks:    not yet measured on real trained checkpoints
```

State RAM is a real, clean win. Quality is a real, clean miss, on the
one metric measured cleanly (VB vs exact BDH, no confound). This does
NOT automatically kill HZ-Core-1 -- the gate only requires more than one
advantage, not all of them, and stateful-task competitiveness (the
original motivating use case for a compressed state) hasn't been
checked yet on real checkpoints -- but the quality picture is real,
disclosed, and worse than every synthetic-task result this session
produced. Not spun as a win; reported as found.
