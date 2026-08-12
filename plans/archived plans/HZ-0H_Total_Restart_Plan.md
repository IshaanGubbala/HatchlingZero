# HZ-0H Total Restart Plan

Date: 2026-08-11. Written after discovering that `reference/hz0h_bdh_torch.py`
and `reference/hz0h_bdh_mlx.py` had two real, confirmed bugs since H1 (missing
RoPE cycles->radians conversion, missing embed-init-scale override -- see
`docs/restart/hz0h_rope_bug_critical_correction.md`), and that 8 of 9 H3-T
scripts additionally had the same-sequence-target bug H5 already found and
fixed once for this exact `BDH` class. Per explicit user direction: stop
patching individual results one at a time, and instead work back up from a
foundation that is directly, provably correct against the real upstream
source -- the same way `reference/hz0h_bdh_torch.py` itself was rewritten as
a verbatim transcription rather than a further-patched hand-port.

This plan does not replace `plans/HZ-0H_BDH_Reconciliation_Plan.md` -- that
document's objective, constraints, and phase definitions (H0-H8, T0-T4) still
apply. This plan only governs the ORDER and METHOD of re-deriving every H2+
result against the two now-verified oracle files.

## Starting point: what is actually trusted right now

Only these two files are trusted as correct against upstream, and only
because each was checked by a fresh, complete, verbatim WebFetch of the real
source and diffed line-by-line (not from memory, not from a prior summary):

- `reference/hz0h_bdh_torch.py` -- verified 2026-08-11 against a fresh fetch
  of `github.com/pathwaycom/bdh/bdh.py`. Diff against the real file shows
  exactly two deltas, both intentional and marked: `BDHConfig.ternary`
  (appended field) and `self._w(...)` wrapping the three shared-weight reads
  (the ternary quantization hook). Everything else -- `get_freqs`,
  `Attention` (including the RoPE fix), `BDH.__init__` (real parameter order,
  `_init_weights`/`self.apply`), `forward`, `generate` -- is byte-identical.
- `reference/hz0h_bdh_train_torch.py` -- NEW 2026-08-11, verbatim port of
  `github.com/pathwaycom/bdh/train.py`. Confirms the real recipe: plain
  `AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)` over EVERY
  parameter (no separate treatment of `encoder`/`encoder_v`/`decoder`, no LR
  schedule), real shifted-target convention (`x = data[i:i+T]`,
  `y = data[i+1:i+1+T]`), byte-level tinyshakespeare data, `torch.compile`,
  CUDA-only autocast/GradScaler (no-ops on Mac MPS/CPU, which is what this
  project actually runs on). Extension functions `shifted_target_batch`,
  `build_optimizer`, `train_step` factor the real per-step logic out for
  reuse on synthetic data, so every rebuilt script gets the real convention
  by construction.
- `reference/hz0h_bdh_mlx.py` -- RoPE fix verified 2026-08-10/11 (embed init
  was already correct in this port, no second bug there). Not yet re-checked
  against MLX-specific real-upstream details beyond the RoPE formula itself
  -- lower priority, this project's HZ-0H work runs primarily on the Torch
  oracle.

Everything else -- every H2/H5/H6/T0-T2/H3-T *result* (not necessarily the
*code*, several files are convention-agnostic and stay correct once called
correctly) -- is UNVERIFIED against the corrected model until re-run. Treat
prior numbers as historical record of what was found on the broken model,
not as current evidence.

## Rule for every rebuilt or re-run script

Any script that trains or evaluates a loss on `BDH` MUST get its `(x, y)`
pair from `reference.hz0h_bdh_train_torch.shifted_target_batch` (or an
equivalent explicit shift with a comment justifying it), and MUST use
`build_optimizer`/`train_step` unless there's a specific, stated reason to
deviate (e.g. a training-law arm that substitutes a different gradient
before the optimizer step). No script may call `model(idx, targets=idx)`.
This is now enforced by construction for anything built on top of
`hz0h_bdh_train_torch.py`, not just a convention to remember.

## Re-derivation order

Cheapest / most structurally-likely-to-survive first, so an early collapse
is found before sinking time into later phases that depend on it.

### Step 1 -- H2 (streaming/parallel equivalence)

**Why first:** H2's proof is algebraic (no-softmax attention decomposes
exactly into a running outer-product state) and doesn't depend on WHAT the
RoPE formula is, only that both the parallel and streaming paths use the
SAME one -- which they provably do (`bdh_stream_chunk` calls
`model.attn.rope`/`phases_cos_sin` directly, no separate implementation).
Real risk is low, but "likely still valid" was explicitly flagged as NOT
independently re-verified in the correction doc -- confirm, don't assume.

**Action:** re-run `tests/reference/test_hz0h_bdh_h2_streaming.py` and
`tests/reference/test_hz0h_bdh_streaming.py` as-is (they already exist,
already correct-by-construction, and already ran clean in every full-suite
pass since the RoPE fix) -- this step is really just formally re-confirming
via the full-suite results already in hand, plus explicitly re-stating the
max-abs-diff numbers in the tracker as post-fix numbers, not re-citing the
pre-fix ones.

**Exit gate:** streaming and parallel forms still agree at the same
precision characterized before (Torch ~1.5e-7 float32, MLX ~0.07-0.09%
relative float32, exact at float64).

### Step 2 -- H5 (synaptic memory: passkey, reassignment)

**Why second:** `reference/hz0h_bdh_h5_memory_tasks.py` already uses the
correct shifted-target convention (confirmed this session) and its own
trainer already matches the real optimizer family (AdamW). Only the
underlying model changed (RoPE + embed-init fix) -- re-running is a matter
of re-executing existing scripts, not rewriting them.

**Action:** re-run the passkey and reassignment/overwrite evaluations from
`docs/restart/hz0h_h5_state_ablation_results.md` against the corrected
model. Real risk: the RoPE fix changes how position information propagates,
which passkey retrieval directly depends on -- do not assume the 1.00/0.109
accuracy split survives unchanged.

**Exit gate:** real-state vs. zeroed-state accuracy gap re-measured and
reported, whichever direction it goes.

### Step 3 -- H6 (effective graph structure)

**Why third:** depends on a trained model (H6 trains 3 seeds to compare
modularity), so depends on Step 2's re-confirmation that training still
produces a model with real learned structure elsewhere (H5's passkey task)
before trusting a graph-structure measurement on it.

**Action:** re-run the trained-vs-untrained-vs-shuffled modularity
comparison from `docs/restart/hz0h_h6_graph_structure_results.md`.

**Exit gate:** modularity comparison re-measured; H6's original finding was
already a disclosed negative result (no structure beyond chance at tiny
scale) -- re-confirm whether that still holds or whether the RoPE fix
changes it either direction.

### Step 4 -- T0-T2 (ternary)

**Why fourth:** ternary quantization is applied via `model._w(...)`, which
now runs on the corrected verbatim model -- the STE mechanism itself
(`_ternary_ste`) is untouched and architecture-independent, so T0's contract
doc doesn't need rework, but T1/T2's actual numbers (convergence gap,
throughput, memory) need re-measurement on the corrected base model per the
ternary guardrail (never treat ternary results as evidence independent of a
known-correct full-precision control).

**Action:** re-run T1 sandbox and T2 FP-vs-ternary paired comparisons for
BDH-GPU (`tests/reference/test_hz0h_bdh_ternary.py`,
`docs/restart/hz0h_t2_bdh_fp_vs_ternary.md`).

**Exit gate:** T1 stability bar and T2 convergence/throughput/memory
comparison re-measured on the corrected model.

### Step 5 -- H3-T (training-law search), rebuilt clean

**Why last:** the most re-work, since 8 of 9 scripts had the
same-sequence-target bug baked in AND ran on the pre-RoPE-fix model --
double-invalid, not just single-invalid like H2/H5/H6/T0-T2. Archived to
`archive2/scripts/` and `archive2/tests/reference/` (8 scripts + 5 test
files + the old `plans/HZ-0H_H3T_Training_Law_Search.md`), not deleted --
still readable for the original reasoning/design even though the numbers
don't hold.

**Kept, not archived** (already correct, or convention-agnostic and reused
correctly by what stays):
- `scripts/hz0h_h3t_stage1_redo_real_task.py` -- already uses the real
  shifted-target convention and H5's real passkey task; ran on the
  corrected model gives cos(raw_hebbian)=0.0149, cos(local_signal)=-0.6443
  (the sign-flip finding, see the correction doc) -- **this is the one real
  H3-T number that has already been checked against the fully-corrected
  model.** Treat as the current best evidence for Stage 1, not as
  historical.
- `scripts/hz0h_h3t_eligibility_gate.py` -- `compute_eligibility_trace`
  (pure forward-pass Hebbian statistic, no targets involved, was never
  buggy) and `cosine` are real, reusable, kept. `compute_true_gradient` DID
  have the same-sequence-target bug baked in -- fixed 2026-08-11 to take an
  explicit `targets` argument instead of assuming `idx`; its own `main()`
  demo updated to use `shifted_target_batch`.
- `scripts/hz0h_h3t_eligibility_gate_v2.py` -- `compute_local_signal_pseudo_
  gradient`/`compute_true_gradient` here already took `idx`/`targets` as
  separate parameters (convention-agnostic); only its own `main()` demo used
  `targets = idx` -- the reusable functions were never the bug, callers
  matter (stage1_redo already calls them correctly).
- `scripts/hz0h_h3t_sg_global.py`, `scripts/hz0h_h3t_arm_b_efficiency.py`,
  `scripts/hz0h_h3t_arm_b_all_shared_params_efficiency.py` -- audited
  2026-08-11, none call `model(..., targets=...)` at all (they measure
  pseudo-gradients/timing, not trained loss), so the same-sequence-target
  bug doesn't apply to them specifically. Still ran on the pre-RoPE-fix
  model where used for a result -- re-run before trusting any number.

**Action, in order:**
1. Re-run `scripts/hz0h_h3t_stage1_redo_real_task.py`'s check is already
   done (see above) -- start from the -0.6443 sign-flip result, not the
   original +0.5283/+0.6659 numbers.
2. Rebuild Arms A/B/C using `build_optimizer`/`train_step`/
   `shifted_target_batch` from `reference/hz0h_bdh_train_torch.py` instead
   of hand-rolled loops -- given Stage 1's core signal now has NEGATIVE
   cosine to the true gradient, Arm A (which substitutes the local signal
   directly for `encoder.grad`) should be expected to actively hurt
   training, not merely trail true BPTT as originally reported. Confirm or
   refute this directly rather than assuming.
3. Only rebuild SG-global / the calibration sweep / the three-parameter
   extension if Arms A/B/C's rebuilt Stage 2 results still motivate them --
   don't invest in Stage 3+ machinery ahead of confirming Stage 1/2 survive
   the fix.

**Exit gate:** a real, current answer to H3-T's original question (does
BDH's shared/tied-parameter structure support a training rule cheaper than
full BPTT+AdamW) grounded entirely in scripts built on the verbatim
`hz0h_bdh_torch.py` + `hz0h_bdh_train_torch.py` foundation.

## Non-goals for this restart

- Not re-litigating H0/H1's provenance work (unaffected -- H0 doesn't run
  the model; H1's own parity-test infrastructure is what caught the RoPE
  bug in the first place, once pointed at the real source).
- Not re-opening H3/H4/H7/T3/T4 (correctly still blocked on HZ-0G's G1 gate,
  unrelated to this restart).
- Not modifying canonical HZ backbone -- HZ-0H remains an isolated oracle,
  same contract as before.
