# HZ-0B History Audit

Date: July 29, 2026

## Scope

This is the Phase B0 recovery artifact for HZ-0B (session-local memory scratchpad), done while HZ-0A's Stage 2 (100M-token) runs finish in the background. Per the HZ-0B tracker's own working assumption ("all legacy HZ-0B behavior must be re-audited before reuse") and readiness gate ("STOP: implementation before B0/B1"), this treats `archive/` as evidence to verify, not as trusted implementation -- same discipline as the HZ-0A A0 audit.

Every legacy claim below is classified `Confirmed`, `Uncertain`, or `Rejected` based on direct artifacts, not on what the legacy summary documents say about themselves.

## Rejected

### 1. `archive/src/hz0/scratchpad_lab/HZ0B_FINAL_SUMMARY.md`'s "4/4 memory gates validated, 100% recall, ready for production" claim

This is the single most important finding of this audit: the legacy summary document's headline claims are **not supported by the legacy project's own probe data**, checked directly.

Evidence:
- `archive/docs/hz0b/memory-probe-{associative,distance,overwrite,protected}-step425.json` -- probes of `outputs/hz0b-mac-110m-scratchpad-ft/step_0000425.pt`, the actual 110M-scale fine-tuned checkpoint the summary's "Phase 5A/5B" sections describe.
- `archive/docs/hz0b/v1-memory-probe-associative-step350.json`, `archive/docs/hz0b/v2-memory-probe-associative-step325.json` -- earlier checkpoints in the same lineage.

Recovered facts:
- All four gate probes at step 425 (associative, distance, overwrite, protected) report `accuracy: 0.0` **both before and after** the probe fine-tuning step, with 64 samples each -- not partial, not noisy, exactly zero.
- The same zero result holds at step 350 (v1) and step 325 (v2) -- this is not a one-checkpoint anomaly, it is the result at every recorded probe of the real-scale system across the whole iteration history.
- `HZ0B_FINAL_SUMMARY.md` itself, in its "Known Issues" section, separately admits the full 110M/GDN-2 backbone integration never worked at all ("Backbone (`create_hz_36m_mlx`) produces all-NaN output... Cannot use full 110M backbone yet") -- meaning the "100% recall" and "Hybrid architecture validated" claims elsewhere in the same document can only be referring to a **tiny 1-5M-parameter toy backbone** (2 layers x 128 dim, per the document's own "Architecture Overview" diagram), not the system the probe files above were actually run against.
- The summary's own "Phase 4: Oracle Ablations" row is marked `⚠️` with "signal weak," meaning even the toy-scale diagnostic designed to isolate whether routing, storage, or readout was the broken component never produced a usable answer.

Conclusion: the "production ready" framing is a real, direct overclaim by the legacy material, of the same kind and severity this whole restart's A0-A12 audits have repeatedly found and corrected in the HZ-0A legacy history. Nothing about the scratchpad's actual memory *function* at any scale beyond toy synthetic tasks is validated. This does not mean the underlying idea or code is worthless -- the design rationale and the diagnostic instrumentation (see below) are real, salvageable contributions -- but the claim that it works is false as stated.

### 2. Implicit claim that the SessionScratchpad module is ready to reuse as-is

`archive/src/hz0/model/session_scratchpad.py` is PyTorch, not MLX -- inconsistent with the now-MLX-native HZ-0A backbone this session built. It would need a real port (mirroring how HZ-0A's own reference/kernel code was rebuilt in MLX + Metal from a PyTorch-era archive), not a direct import.

## Confirmed

### 1. `archive/docs/hz0b/mem-fix-plan-2026-07-26.md` accurately diagnoses the real failure mode, and predates the overclaiming summary

Evidence: the mem-fix-plan is dated the same day as `HZ0B_FINAL_SUMMARY.md` and states plainly, in its own words: "held-out synthetic memory recall is still `0.0 -> 0.0` on every committed checkpoint." This matches the probe JSON evidence above exactly and is the more trustworthy of the two same-day documents -- it describes failure with specific measurements, rather than asserting success without citing the measurements that would support it.

Recovered facts (five real, specific blockers identified, all worth carrying forward as known risks for B1+):
- **Speed**: the scratchpad's per-token Python loop over ~128 sequence positions took training from ~5s/step to ~50s/step (10x overhead) -- a genuine backend blocker for iteration speed, not a minor inconvenience. Any MLX/Metal reimplementation should be designed vectorized/kernel-based from the start, not ported as a Python loop first.
- **Joint-learning difficulty**: the scratchpad must jointly learn key/query projections, slot-address embeddings, routing normalization, value storage, and readout injection simultaneously from a small number of updates on top of an already-trained backbone -- plausibly too much to learn at once, a real architectural risk for B1's memory contract design, not just a training-duration problem.
- **No route-match diagnostic existed** at the time recall was 0.0 -- meaning the project could not tell whether the write-time and read-time hard-routing decisions for the same key ever actually agreed. This diagnostic gap is exactly what `session_scratchpad.py`'s `read_hard_idx`/`write_hard_idx` logging (see Confirmed #2 below) was later built to close.
- **Checkpointing was measurably unreliable**: `step_0000350.pt` failed to reload (`PytorchStreamReader failed reading zip archive: failed finding central directory`) -- a real corrupted checkpoint, not a hypothetical risk.
- **Data quality was a separate, compounding issue**: the seed corpus was too small/degenerate for general language quality, independent of the memory mechanism's own failure.

### 2. `archive/src/hz0/model/session_scratchpad.py`'s design and diagnostics are real, coherent, and worth carrying forward as a starting contract, not as working code

Evidence: direct code read, cross-checked against the mem-fix-plan's stated goals.

Recovered facts:
- The module implements the mem-fix-plan's Phase 3 (hard-route diagnostics: `read_hard_idx`/`write_hard_idx` surfaced per token) and Phase 4 (`oracle_slot` bypass parameter enabling oracle-routing/oracle-storage/oracle-read ablations) -- meaning at least two of the plan's five "immediate next tasks" were genuinely implemented at the code level, whatever the empirical outcome (Rejected #1 above) ended up being.
- Slot addresses are orthogonally initialized (`nn.init.orthogonal_`) specifically to avoid dead-slot collapse under hard argmax routing -- a real, specific, documented design decision with a stated rationale, not an arbitrary choice.
- Reset is explicit (`reset()` returns zeros, called at the start of every forward pass) and writes are slot-local (unselected slots pass through unchanged) -- both match the HZ-0B tracker's current "Known Constraints" (`Memory must stay distinct from recurrent state`, `Reset must be explicit`) already, without needing to be re-derived.
- The `momentum` parameter's semantics changed mid-development (comment explicitly notes it: "momentum is now an *intra-slot persistence* knob" as of this version, distinct from an earlier meaning) -- a real, disclosed API evolution, useful context if any B1 contract references old configs (`hz0b-mac-110m-scratchpad-ft.yaml` etc.) that may assume the earlier semantics.

### 3. `archive/src/hz0/scratchpad_lab/HZ0B_FINAL_SUMMARY.md`'s vectorization figure (Phase 7, 6.0x speedup) is a real, separate, checkable claim -- not verified true or false by this audit, but distinct in kind from the recall claims

This is the one FINAL_SUMMARY claim not contradicted by the probe evidence above -- it measures a wall-clock property (Python-loop vs. vectorized scratchpad execution time), not a memory-function property, so the 0.0 recall finding doesn't bear on it either way. Genuinely unverified here (would need the actual benchmark script re-run, and it is PyTorch/MPS-specific, not MLX), but a plausible, self-contained, lower-stakes claim than "the memory works."

## Uncertain

### 1. Whether the architecture (slot-addressed hard routing + STE) is fundamentally viable, or whether the joint-learning difficulty identified in the mem-fix-plan (Confirmed #1) is a hard blocker

The mem-fix-plan's own Phase 4 oracle ablations -- specifically designed to answer this by isolating routing failure from storage failure from readout failure -- were never conclusively run to completion (FINAL_SUMMARY marks this phase `⚠️ weak signal`). No artifact in `archive/` answers whether, e.g., oracle-routing alone (bypassing the learned routing entirely) would have produced non-zero recall. This is the single most important open question for B1 (memory contract) to resolve early, ideally via a small, honest, from-scratch oracle-ablation experiment before committing to hard-routing-with-STE as the mechanism.

### 2. Whether the `random_values` curriculum stage's near-zero recall (FINAL_SUMMARY Phase 1-2, 5%) reflects a real task-design flaw (as the summary claims: "intentionally hard... no curriculum signal") or an actual failure mode worth investigating

Not independently checkable from the artifacts alone; the summary's explanation is plausible on its face (a task with genuinely no learnable signal should fail) but given Rejected #1's finding that this same document's other claims don't hold up, its explanation for this one is not automatically trustworthy either.

## What This Means for B1 (Next)

- Do not port `session_scratchpad.py` as "working code" -- port its **contract** (explicit reset, slot-local writes, hard-route diagnostics, oracle-bypass hooks) as a starting design, and re-derive/re-verify recall from a genuinely tiny, fast, MLX-native lab model before attempting any backbone integration, mirroring the mem-fix-plan's own (sound) Phase 1-2 recommendation.
- Budget real effort for the oracle-ablation experiment (Uncertain #1) early -- it is cheap to build and was the one diagnostic that could have told the legacy project where the 0.0 recall actually came from, and never got a clean answer.
- Treat vectorization/backend performance as a from-the-start design constraint (the mem-fix-plan's own Phase 7 goal: scratchpad overhead `<2x`, not the `~10x` the Python-loop version measured), not an optimization to bolt on after correctness -- directly informed by this session's HZ-0A experience, where kernel-dispatch overhead was repeatedly the dominant, hard-to-retrofit cost.
- Do not reuse any `associative_recall = 100%` / `"4/4 gates validated"` framing anywhere in new HZ-0B documentation without a fresh, cited measurement -- the legacy number is contradicted by the legacy project's own probe files.
