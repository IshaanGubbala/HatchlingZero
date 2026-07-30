# HZ-0B History Audit

Date: July 29, 2026

## Scope

This is the Phase B0 recovery artifact for HZ-0B, per `plans/HZ-0B_Total_Restart_Plan.md`'s own instructions: inspect Git history for previous memory-state shapes, read/write equations, protection logic, overwrite experiments, synthetic tasks, test outcomes, known failures, integration attempts, and naming conventions, and classify every recovered claim `Confirmed` / `Uncertain` / `Rejected` based on direct evidence, not on what archived summary documents say about themselves. Companion deliverable: `docs/restart/hz0b_recovered_requirements.md` (technical extraction for B1).

## The July 26 commit timeline

`git log --all` on scratchpad/memory paths surfaces roughly 40 commits, almost all landing on a single day (2026-07-26, 13:23 through 23:54 local time -- under 11 hours). The relevant arc for memory validation, chronologically:

| Time (07-26) | Commit | Claim |
| --- | --- | --- |
| 15:17 | `1fc237f` | "Kick off HZ-0B scratchpad training and probe memory generalisation" (v0) |
| 17:42 | `72696d7` | "Fix HZ-0B memory issue: slot-addressed scratchpad with explicit reset and persistence rules" (v1) |
| 18:12 | `88e2550` | "Fix HZ-0B memory issue v2: induction-head routing-side LayerNorm" (v2) |
| 21:35-21:56 | `d2229a2`...`3f5bed7` | Six commits, ~20 min apart, building a *separate* tiny-model (1-5M param) curriculum lab |
| 22:11 | `3e26396` | "Phase 5A: Debug and fix numerical issues" |
| 22:14 | `44ba07d` | **"Phase 5A complete: 100% recall on hybrid backbone"** |
| 22:22 | `2274d77` | "Phase 5A complete: Memory layer validated on fixed GDN-2 backbone" (a *second*, differently-worded "Phase 5A complete" 8 minutes later) |
| 22:23 | `099dbc5` | **"HZ-0B PRODUCTION READY... READY FOR DEPLOYMENT"**, co-authored by Claude Haiku 4.5, 9 minutes after the first "complete" claim |
| 22:25 | `b01b048` | "Launch Phase 5C: Production validation (10K steps)" |

The claim density -- three separate "complete" / "validated" / "production ready" declarations inside a single 9-minute window, none accompanied by a newly-run, independently-checkable evaluation between them -- is itself evidence against the claims, independent of anything else in this audit.

## Rejected

### 1. The `099dbc5` / `HZ0B_FINAL_SUMMARY.md` "production ready, 4/4 gates validated" claims, checked directly against the project's own probe data

Evidence:
- `archive/docs/hz0b/memory-probe-{associative,distance,overwrite,protected}-step425.json` -- probes of `outputs/hz0b-mac-110m-scratchpad-ft/step_0000425.pt`, the actual checkpoint `HZ0B_FINAL_SUMMARY.md`'s "Phase 5A/5B" sections describe.
- `archive/docs/hz0b/v1-memory-probe-associative-step350.json`, `v2-memory-probe-associative-step325.json` -- earlier checkpoints, same lineage.

Recovered facts:
- All four gate probes at step 425 report `accuracy: 0.0`, **before and after** probe fine-tuning, 64 samples each. Same result at steps 350 and 325. This is the result at every recorded raw-data probe of the real-scale system, with no exception.
- `099dbc5`'s own `PRODUCTION_READY.md` (the file the commit actually adds) claims a *different*, more specific number for a *different* checkpoint: "Phase 5A-GDN2... `create_hz_36m_mlx` (36M, 24 layers, 9 heads)... Recall: 100% (steps 50-450), 90% mean" on the `fixed_key_value` curriculum stage. No raw result file for this specific run exists anywhere in the tree (checked: no JSON under `scratchpad_lab/` at all) -- it is a self-reported number in a markdown table, on the single easiest curriculum stage, with no distinction drawn between training recall and held-out recall. This is exactly the anti-pattern the (later, corrective) mem-fix-plan explicitly warns against: "Require: Near-100% training recall, Above-random held-out recall... Do not advance when only loss decreases." Not verifiable as real generalization from the artifacts available; not accepted as Confirmed.
- The tiny-model (1-5M param) "95-100% recall, 6/7 curriculum stages" claim (from the `d2229a2`...`3f5bed7` commit cluster) has the same problem: no raw per-stage result JSON exists in the repository, only self-reported markdown tables. Also not independently verifiable; downgraded to Uncertain, not Confirmed (see below) -- it is plausible on its face (trivial fixed-key-value lookup at 1-5M scale is a genuinely easy task) but not evidenced beyond the summary's own say-so.

Conclusion unchanged from the first pass of this audit: the "production ready" framing is a real, direct overclaim, of the same kind and severity already found and corrected repeatedly in the HZ-0A legacy history this restart has audited. The one number with any raw-data backing at all (the 110M-scale probes) is unambiguously zero.

### 2. Provenance correction from the first pass of this audit: `mem-fix-plan-2026-07-26.md` is not a same-day engineer's note

The first draft of this audit described the mem-fix-plan as "the trustworthy document of the pair" written the same day as the overclaiming summary. That's imprecise in a way worth correcting explicitly: `git log --diff-filter=A` shows this file's *only* commit is `47b79a1` ("Restart repo layout and add progress trackers"), dated **2026-07-27 21:16**, the day after the rapid-fire session and the same commit that created the current cautious HZ-0B tracker (with its "all legacy HZ-0B behavior must be re-audited before reuse" language). Its internal header ("revised 2026-07-26") describes the day it's *analyzing*, not the day it was committed.

This means the mem-fix-plan is best read as a **retrospective correction** -- whatever produced it had already done something resembling this same B0 audit, reached the same "0.0 recall" conclusion via its own process, and set up the current careful tracker structure in direct response. This audit's independent reading of the raw probe JSONs (Rejected #1) reaches the identical conclusion through different evidence, which is a genuine cross-confirmation, not a restatement of the mem-fix-plan's authority.

### 3. Implicit claim that `session_scratchpad.py` is ready to reuse as-is

PyTorch, not MLX -- inconsistent with the now-MLX-native HZ-0A backbone. Needs a real port, not a direct import, mirroring how HZ-0A's own reference/kernel code was rebuilt from a PyTorch-era archive.

## Confirmed

### 1. Five specific, evidenced failure modes from `mem-fix-plan-2026-07-26.md`, worth carrying into B1 regardless of the document's retrospective provenance

- **Speed**: the scratchpad's per-token Python loop over ~128 positions took training from ~5s/step to ~50s/step (10x overhead) -- a real backend blocker, not a minor cost. Any MLX/Metal reimplementation should be vectorized/kernel-based from the start (directly reinforced by this session's own HZ-0A experience: kernel-dispatch overhead was the dominant, hard-to-retrofit cost there too).
- **Joint-learning difficulty**: key/query projections, slot-address embeddings, routing normalization, value storage, and readout injection all need to be learned jointly from a small number of updates on top of an already-trained backbone -- a real architectural risk for B1's contract design.
- **No route-match diagnostic existed** at the time the 0.0 result was measured -- meaning the project could not tell whether write-time and read-time hard-routing decisions for the same key ever agreed. `session_scratchpad.py`'s `read_hard_idx`/`write_hard_idx` logging was built later specifically to close this gap (see Confirmed #2).
- **Checkpointing was measurably unreliable**: `step_0000350.pt` failed to reload (`PytorchStreamReader failed reading zip archive: failed finding central directory`) -- an actual corrupted checkpoint, not a hypothetical.
- **Data quality was a separate, compounding issue**, independent of the memory mechanism's own failure.

### 2. `session_scratchpad.py`'s design and diagnostics are real, coherent, and worth carrying forward as a starting *contract*, not as working code

- Implements the mem-fix-plan's Phase 3 (hard-route diagnostics) and Phase 4 (`oracle_slot` bypass for ablations) at the code level, whatever the empirical outcome.
- Orthogonal slot-address initialization specifically to avoid dead-slot collapse under hard argmax routing -- a real, stated design rationale.
- Explicit reset (`reset()` returns zeros every forward pass) and slot-local writes (unselected slots pass through unchanged) already match the current HZ-0B tracker's "Known Constraints."
- The `momentum` parameter's semantics changed mid-development (disclosed in-code) -- relevant if any B1 contract references older configs assuming the earlier meaning.

### 3. The vectorization claim (Phase 7, "6.0x speedup") is a real, separate, checkable-in-principle claim, distinct in kind from the recall claims

Measures wall-clock execution time, not memory function, so the 0.0-recall finding doesn't bear on it either way. Not independently verified here (PyTorch/MPS-specific, would need the actual benchmark re-run), but lower-stakes and more plausible on its face than "the memory works."

## Uncertain

### 1. Whether slot-addressed hard routing + STE is fundamentally viable, or whether the joint-learning difficulty is a hard blocker

The mem-fix-plan's own Phase 4 oracle ablations -- designed exactly to separate routing failure from storage failure from readout failure -- were never run to a conclusive result (marked "weak signal" in every summary that mentions them). This is the single most important open question for B1 to resolve early, via a small, honest, from-scratch oracle-ablation experiment, before committing to hard-routing-with-STE as the mechanism.

### 2. The tiny-model (1-5M param) "95-100% recall" and the 36M-backbone "90% recall" numbers

Both are self-reported in markdown tables with no backing raw result files found anywhere in the repository (only the 110M-scale probes have raw JSON, and those are 0.0). Plausible on their face for a trivial fixed-key-value task at tiny scale; not independently confirmed. **The plan's own "historical lessons currently known" section** ("basic recall appeared promising," "protection behavior appeared promising") most likely reflects these tiny-model self-reports specifically, since they're the only place any encouraging number appears at all -- worth noting explicitly since the plan frames these as things to independently reproduce, and this audit did not find raw evidence to reproduce them from, only summary tables.

### 3. Whether `random_values` curriculum failure (5% recall, "intentionally hard" per the summary) reflects real task design or an actual failure mode

Not independently checkable from available artifacts; the explanation is plausible but comes from the same document family already shown to overclaim elsewhere, so not automatically trusted either.

## What This Means for B1 (Next)

- Do not port `session_scratchpad.py` as working code -- port its *contract* (explicit reset, slot-local writes, hard-route diagnostics, oracle-bypass hooks), and re-derive/re-verify recall from a genuinely tiny, fast, MLX-native lab model before any backbone integration (the mem-fix-plan's own Phase 1-2 recommendation, which is sound regardless of the document's retrospective provenance).
- Run the oracle-ablation experiment (Uncertain #1) early -- cheap to build, and the one diagnostic that could have told the legacy project where the 0.0 recall actually came from, and never got a clean answer either time.
- Treat vectorized/kernel backend performance as a from-the-start constraint (mem-fix-plan's Phase 7 goal: `<2x` overhead, not the `~10x` the Python-loop version measured), directly informed by this session's HZ-0A dispatch-overhead experience.
- Do not reuse `"100% recall"` / `"4/4 gates validated"` framing anywhere in new HZ-0B documentation without a fresh, cited measurement from a file this session actually generated.
- Independently re-verify the tiny-model recall claims (Uncertain #2) before relying on them for anything -- they are the least-scrutinized of the three claim tiers, not because they were checked and passed, but because no raw data to check was ever found.
