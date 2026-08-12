# HZ Phase 4 (BlockBDH): real speedup confirmed, zero-shot quality fails badly

## ⚠️ Update 2: does NOT generalize to the harder reassignment task — real degradation, not 0%

Same day, same trained-end-to-end methodology, applied immediately
(before trusting the passkey-only "0% degradation at every fraction"
claim below) to H5's harder reassignment/overwrite task, exactly the
check that already once corrected an over-optimistic easy-task result
in this session (`docs/restart/hz0h_phase2r_reassignment_task_results.md`).
Same config, 2500 training steps (confirmed sufficient for dense exact
BDH elsewhere this session):

| Active fraction | Accuracy (trained end-to-end) |
| --- | --- |
| 50% | **0.74** (real degradation, -26 points) |
| 25% | **1.00** |
| 12.5% | **0.135** (near the 0.125 chance floor — collapse) |

**Non-monotonic and, at 50%/12.5%, real degradation — the passkey
result's "0% degradation at every fraction" does not hold here.**

## Update 4 (`plans/HZ Integrated Candidate Plan.md` Step 3): 5-seed picture — majority-reliable, not yet consistent

Extended to 5 total seeds at 50% active on reassignment, the concrete
target set in `plans/HZ Integrated Candidate Plan.md` ("turn 0.60-1.00
into consistently ~0.95-1.00"):

| Seed | Accuracy at 50% active |
| --- | --- |
| 0 | 0.74 |
| 1 | 0.60 |
| 2 | 1.00 |
| 3 | **1.00** |
| 4 | **1.00** |

**3 of 5 seeds reach perfect (1.00) accuracy; 2 of 5 show real
degradation (0.60, 0.74).** More precise than the earlier 3-seed picture
suggested: this is not a coin-flip between "works" and "fails" — the
majority of initializations DO find the fully-correct solution, but the
training recipe still doesn't reliably reach it every time. **Target not
yet met** (this is not yet "consistently ~0.95-1.00") but the gap is
narrower than it first looked: 60% of seeds already hit the target
outcome exactly. Real next step per the plan: diagnose what
differentiates the 3 successful seeds from the 2 unsuccessful ones
(e.g. early-training loss trajectory, gradient norms) before trying a
blind fix (more steps, different LR) — same discipline as every other
training-instability diagnosis this session.

## Update 3: initial 3-seed picture (kept for the record, see Update 4 for the fuller 5-seed result)

Ran 2 more seeds at the ambiguous 50%-active setting to check whether
the non-monotonic shape above (50% worse than 25%) was a real capacity
effect or training variance:

| Seed | Accuracy at 50% active |
| --- | --- |
| 0 (original) | 0.74 |
| 1 | 0.60 |
| 2 | 1.00 |

High variance, 0.60-1.00 range across 3 seeds at the identical config —
settled the earlier ambiguity (real instability, not a clean capacity
trend), though the fuller 5-seed picture in Update 4 shows this is
majority-reliable (3/5 perfect) rather than a coin flip. Read the
passkey-only numbers below as one real, positive, single-seed data
point on an easy task, not a general or seed-stable property of trained
BlockBDH.

## ✅ Update 1: trained end-to-end on PASSKEY, both halves of the exit gate real — 100% accuracy at every fraction tested, up to 6.20x speedup

`scripts/hz0h_bdh_blocksparse_passkey_eval.py`, same day. Direct fix for
the zero-shot failure below, matching the exact pattern that made 2R-B's
value bottleneck work: train THROUGH the real block-sparse forward path
from step 0 (active blocks recomputed fresh each step from the model's
current weights; gradient reaches only the columns actually selected
that step — `tests/reference/test_hz0h_bdh_blocksparse_torch.py`'s new
`test_gradients_flow_through_selected_columns_only` confirms this
directly: selected columns get real, nonzero gradient, unselected
columns get exactly zero that step).

Real result, 800 training steps (same budget dense exact BDH needs for
1.00 accuracy on this task elsewhere in this session):

| Active fraction | Accuracy (trained end-to-end) | Measured speedup (from above) |
| --- | --- | --- |
| 50% | **1.00** | 1.95x |
| 25% | **1.00** | 3.75x |
| 12.5% | **1.00** | 6.20x |

**Perfect accuracy at every fraction tested, including the most
aggressive one (12.5% active, 6.20x measured speedup).** Both halves of
Phase 4's exit gate ("real end-to-end speed... AND acceptable quality")
are now real and measured together, not just the speed half. This is a
clean, complete, positive Phase 4 result — read the caveats below
(still real and still apply) before generalizing past what was actually
tested.

Date: 2026-08-11. First real experiment under
`plans/HatchlingZero_Reality_Plan.md`'s Phase 4 ("BlockBDH: Turn
Sparsity Into Real Compute Savings"), whose own exit gate demands real
end-to-end speed/energy improvement, not a FLOP-counting argument
("Never claim a speedup from FLOP reduction alone" — Risk 2).

## What was built

`reference/hz0h_bdh_blocksparse_torch.py`: a cheap, coarse (whole-call,
not per-token — per the plan's own "avoid arbitrary fine-grained
sparsity until it produces real wall-clock wins") block router selects
the top-k active blocks of `encoder`'s N-dimension by mean activation
magnitude, then `encoder`/`encoder_v`/`decoder` are multiplied through
ONLY the selected columns (`index_select`, a real smaller matmul, not a
full computation with a mask applied after). Real correctness subtlety
found and fixed: RoPE's frequency buffer must be gathered with the SAME
column indices as the activations, or the surviving columns get rotated
by the wrong phase — bypassed `Attention.forward`'s black-box call and
reimplemented the attention math directly with the gathered
`sparse_freqs`. Verified byte-exact against dense `BDH.forward` at 100%
active (the real validation this whole implementation depends on) —
`tests/reference/test_hz0h_bdh_blocksparse_torch.py`, 5 tests.

## Real result 1: the speedup is real and strong, not a FLOP artifact

Matched ~4.8M-param pilot config (D=256, n_layer=6), Mac MPS, batch=8,
seq_len=256, real wall-clock per forward call:

| Active fraction | ms/call | Speedup vs. dense |
| --- | --- | --- |
| 100% (sanity check) | 32.80 | 0.99x (correctly ~1x, no overhead) |
| 50% | 16.73 | **1.95x** |
| 25% | 8.68 | **3.75x** |
| 12.5% | 5.26 | **6.20x** |

**Near-proportional real speedup, not the disappointing result the
earlier BDH-KV-cache experiment found** (`docs/restart/hz0h_phase1_kv_cache_bdh_results.md`
-- that approach's per-token gather/scatter overhead ate most of the
theoretical win). The mechanistic difference: this reduces the WIDTH of
large matmuls in the full dense parallel forward pass (a genuine
GEMM-shape reduction), not irregular per-step indexing on small tensors
-- shape reduction maps cleanly to real hardware speedup in a way
per-token dynamic gather doesn't.

## Real result 2: zero-shot quality collapses badly — the router needs training, not a heuristic

Same H5 passkey-retrieval methodology used throughout this session,
same "zero-shot" pattern as the grouped-state check
(`docs/restart/hz0h_phase2r_grouped_state_results.md`) — a real trained
exact-BDH model (400 steps, 1.00 dense accuracy), evaluated through the
block-sparse forward path WITHOUT retraining, using the cheap
mean-activation router:

| Active fraction | Accuracy | Degradation vs. dense |
| --- | --- | --- |
| 100% | 1.00 | 0% (exact, sanity check) |
| 50% | 0.505 | **-49.5%** |
| 25% | 0.625 | -37.5% |
| 12.5% | 0.130 | -87.0% (near the 0.125 chance floor) |

**Badly, non-monotonically broken** (25% scoring higher than 50% is
itself a signal something is off, not a real trend — most likely
reflects that this specific cheap router isn't reliably finding the
task-relevant blocks at either granularity, more than a genuine
"less-is-more" effect). This is the SAME lesson 2R-C's plain zero-shot
state grouping already taught: dropping/merging information the model
was never trained to expect dropped or merged destroys real
task-critical signal, and a cheap post-hoc heuristic router is not a
substitute for the model (and ideally the router itself) being trained
under the actual compute-reduction regime it will run under.

## Real, honest verdict

**The speed half of Phase 4's exit gate is solidly real**: measured,
non-FLOP-counting wall-clock speedup (1.95x-6.20x), confirmed at a
matched pilot scale. **The quality half is real but task-dependent, not
universal**: training through the actual block-sparse path (rather than
zero-shot) fixed quality completely on passkey retrieval (1.00 at every
fraction) but only partially on the harder reassignment task (0.74 at
50%, collapsing to 0.135 at 12.5% — see Update 2 above). The zero-shot
failure was real and important to disclose, and training-through-the-
path is a real, necessary fix (not optional, as passkey alone might have
suggested) — but it is not sufficient on its own to guarantee quality at
every compression level on every task. The safe operating fraction is
task-dependent, same conclusion 2R-B/2R-E already reached for state
compression.

## Real, honest caveats — still apply, read before generalizing

1. Single trained model, one task, tiny scale (n_embd=32, 2 layers) —
   same limitations as every other Phase 2R/4 result this session. The
   speed numbers were separately measured at a bigger matched pilot
   scale (D=256, n_layer=6); the QUALITY numbers were only measured at
   this small scale — not yet confirmed together at the same scale.
2. The router itself is a minimal, un-learned heuristic (mean absolute
   activation of the first layer only, aggregated once per whole
   call) — worked here, but the plan's own candidate design implies a
   "cheap block router" could itself be a small learned component; not
   built or compared against here.
3. Coarse (whole-call) granularity, not the plan's eventual per-token
   design — a real, deliberate scope limit for this first experiment,
   matching the plan's own "avoid fine-grained sparsity until it proves
   a real win" guidance, but per-token routing (with its own real
   dispatch-overhead risk, same class of risk the BDH-KV-cache
   experiment ran into) is still real, undone future work.
4. Only tested on passkey retrieval — the same task that gave 2R-B's
   own too-optimistic-looking initial "32x, 0% degradation" result later
   corrected by the harder reassignment task
   (`docs/restart/hz0h_phase2r_reassignment_task_results.md`). Real,
   important next step: re-run this exact check on reassignment before
   trusting "0% degradation at 6.20x speedup" as a general property
   rather than a passkey-specific one.
5. No energy measurement here (same CUDA-only/Mac-has-no-`powermetrics`-
   path gap as every other inference benchmark this session).

## Real next steps

1. ~~Re-run on the harder reassignment/overwrite task~~ — done, see
   Update 2 above: real degradation at 50%/12.5%, not the clean 0% the
   passkey-only result suggested.
2. ~~Diagnose WHY reassignment fails non-monotonically~~ — done, see
   Update 3 above: 50% active is genuinely seed-unstable (0.60-1.00
   across 3 seeds), a training-recipe robustness issue, not a fixed
   architectural ceiling.
3. Fix the training-recipe instability itself (more steps, a different
   LR/schedule, or averaging/ensembling across seeds and accepting the
   best) before trying a learned router — the current failure mode looks
   like an optimization-robustness problem first, an architecture
   problem second.
4. Try a real learned router (small linear scorer) instead of the mean-
   activation heuristic, per the plan's own candidate design — may
   matter more for reassignment than it did for passkey, where the
   simple heuristic already sufficed, but lower priority than item 3.
5. Confirm speed AND quality together at the same (larger) scale — the
   6.20x number and the accuracy numbers were measured at different
   model sizes in this pass.
6. Extend to per-token (not per-call) routing once a call-level version
   is quality-validated at more than one task/scale, with real
   grouped/sorted dispatch (the plan's own recommended implementation
   priority) to keep dispatch overhead bounded.
