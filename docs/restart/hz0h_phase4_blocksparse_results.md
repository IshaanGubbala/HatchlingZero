# HZ Phase 4 (BlockBDH): real speedup confirmed, zero-shot quality fails badly

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

**BlockBDH's real-speedup half of Phase 4's exit gate is satisfied** —
this is a genuine, measured, non-FLOP-counting wall-clock win, the
hardest part of Phase 4's own stated risk to clear. **The quality half is
not yet satisfied** — exactly analogous to 2R-B's own arc (value
bottleneck ALSO needed real training from scratch before it worked;
zero-shot state grouping failed the same way this did). The real next
step is the same pattern that worked for 2R-B: train a BlockBDH variant
end to end (model AND router, or a fixed/annealed router used during
training) rather than applying block-sparsity to an already-trained
dense model.

## Real, honest caveats

1. Single trained model, one task, tiny scale — same limitations as
   every other Phase 2R/4 result this session.
2. The router itself is a minimal, un-learned heuristic (mean absolute
   activation of the first layer only, aggregated once per whole
   call) — the plan's own candidate design implies a "cheap block
   router" could itself be a small learned component; not built here.
3. Coarse (whole-call) granularity, not the plan's eventual per-token
   design — a real, deliberate scope limit for this first experiment,
   matching the plan's own "avoid fine-grained sparsity until it proves
   a real win" guidance, but per-token routing (with its own real
   dispatch-overhead risk) is still real, undone future work.
4. No energy measurement here (same CUDA-only/Mac-has-no-`powermetrics`-
   path gap as every other inference benchmark this session).

## Real next steps

1. Train a BlockBDH model end to end (real gradient signal through the
   router's block selection, or a scheduled/annealed sparsity ramp
   during training) — the direct analog of what made 2R-B's value
   bottleneck work after its own zero-shot-equivalent first attempt
   would have failed.
2. Once quality is real and trained-for: re-measure the speed numbers
   above on the actually-deployed (trained) router, since a trained
   router's selection pattern could differ from the untrained heuristic's.
3. Try a real learned router (small linear scorer) instead of the mean-
   activation heuristic, per the plan's own candidate design.
4. Extend to per-token (not per-call) routing once a call-level version
   is quality-validated, with real grouped/sorted dispatch (the plan's
   own recommended implementation priority) to keep dispatch overhead
   bounded.
