# Factorized BDH Under the Real Curriculum Recipe: Real, Decisive Result

Status: real, full 25M-token training run on real data, independently
downloaded and verified. **This resolves the open question left by the
earlier short quality probe** (`docs/restart/hz0h_factorized_quality_probe_results.md`):
does factorization's apparent quality edge survive BDH's actual best
training recipe (the recurrent-depth curriculum, full token budget)?

**Real answer: no.** Under the real, full recipe, factorization loses
its earlier apparent quality edge and ends up worse than both dense BDH
and the matched Transformer.

## What was run

`scripts/hz0h_factorized_curriculum_full_comparison.py` (built by the
concurrent codex-mac/mac-agent thread, commits `067c0cf`/`76f9fd8`,
independently verified here): 4 arms, same real data
(`data/packed/hz0h_bytes_25m_{train,val}.jsonl`), same real
recurrent-depth curriculum (`2->4->6->8` layers over the token budget,
this project's own established best BDH recipe) for both BDH arms,
same batch=12/seq=256/D=512/8H/mult=32, seed=7, bf16, RTX3060, real
energy tracking, 25M training tokens each -- the same scale as the
project's own canonical Phase F comparison.

## Correctness, independently re-verified (not just trusted from the report)

Ran `tests/reference/test_hz0i_factorized_bdh.py` locally (10/10 pass),
including two load-bearing checks specific to this comparison:

- `test_variable_depth_matches_factorized_full_forward_and_gradients`:
  `factorized_variable_depth_forward` with `n_iterations == n_layer`
  matches the full fixed-depth forward exactly (logits, loss,
  gradients), and shorter depths produce genuinely different (not
  no-op) computation -- the curriculum plumbing itself is correct.
- `test_full_rank_factorized_matches_dense_at_each_curriculum_depth`:
  a full-rank (uncompressed) factorized model exactly reproduces dense
  BDH's math at every curriculum depth -- isolates "factorization
  introduces error" from "rank reduction introduces error." The
  factorization framework itself is exact; only the deliberate rank
  reduction (64, 128) is the variable under test below.

Also confirmed `best_validation_loss == final_validation_loss` for all
3 BDH-family arms (and negligibly different for the Transformer) --
each arm's reported "best" is genuinely its fully-trained final state,
not an early-training artifact from a shallower curriculum stage.

## Real results

| arm | params | best val loss | tok/s | peak memory | J/token |
|---|---:|---:|---:|---:|---:|
| **dense BDH (curriculum)** | 25,427,968 | **1.3848** | 10,559 | 7.53 GiB | 0.01594 |
| factorized rank=128 | 8,126,464 | 1.7539 | 14,176 | 6.14 GiB | 0.01153 |
| factorized rank=64 | 4,194,304 | 1.8984 | 14,725 | 5.34 GiB | 0.01112 |
| matched Transformer | 25,343,488 | 1.5141 | 74,400 | 0.72 GiB | 0.00223 |

**Real, decisive ordering on quality**: dense BDH (1.3848) < matched
Transformer (1.5141) < factorized rank=128 (1.7539) < factorized
rank=64 (1.8984). Both factorized variants land clearly worse than
BOTH dense BDH and the Transformer once trained on the real curriculum
recipe at full token budget -- the opposite of the short, no-curriculum,
500-step probe's finding (`hz0h_factorized_quality_probe_results.md`),
which had factorized rank=64 *beating* dense BDH (2.4156 vs 2.5578).

**Real, consistent with every prior measurement**: factorization still
wins decisively on speed (1.34-1.39x), memory (0.71-0.82x), and energy
(0.70-0.72x joules/token) vs dense BDH. Those systems-level wins were
never in question; what's now resolved is that they come with a real,
non-trivial quality cost once training is done properly, not for free
as the short probe's surprising result had suggested.

## Reconciling with the earlier short-probe finding

The two results are not a contradiction -- they measured genuinely
different regimes, and the discrepancy itself is the real finding:
**a quality advantage measured on a short, fixed-depth, no-curriculum
budget does not predict what happens under the architecture's own real,
full training recipe.** Plausible, real, undistinguished explanations
(not yet isolated by any single experiment): the earlier probe's
500-step budget may have favored the much smaller (6x fewer parameter)
factorized model's faster early convergence, a pattern that reverses
once training continues long enough for dense BDH's extra capacity to
matter; or the depth curriculum itself may benefit disproportionately
from the extra capacity dense BDH has at each depth stage, in a way a
low-rank model cannot exploit as well. Either way, **short-budget
probes of architecture changes are not a reliable substitute for the
real training recipe** -- a real, useful methodological lesson for any
future architecture-change quality claim in this project.

## Real, open, unresolved discrepancy (disclosed, not glossed over)

This run's `dense_bdh` best validation loss (**1.3848**) is
meaningfully *better* than this project's own previously-documented
Phase F headline number for what should be the identical recipe (real
data, same batch/seq/curriculum/seed/lr,
`docs/restart/hz0h_phase_f_same_gpu_comparison_results.md`: **1.58203125**).
Both this run's `best_validation_loss == final_validation_loss` (ruling
out an early-shallow-depth eval artifact as the cause), so this is a
real, unexplained ~0.2 nat difference between two runs that both claim
to use the same real recipe. Not yet root-caused -- candidates include
a genuine difference in the two scripts' own eval-batch sampling/epoch
alignment, or real run-to-run training variance at this token budget
that hasn't been characterized with repeats. Flagging honestly rather
than silently picking whichever number is more convenient; the
*relative* ordering within this one self-consistent, single run (which
is what this document's real conclusion rests on) is unaffected either
way, since all 4 arms here were trained and evaluated identically
within the same run.

## Real conclusion

Factorization at rank 64/128 is a genuine, real systems win (speed,
memory, energy) but is **not** currently a quality-neutral substitute
for dense BDH once trained under the architecture's own real best
recipe -- it loses BDH's real, decisive quality edge over the
Transformer entirely, landing behind both. The short-probe result that
suggested otherwise was a real but misleading artifact of an
insufficiently long, non-curriculum training budget. Any future work
treating factorization as a drop-in efficiency win for BDH needs to
either accept this quality cost, find a rank/capacity regime that
narrows it, or treat factorization as a compute/inference-cost lever
for regimes where the Transformer's quality is acceptable but BDH's
memory footprint isn't -- not as a way to keep BDH's own quality
advantage while shrinking it.
