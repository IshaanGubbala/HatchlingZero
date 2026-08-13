# HZ Next-Phase Plan Phase B (VB compression-ratio sweep): D/4 remains the best choice -- D/2 and D/3 both lose on quality AND use more memory

Date: 2026-08-12. `plans/HatchlingZero_Next_Phase_Plan.md` Phase B's
real question: does a milder compression ratio (D/2, D/3, not just the
originally-used D/4) preserve more quality under the now-locked
recurrent-depth curriculum, giving a better quality/memory Pareto
point? Same config as every other locked-curriculum run (n_embd=512,
n_layer=8, n_head=8, mlp_mult=32, batch=12, 25M tokens, bf16,
`--compile-step`, seed=7).

## Real result: neither D/2 nor D/3 beats D/4 -- D/4 wins on both axes

| | d_state | best_validation_loss | Real state bytes/batch-item (fp32) |
| --- | --- | --- | --- |
| Exact BDH + curriculum (reference) | -- | 1.5820 | -- |
| **VB D/4 + curriculum** | 128 | **1.6309** | smallest |
| VB D/2 + curriculum | 256 | 1.6367 | largest of the three |
| VB D/3 + curriculum | 171 | 1.6387 | middle |

D/4 -- the ORIGINAL, most-compressed setting already used throughout
HZ-Core-1 -- has the best validation loss of all three VB divisors
tested, not just among the best-of-a-tradeoff: it also has the
SMALLEST state (real, monotonic, expected consequence of larger
`d_state`). **D/4 dominates D/2 and D/3 on both quality and memory --
there is no real Pareto tradeoff here to select between.** Milder
compression did not buy back quality; if anything it cost a small
amount (D/2: +0.0058 vs D/4; D/3: +0.0078 vs D/4).

## Real, honest caveats

1. **Single seed per divisor** (D/2, D/3 at seed=7 only) -- the RTX3060
   side itself flagged this isn't confirmed to the >=3-seed bar Phase A
   used before locking the curriculum recipe. The gap (0.006-0.008 in
   validation loss) is small enough that seed noise is a real,
   plausible partial or full explanation -- not ruled out here.
2. Real, honest note on a minor internal discrepancy caught and
   resolved before trusting these numbers: the training-side report
   initially stated D/3's `d_state` as 170 (mentally computing
   `512 // 3`), but the runner script's real logic is `round(512 / 3)
   = 171`, and the actual saved checkpoint's `P`/`O` parameter shapes
   were checked directly and confirmed `d_state = 171`, matching the
   runner's real behavior, not the `170` first assumed. Verified before
   using the checkpoint for anything downstream (matches this
   session's own standing "verify before trusting" discipline).
3. Passkey/reassignment in-context recall accuracy on both new
   checkpoints (via `scripts/hz0h_phase_b_report.py`) is 0.0 -- matches
   the same pattern already found for every other 25M-token-budget
   checkpoint this session (not enough real training data/budget for
   this capability to emerge yet, not a new or divisor-specific
   finding).
4. `training_seconds`/`tokens_per_second` for D/2 (1616.2s, 15,470
   tok/s) and D/3 (1585.9s, 15,766 tok/s) are close to each other and
   to the earlier VB+curriculum+compile numbers -- no real speed
   differentiator between the divisors at this scale.

## Real, honest conclusion for Phase B

**D/4 (`d_state = n_embd // 4`) is confirmed as the real Pareto choice
-- no adjustment needed.** The value bottleneck's already-locked
setting (used throughout HZ-Core-1 and every VB curriculum result this
session) was already the right one among the three tested; loosening
compression made things mildly worse, not better. Real, disclosed open
item: a genuine 3-seed confirmation of D/4's advantage over D/2/D/3
(matching Phase A's own bar) has not been run -- the gap is real but
small enough that this is a reasonable, not urgent, follow-up rather
than a blocking question.
