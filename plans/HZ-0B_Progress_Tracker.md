# HZ-0B Progress Tracker

Updated: July 29, 2026 (B0 history audit done)

## Mission

Add explicit, controllable, session-local associative memory to a finished and frozen HZ-0A base.

## Current Status

- Overall phase: B0-B3 complete; B4+ not started. Plan's own "first concrete milestone" reached at B2: "a standalone, deterministic memory simulator with explicit read, write, update, protect, forget, delete, reset, and restore operations that passes synthetic tests for recall, overwrite, collision, and protection" -- `reference/hz0b_memory_simulator.py`, 20/20 tests passing, no LM attached.
- Dependency status: B0/B1 (audit/contract) unblocked and in progress per the "GO LATER... when parallel research time is available" gate (HZ-0A A1-A3 have been done since early in the restart); B6+ (integration) still blocked on frozen HZ-0A, which is not yet frozen (Stage 2/100M still running)
- Working assumption: all legacy HZ-0B behavior must be re-audited before reuse -- **this was not a hypothetical caution**, see B0 below

## Phase Tracker

| Phase | Deliverable | Status | Notes |
| --- | --- | --- | --- |
| B0 | history audit | **Complete** (both required deliverables) | `docs/restart/hz0b_history_audit.md` + `docs/restart/hz0b_recovered_requirements.md`. Full `git log --all` sweep (not just the checked-out archive/ tree) found ~40 commits, almost all within one 11-hour burst on 2026-07-26, with three separate "complete"/"production ready" declarations inside a single 9-minute window (22:14-22:23) -- claim density alone is evidence against trusting them. Checked directly against raw probe data: all four gate probes (`memory-probe-*-step425.json`, plus earlier v1/v2 checkpoints) show `0.0` recall before and after fine-tuning on the real 110M-scale system, every time it was measured. The one narrower positive number (`PRODUCTION_READY.md`'s "36M backbone, 90% recall") has no raw result file anywhere in the tree and conflates training recall with held-out recall on the single easiest curriculum stage. `mem-fix-plan-2026-07-26.md` -- corrected from the first audit pass -- is not a same-day note but a **retrospective correction**, committed the next day (2026-07-27) as part of the same reorg that created this tracker; it independently reaches the same 0.0 conclusion this audit's own raw-JSON read reached, a genuine cross-confirmation. Recovered requirements doc extracts the real read/write/gating equations from `hybrid_lm.py`/`session_scratchpad.py` (cited to file:line, not assumed) and finds the legacy code implements only 3 of the 10 operations B1 requires (`read`/`write`/`reset`; no `reinforce`/`update`/`protect`/`forget`/`delete`/`serialize`/`restore`), plus a real lifecycle scope gap (legacy memory resets every `forward()` call -- it is sequence-local, not cross-call session-persistent, despite the "session" framing). |
| B1 | memory contract | **Complete** | `docs/restart/hz0b_b1_memory_contract.md`. Six state tensors (`keys`, `values`, `confidence`, `age`, `protection_strength`, `write_metadata`), ten operations with real signatures, all 14 required design decisions made with stated rationale. Key deviation from legacy, motivated by the B0 audit: memory is now genuinely content-addressable (`keys` is real per-slot state, written and matched against) instead of legacy's fixed learned `slot_addresses` routing parameter that never connected to what was actually stored -- a direct, targeted response to B0's still-open "was hard routing itself the broken link" question. `reinforce`/`update`/`protect` kept as three separate operations (legacy collapsed all of this into one `momentum` knob, which its own config comments show was an active footgun). Soft addressing decided first, hard/STE routing deferred to a later explicitly-gated experiment rather than repeating legacy's unverified jump straight to it. Session lifecycle scoped honestly to sequence-local (matching what legacy code actually did, not the more ambitious "session" framing in the plan's Objective) with cross-call persistence named as explicit future work, not assumed. |
| B2 | isolated simulator | **Complete** | `reference/hz0b_memory_simulator.py` + `tests/reference/test_hz0b_memory_simulator.py`. All 14 of the plan's own listed B2 tasks pass (store/retrieve exact+noisy/multiple keys/similar keys/overwrite/reinforce/protect/overwrite-different/forget-by-age/overflow/reset+restore/chained retrieval/conflicting writes), plus delete, determinism, and learned-mode differentiability checks -- 20 tests total. Found 3 real bugs via the test run (not assumed correct on first write): a write-collision case that let protected memories get silently duplicated instead of actually protected, a non-differentiable similarity normalization that produced NaN gradients at empty (all-zero) slots, and a flawed test fixture that accidentally created near-duplicate keys. All fixed and re-verified. Exit gate ("the simulator passes predefined tests without any language model attached") satisfied -- no LM anywhere in the file. |
| B3 | memory semantics | **Complete** | `reference/hz0b_memory_semantics.py` + `tests/reference/test_hz0b_memory_semantics.py`. Directly answers the plan's own critique of B2 ("Do not rely on one similarity threshold for all behaviors" -- B2's write() used a single 0.95 cutoff for everything): separate, named thresholds (update-match 0.9, interference 0.6, staleness half-life) and graded (sigmoid) control signals -- write/update/reinforce/contradiction probability, erase_strength, soft slot_selection_distribution, memory_validity_confidence distinct from raw occupancy confidence -- plus all 5 required penalty functions (excessive writes, uncontrolled growth, protected-memory corruption, duplicate memories, unstable norms) as pure, independently-tested functions. 11/11 tests verify the exit gate directly: new writes, reinforcement, updates, deletion, and protection each produce measurably different signal values. Found a real test-fixture bug along the way (a "near-duplicate" test key was accidentally similar enough to trigger B2's own write-match threshold) and fixed it. |
| B4 | training curriculum | Not started | Synthetic tasks and schedules |
| B5 | fair baselines | Not started | Compare against no-memory and context-only |
| B6 | HZ-0A integration | Blocked | Wait for frozen HZ-0A checkpoint/spec |
| B7 | PMetal implementation | Blocked | After validated simulator + integration |
| B8 | evaluation suite | Not started | Must verify useful memory behaviors directly |

## Known Constraints

- Memory must stay distinct from recurrent state.
- Reset must be explicit.
- State must be serializable.
- No cross-user persistent storage in the first implementation.

## Historical Caution

- Legacy material suggests recall and protection may have looked promising.
- Legacy material also says overwrite remained unresolved.
- No historical claim is accepted until reproduced with new tests.

## Readiness Gates

- `STOP`: implementation before B0/B1
- `STOP`: any integration into HZ-0A before HZ-0A is frozen
- `GO LATER`: begin B0 after HZ-0A A1-A3 are established or when parallel research time is available
