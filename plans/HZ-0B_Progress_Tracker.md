# HZ-0B Progress Tracker

Updated: July 29, 2026 (B0 history audit done)

## Mission

Add explicit, controllable, session-local associative memory to a finished and frozen HZ-0A base.

## Current Status

- Overall phase: B0 complete; B1+ not started
- Dependency status: B0/B1 (audit/contract) unblocked and in progress per the "GO LATER... when parallel research time is available" gate (HZ-0A A1-A3 have been done since early in the restart); B6+ (integration) still blocked on frozen HZ-0A, which is not yet frozen (Stage 2/100M still running)
- Working assumption: all legacy HZ-0B behavior must be re-audited before reuse -- **this was not a hypothetical caution**, see B0 below

## Phase Tracker

| Phase | Deliverable | Status | Notes |
| --- | --- | --- | --- |
| B0 | history audit | **Complete** | `docs/restart/hz0b_history_audit.md`. Headline finding: the legacy `HZ0B_FINAL_SUMMARY.md` ("4/4 memory gates validated, 100% recall, ready for production," commit `44ba07d` 2026-07-26) is directly contradicted by its own project's probe data -- all four gate probes (`archive/docs/hz0b/memory-probe-*-step425.json`) show `0.0` recall before and after fine-tuning on the real 110M-scale checkpoint, consistently across v1/v2/final iterations. The same-day `mem-fix-plan-2026-07-26.md` says this plainly ("held-out synthetic memory recall is still 0.0 -> 0.0 on every committed checkpoint") and is the trustworthy document of the pair. Real, salvageable: `session_scratchpad.py`'s design (orthogonal slot init, explicit reset, slot-local writes, hard-route diagnostics, oracle-bypass hooks) as a starting *contract* for B1, not as working code (it's PyTorch, needs an MLX port; its own "does it work" claim doesn't hold). Five concrete failure modes carried forward from the mem-fix-plan: 10x training-speed overhead from a per-token Python loop, joint-learning difficulty (routing+storage+readout all learned at once from few updates), no route-match diagnostic existed at the time of the 0.0 result, a corrupted checkpoint, and separately-inadequate training data. |
| B1 | memory contract | Not started | Must define tensors, operations, lifecycle |
| B2 | isolated simulator | Not started | No LM integration yet |
| B3 | memory semantics | Not started | Separate overwrite, reinforce, protect, delete |
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
