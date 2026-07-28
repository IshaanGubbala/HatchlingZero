# HZ-0B Progress Tracker

Updated: July 28, 2026

## Mission

Add explicit, controllable, session-local associative memory to a finished and frozen HZ-0A base.

## Current Status

- Overall phase: not started
- Dependency status: blocked on stable, frozen HZ-0A
- Working assumption: all legacy HZ-0B behavior must be re-audited before reuse

## Phase Tracker

| Phase | Deliverable | Status | Notes |
| --- | --- | --- | --- |
| B0 | history audit | Not started | Recover supported memory lessons only |
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
