# HZ-0D Progress Tracker

Updated: July 28, 2026

## Mission

Add adaptive recurrence and bounded dynamic compute to a completed HZ-0C model.

## Current Status

- Overall phase: not started
- Dependency status: blocked on stable HZ-0C for integration
- Possible independent work: tiny adaptive-compute simulator only

## Phase Tracker

| Phase | Deliverable | Status | Notes |
| --- | --- | --- | --- |
| D0 | history audit | Not started | Recover adaptive-depth intentions |
| D1 | compute contract | Not started | Must define bounds and deterministic inference |
| D2 | simulator | Not started | Synthetic difficulty-controlled tasks |
| D3 | controller training | Not started | Budget-aware controller behavior |
| D4 | fair baselines | Not started | Compare against fixed-depth/equal-FLOP models |
| D5 | wait for stable HZ-0C | Blocked | Explicit dependency gate |
| D6 | integrate with frozen HZ-0C | Blocked | Only after D1-D4 and HZ-0C stability |

## Hard Constraints

- No unbounded compute.
- Inference policy must be deterministic and measurable.
- Quality claims must be normalized by compute, not only by parameter count.

## Readiness Gates

- `STOP`: adaptive integration before HZ-0C exists
- `GO LATER`: simulator work can happen in isolation after earlier stages are healthier
