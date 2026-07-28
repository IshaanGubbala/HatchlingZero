# HZ-0C Progress Tracker

Updated: July 28, 2026

## Mission

Add session-local fast weights to a completed HZ-0B system without altering permanent pretrained weights.

## Current Status

- Overall phase: not started
- Dependency status: blocked on HZ-0B contract and stable integration

## Phase Tracker

| Phase | Deliverable | Status | Notes |
| --- | --- | --- | --- |
| C0 | history audit | Not started | Recover only tested fast-weight ideas |
| C1 | fast-weight contract | Not started | Define state, update rule, reset semantics |
| C2 | isolated simulator | Not started | Must prove temporary adaptation cleanly |
| C3 | update-mechanism choice | Not started | Compare at least three approaches |
| C4 | fair baselines | Not started | Must beat simpler alternatives honestly |
| C5 | HZ-0B integration | Blocked | Wait for HZ-0B stability |
| C6 | PMetal implementation | Blocked | After isolated validation |
| C7 | full evaluation | Not started | Adaptation speed, reset fidelity, interference |

## Design Rules

- Fast weights must reset cleanly.
- They must remain distinct from HZ-0B memory and ordinary hidden state.
- Start with low-rank fast adapters, not full dense session matrices.

## Readiness Gates

- `STOP`: any implementation before HZ-0B is stable
- `GO LATER`: isolated history/spec work can begin once HZ-0A is well underway
