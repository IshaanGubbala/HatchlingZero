# HZ-0E Progress Tracker

Updated: July 28, 2026

## Mission

Add a small, Apple-Silicon-friendly micro-MoE layer to a completed HZ-0D model.

## Current Status

- Overall phase: not started
- Dependency status: blocked on stable HZ-0D for integration
- Possible independent work: router simulator and expert-contract design only

## Phase Tracker

| Phase | Deliverable | Status | Notes |
| --- | --- | --- | --- |
| E0 | history audit | Not started | Recover intended expert/routing assumptions |
| E1 | expert contract | Not started | Count, routing, capacity, active params |
| E2 | router simulator | Not started | Must avoid expert collapse |
| E3 | routing losses | Not started | Balance utilization with task loss |
| E4 | fair baselines | Not started | Same active params and same total params comparisons |
| E5 | wait for stable HZ-0D | Blocked | Explicit dependency gate |
| E6 | integrate with frozen HZ-0D | Blocked | Upper-layer MLP replacement first |

## Hard Constraints

- Keep expert count small.
- Keep routing deterministic at inference.
- Report total params and active params separately.
- Do not place experts inside recurrent-state or memory-write internals initially.

## Readiness Gates

- `STOP`: integration before HZ-0D stability
- `GO LATER`: isolated router design/simulation once earlier stages are established
