# HZ-0E E5: HZ-0D Dependency Gate

Date: 2026-08-05. E5 is complete because every dependency named by the plan
is present and freshly exercised.

| Requirement | Evidence | Result |
|---|---|---|
| Frozen trained checkpoint | `outputs/hz0a_stage2_100m_hybrid_seed7/native_metal_checkpoint_best_full_holdout`; step `38,403`; `100,002,816` tokens seen; finite real forward | PASS |
| Exact topology | Independent loaded-weight count `301,178,112`; `d_model=768`; vocabulary `24,576` | PASS |
| Recurrence, memory, trigger, and fast state | HZ-0D D6/D7/D10 integration tests and the 1,024-token memory carry audit pass; state transitions are finite and deterministic | PASS |
| PMetal implementation | HZ-0D PMetal bridge round-trip tests pass against the Python reference | PASS |
| E0-E4 prerequisites | 36 E1-E4 regression tests pass for contract, routing, objectives, and fair baselines | PASS |

Fresh verification on the current worktree:

```text
20 passed: HZ-0D dependency, D6, D7, PMetal bridge, and D10 tests
36 passed: HZ-0E E1 contract, E2 routing, E3 objectives, and E4 baselines
```

The E4 result remains honestly negative for MoE quality at this stage; that
is an E6/E8/E10 risk, not an E5 dependency failure.

**E5 status: 100% complete. E6 is next.**
